import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from patchforge.brain import BrainAgent, extract_code, parse_json
from patchforge.contracts import AgentTask, SandboxResult
from patchforge.jobs import JobRequest, JobRunner
from patchforge.sandbox import FakeSandbox, LocalSandbox
from patchforge.skills import SkillStore
from tests.conftest import FakeGitHub, FakeRegistry, git

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "tests" / "fixtures" / "demo-repo"
V2 = Path(__file__).resolve().parent / "fixtures" / "demo_v2"
ARCHITECT, WORKER = "Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507"

SPEC = {
    "library": "pydantic", "from_version": "1.10.13", "to_version": "2.14.1",
    "changes": [
        {"old": "Field(..., regex=...)", "new": "Field(..., pattern=...)", "note": "renamed", "source": "https://docs.pydantic.dev/migration/"},
        {"old": "User.parse_obj(", "new": "User.model_validate(", "note": "renamed", "source": "https://docs.pydantic.dev/migration/"},
    ],
}


def reply(content, t_in=1000, t_out=200, finish="stop"):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=t_in, completion_tokens=t_out),
                           choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))])


class FakeLLM:
    """Answers each kind of BrainAgent prompt; patches return the hand-migrated v2 files."""

    def __init__(self, patches=None):
        self.calls = []
        self.lock = threading.Lock()
        self.patches = patches or {
            "app/models.py": (V2 / "models.py").read_text(),
            "app/service.py": (V2 / "service.py").read_text(),
        }
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, temperature, max_tokens):
        prompt = messages[0]["content"]
        with self.lock:
            self.calls.append((model, prompt))
        if "Work out which library" in prompt:
            return reply('<think>hmm</think>{"library": "pydantic", "import_name": "pydantic", '
                         '"from_version": "1.10.13", "to_version": "latest"}')
        if "Extract every breaking change" in prompt:
            return reply("Here you go:\n" + json.dumps(SPEC))
        for rel, code in self.patches.items():
            if f"FILE {rel}:" in prompt:
                return reply(f"```python\n{code}```")
        return reply("NO_CHANGE")

    def models_for(self, needle):
        return [m for m, p in self.calls if needle in p]


class FakeTavily:
    def __init__(self):
        self.calls = 0

    def search(self, query, max_results, search_depth, **kw):
        self.calls += 1
        self.queries = getattr(self, "queries", []) + [query]
        return {"results": [{"url": "https://docs.pydantic.dev/migration/", "content": "regex is now pattern...",
                             "raw_content": "# Migration guide\nFull page: `regex` was renamed to `pattern`."}]}


def agent(tmp_path, llm=None, tavily=None, **kw):
    kw.setdefault("registry", FakeRegistry())
    return BrainAgent(model=ARCHITECT, worker_model=WORKER, llm=llm or FakeLLM(), tavily=tavily or FakeTavily(),
                      skills=SkillStore(tmp_path / "skills"), workers=4, **kw)


@pytest.fixture
def demo(tmp_path):
    work = tmp_path / "demo"
    shutil.copytree(DEMO, work)
    return work


def task():
    return AgentTask(repo="me/demo", instruction="Upgrade pydantic to v2", base_branch="main")


def test_parse_helpers():
    assert parse_json('<think>{"x": 0}</think> sure: {"a": 1}') == {"a": 1}
    assert extract_code("```python\nx = 1\n```") == "x = 1\n"
    assert extract_code("NO_CHANGE") is None


def test_run_bumps_pin_patches_in_parallel_and_routes_models(tmp_path, demo):
    llm, events = FakeLLM(), []
    out = agent(tmp_path, llm=llm).run(task(), demo, FakeSandbox([SandboxResult(passed=True)]), events.append)

    assert (demo / "requirements.txt").read_text() == "pydantic==2.14.1\n"
    assert "model_validate" in (demo / "app/service.py").read_text()
    assert out.changed_files == ["app/models.py", "app/service.py", "requirements.txt"]
    assert out.test_results.passed
    # Planning on the architect, per-file edits on the cheap worker model.
    assert set(llm.models_for("Work out which library")) == {ARCHITECT}
    assert set(llm.models_for("Update this file")) == {WORKER}
    priced = [e for e in events if e.model]
    assert priced and all(e.cost_usd > 0 for e in priced)
    assert any(e.stage == "skill" and "saved skill" in e.message.lower() for e in events)


def test_retry_escalates_to_the_fixer_model(tmp_path, demo):
    fail = SandboxResult(passed=False, failed_tests=["tests.test_users"], error_log="app/models.py: boom")
    llm = FakeLLM()
    out = agent(tmp_path, llm=llm).run(task(), demo, FakeSandbox([fail, SandboxResult(passed=True)]), lambda e: None)
    patch_models = llm.models_for("Update this file")
    assert patch_models.count(WORKER) == 3  # first pass on both files, then the retry: the fixer is the worker model by default
    assert out.test_results.passed


def test_skill_is_reused_on_second_repo(tmp_path, demo):
    tavily = FakeTavily()
    first = agent(tmp_path, tavily=tavily)
    first.run(task(), demo, FakeSandbox([SandboxResult(passed=True)]), lambda e: None)
    assert tavily.calls == 1

    second_repo = tmp_path / "demo2"
    shutil.copytree(DEMO, second_repo)
    llm, events = FakeLLM(), []
    out = agent(tmp_path, llm=llm, tavily=tavily).run(task(), second_repo, FakeSandbox([SandboxResult(passed=True)]),
                                                      events.append)
    assert tavily.calls == 1  # no new web search
    assert not llm.models_for("Extract every breaking change")  # no research call
    assert any(e.stage == "skill" and "reusing" in e.message.lower() for e in events)
    assert "Reused saved skill" in out.summary
    assert SkillStore(tmp_path / "skills").all()[0].uses == 1


def test_no_skill_saved_when_tests_fail(tmp_path, demo):
    fail = SandboxResult(passed=False, error_log="nope")
    agent(tmp_path, max_retries=1).run(task(), demo, FakeSandbox([fail]), lambda e: None)
    assert SkillStore(tmp_path / "skills").all() == []


def test_cut_off_reply_leaves_file_alone(tmp_path, demo):
    llm = FakeLLM()
    original = llm.create

    def create(model, messages, temperature, max_tokens):
        r = original(model, messages, temperature, max_tokens)
        if "Update this file" in messages[0]["content"]:
            r.choices[0].finish_reason = "length"
        return r

    llm.chat.completions.create = create
    before = (demo / "app/models.py").read_text()
    agent(tmp_path, llm=llm, max_retries=0).run(task(), demo, FakeSandbox([SandboxResult(passed=False)]), lambda e: None)
    assert (demo / "app/models.py").read_text() == before


@pytest.fixture
def demo_remote(tmp_path):
    seed = tmp_path / "seed"
    shutil.copytree(DEMO, seed)
    git("init", "-q", "-b", "main", cwd=seed)
    git("add", ".", cwd=seed)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=seed)
    bare = tmp_path / "demo.git"
    git("clone", "-q", "--bare", str(seed), str(bare))
    return bare


@pytest.mark.skipif(not shutil.which("python3.12"), reason="needs python3.12 (pydantic 1.10 doesn't run on 3.14)")
def test_full_job_on_demo_repo_opens_pr_then_reuses_skill(tmp_path, demo_remote):
    """Real pip installs and real pytest runs of the demo repo, before and after; fake LLM and GitHub."""
    github = FakeGitHub(demo_remote)
    tavily = FakeTavily()
    runner = JobRunner(github=github, sandbox_factory=lambda: LocalSandbox(python="python3.12"),
                       agent_factory=lambda: agent(tmp_path, tavily=tavily), clone_depth=None)

    job = runner.run_sync(JobRequest(repo="me/demo", instruction="Upgrade pydantic to v2"))
    assert job.status == "succeeded", job.error
    baseline = next(e for e in job.events if e["message"].startswith("Before changes"))
    assert "All tests passed" in baseline["message"]
    assert "pydantic==2.14.1" in git("show", f"{job.branch}:requirements.txt", cwd=demo_remote)
    assert "Upgrade `pydantic`" in github.prs[0]["body"]

    again = runner.run_sync(JobRequest(repo="me/demo", instruction="Upgrade pydantic to v2"))
    assert again.status == "succeeded", again.error
    assert tavily.calls == 1
    assert again.totals()["cost_usd"] < job.totals()["cost_usd"]


def test_test_files_are_never_patched(tmp_path, demo):
    from patchforge.brain import is_test_file
    assert is_test_file(Path("tests/test_users.py")) and is_test_file(Path("pkg/conftest.py"))
    assert is_test_file(Path("app/models_test.py")) and not is_test_file(Path("app/models.py"))
    files = agent(tmp_path).find_files(demo, {"import_name": "pydantic"}, SPEC)
    assert [str(p.relative_to(demo)) for p in files] == ["app/models.py", "app/service.py"]


def test_retry_shows_the_failing_test_to_the_model(tmp_path, demo):
    from patchforge.brain import related_tests
    fail = SandboxResult(passed=False, failed_tests=["tests.test_users::test_from_record"],
                         error_log="tests/test_users.py:40: in test_from_record\nE AttributeError")
    text = related_tests(demo, fail)
    assert text.count("TEST FILE tests/test_users.py") == 1 and "def test_from_record" in text

    llm = FakeLLM()
    agent(tmp_path, llm=llm).run(task(), demo, FakeSandbox([fail, SandboxResult(passed=True)]), lambda e: None)
    retry_prompts = [p for m, p in llm.calls if "Update this file" in p and "previous edit" in p]
    assert retry_prompts and all("TEST FILE tests/test_users.py" in p and "the test wins" in p for p in retry_prompts)


def test_install_failure_stops_the_retries(tmp_path, demo):
    broken = SandboxResult(passed=False, error_log="[sandbox] dependency install failed\nResolutionImpossible")
    llm, events = FakeLLM(), []
    agent(tmp_path, llm=llm).run(task(), demo, FakeSandbox([broken]), events.append)
    assert not [p for m, p in llm.calls if "previous edit" in p]  # no wasted re-patching
    assert any("don't install" in e.message for e in events)


EXPRESS_LOG = """TAP version 13
# /work/node_modules/path-to-regexp/dist/index.js:108
#                     throw new PathError(`Missing parameter name at index ${index}`, str);
#                     ^
# PathError [TypeError]: Missing parameter name at index 8: /files/*; visit https://git.new/pathToRegexpError for info
#     at consumeUntil (/work/node_modules/path-to-regexp/dist/index.js:108:27)
not ok 1 - test/server.test.js"""


def test_key_error_line_is_what_a_developer_would_search():
    from patchforge.brain import key_error_line
    assert key_error_line(EXPRESS_LOG).startswith("PathError [TypeError]: Missing parameter name at index 8: /files/*")
    assert key_error_line("E   AttributeError: 'types.SimpleNamespace' object has no attribute 'get'") == \
        "AttributeError: 'types.SimpleNamespace' object has no attribute 'get'"
    assert key_error_line("all good\nnothing here") is None


def test_research_uses_full_pages_and_retries_when_empty(tmp_path):
    from patchforge.brain import BrainAgent
    prompts = []

    class EmptyThenSpec(FakeLLM):
        def create(self, model, messages, temperature, max_tokens):
            prompt = messages[0]["content"]
            if "Extract every breaking change" in prompt:
                prompts.append(prompt)
                return reply(json.dumps({"changes": []}) if len(prompts) == 1 else json.dumps(SPEC))
            return super().create(model, messages, temperature, max_tokens)

    tavily, events = FakeTavily(), []
    spec = agent(tmp_path, llm=EmptyThenSpec(), tavily=tavily).research(
        {"library": "pydantic", "from_version": "1.10.13", "to_version": "2.14.1", "ecosystem": "python"}, events.append)
    assert len(spec["changes"]) == 2 and tavily.calls == 2
    assert "Full page: `regex` was renamed" in prompts[0]  # the page's full text, not just the snippet
    assert "Python package" in tavily.queries[0]
    assert any("trying another search" in e.message for e in events)


def test_retry_searches_the_web_for_the_error(tmp_path, demo):
    fail = SandboxResult(passed=False, error_log=EXPRESS_LOG.replace("/files/*", "/files/x"))
    llm, tavily = FakeLLM(), FakeTavily()
    agent(tmp_path, llm=llm, tavily=tavily).run(task(), demo, FakeSandbox([fail, fail, SandboxResult(passed=True)]),
                                                lambda e: None)
    error_queries = [q for q in tavily.queries if "Missing parameter name" in q]
    assert len(error_queries) == 1  # the same error is looked up once, not on every retry
    retry_prompts = [p for m, p in llm.calls if "previous edit" in p]
    assert retry_prompts and all("trust them over your memory" in p for p in retry_prompts)


def test_slow_code_model_falls_back_to_the_research_model(tmp_path, demo):
    import httpx
    from openai import APITimeoutError

    class SlowWorker(FakeLLM):
        def create(self, model, messages, temperature, max_tokens):
            if model == WORKER:
                with self.lock:
                    self.calls.append((model, messages[0]["content"]))
                raise APITimeoutError(request=httpx.Request("POST", "https://x"))
            return super().create(model, messages, temperature, max_tokens)

    llm, events = SlowWorker(), []
    out = agent(tmp_path, llm=llm).run(task(), demo, FakeSandbox([SandboxResult(passed=True)]), events.append)
    assert out.test_results.passed and "model_validate" in (demo / "app/service.py").read_text()
    assert any("didn't answer within" in e.message for e in events)
    assert ARCHITECT in llm.models_for("Update this file")
