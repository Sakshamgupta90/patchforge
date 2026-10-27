import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from patchforge import deps
from patchforge.agent_stub import StubAgent
from patchforge.brain import BrainAgent, is_test_file
from patchforge.contracts import AgentTask, SandboxResult
from patchforge.jobs import JobRequest, JobRunner, clean_path
from patchforge.registry import latest_version
from patchforge.sandbox import AlwaysPassSandbox, FakeSandbox
from patchforge.skills import SkillStore
from tests.conftest import SAMPLE, FakeGitHub, FakeRegistry, git


# --- package.json version bumps -------------------------------------------------------

def test_npm_bump_keeps_style_and_formatting(tmp_path):
    (tmp_path / "package.json").write_text(
        '{\n    "name": "x",\n    "dependencies": {\n        "express": "^4.17.1",\n        "lodash": "4.17.15"\n    },\n'
        '    "devDependencies": {\n        "typescript": "~4.9.5"\n    }\n}\n')
    assert deps.bump(tmp_path, "express", "5.1.0", ecosystem="node") == ["package.json"]
    assert deps.bump(tmp_path, "lodash", "4.17.21", ecosystem="node") == ["package.json"]
    assert deps.bump(tmp_path, "typescript", "5.9.3", ecosystem="node") == ["package.json"]
    text = (tmp_path / "package.json").read_text()
    data = json.loads(text)
    assert data["dependencies"] == {"express": "^5.1.0", "lodash": "4.17.21"}
    assert data["devDependencies"]["typescript"] == "~5.9.3"
    assert text.startswith('{\n    "name"') and text.endswith("}\n")  # 4-space indent and newline kept


def test_npm_bump_unknown_version_and_missing_package(tmp_path):
    (tmp_path / "package.json").write_text('{"dependencies": {"vue": "2.6.14"}}')
    assert deps.bump(tmp_path, "vue", "latest", ecosystem="node") == ["package.json"]
    assert json.loads((tmp_path / "package.json").read_text())["dependencies"]["vue"] == "latest"
    assert deps.bump(tmp_path, "react", "19.2.0", ecosystem="node") == []


# --- registry lookups -------------------------------------------------------------------

def registry(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_latest_version_npm_and_pypi():
    def handler(req):
        if req.url.host == "registry.npmjs.org":
            assert req.url.path == "/@angular/core"
            return httpx.Response(200, json={"dist-tags": {"latest": "20.3.1"},
                                             "versions": {"19.2.0": {}, "19.2.14": {}, "20.3.1": {}, "21.0.0-rc.1": {}}})
        return httpx.Response(200, json={"info": {"version": "2.14.1"},
                                         "releases": {"1.10.13": [{}], "2.9.2": [{}], "2.14.1": [{}], "2.15.0b1": [{}], "3.0.0": []}})

    c = registry(handler)
    assert latest_version("node", "@angular/core", client=c) == "20.3.1"
    assert latest_version("node", "@angular/core", major="19", client=c) == "19.2.14"
    assert latest_version("python", "pydantic", client=c) == "2.14.1"
    assert latest_version("python", "pydantic", major="2", client=c) == "2.14.1"  # pre-release skipped
    assert latest_version("python", "pydantic", major="3", client=c) is None  # no files = not released


def test_latest_version_network_error():
    def boom(req):
        raise httpx.ConnectError("offline")

    assert latest_version("node", "express", client=registry(boom)) is None


# --- the agent on a Node project ---------------------------------------------------------

def reply(content):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=500, completion_tokens=100),
                           choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))])


class NodeLLM:
    def __init__(self):
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, temperature, max_tokens):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if "Work out which library" in prompt:
            assert "package name on npm" in prompt
            return reply('{"library": "express", "import_name": "express", "from_version": "4.17.1", '
                         '"to_version": "latest", "companions": ["body-parser", "express"]}')
        if "Extract every breaking change" in prompt:
            return reply(json.dumps({"changes": [
                {"old": "app.del(", "new": "app.delete(", "note": "removed", "source": "https://expressjs.com/en/guide/migrating-5.html"},
                {"old": "res.sendfile(", "new": "res.sendFile(", "note": "removed", "source": "https://expressjs.com"}]}))
        if "FILE server.js:" in prompt:
            assert "```javascript" in prompt
            return reply("```javascript\nconst express = require('express');\nconst app = express();\napp.delete('/x', () => {});\n```")
        return reply("NO_CHANGE")


@pytest.fixture
def node_project(tmp_path):
    root = tmp_path / "proj"
    (root / "test").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "package.json").write_text(json.dumps({"name": "api", "scripts": {"test": "node --test"},
                                                   "dependencies": {"express": "4.17.1", "body-parser": "1.19.0"}}, indent=2) + "\n")
    (root / ".nvmrc").write_text("16\n")
    (root / "server.js").write_text("const express = require('express');\nconst app = express();\napp.del('/x', () => {});\n")
    (root / "lib" / "util.ts").write_text("export const add = (a: number, b: number) => a + b;\n")
    (root / "test" / "server.test.js").write_text("const express = require('express');\n")
    (root / "next.config.js").write_text("module.exports = { target: 'serverless' };\n")
    return root


def test_node_find_files_skips_tests_and_includes_runtime_pin(tmp_path, node_project):
    agent = BrainAgent(llm=NodeLLM(), tavily=None, skills=SkillStore(tmp_path / "s"), registry=FakeRegistry())
    info = {"ecosystem": "node", "import_name": "express", "companions": []}
    files = [str(p.relative_to(node_project)) for p in agent.find_files(node_project, info, {"changes": []})]
    assert files == ["server.js"]  # tests are never patched; the runtime pin is handled by code, not the model
    assert is_test_file(Path("src/app.spec.ts")) and is_test_file(Path("__tests__/a.js"))


def test_node_run_resolves_versions_and_bumps_companions(tmp_path, node_project):
    registry = FakeRegistry(
        versions={"express": "5.1.0", "body-parser": lambda major, runtime: None if major else "2.2.0"},
        runtimes={"express": "18"})

    tavily = SimpleNamespace(search=lambda **k: {"results": [{"url": "https://expressjs.com", "content": "app.del removed"}]})
    events = []
    out = BrainAgent(llm=NodeLLM(), tavily=tavily, skills=SkillStore(tmp_path / "s"), registry=registry) \
        .run(AgentTask(repo="me/api", instruction="Upgrade Express", base_branch="main"), node_project,
             FakeSandbox([SandboxResult(passed=True)]), events.append)

    pkg = json.loads((node_project / "package.json").read_text())
    assert pkg["dependencies"] == {"express": "5.1.0", "body-parser": "2.2.0"}
    assert "app.delete" in (node_project / "server.js").read_text()
    assert "test/server.test.js" not in out.changed_files
    assert set(out.changed_files) == {"package.json", "server.js", ".nvmrc"}
    assert (node_project / ".nvmrc").read_text() == "18\n"  # express 5 needs Node 18; the project pinned 16
    assert any("raised the project from 16 to 18" in e.message for e in events)
    # companions run on the raised runtime; body-parser isn't an express sibling, so no major-version lock
    assert ("node", "body-parser", None, "18") in registry.calls
    assert not any(c[1] == "body-parser" and c[2] for c in registry.calls)
    assert any("Latest express on npm: 5.1.0" in e.message for e in events)
    assert "with `body-parser`" in out.summary


def test_config_file_found_by_option_name(tmp_path):
    root = tmp_path / "ts"
    root.mkdir()
    (root / "package.json").write_text('{"devDependencies": {"typescript": "4.9.5"}}')
    (root / "tsconfig.json").write_text('{\n  "compilerOptions": {\n    "suppressImplicitAnyIndexErrors": true\n  }\n}\n')
    (root / "index.ts").write_text("export const x = 1;\n")
    agent = BrainAgent(llm=NodeLLM(), tavily=None, skills=SkillStore(tmp_path / "s"), registry=FakeRegistry())
    spec = {"changes": [{"old": '"suppressImplicitAnyIndexErrors": true', "new": "remove it"}]}
    files = agent.find_files(root, {"ecosystem": "node", "import_name": "typescript", "companions": []}, spec)
    assert [p.name for p in files] == ["tsconfig.json"]


# --- folders inside a repo ---------------------------------------------------------------

def test_clean_path():
    assert clean_path("") == ""
    assert clean_path("./python-flask/") == "python-flask"
    assert clean_path("services\\\\api") == "services/api"
    for bad in ("../etc", "a/../../b", "/abs", ".git/hooks"):
        with pytest.raises(ValueError):
            clean_path(bad)


@pytest.fixture
def monorepo(tmp_path):
    seed = tmp_path / "seed"
    shutil.copytree(SAMPLE, seed / "services" / "calc")
    (seed / "README.md").write_text("monorepo\n")
    git("init", "-q", "-b", "main", cwd=seed)
    git("add", ".", cwd=seed)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=seed)
    bare = tmp_path / "mono.git"
    git("clone", "-q", "--bare", str(seed), str(bare))
    return bare


def test_job_runs_in_a_folder(monorepo):
    seen = []

    class RecordingSandbox(AlwaysPassSandbox):
        def run_tests(self, folder):
            seen.append(Path(folder).name)
            return super().run_tests(folder)

    github = FakeGitHub(monorepo)
    runner = JobRunner(github=github, sandbox_factory=RecordingSandbox, agent_factory=lambda: StubAgent(delay_s=0),
                       clone_depth=None)
    job = runner.run_sync(JobRequest(repo="me/mono", instruction="Upgrade x", path="./services/calc/"))
    assert job.status == "succeeded", job.error
    # "before" is the snapshot of the folder used for the parallel before-check
    assert job.request.path == "services/calc" and set(seen) <= {"calc", "before"} and "calc" in seen
    tree = git("ls-tree", "-r", "--name-only", job.branch, cwd=monorepo)
    assert "services/calc/PATCHFORGE_STUB.md" in tree  # the agent worked inside the folder
    assert github.prs[0]["title"].endswith("(services/calc)")
    assert job.snapshot()["path"] == "services/calc"


def test_missing_folder_fails_cleanly(monorepo):
    runner = JobRunner(github=FakeGitHub(monorepo), sandbox_factory=AlwaysPassSandbox,
                       agent_factory=lambda: StubAgent(delay_s=0), clone_depth=None)
    job = runner.run_sync(JobRequest(repo="me/mono", instruction="x", path="nope"))
    assert job.status == "failed" and "folder 'nope' doesn't exist" in job.error


# --- runtime-aware versions ------------------------------------------------------------------

def test_runtime_range_parsing():
    from patchforge.registry import node_allows, python_allows
    assert python_allows(">=3.11", "3.11") and not python_allows(">=3.11", "3.10")
    assert python_allows(None, "3.8") and python_allows(">=3.7,<3.11", "3.10") and not python_allows(">=3.7,<3.11", "3.12")
    assert node_allows(">=18", "20") and not node_allows(">=18", "16")
    assert node_allows("^18.18.0 || ^19.8.0 || >= 20.0.0", "22") and node_allows("^18.18.0 || >=20", "18")
    assert not node_allows("^18.18.0 || ^19.8.0 || >= 20.0.0", "17")
    assert node_allows("14 - 16", "16") and not node_allows("14 - 16", "18")
    assert node_allows("18.x", "18") and not node_allows("18.x", "20") and node_allows("*", "10")


def test_registry_picks_versions_that_fit_the_runtime():
    from patchforge.registry import PackageRegistry

    def handler(req):
        if req.url.host == "registry.npmjs.org":
            return httpx.Response(200, json={"dist-tags": {"latest": "16.0.0"}, "versions": {
                "15.5.0": {"engines": {"node": ">=18.18.0"}}, "16.0.0": {"engines": {"node": ">=20.9.0"}}}})
        return httpx.Response(200, json={"info": {"version": "2.5.4"}, "releases": {
            "2.4.6": [{"requires_python": ">=3.11"}], "2.5.4": [{"requires_python": ">=3.12"}],
            "2.5.5": [{"requires_python": ">=3.12", "yanked": True}]}})

    reg = PackageRegistry(registry(handler))
    assert reg.latest_version("python", "numpy") == "2.5.4"
    assert reg.latest_version("python", "numpy", runtime="3.11") == "2.4.6"
    assert reg.min_runtime("python", "numpy", "2.5.4") == "3.12"
    assert reg.latest_version("node", "next", runtime="18") == "15.5.0"
    assert reg.min_runtime("node", "next", "16.0.0") == "20"


def test_python_runtime_is_raised_for_the_new_version(tmp_path):
    root = tmp_path / "py"
    root.mkdir()
    (root / ".python-version").write_text("3.10\n")
    (root / "requirements.txt").write_text("pandas==1.5.3\nnumpy==1.24.4\n")
    reg = FakeRegistry(versions={"numpy": lambda major, runtime: {"3.11": "2.4.6"}.get(runtime)},
                       runtimes={"pandas": "3.11"})
    agent = BrainAgent(llm=NodeLLM(), tavily=None, skills=SkillStore(tmp_path / "s"), registry=reg)
    events = []
    targets, files = agent.plan_versions(root, {"ecosystem": "python", "library": "pandas", "companions": ["numpy"]},
                                         "3.1.0", events.append)
    assert targets == {"pandas": "3.1.0", "numpy": "2.4.6"}
    assert files == [".python-version"] and (root / ".python-version").read_text() == "3.11\n"


def test_runtime_left_alone_when_new_enough(tmp_path):
    root = tmp_path / "py"
    root.mkdir()
    agent = BrainAgent(llm=NodeLLM(), tavily=None, skills=SkillStore(tmp_path / "s"),
                       registry=FakeRegistry(runtimes={"flask": "3.9"}))
    _, files = agent.plan_versions(root, {"ecosystem": "python", "library": "flask", "companions": []}, "3.1.3",
                                   lambda e: None)
    assert files == [] and not (root / ".python-version").exists()  # sandbox default 3.12 is already enough


def test_check_weakening_edits_are_rejected(tmp_path):
    from patchforge.brain import weakening
    assert weakening('{"strict": true}', '{"strict": true, "noImplicitAny": false}')
    assert weakening("x = f()", "x = f()  # type: ignore")
    assert weakening("const a = b;", "// @ts-ignore\nconst a = b;")
    assert weakening("def test_x(): ...", "@pytest.mark.skip\ndef test_x(): ...")
    assert weakening('{"strict": false}', '{"strict": false, "target": "es2022"}') is None  # already there
    assert weakening("x = 1", "x = 2") is None


def test_only_lockstep_siblings_share_a_major():
    from patchforge.brain import siblings
    assert siblings("react", "react-dom") and siblings("@angular/core", "@angular/forms") and siblings("vue", "vue-router")
    assert not siblings("typescript", "@types/node") and not siblings("pandas", "numpy") and not siblings("next", "react")


def test_files_named_in_errors_can_be_patched(tmp_path):
    from patchforge.brain import files_in_error
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "index.ts").write_text("export const x = 1;\n")
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "a.test.ts").write_text("x\n")
    log = "src/index.ts(12,10): error TS7053: Element implicitly has an 'any' type\n/work/test/a.test.ts:3 boom\n"
    assert [p.name for p in files_in_error(tmp_path, log, "node")] == ["index.ts"]  # never the tests


def test_files_in_error_keep_the_callers_path_spelling(tmp_path):
    """On macOS /var is a symlink to /private/var; returned paths must be relative_to() the given workdir."""
    from patchforge.brain import files_in_error
    real = tmp_path / "real"
    (real / "app").mkdir(parents=True)
    (real / "app" / "views.py").write_text("x = 1\n")
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    log = f"File \"{real}/app/views.py\", line 1\nImportError: boom\n"
    [hit] = files_in_error(alias, log, "python")
    assert hit.relative_to(alias) == Path("app/views.py")


# --- conflicting pins, resolved from registry data -------------------------------------------

def test_npm_ranges():
    from patchforge.registry import npm_satisfies as ok
    assert ok("17.0.2", "^16.0.0 || ^17.0.0") and not ok("19.2.0", "^16.0.0 || ^17.0.0")
    assert ok("4.17.21", "^4.17.1") and not ok("5.0.0", "^4.17.1")
    assert ok("0.2.5", "^0.2.3") and not ok("0.3.0", "^0.2.3")
    assert ok("1.2.9", "~1.2.3") and not ok("1.3.0", "~1.2.3")
    assert ok("18.2.0", ">=16.8 <19") and not ok("19.0.0", ">=16.8 <19")
    assert ok("2.5.0", "1.2 - 2.x") and not ok("3.0.0", "1.2 - 2.x")
    assert ok("5.0.0", "*") and ok("1.0.0", "") and ok("1.0.0", "npm:other@1")
    assert ok("16.4.0", "16.x") and not ok("17.0.0", "16.x")


def test_python_dependencies_follow_markers(tmp_path):
    from patchforge.registry import PackageRegistry

    def handler(req):
        assert req.url.path == "/pypi/Flask/3.1.3/json"
        return httpx.Response(200, json={"info": {"requires_dist": [
            "Werkzeug>=3.1.0", "Jinja2>=3.1.2", "itsdangerous>=2.2.0", "click>=8.1.3",
            "importlib-metadata>=3.6; python_version < '3.10'", "asgiref>=3.2; extra == 'async'"]}})

    reg = PackageRegistry(registry(handler))
    names = dict(reg.dependencies("python", "Flask", "3.1.3", runtime="3.12"))
    assert names == {"Werkzeug": ">=3.1.0", "Jinja2": ">=3.1.2", "itsdangerous": ">=2.2.0", "click": ">=8.1.3"}
    assert "importlib-metadata" in dict(reg.dependencies("python", "Flask", "3.1.3", runtime="3.9"))


def test_flask_upgrade_moves_the_pins_it_conflicts_with(tmp_path):
    """The real failure: "Upgrade Flask to the latest version" bumped Flask only, and pip refused Jinja2==2.11.3."""
    root = tmp_path / "flask"
    root.mkdir()
    (root / "requirements.txt").write_text(
        "Flask==1.1.4\nWerkzeug==1.0.1\nJinja2==2.11.3\nMarkupSafe==1.1.1\nitsdangerous==1.1.0\nrequests==2.32.0\n")
    reg = FakeRegistry(versions={"Werkzeug": "3.1.9", "Jinja2": "3.1.6", "MarkupSafe": "3.0.4", "itsdangerous": "2.2.0"})
    reg.pins = {"Werkzeug": "1.0.1", "Jinja2": "2.11.3", "MarkupSafe": "1.1.1", "itsdangerous": "1.1.0", "requests": "2.32.0"}
    reg.deps = {("Flask", "3.1.3"): [("Werkzeug", ">=3.1.0"), ("Jinja2", ">=3.1.2"), ("itsdangerous", ">=2.2.0")],
                ("Jinja2", "3.1.6"): [("MarkupSafe", ">=2.0")],
                ("Werkzeug", "3.1.9"): [("MarkupSafe", ">=2.1.1")]}
    agent = BrainAgent(llm=NodeLLM(), tavily=None, skills=SkillStore(tmp_path / "s"), registry=reg)
    events = []
    targets, _ = agent.plan_versions(root, {"ecosystem": "python", "library": "Flask", "companions": []}, "3.1.3",
                                     events.append)
    assert targets == {"Flask": "3.1.3", "Werkzeug": "3.1.9", "Jinja2": "3.1.6", "itsdangerous": "2.2.0",
                       "MarkupSafe": "3.0.4"}  # requests isn't touched: nothing conflicts with it
    assert any("Also upgrading Jinja2 to 3.1.6: Flask 3.1.3 needs Jinja2 >=3.1.2" in e.message for e in events)


def test_pinned_package_that_rejects_the_new_version_moves_too(tmp_path):
    root = tmp_path / "web"
    root.mkdir()
    (root / "package.json").write_text(json.dumps({"dependencies": {
        "react": "17.0.2", "react-dom": "17.0.2", "react-scripts": "4.0.3", "lodash": "4.17.21"}}))
    reg = FakeRegistry(versions={"react": "19.2.0", "react-dom": "19.2.0", "react-scripts": "5.0.1"})
    reg.pins = {"react-scripts": "4.0.3", "lodash": "4.17.21"}
    reg.deps = {("react-scripts", "4.0.3"): [("react", "^16 || ^17")],  # the pinned version refuses React 19
                ("react-dom", "19.2.0"): [("react", "^19.2.0")]}
    agent = BrainAgent(llm=NodeLLM(), tavily=None, skills=SkillStore(tmp_path / "s"), registry=reg)
    targets, _ = agent.plan_versions(root, {"ecosystem": "node", "library": "react", "companions": ["react-dom"]},
                                     "19.2.0", lambda e: None)
    assert targets["react-scripts"] == "5.0.1" and "lodash" not in targets
