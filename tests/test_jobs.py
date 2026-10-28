import time

from patchforge.agent_stub import StubAgent
from patchforge.contracts import AgentOutput, SandboxResult
from patchforge.jobs import JobRequest, JobRunner, branch_name
from patchforge.sandbox import AlwaysPassSandbox, FakeSandbox, LocalSandbox
from tests.conftest import FAKE_TOKEN, git


def runner(github, sandbox_factory=LocalSandbox, agent=None):
    return JobRunner(github=github, sandbox_factory=sandbox_factory,
                     agent_factory=lambda: agent or StubAgent(delay_s=0, seed=1), clone_depth=None)


def test_happy_path_opens_pr(fake_github, remote_repo):
    job = runner(fake_github).run_sync(JobRequest(repo="me/demo", instruction="Upgrade requests to 2.33"))
    assert job.status == "succeeded", job.error
    assert job.pr_url == "https://github.com/me/demo/pull/1"

    pr = fake_github.prs[0]
    assert pr["base"] == "main" and pr["head"] == job.branch
    assert job.branch.startswith("patchforge/upgrade-requests-to-2-33-")
    assert "PATCHFORGE_STUB.md" in pr["body"] and "All tests passed" in pr["body"]
    assert "PATCHFORGE_STUB.md" in git("ls-tree", "-r", "--name-only", job.branch, cwd=remote_repo)

    stages = [e["stage"] for e in job.events]
    for s in ("job", "clone", "baseline_tests", "research", "plan", "patch", "tests", "push", "pr", "done"):
        assert s in stages
    assert job.totals()["cost_usd"] > 0 and job.totals()["model_calls"] == 3
    # The token never leaks into events.
    assert FAKE_TOKEN not in str(job.events)


def test_failing_tests_block_the_pr(fake_github, remote_repo):
    fail = SandboxResult(passed=False, failed_tests=["t::x"], error_log="boom")
    job = runner(fake_github, sandbox_factory=lambda: FakeSandbox([fail])).run_sync(
        JobRequest(repo="me/demo", instruction="x"))
    assert job.status == "failed"
    assert "refusing to open a PR" in job.error
    assert fake_github.prs == []
    assert "patchforge/" not in git("branch", "--list", cwd=remote_repo)  # nothing pushed


class NoopAgent:
    def run(self, task, workdir, sandbox, emit):
        return AgentOutput(changed_files=[], summary="nothing")


def test_no_changes_fails(fake_github):
    job = runner(fake_github, sandbox_factory=AlwaysPassSandbox, agent=NoopAgent()).run_sync(
        JobRequest(repo="me/demo", instruction="x"))
    assert job.status == "failed" and "no changes" in job.error


def test_agent_crash_is_reported(fake_github):
    class Boom:
        def run(self, *a):
            raise RuntimeError("model said no")

    job = runner(fake_github, sandbox_factory=AlwaysPassSandbox, agent=Boom()).run_sync(
        JobRequest(repo="me/demo", instruction="x"))
    assert job.status == "failed" and "model said no" in job.error
    assert job.events[-1]["stage"] == "error"


def test_token_refetched_before_push(fake_github):
    runner(fake_github, sandbox_factory=AlwaysPassSandbox).run_sync(JobRequest(repo="me/demo", instruction="x"))
    assert fake_github.token_calls == 2


def test_submit_runs_in_background_and_streams(fake_github):
    job = runner(fake_github, sandbox_factory=AlwaysPassSandbox).submit(JobRequest(repo="me/demo", instruction="x"))
    seen, deadline = 0, time.time() + 30
    while not job.done and time.time() < deadline:
        seen += len(job.wait_for_events(seen, timeout=1))
    assert job.status == "succeeded", job.error


def test_bad_repo_name_rejected(fake_github):
    import pytest
    with pytest.raises(ValueError):
        runner(fake_github).submit(JobRequest(repo="not-a-repo", instruction="x"))


def test_branch_name():
    assert branch_name("Patch CVE-2026-1234 in Requests!!", "abcdef123") == "patchforge/patch-cve-2026-1234-in-requests-abcdef"
    assert branch_name("???", "abcdef") == "patchforge/patch-abcdef"



def test_identical_final_check_is_reused(fake_github):
    from patchforge.sandbox import FakeSandbox

    class Counting(FakeSandbox):
        def __init__(self):
            super().__init__([SandboxResult(passed=True)])

    box = Counting()
    job = runner(fake_github, sandbox_factory=lambda: box).run_sync(JobRequest(repo="me/demo", instruction="x"))
    assert job.status == "succeeded", job.error
    # before-check + the agent's check; the final check on the same code reuses the agent's result
    assert len(box.calls) == 2
    assert any("reusing its result" in e["message"] for e in job.events)


def test_project_without_tests_gets_a_draft_pr(fake_github):
    unverified = SandboxResult(passed=True, checked="compile")
    job = runner(fake_github, sandbox_factory=lambda: FakeSandbox([unverified])).run_sync(
        JobRequest(repo="me/demo", instruction="x"))
    assert job.status == "succeeded", job.error
    pr = fake_github.prs[0]
    assert pr["draft"] is True and "Not verified by tests" in pr["body"]
    assert job.snapshot()["verified"] is False and job.snapshot()["checked"] == "compile"


def test_verified_pr_is_not_a_draft(fake_github):
    job = runner(fake_github, sandbox_factory=AlwaysPassSandbox).run_sync(JobRequest(repo="me/demo", instruction="x"))
    assert fake_github.prs[0]["draft"] is False and job.snapshot()["verified"] is True


class TouchesTests:
    def __init__(self, rel, text="x = 1\n"):
        self.rel, self.text = rel, text

    def run(self, task, workdir, sandbox, emit):
        target = workdir / self.rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.text)
        return AgentOutput(changed_files=[self.rel], summary="s")


def guarded(fake_github, remote_repo, rel, text="x = 1\n"):
    return runner(fake_github, sandbox_factory=AlwaysPassSandbox, agent=TouchesTests(rel, text)).run_sync(
        JobRequest(repo="me/demo", instruction="x"))


def test_guard_blocks_edits_to_tests(fake_github, remote_repo):
    job = guarded(fake_github, remote_repo, "tests/test_new.py")
    assert job.status == "failed" and "changes test files" in job.error
    assert fake_github.prs == [] and "patchforge/" not in git("branch", "--list", cwd=remote_repo)
    assert any(e["stage"] == "guard" and e["level"] == "error" for e in job.events)


def test_guard_blocks_workflow_edits(fake_github, remote_repo):
    job = guarded(fake_github, remote_repo, ".github/workflows/ci.yml")
    assert job.status == "failed" and "GitHub configuration" in job.error


def test_guard_blocks_secret_looking_content(fake_github, remote_repo):
    job = guarded(fake_github, remote_repo, "app/conf.py", 'API_KEY = "abcdefghijklmnopqrstuvwxyz0123456789"\n')
    assert job.status == "failed" and "looks like a hard-coded secret" in job.error
    assert fake_github.prs == []


def test_guard_blocks_huge_diffs(fake_github, remote_repo, monkeypatch):
    monkeypatch.setattr("patchforge.guard.MAX_CHANGED_LINES", 5)
    job = guarded(fake_github, remote_repo, "app/big.py", "x = 1\n" * 50)
    assert job.status == "failed" and "lines (limit 5)" in job.error


def test_guard_lets_a_normal_patch_through(fake_github, remote_repo):
    job = guarded(fake_github, remote_repo, "app/new.py")
    assert job.status == "succeeded", job.error
