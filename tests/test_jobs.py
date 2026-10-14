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
