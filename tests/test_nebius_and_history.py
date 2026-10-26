import base64
import io
import json
import subprocess
import tarfile

import pytest

from patchforge.agent_stub import StubAgent
from patchforge.contracts import SandboxResult
from patchforge.jobs import JobRequest, JobRunner
from patchforge.sandbox import AlwaysPassSandbox, FallbackSandbox, NebiusJobSandbox, SandboxError
from patchforge.sandbox.nebius import CODE_PATH, job_script, pack, parse_logs
from patchforge.sandbox.toolchain import detect
from tests.conftest import SAMPLE

JUNIT = ('<testsuites><testsuite><testcase classname="tests.test_calc" name="test_add">'
         '<failure message="x"/></testcase></testsuite></testsuites>')


class FakeCLI:
    """Stands in for the `nebius` command: create -> get (running, then completed) -> logs -> delete."""

    def __init__(self, logs, states=("RUNNING", "COMPLETED")):
        self.calls, self.logs, self.states = [], logs, list(states)

    def __call__(self, cmd, capture_output, text, timeout):
        self.calls.append(cmd)
        sub = cmd[3]
        out = {"create": "job-123\n",
               "get": lambda: json.dumps({"status": {"state": self.states.pop(0)}}),
               "logs": self.logs, "delete": "", "cancel": ""}[sub]
        return subprocess.CompletedProcess(cmd, 0, out() if callable(out) else out, "")


def sandbox(cli):
    return NebiusJobSandbox(platform="cpu-e2", preset="4vcpu-16gb", poll_s=0, runner=cli)


def test_requires_platform():
    with pytest.raises(SandboxError, match="NEBIUS_JOB_PLATFORM"):
        NebiusJobSandbox(platform="", preset="")


def test_pack_skips_junk(tmp_path):
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.pyc").write_text("junk")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    names = tarfile.open(fileobj=io.BytesIO(pack(tmp_path))).getnames()
    assert names == ["a.py"]


def test_parse_logs():
    logs = f"noise\nPF_EXIT_CODE=1\nPF_REPORT_BEGIN\n{base64.b64encode(JUNIT.encode()).decode()}\nPF_REPORT_END\n"
    assert parse_logs(logs) == (1, JUNIT)
    assert parse_logs("PF_EXIT_CODE=install") == ("install", "")
    assert parse_logs("nothing") == (None, "")


def test_run_tests_reads_result_from_logs():
    logs = f"PF_EXIT_CODE=1\nPF_REPORT_BEGIN\n{base64.b64encode(JUNIT.encode()).decode()}\nPF_REPORT_END\n"
    cli = FakeCLI(logs)
    result = sandbox(cli).run_tests(SAMPLE)
    assert not result.passed and result.failed_tests == ["tests.test_calc::test_add"]
    create = cli.calls[0]
    assert create[:3] == ["nebius", "ai", "job"] and create[3] == "create"
    assert create[create.index("--platform") + 1] == "cpu-e2"
    assert create[create.index("--inject-file") + 1].endswith(f":{CODE_PATH}")
    assert "--env" not in create  # no secrets or env go into the job
    assert [c[3] for c in cli.calls] == ["create", "get", "get", "logs", "delete"]


def test_run_tests_pass():
    result = sandbox(FakeCLI("PF_EXIT_CODE=0\nPF_REPORT_BEGIN\n\nPF_REPORT_END")).run_tests(SAMPLE)
    assert result.passed


def test_missing_cli_falls_back():
    def missing(*a, **k):
        raise FileNotFoundError

    fallback = FallbackSandbox(sandbox(missing), AlwaysPassSandbox())
    assert fallback.run_tests(SAMPLE).passed


def test_history_survives_restart(tmp_path, fake_github):
    history = tmp_path / "jobs"
    runner = JobRunner(github=fake_github, sandbox_factory=AlwaysPassSandbox,
                       agent_factory=lambda: StubAgent(delay_s=0), clone_depth=None, history_dir=history)
    job = runner.run_sync(JobRequest(repo="me/demo", instruction="x"))

    reloaded = JobRunner(github=fake_github, sandbox_factory=AlwaysPassSandbox,
                         agent_factory=lambda: StubAgent(delay_s=0), history_dir=history)
    again = reloaded.jobs[job.id]
    assert again.status == "succeeded" and again.pr_url == job.pr_url
    assert len(again.events) == len(job.events)
    assert again.totals() == job.totals()


def test_interrupted_job_marked_failed(tmp_path, fake_github):
    history = tmp_path / "jobs"
    runner = JobRunner(github=fake_github, sandbox_factory=AlwaysPassSandbox,
                       agent_factory=lambda: StubAgent(delay_s=0), history_dir=history)
    job = runner.run_sync(JobRequest(repo="me/demo", instruction="x"))
    record = json.loads((history / f"{job.id}.json").read_text())
    record["snapshot"]["status"] = "running"
    (history / f"{job.id}.json").write_text(json.dumps(record))
    reloaded = JobRunner(github=fake_github, sandbox_factory=AlwaysPassSandbox,
                         agent_factory=lambda: StubAgent(delay_s=0), history_dir=history)
    assert reloaded.jobs[job.id].status == "failed"
    assert "restarted" in reloaded.jobs[job.id].error


def test_job_script_uses_the_project_toolchain(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "node --test"}}')
    script = job_script(detect(tmp_path))
    assert "npm install" in script and "npm test" in script and "npm_config_cache=/cache/npm" in script
