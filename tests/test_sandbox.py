import shutil
import subprocess
from pathlib import Path

import pytest

from patchforge.sandbox import DockerSandbox, FakeSandbox, LocalSandbox
from patchforge.sandbox.common import failed_tests_from_junit, install_command, interpret_pytest_exit
from patchforge.sandbox.fake import CANNED_FAILURE, CANNED_PASS
from tests.conftest import SAMPLE


def test_docker_command_is_locked_down(tmp_path):
    cmd = DockerSandbox().build_command(tmp_path, "pytest", network=False, name="pf-x")
    joined = " ".join(cmd)
    assert cmd[cmd.index("--network") + 1] == "none"
    for flag in ("--cap-drop ALL", "--read-only", "no-new-privileges", "--memory", "--pids-limit"):
        assert flag in joined
    user = cmd[cmd.index("--user") + 1]
    assert not user.startswith("0:")
    # Only these env vars go in: nothing from the host environment.
    envs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]
    assert {e.split("=")[0] for e in envs} == {"HOME", "PIP_NO_CACHE_DIR", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"}


def test_install_step_gets_network(tmp_path):
    cmd = DockerSandbox().build_command(tmp_path, "pip", network=True, name="pf-x")
    assert cmd[cmd.index("--network") + 1] == "bridge"


def test_install_command(tmp_path):
    assert install_command(tmp_path).endswith(" pytest")
    (tmp_path / "pyproject.toml").write_text("")
    assert " . pytest" in install_command(tmp_path)
    (tmp_path / "requirements.txt").write_text("")
    assert "-r requirements.txt pytest" in install_command(tmp_path)


def test_junit_parsing(tmp_path):
    report = tmp_path / "r.xml"
    report.write_text(
        '<testsuites><testsuite><testcase classname="tests.test_a" name="test_ok"/>'
        '<testcase classname="tests.test_a" name="test_bad"><failure message="x"/></testcase>'
        '<testcase classname="tests.test_b" name="test_err"><error message="y"/></testcase>'
        "</testsuite></testsuites>")
    assert failed_tests_from_junit(report) == ["tests.test_a::test_bad", "tests.test_b::test_err"]
    assert failed_tests_from_junit(tmp_path / "missing.xml") == []


def test_no_tests_is_not_a_pass():
    assert interpret_pytest_exit(0, "")[0] is True
    passed, log = interpret_pytest_exit(5, "")
    assert passed is False and "no tests" in log


def test_fake_sandbox_fails_then_passes(tmp_path):
    sb = FakeSandbox()
    assert sb.run_tests(tmp_path) == CANNED_FAILURE
    assert sb.run_tests(tmp_path) == CANNED_PASS
    assert sb.run_tests(tmp_path) == CANNED_PASS
    assert len(sb.calls) == 3


def test_local_sandbox_pass_and_fail(tmp_path):
    ok = LocalSandbox().run_tests(SAMPLE)
    assert ok.passed, ok.error_log

    broken = tmp_path / "broken"
    shutil.copytree(SAMPLE, broken)
    (broken / "calc.py").write_text("def add(a, b):\n    return a - b\n\ndef divide(a, b):\n    return a / b\n")
    bad = LocalSandbox().run_tests(broken)
    assert not bad.passed
    assert "tests.test_calc::test_add" in bad.failed_tests
    # Tests ran on a copy: no junk left behind in the source folder.
    assert not (broken / ".pf-report.xml").exists()


def _docker_up():
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "version"], capture_output=True, timeout=5).returncode == 0
    except subprocess.TimeoutExpired:
        return False


@pytest.mark.skipif(not _docker_up(), reason="Docker isn't running")
def test_docker_sandbox_runs_sample_project():
    result = DockerSandbox().run_tests(SAMPLE)
    assert result.passed, result.error_log
