import shutil
import subprocess
from pathlib import Path

import pytest

from patchforge.sandbox import DockerSandbox, FakeSandbox, LocalSandbox
from patchforge.sandbox.common import failed_tests_from_junit, interpret_pytest_exit
from patchforge.sandbox.toolchain import detect, node_failures, python_install
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
    assert {e.split("=")[0] for e in envs} == {"HOME", "PIP_CACHE_DIR", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"}
    assert "python:3.12-slim" in cmd


def test_install_step_gets_network(tmp_path):
    cmd = DockerSandbox().build_command(tmp_path, "pip", network=True, name="pf-x")
    assert cmd[cmd.index("--network") + 1] == "bridge"


def test_python_install_command(tmp_path):
    assert python_install(tmp_path).endswith(" pytest")
    (tmp_path / "pyproject.toml").write_text("")
    assert " . pytest" in python_install(tmp_path)
    (tmp_path / "requirements.txt").write_text("")
    assert "-r requirements.txt pytest" in python_install(tmp_path)


def test_detect_python_version(tmp_path):
    assert detect(tmp_path).image == "python:3.12-slim"
    (tmp_path / ".python-version").write_text("3.11.9\n")
    assert detect(tmp_path).image == "python:3.11-slim"


def test_detect_node(tmp_path):
    (tmp_path / "package.json").write_text('{"name": "x", "engines": {"node": ">=18"}, "scripts": {"test": "jest"}}')
    tc = detect(tmp_path)
    assert tc.language == "node" and tc.image == "node:18-slim"
    assert tc.install.startswith("npm install") and tc.test == "npm test"
    (tmp_path / ".nvmrc").write_text("v16.20.2\n")
    assert detect(tmp_path).image == "node:16-slim"  # .nvmrc wins over engines


def test_node_docker_command(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    cmd = DockerSandbox().build_command(tmp_path, "npm test", network=False, name="pf-x")
    assert "node:22-slim" in cmd and cmd[cmd.index("--network") + 1] == "none"
    envs = {cmd[i + 1].split("=")[0] for i, a in enumerate(cmd) if a == "-e"}
    assert "npm_config_cache" in envs and "PYTHONPATH" not in envs


def test_node_results():
    tap = "TAP version 13\nok 1 - adds\nnot ok 2 - divides by zero\n# fail 1"
    assert node_failures(tap) == ["divides by zero"]
    spec = "✔ adds (0.5ms)\n✖ divides by zero (1.2ms)\n"
    assert node_failures(spec) == ["divides by zero"]
    tc = detect.__globals__["Toolchain"]("node", "node:22-slim", "npm install", "npm test")
    passed, failed, log = tc.interpret(1, 'npm error Missing script: "test"', Path("."))
    assert not passed and "no `npm test` script" in log


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


def test_download_cache_is_per_sandbox_and_cleaned_up(tmp_path, monkeypatch):
    import gc
    monkeypatch.setenv("PATCHFORGE_SANDBOX_TMP", str(tmp_path))
    a, b = DockerSandbox(), DockerSandbox()
    cmd = a.build_command(tmp_path, "true", network=True, name="pf-x")
    cache = a.cache_dir()
    assert f"{cache.resolve()}:/cache" in cmd
    assert a.cache_dir() == cache and b.cache_dir() != cache  # one cache per sandbox (per job)
    del a, cmd
    gc.collect()
    assert not cache.exists()


def test_projects_without_tests_get_a_build_or_compile_check(tmp_path):
    node = tmp_path / "node"
    node.mkdir()
    (node / "package.json").write_text('{"scripts": {"test": "echo \\"Error: no test specified\\" && exit 1", "build": "next build"}}')
    tc = detect(node)
    assert (tc.test, tc.checked) == ("npm run build", "build")
    (node / "package.json").write_text('{"scripts": {}}')
    (node / "tsconfig.json").write_text("{}")
    assert detect(node).checked == "typecheck"
    (node / "tsconfig.json").unlink()
    assert detect(node).checked == "compile" and "node --check" in detect(node).test

    py = tmp_path / "py"
    py.mkdir()
    (py / "app.py").write_text("x = 1\n")
    assert detect(py).checked == "compile" and "compileall" in detect(py).test
    (py / "tests").mkdir()
    (py / "tests" / "test_app.py").write_text("def test_x(): pass\n")
    assert detect(py).checked == "tests"


def test_deps_key_changes_only_with_manifests(tmp_path):
    (tmp_path / "requirements.txt").write_text("pydantic==1.10.13\n")
    (tmp_path / "app.py").write_text("x = 1\n")
    key = detect(tmp_path).deps_key(tmp_path)
    (tmp_path / "app.py").write_text("x = 2\n")  # code change: same dependencies
    assert detect(tmp_path).deps_key(tmp_path) == key
    (tmp_path / "requirements.txt").write_text("pydantic==2.14.1\n")
    assert detect(tmp_path).deps_key(tmp_path) != key


def test_compile_check_runs_for_python_without_tests(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n")
    assert LocalSandbox(install=False).run_tests(tmp_path).passed
    (tmp_path / "bad.py").write_text("def broken(:\n")
    result = LocalSandbox(install=False).run_tests(tmp_path)
    assert not result.passed and result.checked == "compile"
