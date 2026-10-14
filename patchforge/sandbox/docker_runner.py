"""Run a repo's tests inside a throwaway Docker container.

Two containers per run, both on a private copy of the code:
1. install: network on (pip needs it), installs deps into .pf-deps
2. test:    network OFF, runs pytest against .pf-deps

Both run as a non-root user with all capabilities dropped, a read-only root
filesystem, and memory/CPU/process limits. No environment variables from the
host are passed in, so the repo's code never sees a token or API key.
The install step does run the repo's setup code with network access, which is
acceptable only because the container holds no secrets.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from ..contracts import SandboxResult
from .common import (DEPS_DIR, REPORT_FILE, copy_tree, failed_tests_from_junit, install_command,
                     interpret_pytest_exit, tail)


class SandboxError(RuntimeError):
    pass


class DockerSandbox:
    def __init__(self, image: str = "python:3.12-slim", *, memory: str = "1g", cpus: str = "1.0",
                 pids: int = 256, install_timeout_s: float = 300, test_timeout_s: float = 600,
                 docker: str = "docker"):
        self.image = image
        self.memory, self.cpus, self.pids = memory, cpus, pids
        self.install_timeout_s, self.test_timeout_s = install_timeout_s, test_timeout_s
        self.docker = docker

    # Pure so it can be unit-tested without Docker.
    def build_command(self, workdir: Path, shell_cmd: str, *, network: bool, name: str) -> list[str]:
        uid = os.getuid() if hasattr(os, "getuid") else 1000
        gid = os.getgid() if hasattr(os, "getgid") else 1000
        if uid == 0:  # never run the repo's code as root, even if PatchForge itself is
            uid = gid = 65534
        return [
            self.docker, "run", "--rm", "--name", name,
            "--network", "bridge" if network else "none",
            "--user", f"{uid}:{gid}",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only", "--tmpfs", "/tmp:rw,size=256m",
            "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", str(self.pids),
            "-e", "HOME=/tmp", "-e", "PIP_NO_CACHE_DIR=1", "-e", f"PYTHONPATH=/work/{DEPS_DIR}:/work",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{workdir.resolve()}:/work", "-w", "/work",
            self.image, "sh", "-c", shell_cmd,
        ]

    def check_available(self) -> None:
        try:
            proc = subprocess.run([self.docker, "version", "--format", "{{.Server.Version}}"],
                                  capture_output=True, text=True, timeout=10)
        except FileNotFoundError as e:
            raise SandboxError("Docker isn't installed (no `docker` command found)") from e
        except subprocess.TimeoutExpired as e:
            raise SandboxError("Docker didn't answer; is Docker Desktop running?") from e
        if proc.returncode != 0:
            raise SandboxError(f"Docker isn't running; start Docker Desktop. ({proc.stderr.strip()[:200]})")

    def _run(self, cmd: list[str], name: str, timeout: float) -> tuple[int | None, str]:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return proc.returncode, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as e:
            subprocess.run([self.docker, "rm", "-f", name], capture_output=True)
            out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return None, out + f"\n[sandbox] timed out after {timeout:.0f}s"

    def run_tests(self, code_folder: Path) -> SandboxResult:
        self.check_available()
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="pf-sbx-") as tmp:
            work = copy_tree(Path(code_folder), Path(tmp) / "work")
            run_id = uuid.uuid4().hex[:10]

            name = f"pf-install-{run_id}"
            code, log = self._run(self.build_command(work, install_command(work), network=True, name=name),
                                  name, self.install_timeout_s)
            if code != 0:
                return SandboxResult(passed=False, error_log=tail("[sandbox] dependency install failed\n" + log),
                                     duration=time.monotonic() - start)

            name = f"pf-test-{run_id}"
            test_cmd = f"python -m pytest -q -p no:cacheprovider --junitxml={REPORT_FILE}"
            code, log = self._run(self.build_command(work, test_cmd, network=False, name=name),
                                  name, self.test_timeout_s)
            failed = failed_tests_from_junit(work / REPORT_FILE)
            passed, log = (False, log) if code is None else interpret_pytest_exit(code, log)
            return SandboxResult(passed=passed, failed_tests=failed, error_log="" if passed else tail(log),
                                 duration=time.monotonic() - start)
