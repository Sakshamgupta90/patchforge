"""Run a project's tests inside a throwaway Docker container.

Two containers per run, both on a private copy of the code:
1. install: network on (pip/npm need it): pip into .pf-deps, or npm install
2. test:    network OFF: pytest, or npm test

The image follows the project (python:3.x-slim or node:NN-slim, see toolchain.py).

Both run as a non-root user with all capabilities dropped, a read-only root
filesystem, and memory/CPU/process limits. No environment variables from the
host are passed in, so the repo's code never sees a token or API key.
The install step does run the repo's setup code with network access, which is
acceptable only because the container holds no secrets.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import weakref
from pathlib import Path

from ..contracts import ProgressFn, SandboxResult
from .common import copy_tree, tail
from .toolchain import Toolchain, detect


class SandboxError(RuntimeError):
    pass


HEARTBEAT_S = 20


def elapsed(seconds: float) -> str:
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s" if seconds >= 60 else f"{int(seconds)}s"


def scratch_root() -> Path:
    """Where sandbox copies go. Must be a folder Docker can mount: Docker Desktop shares the system
    temp folder, but Colima and most Linux setups only share the home folder, so default to that."""
    root = Path(os.getenv("PATCHFORGE_SANDBOX_TMP") or Path.home() / ".cache" / "patchforge" / "sandbox")
    root.mkdir(parents=True, exist_ok=True)
    return root


class DockerSandbox:
    def __init__(self, image: str = "python:3.12-slim", *, memory: str = "2g", cpus: str = "2.0",
                 pids: int = 512, install_timeout_s: float = 600, test_timeout_s: float = 900,
                 docker: str = "docker"):
        self.image = image  # default for Python projects; Node projects pick node:NN-slim
        self.memory, self.cpus, self.pids = memory, cpus, pids
        self.install_timeout_s, self.test_timeout_s = install_timeout_s, test_timeout_s
        self.docker = docker
        self._cache: Path | None = None
        self._lock = threading.Lock()  # a job may run its "before" check and the agent's check at once
        self._env_locks: dict[str, threading.Lock] = {}

    def cache_dir(self) -> Path:
        """Download cache for this sandbox's lifetime (one job), removed when the sandbox is discarded."""
        with self._lock:
            if self._cache is None:
                self._cache = Path(tempfile.mkdtemp(prefix="pf-cache-", dir=scratch_root()))
                weakref.finalize(self, shutil.rmtree, self._cache, True)
            return self._cache

    def _env(self, key: str) -> tuple[Path, threading.Lock]:
        """Installed dependencies for one set of manifests, reused by every check in the job that
        has the same requirements.txt / package.json (e.g. retries that only changed code)."""
        env = self.cache_dir() / f"env-{key}"
        env.mkdir(exist_ok=True)
        with self._lock:
            return env, self._env_locks.setdefault(key, threading.Lock())

    # Pure so it can be unit-tested without Docker.
    def build_command(self, workdir: Path, shell_cmd: str, *, network: bool, name: str,
                      toolchain: Toolchain | None = None, deps: Path | None = None) -> list[str]:
        tc = toolchain or detect(workdir, self.image)
        deps_mount = ["-v", f"{deps.resolve()}:/work/{tc.deps_dir}"] if deps else []
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
            "-e", "HOME=/tmp", *[arg for k, v in tc.env.items() for arg in ("-e", f"{k}={v}")],
            "-v", f"{workdir.resolve()}:/work", *deps_mount, "-v", f"{self.cache_dir().resolve()}:/cache", "-w", "/work",
            tc.image, "sh", "-c", shell_cmd,
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

    def _run(self, cmd: list[str], name: str, timeout: float, progress: ProgressFn | None = None,
             doing: str = "Working") -> tuple[int | None, str]:
        """Run one container, calling `progress` every HEARTBEAT_S seconds while it's still going."""
        start = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            try:
                out, _ = proc.communicate(timeout=HEARTBEAT_S)
                return proc.returncode, out or ""
            except subprocess.TimeoutExpired:
                spent = time.monotonic() - start
                if spent > timeout:
                    subprocess.run([self.docker, "rm", "-f", name], capture_output=True)
                    proc.kill()
                    out, _ = proc.communicate()
                    return None, (out or "") + f"\n[sandbox] timed out after {timeout:.0f}s"
                if progress:
                    progress(f"Still {doing.lower()}… {elapsed(spent)}")

    def run_tests(self, code_folder: Path, progress: ProgressFn | None = None) -> SandboxResult:
        say = progress or (lambda msg: None)
        self.check_available()
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="pf-sbx-", dir=scratch_root()) as tmp:
            work = copy_tree(Path(code_folder), Path(tmp) / "work")
            tc = detect(work, self.image)
            run_id = uuid.uuid4().hex[:10]

            env, env_lock = self._env(tc.deps_key(work))
            with env_lock:
                if (env / ".pf-ready").exists():
                    say("Dependencies unchanged since the last check; reusing the installed packages")
                else:
                    name = f"pf-install-{run_id}"
                    say(f"Installing dependencies in a {tc.image} sandbox ({tc.install.split(' --')[0]})…")
                    code, log = self._run(self.build_command(work, tc.install, network=True, name=name,
                                                             toolchain=tc, deps=env),
                                          name, self.install_timeout_s, progress, "Installing dependencies")
                    if code != 0:
                        return SandboxResult(passed=False, checked=tc.checked, duration=time.monotonic() - start,
                                             error_log=tail("[sandbox] dependency install failed\n" + log))
                    (env / ".pf-ready").touch()
                    say(f"Dependencies installed ({elapsed(time.monotonic() - start)})")

            what = {"tests": "Running the project's tests", "build": "No tests found; building the project",
                    "typecheck": "No tests found; type-checking the project",
                    "compile": "No tests found; checking every file compiles"}[tc.checked]
            name = f"pf-test-{run_id}"
            say(f"{what} ({tc.test.split(' -')[0].split(' |')[0]}), network off…")
            code, log = self._run(self.build_command(work, tc.test, network=False, name=name, toolchain=tc, deps=env),
                                  name, self.test_timeout_s, progress, what.split("; ")[-1])
            passed, failed, log = (False, [], log) if code is None else tc.interpret(code, log, work)
            return SandboxResult(passed=passed, failed_tests=failed, error_log="" if passed else tail(log),
                                 duration=time.monotonic() - start, checked=tc.checked)
