"""Runs pytest directly on this machine, on a copy of the code. NO isolation.

Only for repos you wrote yourself, when Docker isn't available. Host secrets are
still stripped from the environment, but the code can read your files and use
the network.

The repo's pinned dependencies are installed into the copy (.pf-deps) and put
first on PYTHONPATH, so tests see the repo's versions, not PatchForge's own.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import weakref
from pathlib import Path

from ..contracts import ProgressFn, SandboxResult
from .common import copy_tree, tail
from .toolchain import detect

# Only these host variables reach the test process.
SAFE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")


def _local_node_major() -> str | None:
    try:
        out = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=10).stdout
        return out.strip().lstrip("v").split(".")[0] or None
    except (OSError, subprocess.TimeoutExpired):
        return None


class LocalSandbox:
    def __init__(self, python: str | None = None, timeout_s: float = 600, install: bool = True):
        # PATCHFORGE_LOCAL_PYTHON picks the interpreter tests run on (e.g. python3.12), since old
        # libraries often don't support the newest Python. pytest is installed per run, so any works.
        self.python = python or os.getenv("PATCHFORGE_LOCAL_PYTHON") or sys.executable
        self.timeout_s = timeout_s
        self.install = install
        self._cache = Path(tempfile.mkdtemp(prefix="pf-local-cache-"))
        weakref.finalize(self, shutil.rmtree, self._cache, True)

    def _sh(self, cmd: str, work: Path, env: dict) -> tuple[int | None, str]:
        try:
            proc = subprocess.run(cmd, shell=True, cwd=work, env=env, capture_output=True, text=True,
                                  timeout=self.timeout_s)
            return proc.returncode, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            return None, f"[sandbox] timed out after {self.timeout_s:.0f}s"

    def run_tests(self, code_folder: Path, progress: ProgressFn | None = None) -> SandboxResult:
        say = progress or (lambda msg: None)
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="pf-local-") as tmp:
            work = copy_tree(Path(code_folder), Path(tmp) / "work")
            tc = detect(work)
            env = {k: v for k, v in os.environ.items() if k in SAFE_ENV}
            # The toolchain's paths assume the Docker mount point /work.
            env.update({k: v.replace("/work", str(work)).replace("/cache", str(self._cache)) for k, v in tc.env.items()})
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = env["PYTHONPATH"].replace(":", os.pathsep)
            note = ""
            if tc.language == "python":
                run = lambda c: c.replace("python -m", f'"{self.python}" -m', 1)  # noqa: E731
            else:
                run = lambda c: c  # noqa: E731
                wanted, have = tc.image.split(":")[1].split("-")[0], _local_node_major()
                if have and wanted != have:
                    note = f"[sandbox] project wants Node {wanted} but this machine has Node {have}\n"

            if self.install:
                say(f"Installing dependencies on this machine ({tc.install.split(' --')[0]})…")
                code, log = self._sh(run(tc.install), work, env)
                if code != 0:
                    return SandboxResult(passed=False, duration=time.monotonic() - start,
                                         error_log=tail(note + "[sandbox] dependency install failed\n" + log))
            say("Running the project's tests…" if tc.checked == "tests" else f"No tests found; running a {tc.checked} check…")
            code, log = self._sh(run(tc.test), work, env)
            if code is None:
                return SandboxResult(passed=False, error_log=log, duration=time.monotonic() - start)
            passed, failed, log = tc.interpret(code, log, work)
            return SandboxResult(passed=passed, failed_tests=failed, checked=tc.checked, error_log="" if passed else tail(note + log),
                                 duration=time.monotonic() - start)
