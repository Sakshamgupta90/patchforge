"""Runs pytest directly on this machine, on a copy of the code. NO isolation.

Only for repos you wrote yourself, when Docker isn't available. Host secrets are
still stripped from the environment, but the code can read your files and use
the network.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..contracts import SandboxResult
from .common import REPORT_FILE, copy_tree, failed_tests_from_junit, interpret_pytest_exit, tail

# Only these host variables reach the test process.
SAFE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")


class LocalSandbox:
    def __init__(self, python: str = sys.executable, timeout_s: float = 300):
        self.python = python
        self.timeout_s = timeout_s

    def run_tests(self, code_folder: Path) -> SandboxResult:
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="pf-local-") as tmp:
            work = copy_tree(Path(code_folder), Path(tmp) / "work")
            env = {k: v for k, v in os.environ.items() if k in SAFE_ENV}
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = str(work)
            cmd = [self.python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={REPORT_FILE}"]
            try:
                proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True,
                                      timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                return SandboxResult(passed=False, error_log=f"timed out after {self.timeout_s:.0f}s",
                                     duration=time.monotonic() - start)
            passed, log = interpret_pytest_exit(proc.returncode, proc.stdout + proc.stderr)
            return SandboxResult(passed=passed, failed_tests=failed_tests_from_junit(work / REPORT_FILE),
                                 error_log="" if passed else tail(log), duration=time.monotonic() - start)
