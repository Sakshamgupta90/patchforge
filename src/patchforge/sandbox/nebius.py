"""Run a repo's tests as a Nebius Serverless AI Job, through the `nebius` CLI.

How it works:
1. The code is packed into a .tar.gz and injected into the job with
   `--inject-file` (Nebius limits injected files to 64 KiB, which fits a small
   demo repo; bigger repos should use the Docker sandbox).
2. The job unpacks it, installs dependencies, runs the tests (pytest or
   npm test, see toolchain.py), then prints the exit code and, for Python,
   the JUnit report between markers.
3. PatchForge polls `nebius ai job get` until the job finishes, reads
   `nebius ai job logs`, and parses the markers.

Like the Docker sandbox, the job gets no environment variables or credentials.
Unlike Docker, the tests run with network access (Nebius jobs have no
"network off" switch), so this is weaker isolation.

Configure with NEBIUS_JOB_PLATFORM and NEBIUS_JOB_PRESET. There is no default:
Nebius defaults to an H100 GPU VM, which would be an expensive way to run pytest.
"""

from __future__ import annotations

import base64
import io
import json
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from ..contracts import ProgressFn, SandboxResult
from .common import IGNORE, REPORT_FILE, tail
from .docker_runner import SandboxError
from .toolchain import Toolchain, detect

INJECT_LIMIT = 64 * 1024
CODE_PATH = "/pf/code.tar.gz"
EXIT_MARK = "PF_EXIT_CODE="
REPORT_BEGIN, REPORT_END = "PF_REPORT_BEGIN", "PF_REPORT_END"
# Job states after which nothing more will happen.
TERMINAL = {"COMPLETED", "SUCCEEDED", "FAILED", "ERROR", "CANCELLED", "CANCELED", "DELETED", "TERMINATED"}


def pack(folder: Path) -> bytes:
    """tar.gz of the folder, skipping the same junk the other sandboxes skip."""
    folder = Path(folder)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(folder.rglob("*")):
            rel = path.relative_to(folder)
            if any(IGNORE("", [part]) for part in rel.parts):
                continue
            tar.add(path, arcname=str(rel), recursive=False)
    return buf.getvalue()


def job_script(tc: Toolchain) -> str:
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in {"HOME": "/tmp", **tc.env}.items())
    return " && ".join([
        "mkdir -p /work /cache", f"tar xzf {CODE_PATH} -C /work", "cd /work",
        f"export {exports}",
        f"( {tc.install} ) || {{ echo '[sandbox] dependency install failed'; echo {EXIT_MARK}install; exit 0; }}",
    ]) + (f"; {tc.test}; code=$?; echo {EXIT_MARK}$code; echo {REPORT_BEGIN}; "
          f"[ -f {REPORT_FILE} ] && base64 {REPORT_FILE} | tr -d '\\n'; echo; echo {REPORT_END}")


def parse_logs(logs: str) -> tuple[int | None | str, str]:
    """(exit code, or 'install' if deps failed, or None if missing; JUnit XML)."""
    code = None
    m = re.findall(rf"{EXIT_MARK}(\w+)", logs)
    if m:
        code = int(m[-1]) if m[-1].isdigit() else m[-1]
    report = ""
    r = re.search(rf"{REPORT_BEGIN}\s*([A-Za-z0-9+/=]*)\s*{REPORT_END}", logs)
    if r and r.group(1):
        try:
            report = base64.b64decode(r.group(1)).decode(errors="replace")
        except ValueError:
            report = ""
    return code, report


class NebiusJobSandbox:
    def __init__(self, *, platform: str, preset: str, image: str = "python:3.12-slim", cli: str = "nebius",
                 poll_s: float = 10, timeout_s: float = 1800, runner=subprocess.run):
        if not platform or not preset:
            raise SandboxError("set NEBIUS_JOB_PLATFORM and NEBIUS_JOB_PRESET for the Nebius sandbox "
                               "(a CPU platform; the Nebius default is an H100 GPU)")
        self.platform, self.preset, self.image, self.cli = platform, preset, image, cli
        self.poll_s, self.timeout_s = poll_s, timeout_s
        self._run = runner

    def _cli(self, *args: str, timeout: float = 120) -> str:
        try:
            proc = self._run([self.cli, *args], capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as e:
            raise SandboxError("the `nebius` CLI isn't installed; see docs.nebius.com/cli") from e
        if proc.returncode != 0:
            raise SandboxError(f"nebius {' '.join(args[:3])} failed: {(proc.stderr or proc.stdout).strip()[:500]}")
        return proc.stdout

    def create_args(self, name: str, code_file: Path, tc: Toolchain) -> list[str]:
        return [
            "ai", "job", "create", "--name", name, "--image", tc.image,
            "--platform", self.platform, "--preset", self.preset,
            "--inject-file", f"{code_file}:{CODE_PATH}",
            "--container-command", "sh", "--args", f"-c {shlex.quote(job_script(tc))}",
            "--timeout", "1h", "--restart-policy", "never",
            "--async", "--format", "jsonpath={.resource_id}",
        ]

    def run_tests(self, code_folder: Path, progress: ProgressFn | None = None) -> SandboxResult:
        say = progress or (lambda msg: None)
        start = time.monotonic()
        tc = detect(Path(code_folder), self.image)
        archive = pack(code_folder)
        if len(archive) > INJECT_LIMIT:
            raise SandboxError(f"repo is {len(archive) // 1024} KiB packed; Nebius injected files are limited "
                               "to 64 KiB. Use PATCHFORGE_SANDBOX=docker for this repo")
        with tempfile.TemporaryDirectory(prefix="pf-nebius-") as tmp:
            code_file = Path(tmp) / "code.tar.gz"
            code_file.write_bytes(archive)
            name = f"pf-tests-{uuid.uuid4().hex[:10]}"
            job_id = self._cli(*self.create_args(name, code_file, tc)).strip()
        if not job_id:
            raise SandboxError("nebius ai job create returned no job ID")
        say(f"Started Nebius job {job_id} ({self.platform}); it installs dependencies and runs the tests…")

        try:
            state = ""
            while time.monotonic() - start < self.timeout_s:
                info = json.loads(self._cli("ai", "job", "get", job_id, "--format", "json") or "{}")
                new_state = str(info.get("status", {}).get("state", "")).upper()
                if new_state != state:
                    say(f"Nebius job {new_state.lower() or 'pending'}…")
                state = new_state
                if state in TERMINAL:
                    break
                time.sleep(self.poll_s)
            else:
                self._cli("ai", "job", "cancel", job_id)
                return SandboxResult(passed=False, error_log=f"[sandbox] Nebius job {job_id} timed out ({state})",
                                     duration=time.monotonic() - start)

            logs = self._cli("ai", "job", "logs", job_id, timeout=300)
        finally:
            try:
                self._cli("ai", "job", "delete", job_id)
            except SandboxError:
                pass

        code, report_xml = parse_logs(logs)
        if code == "install":
            return SandboxResult(passed=False, error_log=tail("[sandbox] dependency install failed\n" + logs),
                                 duration=time.monotonic() - start)
        if code is None:
            return SandboxResult(passed=False, error_log=tail(f"[sandbox] job ended in state {state} with no "
                                                              f"test result\n" + logs),
                                 duration=time.monotonic() - start)
        with tempfile.TemporaryDirectory(prefix="pf-nebius-report-") as tmp:
            if report_xml:
                (Path(tmp) / REPORT_FILE).write_text(report_xml)
            passed, failed, log = tc.interpret(code, logs, Path(tmp))
        return SandboxResult(passed=passed, failed_tests=failed, checked=tc.checked, error_log="" if passed else tail(log),
                             duration=time.monotonic() - start)
