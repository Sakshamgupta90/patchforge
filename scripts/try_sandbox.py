"""Run a folder's tests in a sandbox and print the result.

    python scripts/try_sandbox.py tests/fixtures/demo-repo            # Docker
    python scripts/try_sandbox.py tests/fixtures/demo-repo --local    # no isolation
"""

import argparse
from pathlib import Path

from patchforge.sandbox import DockerSandbox, LocalSandbox, SandboxError

p = argparse.ArgumentParser()
p.add_argument("folder", type=Path)
p.add_argument("--local", action="store_true", help="run on this machine without Docker (trusted code only)")
p.add_argument("--image", default="python:3.12-slim")
args = p.parse_args()

sandbox = LocalSandbox() if args.local else DockerSandbox(args.image)
try:
    result = sandbox.run_tests(args.folder)
except SandboxError as e:
    raise SystemExit(f"sandbox error: {e}")
print(result.model_dump_json(indent=2))
raise SystemExit(0 if result.passed else 1)
