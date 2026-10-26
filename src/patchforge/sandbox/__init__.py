from __future__ import annotations

import logging
import os
from pathlib import Path

from ..contracts import Sandbox, SandboxResult
from .docker_runner import DockerSandbox, SandboxError
from .fake import AlwaysPassSandbox, FakeSandbox
from .local import LocalSandbox
from .nebius import NebiusJobSandbox

__all__ = ["DockerSandbox", "SandboxError", "FakeSandbox", "AlwaysPassSandbox", "LocalSandbox",
           "NebiusJobSandbox", "FallbackSandbox", "make_sandbox"]

log = logging.getLogger("patchforge")


class FallbackSandbox:
    """Try the primary sandbox; if it can't run at all (not a test failure), use the fallback."""

    def __init__(self, primary: Sandbox, fallback: Sandbox):
        self.primary, self.fallback = primary, fallback

    def run_tests(self, code_folder: Path, progress=None) -> SandboxResult:
        try:
            return self.primary.run_tests(code_folder, progress=progress)
        except SandboxError as e:
            log.warning("primary sandbox unavailable, falling back: %s", e)
            if progress:
                progress(f"Nebius unavailable ({e}); using Docker instead")
            return self.fallback.run_tests(code_folder, progress=progress)


def make_sandbox(kind: str, *, image: str = "python:3.12-slim") -> Sandbox:
    if kind == "docker":
        return DockerSandbox(image)
    if kind == "nebius":
        # Docker stays the fallback, per the plan: a Nebius outage shouldn't break the demo.
        try:
            nebius = NebiusJobSandbox(platform=os.getenv("NEBIUS_JOB_PLATFORM", ""),
                                      preset=os.getenv("NEBIUS_JOB_PRESET", ""), image=image)
        except SandboxError as e:
            log.warning("Nebius sandbox not configured, using Docker: %s", e)
            return DockerSandbox(image)
        return FallbackSandbox(nebius, DockerSandbox(image))
    if kind == "local":
        return LocalSandbox()
    if kind == "fake":
        return AlwaysPassSandbox()
    raise ValueError(f"unknown PATCHFORGE_SANDBOX={kind!r}; use docker, nebius, local or fake")
