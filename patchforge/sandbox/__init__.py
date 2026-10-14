from __future__ import annotations

from ..contracts import Sandbox
from .docker_runner import DockerSandbox, SandboxError
from .fake import AlwaysPassSandbox, FakeSandbox
from .local import LocalSandbox

__all__ = ["DockerSandbox", "SandboxError", "FakeSandbox", "AlwaysPassSandbox", "LocalSandbox", "make_sandbox"]


def make_sandbox(kind: str, *, image: str = "python:3.12-slim") -> Sandbox:
    if kind == "docker":
        return DockerSandbox(image)
    if kind == "local":
        return LocalSandbox()
    if kind == "fake":
        return AlwaysPassSandbox()
    raise ValueError(f"unknown PATCHFORGE_SANDBOX={kind!r}; use docker, local or fake")
