"""The three shared contracts between Person A (agent brain) and Person B (platform).

Both halves build against these, so change them only together:

1. Sandbox:  run_tests(code_folder) -> SandboxResult
2. Events:   every step emits an AgentEvent (the dashboard renders these)
3. Output:   the agent returns an AgentOutput; the platform turns it into a branch + PR

One deliberate difference from the Day 1 sketch: `tokens` is split into
`tokens_in` and `tokens_out`, because providers price input and output differently.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel, Field

# Stages the platform emits itself. The agent may emit any string, but these
# are the ones the dashboard knows how to group.
PLATFORM_STAGES = ("job", "clone", "baseline_tests", "tests", "guard", "push", "pr", "done", "error")
AGENT_STAGES = ("research", "plan", "patch", "retry", "skill")


# --- 1. Sandbox -------------------------------------------------------------


class SandboxResult(BaseModel):
    passed: bool
    failed_tests: list[str] = Field(default_factory=list)
    error_log: str = ""
    duration: float = 0.0  # seconds
    # What was actually run: "tests" (the project's own tests), or, for a project without tests,
    # "build" (npm run build), "typecheck" (tsc) or "compile" (syntax check). Only "tests"
    # verifies behaviour; anything else gets a PR marked "not verified by tests".
    checked: str = "tests"


ProgressFn = Callable[[str], None]


class Sandbox(Protocol):
    def run_tests(self, code_folder: Path, progress: ProgressFn | None = None) -> SandboxResult:
        """Run the project's tests against a copy of `code_folder`.

        Must never modify `code_folder`, and must never receive credentials.
        `progress`, if given, receives short status lines ("Installing dependencies…") while it runs,
        so the dashboard isn't silent during a long install.
        """
        ...


# --- 2. Events ---------------------------------------------------------------


class AgentEvent(BaseModel):
    stage: str
    message: str
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float | None = None
    level: str = "info"  # info | warning | error
    ts: float = Field(default_factory=time.time)
    data: dict = Field(default_factory=dict)  # free-form extras (file names, test counts...)


EmitFn = Callable[[AgentEvent], None]


# --- 3. Output -----------------------------------------------------------------


class AgentTask(BaseModel):
    repo: str  # "owner/name"
    instruction: str  # e.g. "Patch CVE-2026-1234 in requests" or "Upgrade pydantic to v2"
    base_branch: str


class AgentOutput(BaseModel):
    changed_files: list[str]
    summary: str  # markdown; goes into the PR body
    test_results: SandboxResult | None = None


class AgentFailed(Exception):
    """The agent stopped on purpose, with a reason worth showing the user as is (no stack trace)."""


class Agent(Protocol):
    def run(self, task: AgentTask, workdir: Path, sandbox: Sandbox, emit: EmitFn) -> AgentOutput:
        """Edit files in `workdir` in place and describe what changed.

        The platform re-runs the tests itself before opening a PR, and reads the
        real changed files from git, so `AgentOutput` is a description, not proof.
        The agent never sees the GitHub token.
        """
        ...
