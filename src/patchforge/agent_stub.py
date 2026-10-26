"""Placeholder for Person A's agent.

It follows the Agent contract exactly: emits research/plan/patch events with
token counts and costs, runs the sandbox once, and makes one small real change
so a real PR can be opened end to end. Replace it by implementing the same
`run()` signature (see contracts.Agent) and setting PATCHFORGE_AGENT.

The model names and prices below are made up for the stub; they are not real
Nebius Token Factory models or rates.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from .contracts import AgentEvent, AgentOutput, AgentTask, EmitFn, Sandbox

# (model, $ per 1M input tokens, $ per 1M output tokens). Placeholder values.
FAKE_MODELS = {
    "architect": ("stub-architect-large", 1.00, 3.00),
    "worker": ("stub-worker-small", 0.05, 0.20),
}


class StubAgent:
    def __init__(self, delay_s: float = 0.4, seed: int | None = None):
        self.delay_s = delay_s
        self.rng = random.Random(seed)

    def _call(self, emit: EmitFn, stage: str, tier: str, message: str, **data) -> None:
        model, price_in, price_out = FAKE_MODELS[tier]
        t_in, t_out = self.rng.randint(800, 6000), self.rng.randint(150, 1500)
        start = time.monotonic()
        time.sleep(self.delay_s)
        emit(AgentEvent(
            stage=stage, message=message, model=model, tokens_in=t_in, tokens_out=t_out,
            cost_usd=round(t_in / 1e6 * price_in + t_out / 1e6 * price_out, 6),
            latency_ms=round((time.monotonic() - start) * 1000, 1), data=data,
        ))

    def run(self, task: AgentTask, workdir: Path, sandbox: Sandbox, emit: EmitFn) -> AgentOutput:
        emit(AgentEvent(stage="research", message=f"(stub) would search the web for: {task.instruction}"))
        self._call(emit, "research", "architect", "(stub) built a Delta Spec from 3 sources")

        py_files = sorted(str(p.relative_to(workdir)) for p in workdir.rglob("*.py")
                          if ".git" not in p.parts)[:5]
        self._call(emit, "plan", "architect", f"(stub) planned changes for {len(py_files)} files",
                   files=py_files)

        note = workdir / "PATCHFORGE_STUB.md"
        note.write_text(
            "# PatchForge stub run\n\n"
            f"Instruction: {task.instruction}\n\n"
            "This file was written by the stub agent to prove the pipeline works end to end.\n"
            "The real agent will edit source files instead.\n"
        )
        self._call(emit, "patch", "worker", "(stub) wrote PATCHFORGE_STUB.md", files=["PATCHFORGE_STUB.md"])

        result = sandbox.run_tests(workdir)
        emit(AgentEvent(stage="tests", message=f"agent check: {'passed' if result.passed else 'failed'}",
                        level="info" if result.passed else "warning",
                        data={"failed_tests": result.failed_tests, "duration": result.duration}))

        return AgentOutput(
            changed_files=["PATCHFORGE_STUB.md"],
            summary=(f"Stub run for: **{task.instruction}**\n\n"
                     "No source files were changed; this PR only proves the platform can clone, "
                     "test, push and open a pull request."),
            test_results=result,
        )
