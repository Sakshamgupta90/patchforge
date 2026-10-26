"""A sandbox that returns canned results. For Person A to develop the retry loop without Docker."""

from __future__ import annotations

from pathlib import Path

from ..contracts import SandboxResult

CANNED_FAILURE = SandboxResult(
    passed=False,
    failed_tests=["tests/test_client.py::test_get_user"],
    error_log=(
        "E   AttributeError: module 'httpclient' has no attribute 'get_json'\n"
        "tests/test_client.py:12: AttributeError\n"
        "1 failed, 2 passed in 0.31s"
    ),
    duration=0.3,
)
CANNED_PASS = SandboxResult(passed=True, duration=0.3)


class FakeSandbox:
    """Returns `results` in order, then keeps returning the last one.

    Default: one failure, then passes, which exercises a retry loop exactly once.
    """

    def __init__(self, results: list[SandboxResult] | None = None):
        self.results = list(results) if results else [CANNED_FAILURE, CANNED_PASS]
        self.calls: list[Path] = []

    def run_tests(self, code_folder: Path, progress=None) -> SandboxResult:
        self.calls.append(Path(code_folder))
        i = min(len(self.calls) - 1, len(self.results) - 1)
        return self.results[i]


class AlwaysPassSandbox(FakeSandbox):
    def __init__(self):
        super().__init__([CANNED_PASS])
