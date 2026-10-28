"""Checks on the finished diff, run before anything is pushed.

Tests are the referee, so the agent must never edit them; workflows run with the repo's own
permissions, so it must not touch those either; and a patch that adds something shaped like a
credential, or rewrites half the repo, is not an upgrade. The agent's prompts already steer away
from all of this: this is the hard rule behind the prompts.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

TEST_DIRS = {"tests", "test", "__tests__"}

MAX_FILES = int(os.getenv("PATCHFORGE_MAX_FILES", "40"))
MAX_CHANGED_LINES = int(os.getenv("PATCHFORGE_MAX_CHANGED_LINES", "2000"))

LOCKFILE = re.compile(r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Pipfile\.lock|uv\.lock)$")

SECRET_PATTERNS = {
    "a private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "a GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "an AWS access key": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "an API key": re.compile(r"\b(sk|tvly|xox[bap])-[A-Za-z0-9-]{20,}\b"),
    "a hard-coded secret": re.compile(
        r"""(?i)\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*['"][A-Za-z0-9/+_=-]{20,}['"]"""),
}


def is_test_file(rel: Path) -> bool:
    """Tests are the referee: the agent must not edit them, or it could make them pass by weakening them."""
    name = rel.name
    return (bool(TEST_DIRS & set(rel.parts[:-1])) or name.startswith("test_") or name.endswith("_test.py")
            or name == "conftest.py" or re.search(r"\.(test|spec)\.[cm]?[jt]sx?$", name) is not None)


@dataclass
class FileDiff:
    path: str
    added: list[str]
    changed_lines: int  # added + removed


def check_diff(diffs: list[FileDiff]) -> list[str]:
    """What is wrong with this diff, as sentences for the dashboard; [] when it is fine to push."""
    problems = []
    tests = [d.path for d in diffs if is_test_file(Path(d.path))]
    if tests:
        problems.append("it changes test files (" + ", ".join(tests[:5]) + "): the tests must stay as they are")
    flows = [d.path for d in diffs if d.path.startswith(".github/") or "/.github/" in d.path]
    if flows:
        problems.append("it changes GitHub configuration (" + ", ".join(flows[:5]) + ")")
    for d in diffs:
        for what, pattern in SECRET_PATTERNS.items():
            if any(pattern.search(line) for line in d.added):
                problems.append(f"{d.path} adds what looks like {what}")
                break
    # Lockfiles are generated, and can be thousands of lines for a one-line change.
    lines = sum(d.changed_lines for d in diffs if not LOCKFILE.search(d.path))
    if len(diffs) > MAX_FILES:
        problems.append(f"it changes {len(diffs)} files (limit {MAX_FILES})")
    if lines > MAX_CHANGED_LINES:
        problems.append(f"it changes {lines:,} lines (limit {MAX_CHANGED_LINES:,})")
    return problems
