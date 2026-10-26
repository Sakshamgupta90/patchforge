"""Helpers shared by the sandbox runners."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

LOG_LIMIT = 8000  # characters of test output kept in error_log
DEPS_DIR = ".pf-deps"
REPORT_FILE = ".pf-report.xml"
# Not copied into the sandbox: git history, virtualenvs, caches.
IGNORE = shutil.ignore_patterns(".git", ".venv", "venv", "node_modules", "__pycache__",
                                ".pytest_cache", ".next", DEPS_DIR, REPORT_FILE, ".pf-npm-cache")


def copy_tree(src: Path, dest: Path) -> Path:
    """Tests run on a copy, so they can't modify the agent's working tree."""
    shutil.copytree(src, dest, ignore=IGNORE, symlinks=True)
    return dest


def tail(text: str, limit: int = LOG_LIMIT) -> str:
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


def failed_tests_from_junit(report: Path) -> list[str]:
    if not report.exists():
        return []
    try:
        root = ET.parse(report).getroot()
    except ET.ParseError:
        return []
    failed = []
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            cls, name = case.get("classname", ""), case.get("name", "")
            failed.append(f"{cls}::{name}" if cls else name)
    return failed


def interpret_pytest_exit(code: int, log: str) -> tuple[bool, str]:
    """pytest exit codes: 0 ok, 1 failures, 2 interrupted, 3 internal, 4 usage, 5 no tests.

    "No tests" counts as a failure: a green result must mean something was checked.
    """
    if code == 0:
        return True, log
    if code == 5:
        return False, "pytest collected no tests.\n" + log
    return False, log
