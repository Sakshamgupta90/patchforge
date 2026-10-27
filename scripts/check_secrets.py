"""Refuse to commit real credentials.

Checks every file git would commit for things that look like real keys, and that
.env.example holds placeholders only (real values belong in .env, which is git-ignored).

    python scripts/check_secrets.py            # check tracked + staged files
    python scripts/check_secrets.py --install  # run it automatically before every commit
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Variables in .env.example that must stay empty or a placeholder.
SECRET_VARS = {"GITHUB_CLIENT_SECRET", "NEBIUS_API_KEY", "TAVILY_API_KEY", "PATCHFORGE_API_KEY", "GITHUB_WEBHOOK_SECRET",
               "GITHUB_PRIVATE_KEY"}
PLACEHOLDERS = {"", "change-me", "changeme", "your-key-here", "xxx"}

PATTERNS = [
    (re.compile(r"\btvly-[A-Za-z0-9_-]{16,}"), "Tavily API key"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "GitHub token"),
    (re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private key"),
    (re.compile(r"\bv1\.[A-Za-z0-9_-]{60,}"), "Nebius API key"),
    (re.compile(r"(?i)\b(client_secret|api_key|secret_key)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{24,}"), "secret value"),
]
SKIP = {"scripts/check_secrets.py", "tests/test_secrets.py"}


def files_to_check() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [ROOT / f for f in out.splitlines() if f not in SKIP]


def problems() -> list[str]:
    found = []
    for path in files_to_check():
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(ROOT)
        for pattern, what in PATTERNS:
            if pattern.search(text):
                found.append(f"{rel}: looks like a {what}")
        if path.name == ".env.example":
            for line in text.splitlines():
                key, _, value = line.partition("=")
                if key.strip() in SECRET_VARS and value.strip().strip("\"'") not in PLACEHOLDERS:
                    found.append(f"{rel}: {key.strip()} has a real value; keep it empty and put the real one in .env")
    return found


def install() -> None:
    hook = ROOT / ".git" / "hooks" / "pre-commit"
    hook.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{Path(__file__).resolve()}"\n')
    hook.chmod(0o755)
    print(f"installed: {hook}")


if __name__ == "__main__":
    if "--install" in sys.argv:
        install()
        sys.exit(0)
    issues = problems()
    for issue in issues:
        print("SECRET CHECK:", issue)
    if issues:
        print("Commit blocked: remove the values above (real keys go in .env, which is never committed).")
    sys.exit(1 if issues else 0)
