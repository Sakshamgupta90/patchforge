"""Thin wrappers around the git CLI.

The token is passed to git as an HTTP header through environment variables
(GIT_CONFIG_COUNT/KEY/VALUE), so it never appears in the clone URL, in
.git/config, in the process list, or in error messages.
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def _auth_env(token: str | None) -> dict[str, str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update({
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraheader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
        })
    return env


def _scrub(text: str, token: str | None) -> str:
    if token:
        text = text.replace(token, "***")
        text = text.replace(base64.b64encode(f"x-access-token:{token}".encode()).decode(), "***")
    # Belt and braces: hide anything that looks like a GitHub token.
    return re.sub(r"gh[pousr]_[A-Za-z0-9]{20,}", "***", text)


def _git(args: list[str], *, cwd: Path | None = None, token: str | None = None,
         timeout: float = 300) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=cwd, env=_auth_env(token), capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise GitError(f"git {args[0]} timed out after {timeout:.0f}s") from e
    if proc.returncode != 0:
        msg = _scrub((proc.stderr or proc.stdout).strip(), token)
        raise GitError(f"git {args[0]} failed: {msg}")
    return proc.stdout


def clone(url: str, dest: Path, *, token: str | None = None, branch: str | None = None,
          depth: int | None = None) -> Path:
    args = ["clone", "--quiet"]
    if branch:
        args += ["--branch", branch]
    if depth:
        args += ["--depth", str(depth)]
    _git([*args, url, str(dest)], token=token)
    return dest


def create_branch(repo: Path, name: str) -> None:
    _git(["checkout", "--quiet", "-b", name], cwd=repo)


def changed_files(repo: Path) -> list[str]:
    out = _git(["status", "--porcelain", "--untracked-files=all"], cwd=repo)
    files = []
    for line in out.splitlines():
        path = line[3:]
        if " -> " in path:  # renames
            path = path.split(" -> ", 1)[1]
        files.append(path.strip('"'))
    return sorted(files)


def commit_all(repo: Path, message: str, *, author_name: str, author_email: str) -> str:
    _git(["add", "--all"], cwd=repo)
    _git(["-c", f"user.name={author_name}", "-c", f"user.email={author_email}",
          "commit", "--quiet", "--no-verify", "-m", message], cwd=repo)
    return _git(["rev-parse", "HEAD"], cwd=repo).strip()


def push(repo: Path, branch: str, *, token: str | None = None, remote: str = "origin") -> None:
    _git(["push", "--quiet", remote, f"HEAD:refs/heads/{branch}"], cwd=repo, token=token)


def staged_diff(repo: Path) -> list[tuple[str, list[str], int]]:
    """Every pending change as (path, added lines, added + removed line count). Stages everything first."""
    _git(["add", "--all"], cwd=repo)
    counts = {}
    for line in _git(["diff", "--cached", "--numstat", "--no-renames", "-z"], cwd=repo).split("\0"):
        parts = line.split("\t", 2)
        if len(parts) == 3:
            counts[parts[2]] = sum(int(n) for n in parts[:2] if n.isdigit())  # "-" for binary files
    added: dict[str, list[str]] = {}
    path = None
    for line in _git(["diff", "--cached", "-U0", "--no-color", "--no-renames"], cwd=repo).splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            added.setdefault(path, [])
        elif line.startswith("+") and not line.startswith("+++") and path:
            added[path].append(line[1:])
    return [(p, added.get(p, []), n) for p, n in counts.items()]
