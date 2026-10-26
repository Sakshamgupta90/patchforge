"""Work out how to install and test a project: Python (pip + pytest) or Node.js (npm).

Every sandbox (Docker, Nebius, local) runs the same two steps from here, so a
new language only needs a new branch in `detect()`.

The runtime version comes from the project itself, the way its own developers
pin it: `.nvmrc` / `.node-version` / `engines.node` for Node, `.python-version`
for Python. Old projects often only build on an old runtime, and an upgrade
that moves to a new runtime should change that file too.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .common import DEPS_DIR, REPORT_FILE, failed_tests_from_junit, interpret_pytest_exit

# Package download cache, mounted at /cache by the sandbox. One per job (never shared between
# jobs or users), so the 3+ test runs in a job download each package once.
CACHE_DIR = "/cache"
DEFAULT_NODE = "22"
DEFAULT_PYTHON_IMAGE = "python:3.12-slim"
NPM_DEFAULT_TEST = "no test specified"
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", DEPS_DIR, "__pycache__", "dist", "build", ".next"}
PY_MANIFESTS = ("requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.py", "setup.cfg", ".python-version")
NODE_MANIFESTS = ("package.json", "package-lock.json", "npm-shrinkwrap.json", ".nvmrc", ".node-version")


@dataclass
class Toolchain:
    language: str  # "python" | "node"
    image: str
    install: str  # shell command, run with network on
    test: str  # shell command, run with network off
    env: dict[str, str] = field(default_factory=dict)
    checked: str = "tests"  # tests | build | typecheck | compile (see SandboxResult.checked)
    deps_dir: str = DEPS_DIR  # where install puts dependencies, relative to the project
    manifests: tuple[str, ...] = ()  # files that decide what gets installed

    def deps_key(self, folder: Path) -> str:
        """Same key = same dependencies, so an earlier install can be reused."""
        h = hashlib.sha256(self.image.encode())
        for name in self.manifests:
            f = Path(folder) / name
            h.update(name.encode() + b"\0" + (f.read_bytes() if f.is_file() else b"-") + b"\0")
        return h.hexdigest()[:16]

    def interpret(self, code: int, log: str, work: Path) -> tuple[bool, list[str], str]:
        """(passed, failed test names, log) from the test step's exit code and output."""
        if self.checked != "tests":
            return code == 0, [], log
        if self.language == "python":
            passed, log = interpret_pytest_exit(code, log)
            return passed, failed_tests_from_junit(work / REPORT_FILE), log
        if "Missing script: \"test\"" in log or "no test specified" in log:
            return False, [], "The project has no `npm test` script.\n" + log
        return code == 0, node_failures(log), log


def node_failures(log: str) -> list[str]:
    """Failing test names from node:test (TAP or spec reporter) or Jest/Vitest output."""
    names = re.findall(r"^\s*not ok \d+ - (.+?)\s*$", log, flags=re.MULTILINE)
    names += re.findall(r"^\s*[✖✕×]\s+(.+?)(?:\s+\(\d+(?:\.\d+)?\s*m?s\))?\s*$", log, flags=re.MULTILINE)
    seen, out = set(), []
    for n in names:
        if n not in seen and not n.startswith("failing tests"):
            seen.add(n)
            out.append(n)
    return out


def node_major(folder: Path) -> str:
    for name in (".nvmrc", ".node-version"):
        f = folder / name
        if f.is_file():
            m = re.search(r"(\d+)", f.read_text())
            if m:
                return m.group(1)
    try:
        engines = json.loads((folder / "package.json").read_text()).get("engines", {}).get("node", "")
        m = re.search(r"(\d+)", engines)
        if m:
            return m.group(1)
    except (OSError, ValueError, AttributeError):
        pass
    return DEFAULT_NODE


def python_image(folder: Path, default: str) -> str:
    f = folder / ".python-version"
    if f.is_file():
        m = re.search(r"(\d+\.\d+)", f.read_text())
        if m:
            return f"python:{m.group(1)}-slim"
    return default


def _files(folder: Path, pattern: str):
    for p in folder.rglob(pattern):
        if not SKIP_DIRS & set(p.relative_to(folder).parts):
            yield p


def has_python_tests(folder: Path) -> bool:
    return any(True for _ in _files(folder, "test_*.py")) or any(True for _ in _files(folder, "*_test.py"))


def node_scripts(folder: Path) -> dict:
    try:
        return json.loads((folder / "package.json").read_text()).get("scripts") or {}
    except (OSError, ValueError):
        return {}


def node_check(folder: Path) -> tuple[str, str]:
    """(command, what it checks) for a Node project: its tests, else its build, else a type/syntax check."""
    scripts = node_scripts(folder)
    test = scripts.get("test", "")
    if test and NPM_DEFAULT_TEST not in test:
        return "npm test", "tests"
    if scripts.get("build"):
        return "npm run build", "build"
    if (folder / "tsconfig.json").is_file():
        return "npx --no-install tsc --noEmit -p .", "typecheck"
    syntax = ("find . -path ./node_modules -prune -o \\( -name '*.js' -o -name '*.cjs' -o -name '*.mjs' \\) "
              "-type f -print | xargs -r -n1 node --check")
    return syntax, "compile"


def python_install(folder: Path) -> str:
    pip = f"python -m pip install --quiet --disable-pip-version-check --target {DEPS_DIR}"
    if (folder / "requirements.txt").exists():
        return f"{pip} -r requirements.txt pytest"
    if (folder / "pyproject.toml").exists() or (folder / "setup.py").exists():
        return f"{pip} . pytest"
    return f"{pip} pytest"


def detect(folder: Path, python_default: str = DEFAULT_PYTHON_IMAGE) -> Toolchain:
    folder = Path(folder)
    if (folder / "package.json").is_file():
        test, checked = node_check(folder)
        return Toolchain(
            language="node",
            image=f"node:{node_major(folder)}-slim",
            # `npm install`, not `npm ci`: the agent may have just changed package.json.
            install="npm install --no-audit --no-fund --loglevel=error",
            test=test, checked=checked, deps_dir="node_modules", manifests=NODE_MANIFESTS,
            env={"npm_config_cache": f"{CACHE_DIR}/npm", "npm_config_update_notifier": "false",
                 "npm_config_prefer_offline": "true", "CI": "true", "NEXT_TELEMETRY_DISABLED": "1"},
        )
    if has_python_tests(folder):
        test, checked = f"python -m pytest -q --tb=short -p no:cacheprovider --junitxml={REPORT_FILE}", "tests"
    else:
        # No tests: at least prove every file still compiles under the new versions' Python.
        test, checked = f"python -m compileall -q -x '(^|/)(\\.pf-deps|\\.venv|venv|node_modules)/' .", "compile"
    return Toolchain(
        language="python",
        image=python_image(folder, python_default),
        install=python_install(folder),
        test=test, checked=checked, deps_dir=DEPS_DIR, manifests=PY_MANIFESTS,
        env={"PIP_CACHE_DIR": f"{CACHE_DIR}/pip", "PYTHONPATH": f"/work/{DEPS_DIR}:/work",
             "PYTHONDONTWRITEBYTECODE": "1"},
    )
