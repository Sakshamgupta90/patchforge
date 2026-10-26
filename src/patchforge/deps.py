"""Bump a library's version in a project's dependency files (Python or Node).

Done deterministically, not by the model: rewriting code for v2 is useless if
requirements.txt / package.json still pins v1, because the sandbox installs what's pinned.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .skills import CONCRETE_VERSION, normalize

REQUIREMENT_FILES = ("requirements.txt", "requirements-dev.txt", "requirements.in")
# name, optional [extras], optional version spec, optional ; marker or # comment
_REQ_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<extras>\[[^\]]*\])?"
    r"(?P<spec>\s*(?:[<>=!~]=?|===)\s*[^;#\s,]+(?:\s*,\s*(?:[<>=!~]=?|===)\s*[^;#\s,]+)*)?"
    r"(?P<rest>\s*(?:[;#].*)?)$"
)


def version_spec(to_version: str) -> str:
    """'2.9.2' -> '==2.9.2', '2' -> '>=2,<3', 'latest'/'unknown' -> '' (unpinned)."""
    v = (to_version or "").strip().lstrip("v")
    if not CONCRETE_VERSION.match(v):
        return ""
    if "." not in v:
        return f">={v},<{int(v) + 1}"
    return f"=={v}"


def _rewrite(requirement: str, library: str, spec: str) -> str | None:
    m = _REQ_LINE.match(requirement)
    if not m or normalize(m.group("name")) != normalize(library):
        return None
    return f"{m.group('indent')}{m.group('name')}{m.group('extras') or ''}{spec}{m.group('rest')}"


NPM_SECTIONS = ("dependencies", "devDependencies", "optionalDependencies")


def npm_spec(old: str, to_version: str) -> str:
    """Keep the project's pinning style: '^4.17.1' -> '^5.1.0', '4.17.1' -> '5.1.0'; unknown -> 'latest'."""
    v = (to_version or "").strip().lstrip("v")
    if not CONCRETE_VERSION.match(v):
        return "latest"
    if "." not in v:
        return f"^{v}.0.0"
    prefix = old[0] if old[:1] in ("^", "~") else ""
    return prefix + v


def bump_npm(workdir: Path, library: str, to_version: str) -> list[str]:
    path = workdir / "package.json"
    if not path.is_file():
        return []
    text = path.read_text()
    data = json.loads(text)
    changed = False
    for section in NPM_SECTIONS:
        deps = data.get(section) or {}
        if library in deps:
            new = npm_spec(deps[library], to_version)
            changed |= deps[library] != new
            deps[library] = new
    if not changed:
        return []
    m = re.search(r"^([ \t]+)\S", text, flags=re.MULTILINE)
    indent = m.group(1) if m else "  "
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False) + ("\n" if text.endswith("\n") else ""))
    return ["package.json"]


def bump(workdir: Path, library: str, to_version: str, ecosystem: str = "python") -> list[str]:
    """Rewrite every pin of `library`. Returns the files that changed."""
    if ecosystem == "node":
        return bump_npm(workdir, library, to_version)
    spec = version_spec(to_version)
    changed = []

    for name in REQUIREMENT_FILES:
        path = workdir / name
        if not path.is_file():
            continue
        lines = path.read_text().splitlines(keepends=True)
        out = []
        for line in lines:
            body, nl = (line[:-1], "\n") if line.endswith("\n") else (line, "")
            new = _rewrite(body, library, spec)
            out.append((new if new is not None else body) + nl)
        if out != lines:
            path.write_text("".join(out))
            changed.append(name)

    pyproject = workdir / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text()

        def repl(m: re.Match) -> str:
            new = _rewrite(m.group(2), library, spec)
            return m.group(1) + (new if new is not None else m.group(2)) + m.group(1)

        new_text = re.sub(r"""(["'])([A-Za-z0-9][^"'\n]*)\1""", repl, text)
        if new_text != text:
            pyproject.write_text(new_text)
            changed.append("pyproject.toml")

    return changed


# --- runtime pin (.python-version / .nvmrc) ------------------------------------------------

def current_runtime(workdir: Path, ecosystem: str) -> str:
    """The Python ("3.10") or Node ("16") version the project runs on today, as the sandbox will see it."""
    from .sandbox.toolchain import DEFAULT_NODE, node_major

    if ecosystem == "node":
        return node_major(workdir) or DEFAULT_NODE
    f = workdir / ".python-version"
    m = re.search(r"(\d+\.\d+)", f.read_text()) if f.is_file() else None
    return m.group(1) if m else "3.12"


def set_runtime(workdir: Path, ecosystem: str, version: str) -> str:
    """Write the runtime pin the sandbox reads. Returns the file name."""
    name = ".nvmrc" if ecosystem == "node" else ".python-version"
    if ecosystem == "node" and not (workdir / ".nvmrc").is_file() and (workdir / ".node-version").is_file():
        name = ".node-version"
    (workdir / name).write_text(f"{version}\n")
    return name


def runtime_key(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", version)[:2])
