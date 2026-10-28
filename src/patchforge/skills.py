"""Skill store: a migration that worked once is saved and reused on the next repo.

A skill is the Delta Spec (the list of breaking API changes) from a run whose
tests passed. Reusing it skips the web search and the research model call, so
the second repo with the same migration is faster and cheaper.

Layout follows the agent-skill convention, one folder per skill:

    skills/pydantic-2.9.2/SKILL.md

with a small front-matter header and the Delta Spec in a ```json block.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_SPEC = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
CONCRETE_VERSION = re.compile(r"^\d+(\.\d+){0,3}$")


def normalize(name: str) -> str:
    """PyPI names compare case-insensitively with -, _ and . treated alike."""
    return re.sub(r"[-_.]+", "-", name).lower().strip()


@dataclass
class Skill:
    path: Path
    library: str
    from_version: str
    to_version: str
    created: float
    uses: int
    spec: dict


class SkillStore:
    def __init__(self, root: Path, max_age_days: float = 30):
        self.root = Path(root)
        self.max_age_s = max_age_days * 86400
        self._lock = threading.Lock()

    def all(self) -> list[Skill]:
        if not self.root.is_dir():
            return []
        skills = []
        for f in sorted(self.root.glob("*/SKILL.md")):
            skill = self._read(f)
            if skill:
                skills.append(skill)
        return skills

    def find(self, library: str, to_version: str) -> Skill | None:
        """Best saved skill for this library and requested target version.

        "latest" / "unknown" match the newest skill for the library, but only a
        recent one, since "latest" moves over time. A concrete request ("2" or
        "2.9") matches a skill whose version starts with it. A full release
        ("2.11.4") matches a skill for the same major.minor ("2.11.3"): patch
        releases rarely change a migration. The caller keeps the requested
        version as the pin.
        """
        lib = normalize(library)
        wanted = (to_version or "").strip().lower().lstrip("v")
        candidates = [s for s in self.all() if normalize(s.library) == lib]
        if CONCRETE_VERSION.match(wanted):
            minor = wanted.split(".")[:2] if wanted.count(".") >= 2 else None
            candidates = [s for s in candidates
                          if s.to_version == wanted or s.to_version.startswith(wanted + ".")
                          or (minor and s.to_version.split(".")[:2] == minor)]
        else:
            candidates = [s for s in candidates if time.time() - s.created < self.max_age_s]
        return max(candidates, key=lambda s: s.created, default=None)

    def save(self, spec: dict, *, library: str, from_version: str, to_version: str,
             sources: list[str] | None = None) -> Skill:
        to_version = str(spec.get("to_version") or to_version)
        slug = re.sub(r"[^a-z0-9.]+", "-", f"{normalize(library)}-{to_version}".lower()).strip("-")
        folder = self.root / slug
        with self._lock:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "SKILL.md"
            changes = spec.get("changes", [])
            lines = [
                "---",
                f"name: {slug}",
                f"description: Upgrade {library} from {from_version} to {to_version} ({len(changes)} API changes)",
                f"library: {library}",
                f"from_version: {from_version}",
                f"to_version: {to_version}",
                f"created: {time.time():.0f}",
                "uses: 0",
                "---",
                "",
                f"# {library} {from_version} → {to_version}",
                "",
                "Saved by PatchForge after a run whose tests passed. Reused for the same migration in other repos.",
                "",
                "## Changes",
                "",
                *[f"- `{c.get('old', '')}` → `{c.get('new', '')}`: {c.get('note', '')}" for c in changes],
                "",
                "## Sources",
                "",
                *[f"- {u}" for u in (sources or sorted({c.get('source') for c in changes if c.get('source')}))],
                "",
                "## Delta Spec",
                "",
                "```json",
                json.dumps(spec, indent=2),
                "```",
                "",
            ]
            path.write_text("\n".join(lines))
        return self._read(path)

    def record_use(self, skill: Skill) -> None:
        with self._lock:
            text = skill.path.read_text()
            text = re.sub(r"^uses: \d+$", f"uses: {skill.uses + 1}", text, count=1, flags=re.MULTILINE)
            skill.path.write_text(text)
            skill.uses += 1

    @staticmethod
    def _read(path: Path) -> Skill | None:
        try:
            text = path.read_text()
            front = _FRONT.match(text)
            spec = _SPEC.search(text)
            if not front or not spec:
                return None
            meta = dict(line.split(": ", 1) for line in front.group(1).splitlines() if ": " in line)
            return Skill(path=path, library=meta["library"], from_version=meta.get("from_version", "unknown"),
                         to_version=meta["to_version"], created=float(meta.get("created", 0)),
                         uses=int(meta.get("uses", 0)), spec=json.loads(spec.group(1)))
        except (OSError, KeyError, ValueError):
            return None
