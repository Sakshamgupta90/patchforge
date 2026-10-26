"""Ask the package registry (PyPI or npm) about real, current releases.

The model's idea of "latest" stops at its training cutoff; the registry doesn't.
It also says which runtime each release needs (`requires_python`, `engines.node`),
so an upgrade can move the project's Python/Node version along with the library.
"""

from __future__ import annotations

import re
from urllib.parse import quote, urlparse

import httpx
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

_STABLE = re.compile(r"^\d+(\.\d+)*$")
PYTHONS = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]
NODES = [str(n) for n in range(14, 27)]
# Links packages list that aren't documentation (chat, funding, social).
NOT_DOCS = {"pypi.org", "npmjs.com", "www.npmjs.com", "discord.gg", "discord.com", "opencollective.com",
            "tidelift.com", "twitter.com", "x.com", "gitter.im", "patreon.com", "github.com/sponsors"}


def _vkey(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def python_allows(requires: str | None, python: str) -> bool:
    if not requires:
        return True
    try:
        return SpecifierSet(requires).contains(Version(python + ".0"), prereleases=True)
    except (InvalidSpecifier, InvalidVersion):
        return True


def _semver(v: str) -> tuple[int, int, int]:
    parts = [int(p) for p in re.findall(r"\d+", v)[:3]]
    return tuple(parts + [0] * (3 - len(parts)))  # type: ignore[return-value]


def node_allows(engines: str | None, major: str) -> bool:
    """Enough of npm's semver ranges for `engines.node` ('>=18', '^18.18.0 || >=20', '14 - 16', '18.x')."""
    if not engines or not engines.strip() or engines.strip() == "*":
        return True
    v = (int(major), 99, 0)  # "Node N" means the latest N.x
    for clause in engines.split("||"):
        clause = clause.strip()
        if " - " in clause:
            lo, hi = clause.split(" - ", 1)
            if _semver(lo) <= v and v[0] <= _semver(hi)[0]:
                return True
            continue
        ok = True
        for m in re.finditer(r"(>=|<=|>|<|\^|~|=)?\s*v?(\d+(?:\.(?:\d+|x|\*))*)", clause):
            op, ver = m.group(1) or "", m.group(2)
            base = _semver(ver.replace("x", "0").replace("*", "0"))
            if op == ">=":
                ok &= v >= base
            elif op == ">":
                ok &= v > base
            elif op == "<=":
                ok &= v <= base
            elif op == "<":
                ok &= v < base
            elif op in ("^", "~"):
                ok &= v[0] == base[0] and v >= base
            else:  # "=18", bare "18" or "18.x"
                ok &= v[0] == base[0]
        if ok:
            return True
    return False


class PackageRegistry:
    def __init__(self, client: httpx.Client | None = None):
        self.http = client or httpx.Client(timeout=15, headers={"User-Agent": "patchforge"})
        self._cache: dict[tuple[str, str], dict] = {}
        self._meta: dict[tuple[str, str], dict] = {}

    def _releases(self, ecosystem: str, name: str) -> dict:
        """{version: required runtime spec or None} for stable releases, plus the registry's 'latest'."""
        key = (ecosystem, name)
        if key not in self._cache:
            try:
                if ecosystem == "node":
                    data = self.http.get(f"https://registry.npmjs.org/{quote(name, safe='@')}").json()
                    repo = data.get("repository")
                    self._meta[key] = {"urls": [data.get("homepage"), repo.get("url") if isinstance(repo, dict) else repo]}
                    rel = {v: (meta.get("engines") or {}).get("node") if isinstance(meta.get("engines"), dict) else None
                           for v, meta in data.get("versions", {}).items() if _STABLE.match(v)}
                    latest = data.get("dist-tags", {}).get("latest")
                else:
                    data = self.http.get(f"https://pypi.org/pypi/{quote(name)}/json").json()
                    info = data.get("info") or {}
                    self._meta[key] = {"urls": [info.get("home_page"), *(info.get("project_urls") or {}).values()]}
                    rel = {v: files[0].get("requires_python") for v, files in data.get("releases", {}).items()
                           if files and _STABLE.match(v) and not all(f.get("yanked") for f in files)}
                    latest = data.get("info", {}).get("version")
            except (httpx.HTTPError, ValueError):
                rel, latest = {}, None
            self._cache[key] = {"releases": rel, "latest": latest}
        return self._cache[key]

    def latest_version(self, ecosystem: str, name: str, *, major: str | None = None,
                       runtime: str | None = None) -> str | None:
        """Newest stable release, optionally within one major ("2") and runnable on `runtime` ("3.11" / "20")."""
        info = self._releases(ecosystem, name)
        releases = info["releases"]
        allows = node_allows if ecosystem == "node" else python_allows
        candidates = [v for v, req in releases.items()
                      if (not major or v.split(".")[0] == str(major)) and (not runtime or allows(req, runtime))]
        if not candidates:
            return info["latest"] if not major and not runtime else None
        return max(candidates, key=_vkey)

    def official_sites(self, ecosystem: str, name: str) -> list[str]:
        """Domains the package itself lists (homepage, docs, source repo), so research can go to the
        official migration guide first instead of whatever blog ranks highest."""
        self._releases(ecosystem, name)
        domains = []
        for url in self._meta.get((ecosystem, name), {}).get("urls", []):
            if not isinstance(url, str) or not url:
                continue
            url = re.sub(r"^git\+|\.git$", "", url).replace("git://", "https://").replace("git@github.com:", "https://github.com/")
            host = urlparse(url if "://" in url else f"https://{url}").hostname or ""
            host = host.removeprefix("www.")
            if host and host not in domains and host not in NOT_DOCS:
                domains.append(host)
        return domains[:4]

    def min_runtime(self, ecosystem: str, name: str, version: str) -> str | None:
        """Oldest Python ("3.11") or Node ("20") this release supports; None if it doesn't say."""
        req = self._releases(ecosystem, name)["releases"].get(version)
        if not req:
            return None
        allows, choices = (node_allows, NODES) if ecosystem == "node" else (python_allows, PYTHONS)
        return next((c for c in choices if allows(req, c)), None)


def latest_version(ecosystem: str, name: str, *, major: str | None = None,
                   client: httpx.Client | None = None) -> str | None:
    """Kept for simple callers: newest stable release (optionally within a major)."""
    return PackageRegistry(client).latest_version(ecosystem, name, major=major)
