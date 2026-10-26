
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI
from tavily import TavilyClient

from . import deps
from .contracts import AgentEvent, AgentOutput, AgentTask, EmitFn, Sandbox, SandboxResult
from .registry import PackageRegistry
from .skills import CONCRETE_VERSION, SkillStore

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
# Writes the code changes (first patch and fixes). A code-specialised model: on identical jobs it passed
# Flask on the first try and was the only model to pass Express, where Qwen3-30B/235B, GLM-5.3 and
# Nemotron failed. Overridable from .env.
WORKER_MODEL = "moonshotai/Kimi-K2.7-Code"
# Fixes after a failed check. Defaults to the worker model (the code model). Overridable from .env.
FIXER_MODEL = None


#Prices below as on 17Oct. (idk how to implement a always correct prices thing)
PRICES = {
    MODEL: (0.20, 0.60),
    "MiniMaxAI/MiniMax-M3": (0.30, 1.20),
    "NousResearch/Hermes-4-405B": (1.00, 3.00),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (0.10, 0.30),
    "Qwen/Qwen3-Embedding-8B": (0.01, 0.0),  # embeddings only
    "Qwen/Qwen3.5-397B-A17B": (0.60, 3.60),
    "Qwen/Qwen3.8-27B": (0.45, 3.00),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.14, 0.28),
    "deepseek-ai/DeepSeek-V4-Pro": (1.75, 3.50),
    "deepseek-ai/DeepSeek-V4-Pro-0813": (1.32, 3.96),
    "deepseek-ai/DeepSeek-V4.1-Flash": (0.30, 1.20),
    "google/gemma-3-27b-it": (0.10, 0.30),
    "moonshotai/Kimi-K2.6": (0.95, 4.00),
    "moonshotai/Kimi-K2.7-Code": (0.95, 4.00),
    "moonshotai/Kimi-K3": (3.00, 15.00),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
    "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openbmb/MiniCPM-V-4_5": (0.658, 1.11),
    "zai-org/GLM-5.1": (1.40, 4.40),
    "zai-org/GLM-5.2": (1.40, 4.40),
    "zai-org/GLM-5.3": (1.40, 4.40),
    "zai-org/GLM-5.3-Flash": (0.15, 0.50),
}

# Reasoning models (e.g. Kimi-K2.7-Code) think before answering and the thinking counts against this
# limit; 8k cut their replies off mid-file, so the patch was (correctly) discarded.
PATCH_MAX_TOKENS = 32_000
# A code-model call that hasn't answered by then is retried on the research model. Response times vary
# a lot under load (the same 4-5k-token reply took 36s once and 230s the next time).
CALL_TIMEOUT_S = float(os.getenv("PATCHFORGE_MODEL_TIMEOUT", "120"))
MAX_FILE_CHARS = 40_000  #files that are too large will result in too expensive. (kept it short for now)
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__", "site-packages", ".tox", "build", "dist",
             ".next", ".pf-deps", ".pf-npm-cache", "coverage"}
DEP_FILES = ("requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile",
             "package.json", ".nvmrc", ".node-version", ".python-version")
TEST_DIRS = {"tests", "test", "__tests__"}

SOURCE_EXTS = {
    "python": {".py"},
    "node": {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue"},
}
# Config files that migrations often touch (tsconfig options, next.config, angular.json...).
CONFIG_RE = re.compile(r"^(tsconfig.*\.json|jsconfig\.json|angular\.json|\.babelrc|babel\.config\.\w+|"
                       r"[\w-]+\.config\.(js|cjs|mjs|ts))$")
FENCE = {".py": "python", ".js": "javascript", ".jsx": "jsx", ".mjs": "javascript", ".cjs": "javascript",
         ".ts": "typescript", ".tsx": "tsx", ".vue": "vue", ".json": "json"}
LANGUAGE = {"python": "Python", "node": "JavaScript/TypeScript"}
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{5,}")
_COMMON = {"import", "require", "export", "default", "return", "function", "string", "number", "boolean",
           "module", "exports", "object", "client", "server", "config", "options", "value", "values"}


SOURCE_CHARS = 6000  # full page text per search result (snippets are too short to hold a migration guide)
OFFICIAL_CHARS = 40000  # the official migration guide is the most valuable page: keep all of it
_ERROR_HINT = re.compile(r"(Error|Exception|error TS\d+|Cannot |Missing |is not a function|has no attribute|"
                         r"not defined|not supported|deprecated|removed)", re.IGNORECASE)


def source_text(result: dict, limit: int = SOURCE_CHARS) -> str:
    return (result.get("raw_content") or result.get("content") or "")[:limit]


_ERROR_MESSAGE = re.compile(r"(^[\w.]*(Error|Exception)\b[^:]{0,40}:\s*\S|\berror TS\d+:)")
_CODE_LINE = re.compile(r"^(throw |raise |return |const |let |var |def |class |if |assert )|[;{]$")


def failure_excerpt(log: str, limit: int = 5000) -> str:
    """The parts of a test log that explain the failures: each failing test's block, not just the tail."""
    blocks = re.findall(r"^not ok \d+ - .*?(?=^(?:not )?ok \d+ - |^1\.\.\d+|\Z)", log, flags=re.MULTILINE | re.DOTALL)
    if not blocks:  # pytest and others: the summary at the end already lists every failure
        return log[-limit:]
    head = log[:600] if "Error" in log[:600] else ""  # a crash before any test ran
    out = head + "\n".join(b.strip() for b in blocks)
    return out[:limit]


def error_lines(log: str, limit: int = 2) -> list[str]:
    """Up to `limit` distinct error messages worth searching the web for, one per failure block.

    Test assertion failures are left out: they're about this project's behaviour, which no web page
    describes (the failing test's code, shown to the model, is what helps there).
    """
    found = []
    for block in [log] + re.findall(r"^not ok \d+ - .*?(?=^(?:not )?ok \d+ - |\Z)", log, flags=re.MULTILINE | re.DOTALL):
        line = key_error_line(block)
        if line and line not in found and "AssertionError" not in line and not line.startswith("assert "):
            found.append(line)
    return found[:limit]


def key_error_line(log: str) -> str | None:
    """The line of a failure log a developer would paste into a search engine.

    Prefers a real error message ("PathError: …", "AttributeError: …", "error TS2591: …") over
    stack frames and the source lines that tracebacks quote.
    """
    lines = []
    for raw in log.splitlines():
        line = re.sub(r"^[#E>\s|]+", "", raw).strip()
        if len(line) < 12 or line.startswith(("at ", "File ", "Traceback", "npm error")) or "node_modules/" in line \
                or re.match(r"^(name|code|failureType|type|operator|stack|location|expected|actual):", line):
            continue
        lines.append(line)
    clean = lambda l: re.sub(r"\s+", " ", l.replace("/work/", ""))[:200]  # noqa: E731
    for line in lines:
        if _ERROR_MESSAGE.search(line):
            return clean(line)
    for line in lines:
        if _ERROR_HINT.search(line) and not _CODE_LINE.search(line):
            return clean(line)
    return None


_TEST_PATH = re.compile(r"([\w./-]*?(?:test_[\w-]+\.py|[\w-]+_test\.py|[\w.-]+\.(?:test|spec)\.[cm]?[jt]sx?))")


def related_tests(workdir: Path, result: SandboxResult, limit: int = 2, max_chars: int = 6000) -> str:
    """Source of the test files behind a failure, shown to the model read-only on a retry.

    The error alone often isn't enough: the model needs to see what the test expects (for example,
    that a model is built from an object, not a dict) to fix the code rather than guess again.
    """
    candidates = [name.split("::")[0].replace(".", "/") + ".py" for name in result.failed_tests if "::" in name]
    for m in _TEST_PATH.finditer(result.error_log):
        path = m.group(1)
        # "/work/test/x.test.js" (Docker) or "/tmp/pf-local-…/work/test/x.test.js" (local) -> "test/x.test.js"
        candidates.append(path.split("/work/", 1)[1] if "/work/" in path else path.lstrip("./"))
    seen, parts = set(), []
    for rel in candidates:
        path = (workdir / rel).resolve()
        if rel in seen or not path.is_file() or not path.is_relative_to(workdir.resolve()):
            continue
        seen.add(rel)
        parts.append(f"TEST FILE {rel}:\n```\n{path.read_text(errors='replace')[:max_chars]}\n```")
        if len(parts) == limit:
            break
    return "\n\n".join(parts)


# Edits that make checks pass by switching checking off. A patch that newly adds one is rejected:
# the upgrade has to fix the code, not lower the bar.
WEAKENERS = [
    (re.compile(r'"(strict|noImplicitAny|strictNullChecks|noImplicitThis|alwaysStrict)"\s*:\s*false'), "turns off a TypeScript strictness check"),
    (re.compile(r"@ts-(ignore|nocheck|expect-error)"), "silences TypeScript errors"),
    (re.compile(r"#\s*type:\s*ignore"), "silences type errors"),
    (re.compile(r"eslint-disable"), "disables lint rules"),
    (re.compile(r"pytest\.(skip|xfail)|@pytest\.mark\.(skip|xfail)|\b(it|test|describe)\.skip\("), "skips tests"),
    (re.compile(r"filterwarnings\(\s*['\"]ignore"), "hides warnings"),
]


def files_in_error(workdir: Path, log: str, ecosystem: str) -> list[Path]:
    """Project files an error points at ("src/index.ts(12,10): error TS7053", "app/views.py:12"), even ones
    that don't import the library: an upgrade can break code that only *uses* what changed."""
    workdir = Path(workdir).resolve()
    found = []
    for m in re.finditer(r"([\w./-]+\.(?:py|[cm]?[jt]sx?|vue|json))", log):
        raw = m.group(1)
        rel = raw.split("/work/", 1)[1] if "/work/" in raw else raw.lstrip("./")
        path = (workdir / rel).resolve()
        if not path.is_file() or not path.is_relative_to(workdir) or path in found:
            continue
        r = path.relative_to(workdir)
        if SKIP_DIRS & set(r.parts) or is_test_file(r) or path.name in ("package.json", "package-lock.json"):
            continue
        if path.suffix in SOURCE_EXTS[ecosystem] or CONFIG_RE.match(path.name):
            found.append(path)
    return found


def weakening(original: str, new: str) -> str | None:
    """Why a patch only passes by weakening the checks, or None."""
    for pattern, why in WEAKENERS:
        if len(pattern.findall(new)) > len(pattern.findall(original)):
            return why
    return None


def siblings(lib: str, other: str) -> bool:
    """Packages released in lockstep: react/react-dom, @angular/core/@angular/router, vue/vue-router."""
    scope = lambda n: n.split("/")[0] if n.startswith("@") else None  # noqa: E731
    if scope(lib) and scope(lib) == scope(other) and not lib.startswith("@types/"):
        return True
    base = lambda n: re.split(r"[-/]", n.lstrip("@"))[0]  # noqa: E731
    return not other.startswith("@types/") and base(lib) == base(other)


def ecosystem_of(workdir: Path) -> str:
    return "node" if (workdir / "package.json").is_file() else "python"


def is_test_file(rel: Path) -> bool:
    """Tests are the referee: the agent must not edit them, or it could make them pass by weakening them."""
    name = rel.name
    return (bool(TEST_DIRS & set(rel.parts[:-1])) or name.startswith("test_") or name.endswith("_test.py")
            or name == "conftest.py" or re.search(r"\.(test|spec)\.[cm]?[jt]sx?$", name) is not None)


def parse_json(raw: str | None) -> dict:
    if not raw:
        raise ValueError("Empty reply from model")
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in reply")
    return json.loads(raw[start:end + 1])


def extract_code(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    if raw.strip().startswith("NO_CHANGE"):
        return None
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", raw, flags=re.DOTALL)
    return m.group(1) if m else None


class BrainAgent:
    def __init__(self, model: str | None = None, max_retries: int = 5, llm=None, tavily=None,
                 worker_model: str | None = None, workers: int | None = None,
                 skills: SkillStore | None = None, registry=None, fixer_model: str | None = None):
        load_dotenv()
        # model = the "architect": identifies the library, builds the Delta Spec, handles retries.
        self.model = model or os.getenv("PATCHFORGE_ARCHITECT_MODEL") or MODEL
        self.worker_model = worker_model or os.getenv("PATCHFORGE_WORKER_MODEL") or WORKER_MODEL
        self.fixer_model = fixer_model or os.getenv("PATCHFORGE_FIXER_MODEL") or FIXER_MODEL or self.worker_model
        self.workers = workers or int(os.getenv("PATCHFORGE_WORKERS", "4"))
        self.max_retries = max_retries
        self.skills = skills if skills is not None else SkillStore(Path(os.getenv("PATCHFORGE_SKILLS_DIR", "skills")))
        self._llm = llm
        self._tavily = tavily
        self.registry = registry or PackageRegistry()  # PyPI / npm: real versions and the runtime they need

    @property
    def llm(self):
        if self._llm is None:
            load_dotenv()
            self._llm = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=os.getenv("NEBIUS_API_KEY"))
        return self._llm

    @property
    def tavily(self):
        if self._tavily is None:
            load_dotenv()
            self._tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        return self._tavily


    def _chat(self, emit: EmitFn, stage: str, message: str, prompt: str, max_tokens: int = 4000,
              model: str | None = None, **data) -> tuple[str, str]:
        model = model or self.model
        t0 = time.monotonic()
        client = self.llm.with_options(timeout=CALL_TIMEOUT_S, max_retries=0) if hasattr(self.llm, "with_options") else self.llm
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=max_tokens,
            )
        except APITimeoutError:
            if model == self.model:
                raise
            emit(AgentEvent(stage=stage, level="warning",
                            message=f"{model.split('/')[-1]} didn't answer within {CALL_TIMEOUT_S:.0f}s; "
                                    f"asking {self.model.split('/')[-1]} instead"))
            return self._chat(emit, stage, message, prompt, max_tokens=max_tokens, model=self.model, **data)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        t_in, t_out = r.usage.prompt_tokens, r.usage.completion_tokens
        p_in, p_out = PRICES.get(model, (0.0, 0.0))
        finish = r.choices[0].finish_reason
        emit(AgentEvent(
            stage=stage, message=message, model=model, tokens_in=t_in, tokens_out=t_out,
            cost_usd=round(t_in / 1e6 * p_in + t_out / 1e6 * p_out, 6), latency_ms=latency_ms,
            level="warning" if finish == "length" else "info",
            data={**data, "finish_reason": finish},
        ))
        return r.choices[0].message.content or "", finish

    def identify(self, task: AgentTask, workdir: Path, emit: EmitFn) -> dict:
        eco = ecosystem_of(workdir)
        deps = []
        for name in DEP_FILES:
            p = workdir / name
            if p.is_file():
                deps.append(f"FILE {name}:\n{p.read_text(errors='replace')[:3000]}")
        registry = "npm" if eco == "node" else "PyPI"
        import_hint = "name used in `require('x')` / `import ... from 'x'`" if eco == "node" else "name used in `import x`"
        prompt = f"""A developer gave this instruction for their {LANGUAGE[eco]} repository: "{task.instruction}"

Dependency files from the repository:
{chr(10).join(deps) or "(none found)"}

Work out which library is being upgraded or patched. Return ONLY valid JSON, no markdown:
{{"library": "package name on {registry}", "import_name": "{import_hint}", "from_version": "currently pinned or installed version, or 'unknown'", "to_version": "target version, or 'latest' if the instruction gives none", "companions": ["other packages already in the dependency files that must move together with it (for example react-dom with react, or every @angular/* package); [] if none"]}}"""
        raw, _ = self._chat(emit, "research", "Identified the library and target version", prompt, max_tokens=400)
        info = parse_json(raw)
        info["ecosystem"] = eco
        info.setdefault("import_name", info["library"] if eco == "node" else info["library"].replace("-", "_"))
        info["companions"] = [c for c in info.get("companions") or [] if isinstance(c, str) and c != info["library"]]
        self.resolve_version(info, emit)
        return info

    def resolve_version(self, info: dict, emit: EmitFn) -> None:
        """Turn 'latest' or a bare major ('2') into the registry's real newest version."""
        eco, lib = info["ecosystem"], info["library"]
        wanted = str(info.get("to_version") or "latest").strip().lstrip("v")
        if CONCRETE_VERSION.match(wanted) and wanted.count(".") >= 2:
            resolved = wanted
        else:
            major = wanted.split(".")[0] if CONCRETE_VERSION.match(wanted) else None
            resolved = self.registry.latest_version(eco, lib, major=major)
        if resolved:
            info["to_version"] = resolved
            emit(AgentEvent(stage="research", message=f"Latest {lib} on {'npm' if eco == 'node' else 'PyPI'}: {resolved}",
                            data={"library": lib, "version": resolved}))

    def plan_versions(self, workdir: Path, info: dict, to_version: str, emit: EmitFn) -> tuple[dict, list[str]]:
        """Versions to pin for the library and its companions, raising the runtime pin first if needed.

        Returns ({package: version}, files changed). Runtime handling is deterministic, not the model's job:
        pandas 3 needs Python 3.11+, Next.js 16 needs Node 20+; the sandbox reads .python-version / .nvmrc.
        """
        eco, lib = info["ecosystem"], info["library"]
        changed: list[str] = []
        runtime = deps.current_runtime(workdir, eco)
        needed = self.registry.min_runtime(eco, lib, to_version)
        if needed and deps.runtime_key(needed) > deps.runtime_key(runtime):
            changed.append(deps.set_runtime(workdir, eco, needed))
            emit(AgentEvent(stage="patch", message=f"{lib} {to_version} needs {'Node' if eco == 'node' else 'Python'} "
                                                   f"{needed}+; raised the project from {runtime} to {needed}",
                            data={"files": changed}))
            runtime = needed
        targets = {lib: to_version}
        major = to_version.split(".")[0]
        for comp in info.get("companions", []):
            # Only lockstep siblings (react-dom, @angular/*) share the main package's major version:
            # TypeScript 7 does not mean @types/node 7. Every companion must run on the project's runtime.
            lockstep = siblings(lib, comp) and self.registry.latest_version(eco, comp, major=major, runtime=runtime)
            targets[comp] = lockstep or self.registry.latest_version(eco, comp, runtime=runtime) or "latest"
        return targets, changed

    def _search(self, query: str, emit: EmitFn, what: str, max_results: int = 5,
                domains: list[str] | None = None) -> list[dict]:
        t0 = time.monotonic()
        extra = {"include_domains": domains} if domains else {}
        results = self.tavily.search(query=query, max_results=max_results, search_depth="advanced",
                                     include_raw_content="markdown", **extra).get("results", [])
        n = len(results)
        emit(AgentEvent(stage="research", message=f"Searched the web for {what} ({n} source{'' if n == 1 else 's'})",
                        latency_ms=round((time.monotonic() - t0) * 1000, 1),
                        data={"query": query, "sources": [r["url"] for r in results]}))
        return results

    def research(self, info: dict, emit: EmitFn) -> dict:
        lib, frm, to = info["library"], info["from_version"], info["to_version"]
        package = f"the {lib} npm package" if info.get("ecosystem") == "node" else f"the {lib} Python package"
        major = str(to).split(".")[0]
        try:
            official = self.registry.official_sites(info.get("ecosystem", "python"), lib)
        except Exception:  # noqa: BLE001 - research still works without it
            official = []
        # The project's own docs first (that's where the real migration guide lives), then the wider web.
        rounds = []
        # Docs sites beat the source repo here: repo searches surface issues and discussions.
        docs = [d for d in official if d != "github.com"] or official
        if docs:
            # Short, guide-title phrasing ("Migrating to Express 5") finds official guides best.
            rounds.append((f"Migrating to {lib} {major}", f"the official {lib} {major} migration guide on {', '.join(docs)}",
                           docs, 3))
        rounds.append((f"{package} upgrade from {frm} to {to}: migration guide and breaking changes",
                       f"the {lib} {to} migration guide", None, 4))
        rounds.append((f"{lib} {major} official migration guide: what changed and how to update code",
                       "more migration notes", None, 4))

        spec: dict = {"library": lib, "from_version": frm, "to_version": to, "changes": []}
        collected: list[dict] = []
        for i, (query, what, domains, n) in enumerate(rounds):
            seen = {r["url"] for r in collected}
            collected += [r for r in self._search(query, emit, what, n, domains) if r["url"] not in seen]
            # Official pages plus the general search together make the first spec; a third search only
            # if that still found nothing.
            if i == 0 and official and len(rounds) > 1:
                continue
            sources = "\n\n".join(
                f"SOURCE: {r['url']}\n{source_text(r, OFFICIAL_CHARS if any(d in r['url'] for d in docs) else SOURCE_CHARS)}"
                for r in collected[:7])
            spec = self._extract_spec(lib, frm, to, sources, emit)
            if spec.get("changes"):
                return spec
            emit(AgentEvent(stage="research", level="warning",
                            message="Those sources didn't describe any breaking changes; trying another search"))
        return spec

    def _extract_spec(self, lib: str, frm: str, to: str, sources: str, emit: EmitFn) -> dict:
        prompt = f"""You are given web pages about upgrading {lib} from {frm} to {to}.
Extract every breaking change a developer must apply to their code or configuration: renamed or removed functions, changed arguments, new syntax (for example route patterns or config options), and new runtime requirements. Only include changes supported by the sources. Do not invent APIs. Do not include changes that are not breaking.
Copy code VERBATIM from the sources' own before/after examples, character for character (slashes, dots, braces, quotes): small syntax details are usually the whole point of a change. Never paraphrase or simplify an example.
Return ONLY valid JSON, no markdown, no explanation, in this shape:
{{"library": "...", "from_version": "...", "to_version": "...",
  "changes": [{{"old": "old usage, verbatim from the source", "new": "new usage, verbatim from the source", "note": "short explanation", "example": "the source's own before/after code snippet, verbatim, if it has one", "source": "url"}}]}}

{sources}"""
        for attempt in (1, 2):
            raw, finish = self._chat(emit, "research", f"Built the Delta Spec (attempt {attempt})", prompt, max_tokens=6000)
            try:
                spec = parse_json(raw)
                spec["library"], spec["to_version"] = lib, to  # the registry's version is the truth
                spec.setdefault("changes", [])
                emit(AgentEvent(stage="research", message=f"Found {len(spec['changes'])} breaking API changes",
                                data={"changes": len(spec["changes"])}))
                return spec
            except (ValueError, json.JSONDecodeError) as e:
                emit(AgentEvent(stage="research", level="warning", message=f"Could not parse the Delta Spec: {e}",
                                data={"finish_reason": finish, "raw_head": raw[:300]}))
        raise ValueError("model never returned a parseable Delta Spec")

    def lookup_error(self, info: dict, result: SandboxResult, emit: EmitFn) -> str:
        """Search the web for the failure itself, like a developer pasting the error into a search engine.

        This is where post-cutoff knowledge matters most: the model's memory says `/files/*` is a valid
        Express route; the Express 5 docs say it must be `/files/*splat`.
        """
        notes = []
        for line in error_lines(result.error_log):
            if line not in self._error_notes:
                results = self._search(f"{info['library']} {info['to_version']} {line}", emit,
                                       f'the error "{line[:90]}"', max_results=3)
                self._error_notes[line] = "\n\n".join(f"SOURCE: {r['url']}\n{source_text(r, 3000)}" for r in results)
            notes.append(self._error_notes[line])
        return "\n\n".join(n for n in notes if n)


    def find_files(self, workdir: Path, info: dict, spec: dict) -> list[Path]:
        eco = info.get("ecosystem") or ecosystem_of(workdir)
        names = [info["import_name"], *info.get("companions", [])]
        if eco == "node":
            alts = "|".join(re.escape(n) for n in names)
            import_re = re.compile(rf"""(require\(\s*|from\s+|import\s+|import\(\s*)['"]({alts})(/[^'"]*)?['"]""")
        else:
            alts = "|".join(re.escape(n.replace("-", "_")) for n in names)
            import_re = re.compile(rf"^\s*(import|from)\s+({alts})\b", re.MULTILINE)
        olds = [c["old"].strip() for c in spec.get("changes", []) if len(c.get("old", "").strip()) > 6]
        tokens = {t for o in olds for t in _TOKEN.findall(o)} - _COMMON
        hits = []
        for p in sorted(workdir.rglob("*")):
            rel = p.relative_to(workdir)
            if not p.is_file() or SKIP_DIRS & set(rel.parts) or is_test_file(rel) or p.stat().st_size > MAX_FILE_CHARS:
                continue
            is_source, is_config = p.suffix in SOURCE_EXTS[eco], CONFIG_RE.match(p.name) is not None
            if not (is_source or is_config):
                continue
            text = p.read_text(errors="replace")
            if (is_source and import_re.search(text)) or any(o in text for o in olds) \
                    or (is_config and any(re.search(rf"\b{re.escape(t)}\b", text) for t in tokens)):
                hits.append(p)
        return hits


    def patch_file(self, path: Path, workdir: Path, spec: dict, emit: EmitFn, feedback: str | None = None,
                   tests: str = "", web: str = "") -> bool:
        rel = str(path.relative_to(workdir))
        original = path.read_text(errors="replace")
        fix = ""
        if feedback:
            fix = f"""
Your previous edit made the project's checks fail. Fix every failure listed. Output:
{failure_excerpt(feedback)}
"""
            if tests:
                fix += f"""
These are the project's tests. They are read-only and define the correct behaviour: change the code so they pass.
When a test and the Delta Spec disagree, the test wins.

{tests}
"""
            if web:
                fix += f"""
Web pages found by searching for this error. They describe the library as it is today, so trust them over your memory:

{web}
"""
        fence = FENCE.get(path.suffix, "text")
        prompt = f"""Update this file so the project works with the new version of the library. Apply ONLY the changes from the Delta Spec that affect this file. Do not refactor, rename, reformat or touch unrelated code.
Keep the program's behaviour exactly the same: when an API was removed, replace it with its new equivalent that produces the same result (same output formats, same status codes, same values). Never just delete code that customises behaviour.
Never make checks pass by switching checking off: no "strict": false / "noImplicitAny": false, no @ts-ignore or @ts-nocheck, no "# type: ignore", no skipped tests. Fix the code so it passes the checks as they are.

DELTA SPEC:
{json.dumps(spec, indent=2)}
{fix}
FILE {rel}:
```{fence}
{original}
```

Reply with the COMPLETE updated file in a single ```{fence} fenced block and nothing else.
If this file needs no change, reply with exactly: NO_CHANGE"""
        # First pass goes to the cheap worker model; a retry means it got it wrong, so escalate to the fixer.
        model = self.fixer_model if feedback else self.worker_model
        raw, finish = self._chat(emit, "retry" if feedback else "patch", f"{'Re-patched' if feedback else 'Patched'} {rel}",
                                 prompt, max_tokens=PATCH_MAX_TOKENS, model=model, files=[rel])
        if finish == "length":
            emit(AgentEvent(stage="patch", level="warning", message=f"Reply for {rel} was cut off; left the file unchanged"))
            return False
        new = extract_code(raw)
        if new is None or new.strip() == original.strip():
            return False
        why = weakening(original, new)
        if why:
            emit(AgentEvent(stage="patch", level="warning",
                            message=f"Rejected the edit to {rel}: it {why} instead of fixing the code"))
            return False
        path.write_text(new if new.endswith("\n") else new + "\n")
        return True

    def _patch_all(self, paths: list[Path], workdir: Path, spec: dict, emit: EmitFn,
                   feedback: str | None = None, tests: str = "", web: str = "") -> set[str]:
        """Patch files in parallel: each file is an independent worker call."""
        if not paths:
            return set()
        with ThreadPoolExecutor(max_workers=min(self.workers, len(paths)), thread_name_prefix="pf-worker") as pool:
            results = list(pool.map(lambda p: (p, self.patch_file(p, workdir, spec, emit, feedback=feedback,
                                                                  tests=tests, web=web)), paths))
        return {str(p.relative_to(workdir)) for p, changed in results if changed}

    def get_spec(self, info: dict, emit: EmitFn) -> tuple[dict, object | None]:
        """Reuse a saved skill for this migration if there is one; otherwise research it."""
        skill = self.skills.find(info["library"], info["to_version"])
        if skill:
            emit(AgentEvent(stage="skill", message=f"Reusing saved skill {skill.path.parent.name} "
                                                   f"(used {skill.uses} times); skipping web research",
                            data={"skill": skill.path.parent.name, "changes": len(skill.spec.get("changes", []))}))
            return skill.spec, skill
        return self.research(info, emit), None

    def run(self, task: AgentTask, workdir: Path, sandbox: Sandbox, emit: EmitFn) -> AgentOutput:
        workdir = Path(workdir)
        self._error_notes: dict[str, str] = {}  # error line -> web results, so each error is searched once
        info = self.identify(task, workdir, emit)
        spec, skill = self.get_spec(info, emit)
        to_version = str(spec.get("to_version") or info["to_version"])

        files = self.find_files(workdir, info, spec)
        emit(AgentEvent(stage="plan", message=f"{len(files)} files use {info['library']}",
                        data={"files": [str(p.relative_to(workdir)) for p in files]}))
        if not files:
            return AgentOutput(changed_files=[], summary=f"No files in this repo use `{info['library']}`; nothing to change.")

        eco = info.get("ecosystem", "python")
        targets, runtime_files = self.plan_versions(workdir, info, to_version, emit)
        changed: set[str] = set(runtime_files)
        for lib, version in targets.items():
            files_bumped = deps.bump(workdir, lib, version, ecosystem=eco)
            if files_bumped:
                changed |= set(files_bumped)
                shown = version if eco == "node" else (deps.version_spec(version) or " (latest)")
                emit(AgentEvent(stage="patch", message=f"Pinned {lib}{'@' if eco == 'node' else ''}{shown} "
                                                       f"in {', '.join(files_bumped)}", data={"files": files_bumped}))

        changed |= self._patch_all(files, workdir, spec, emit)

        result: SandboxResult | None = None
        last_failure, repeats = None, 0
        for attempt in range(self.max_retries + 1):
            result = sandbox.run_tests(workdir)
            emit(AgentEvent(stage="tests", message=f"Agent check: tests {'passed' if result.passed else 'failed'}"
                            + (f" ({len(result.failed_tests)} failing)" if result.failed_tests else ""),
                            level="info" if result.passed else "warning",
                            data={"failed_tests": result.failed_tests, "duration": result.duration, "attempt": attempt}))
            if result.passed or attempt == self.max_retries:
                break
            # Keep retrying while each attempt changes the outcome; stop when stuck on the same failure.
            failure = (tuple(sorted(result.failed_tests)), key_error_line(result.error_log))
            repeats = repeats + 1 if failure == last_failure else 0
            last_failure = failure
            if repeats >= 2:
                emit(AgentEvent(stage="retry", level="warning",
                                message="The same failure came back three times in a row; stopping the retries"))
                break
            if "dependency install failed" in result.error_log:
                # Code edits can't fix packages that don't install together; don't burn retries on it.
                emit(AgentEvent(stage="retry", level="warning",
                                message="The upgraded dependencies don't install; stopping instead of retrying code changes",
                                data={"log_tail": result.error_log[-800:]}))
                break
            emit(AgentEvent(stage="retry", level="warning", message=f"Retry {attempt + 1} (up to {self.max_retries}): sending "
                                                                    "the failure and the failing tests back to the model",
                            data={"failed_tests": result.failed_tests}))
            # prefer files named in the failure output (including ones that don't import the library,
            # like code broken by a removed compiler option), else everything we touched or considered
            named = files_in_error(workdir, result.error_log, info.get("ecosystem", "python"))
            files += [p for p in named if p not in files]
            tests = related_tests(workdir, result) if result.checked == "tests" else ""
            web = self.lookup_error(info, result, emit)
            changed |= self._patch_all(named or files, workdir, spec, emit, feedback=result.error_log, tests=tests,
                                       web=web)

        if result is not None and result.passed and changed:
            if skill:
                self.skills.record_use(skill)
            else:
                saved = self.skills.save(spec, library=info["library"], from_version=str(info["from_version"]),
                                         to_version=to_version)
                emit(AgentEvent(stage="skill", message=f"Saved skill {saved.path.parent.name} for the next repo",
                                data={"skill": saved.path.parent.name}))

        lines = [f"Upgrade `{info['library']}` {spec.get('from_version', info['from_version'])} -> {to_version}"
                 + (f" (with {', '.join(f'`{c}`' for c in info.get('companions', []))})" if info.get("companions") else ""), "",
                 f"Instruction: **{task.instruction}**", ""]
        if skill:
            lines += [f"Reused saved skill `{skill.path.parent.name}` (no web research needed).", ""]
        lines += ["### API changes applied"]
        lines += [f"- `{c.get('old')}` -> `{c.get('new')}`: {c.get('note', '')} ([source]({c.get('source', '')}))" for c in spec.get("changes", [])]
        lines += ["", f"### Files changed ({len(changed)})"] + [f"- `{f}`" for f in sorted(changed)]
        if result is not None and not result.passed:
            lines += ["", f"**Warning:** checks still failing after the retries: {', '.join(result.failed_tests) or 'see logs'}"]
        return AgentOutput(changed_files=sorted(changed), summary="\n".join(lines), test_results=result)
