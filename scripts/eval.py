"""Run the upgrade cases in evals/cases with the real models and web search, and report how they did.

    python scripts/eval.py                           # every case, once
    python scripts/eval.py --only flask --repeat 3   # models vary between runs: repeat to see through the noise
    python scripts/eval.py --compare evals/results/old.json

Every run starts with an empty skills folder, so the web research always happens. Needs NEBIUS_API_KEY and
TAVILY_API_KEY (from .env) and a sandbox (PATCHFORGE_SANDBOX, default docker). Costs a few cents per case.
Results are saved to evals/results/ so two versions of the code can be compared.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def run_case(case: dict, make_sandbox, skills_root: Path) -> dict:
    from patchforge.brain import BrainAgent
    from patchforge.contracts import AgentTask
    from patchforge.git_ops import staged_diff
    from patchforge.guard import FileDiff, check_diff
    from patchforge.skills import SkillStore

    work = Path(tempfile.mkdtemp(prefix="pf-eval-"))
    try:
        shutil.copytree(ROOT / "evals" / "cases" / case["folder"], work / "repo", ignore=shutil.ignore_patterns(
            ".git", "node_modules", "__pycache__", ".pytest_cache"))
        repo = work / "repo"
        for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=e", "-c", "user.email=e@e", "commit", "-qm", "base"]):
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
        events, sandbox = [], make_sandbox()
        skills = SkillStore(Path(tempfile.mkdtemp(prefix="pf-skills-", dir=skills_root)))
        t0, error = time.monotonic(), None
        try:
            BrainAgent(skills=skills).run(AgentTask(repo="eval/case", instruction=case["instruction"], base_branch="main"),
                                          repo, sandbox, events.append)
            final = sandbox.run_tests(repo)
        except Exception as e:  # noqa: BLE001 - a crash is a failed case, not a failed eval
            final, error = None, f"{type(e).__name__}: {e}"
        seconds = time.monotonic() - t0
        problems = check_diff([FileDiff(p, a, n) for p, a, n in staged_diff(repo)]) if final else []
        searches = [e for e in events if e.message.startswith("Searched the web")]
        return {
            "case": case["name"], "passed": bool(final and final.passed and not problems), "error": error,
            "diff_problems": problems, "seconds": round(seconds, 1),
            "retries": sum(1 for e in events if e.stage == "retry" and e.message.startswith("Retry")),
            "tokens": sum(e.tokens_in + e.tokens_out for e in events), "searches": len(searches),
            "cost_usd": round(sum(e.cost_usd for e in events), 4),
            "changes_found": next((e.data["changes"] for e in events if "breaking API changes" in e.message), 0),
            "warnings": [e.message for e in events if e.level == "warning"][:10],
            "sources": sorted({u for e in searches for u in e.data.get("sources", [])}),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def summarize(runs: list[dict]) -> dict:
    n = len(runs)
    mean = lambda k: sum(r[k] for r in runs) / n
    return {"passed": sum(r["passed"] for r in runs), "runs": n, "retries": mean("retries"), "tokens": mean("tokens"),
            "searches": mean("searches"), "cost_usd": mean("cost_usd"), "seconds": mean("seconds")}


def table(results: list[dict]) -> list[str]:
    by_case: dict[str, list[dict]] = {}
    for r in results:
        by_case.setdefault(r["case"], []).append(r)
    rows = [f"{'case':<20}{'pass':<7}{'retries':<9}{'tokens':<9}{'searches':<10}{'cost':<9}time"]
    for name, runs in by_case.items():
        s = summarize(runs)
        rows.append(f"{name:<20}{str(s['passed']) + '/' + str(s['runs']):<7}{s['retries']:<9.1f}{s['tokens'] / 1000:<9.1f}"
                    f"{s['searches']:<10.1f}${s['cost_usd']:<8.3f}{s['seconds']:.0f}s")
    passed = sum(r["passed"] for r in results)
    rows.append(f"{'total':<20}{str(passed) + '/' + str(len(results)):<7}{'':<9}{sum(r['tokens'] for r in results) / 1000:<9.1f}"
                f"{sum(r['searches'] for r in results):<10}${sum(r['cost_usd'] for r in results):<8.3f}")
    return rows


def compare(old: list[dict], new: list[dict]) -> list[str]:
    a, b = summarize(old), summarize(new)
    pct = lambda x, y: f"{(y - x) / x * 100:+.0f}%" if x else "n/a"
    return [f"vs baseline: pass {a['passed']}/{a['runs']} -> {b['passed']}/{b['runs']}, "
            f"tokens {pct(a['tokens'], b['tokens'])}, searches {pct(a['searches'], b['searches'])}, "
            f"retries {pct(a['retries'], b['retries'])}, cost {pct(a['cost_usd'], b['cost_usd'])}"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", help="run the cases whose name contains this")
    p.add_argument("--repeat", type=int, default=1, help="runs per case (models vary: use 3 to compare versions)")
    p.add_argument("--compare", type=Path, help="a results file from an earlier run")
    p.add_argument("--out", type=Path, help="where to save the results (default evals/results/<time>.json)")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    from patchforge.config import Settings
    from patchforge.sandbox import make_sandbox

    settings = Settings.from_env()
    cases = [c for c in json.loads((ROOT / "evals/cases/cases.json").read_text()) if not args.only or args.only in c["name"]]
    if not cases:
        raise SystemExit("no case matches --only")
    skills_root = Path(tempfile.mkdtemp(prefix="pf-eval-skills-"))
    results = []
    for case in cases:
        for i in range(args.repeat):
            print(f"{case['name']} ({i + 1}/{args.repeat}) ...", flush=True)
            r = run_case(case, lambda: make_sandbox(settings.sandbox, image=settings.sandbox_image), skills_root)
            print(f"  {'passed' if r['passed'] else 'FAILED'}" + (f": {r['error']}" if r["error"] else "")
                  + (f" ({'; '.join(r['diff_problems'])})" if r["diff_problems"] else ""), flush=True)
            results.append(r)
    shutil.rmtree(skills_root, ignore_errors=True)

    print()
    print("\n".join(table(results)))
    if args.compare:
        print("\n".join(compare(json.loads(args.compare.read_text())["results"], results)))
    out = args.out or ROOT / "evals" / "results" / time.strftime("%Y%m%d-%H%M%S.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nsaved {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
