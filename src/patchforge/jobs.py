"""The job pipeline: clone -> baseline tests -> agent -> final tests -> branch -> push -> PR.

Security boundary: this module holds the GitHub token. The sandbox gets only a
copy of the code, and the agent gets only the working tree; neither ever sees
the token. Pushing and opening the PR happen here, after the tests pass.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel

from . import git_ops
from .contracts import Agent, AgentEvent, AgentTask, Sandbox, SandboxResult
from .sandbox.common import IGNORE

log = logging.getLogger("patchforge")


class GitHubLike(Protocol):
    def repo_installation_id(self, owner: str, repo: str) -> int: ...
    def installation_token(self, installation_id: int) -> str: ...
    def get_repo(self, owner: str, repo: str, token: str) -> dict: ...
    def clone_url(self, owner: str, repo: str) -> str: ...
    def create_pull_request(self, owner: str, repo: str, token: str, *, head: str, base: str,
                            title: str, body: str) -> dict: ...


class JobRequest(BaseModel):
    repo: str  # "owner/name"
    instruction: str
    base_branch: str | None = None
    installation_id: int | None = None
    path: str = ""  # project folder inside the repo, e.g. "services/api"; "" = repo root


def clean_path(path: str) -> str:
    """'./services/api/' -> 'services/api'. Rejects anything that could escape the repo."""
    parts = [p for p in (path or "").replace("\\", "/").split("/") if p not in ("", ".")]
    if any(p == ".." or p.startswith(".git") for p in parts) or (path or "").startswith("/"):
        raise ValueError("folder must be a path inside the repository, like services/api")
    return "/".join(parts)


class JobFailed(Exception):
    pass


def tree_hash(folder: Path) -> str:
    """Fingerprint of a project's files (ignoring caches and installed packages)."""
    folder = Path(folder)
    h = hashlib.sha256()
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if p.is_dir() or any(IGNORE("", [part]) for part in rel.parts):
            continue
        h.update(str(rel).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


class LiveSandbox:
    """Wraps the job's sandbox so every test run reports progress to the dashboard as it happens.

    The agent gets this wrapper too, so its own test runs are visible. `stage` says which step
    the messages belong to ("baseline_tests" before the change, "tests" afterwards).

    It also remembers passing results by code fingerprint: the final check on exactly the code the
    agent's last check already passed is answered from that result instead of running again.
    """

    def __init__(self, inner: Sandbox, emit, stage: str = "tests"):
        self.inner, self.emit, self.stage = inner, emit, stage
        self._passed: dict[str, SandboxResult] = {}
        try:
            self._accepts_progress = "progress" in inspect.signature(inner.run_tests).parameters
        except (TypeError, ValueError):
            self._accepts_progress = False

    def _progress(self, message: str) -> None:
        # "Still …" heartbeats update the live status line instead of piling up in the activity list.
        self.emit(AgentEvent(stage=self.stage, message=message,
                             data={"progress": True, "heartbeat": message.startswith("Still ")}))

    def run_tests(self, code_folder: Path, progress=None) -> SandboxResult:
        key = tree_hash(code_folder)
        if key in self._passed:
            self._progress("Same code as the last passing check; reusing its result")
            return self._passed[key]
        if self._accepts_progress:
            result = self.inner.run_tests(code_folder, progress=self._progress)
        else:
            self._progress("Running the tests…")
            result = self.inner.run_tests(code_folder)
        if result.passed:
            self._passed[key] = result
        return result


class Job:
    def __init__(self, request: JobRequest, owner: str | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.owner = owner  # GitHub login of the user who started it; None for API-key jobs
        self.status = "queued"  # queued | running | succeeded | failed
        self.error: str | None = None
        self.pr_url: str | None = None
        self.branch: str | None = None
        self.checked: str | None = None  # what the final check ran: tests | build | typecheck | compile
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.events: list[dict] = []
        self._cond = threading.Condition()

    @property
    def done(self) -> bool:
        return self.status in ("succeeded", "failed")

    def emit(self, event: AgentEvent) -> None:
        with self._cond:
            self.events.append({"id": len(self.events), "job_id": self.id, **event.model_dump()})
            self._cond.notify_all()

    def finish(self, status: str, error: str | None = None) -> None:
        with self._cond:
            self.status, self.error, self.finished_at = status, error, time.time()
            self._cond.notify_all()

    def wait_for_events(self, after: int, timeout: float) -> list[dict]:
        """Events with id >= after; blocks up to `timeout` if there are none yet."""
        with self._cond:
            self._cond.wait_for(lambda: len(self.events) > after or self.done, timeout=timeout)
            return self.events[after:]

    def totals(self) -> dict:
        return {
            "tokens_in": sum(e["tokens_in"] for e in self.events),
            "tokens_out": sum(e["tokens_out"] for e in self.events),
            "cost_usd": round(sum(e["cost_usd"] for e in self.events), 6),
            "model_calls": sum(1 for e in self.events if e["model"]),
        }

    def to_record(self) -> dict:
        return {"snapshot": self.snapshot(), "request": self.request.model_dump(), "owner": self.owner,
                "events": self.events}

    @classmethod
    def from_record(cls, record: dict) -> "Job":
        snap = record["snapshot"]
        job = cls(JobRequest(**record["request"]), owner=record.get("owner"))
        job.id, job.status, job.error = snap["id"], snap["status"], snap["error"]
        job.pr_url, job.branch = snap["pr_url"], snap["branch"]
        job.checked = snap.get("checked")
        job.created_at, job.finished_at = snap["created_at"], snap["finished_at"]
        job.events = record["events"]
        if not job.done:  # the server stopped mid-job
            job.status, job.error = "failed", job.error or "interrupted: the server restarted during this job"
        return job

    def snapshot(self) -> dict:
        return {
            "id": self.id, "status": self.status, "repo": self.request.repo, "path": self.request.path,
            "instruction": self.request.instruction, "branch": self.branch, "pr_url": self.pr_url,
            "error": self.error, "created_at": self.created_at, "finished_at": self.finished_at,
            "checked": self.checked, "verified": self.checked == "tests",
            "event_count": len(self.events), "totals": self.totals(),
        }


def branch_name(instruction: str, job_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", instruction.lower()).strip("-")[:40].rstrip("-") or "patch"
    return f"patchforge/{slug}-{job_id[:6]}"


CHECK_PASSED = {
    "tests": "All tests passed",
    "build": "Build succeeded (this project has no tests)",
    "typecheck": "Type check passed (this project has no tests)",
    "compile": "Every file compiles (this project has no tests)",
}


def _tests_summary(r: SandboxResult, icon: bool = True) -> str:
    if r.passed:
        return f"{'✅ ' if icon else ''}{CHECK_PASSED.get(r.checked, 'Passed')} ({r.duration:.1f}s)"
    if r.failed_tests:
        failing = f"{len(r.failed_tests)} failing: " + ", ".join(r.failed_tests[:10])
    elif "dependency install failed" in r.error_log:
        failing = "dependencies failed to install"
    elif r.checked != "tests":
        failing = f"{r.checked} check failed (this project has no tests)"
    else:
        failing = "tests failed"
    return f"{'❌ ' if icon else ''}{failing}"


class JobRunner:
    def __init__(
        self,
        *,
        github: GitHubLike,
        sandbox_factory: Callable[[], Sandbox],
        agent_factory: Callable[[], Agent],
        workspace_root: Path | None = None,
        git_author: Callable[[], tuple[str, str]] = lambda: ("PatchForge", "patchforge@users.noreply.github.com"),
        max_workers: int = 2,
        clone_depth: int | None = 50,
        history_dir: Path | None = None,
    ):
        self.github = github
        self.sandbox_factory = sandbox_factory
        self.agent_factory = agent_factory
        self.workspace_root = workspace_root
        self.git_author = git_author
        self.clone_depth = clone_depth
        self.jobs: dict[str, Job] = {}
        self.history_dir = Path(history_dir) if history_dir else None
        self._load_history()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pf-job")

    def _load_history(self) -> None:
        if not self.history_dir or not self.history_dir.is_dir():
            return
        for f in self.history_dir.glob("*.json"):
            try:
                job = Job.from_record(json.loads(f.read_text()))
                self.jobs[job.id] = job
            except (ValueError, KeyError) as e:
                log.warning("skipping unreadable job record %s: %s", f.name, e)

    def _save(self, job: Job) -> None:
        if not self.history_dir:
            return
        try:
            self.history_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.history_dir / f"{job.id}.json.tmp"
            tmp.write_text(json.dumps(job.to_record()))
            tmp.replace(self.history_dir / f"{job.id}.json")
        except OSError as e:
            log.warning("could not save job %s: %s", job.id, e)

    def _new_job(self, request: JobRequest, owner: str | None = None) -> Job:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", request.repo):
            raise ValueError("repo must look like 'owner/name'")
        request.path = clean_path(request.path)
        job = Job(request, owner=owner)
        self.jobs[job.id] = job
        job.emit(AgentEvent(stage="job", message=f"Started: {request.instruction}"))
        return job

    def submit(self, request: JobRequest, owner: str | None = None) -> Job:
        job = self._new_job(request, owner)
        self._pool.submit(self._run, job)
        return job

    def run_sync(self, request: JobRequest) -> Job:
        """Same as submit, but blocks until the job finishes (scripts and tests)."""
        job = self._new_job(request)
        self._run(job)
        return job

    def _run(self, job: Job) -> None:
        job.status = "running"
        tmp = Path(tempfile.mkdtemp(prefix="pf-job-", dir=self.workspace_root))
        try:
            self._pipeline(job, tmp)
            job.finish("succeeded")
        except JobFailed as e:
            job.emit(AgentEvent(stage="error", message=str(e), level="error"))
            job.finish("failed", str(e))
        except Exception as e:  # noqa: BLE001 - report anything to the dashboard
            msg = f"{type(e).__name__}: {e}"
            job.emit(AgentEvent(stage="error", message=msg, level="error",
                                data={"traceback": traceback.format_exc(limit=5)}))
            job.finish("failed", msg)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            self._save(job)

    def _pipeline(self, job: Job, tmp: Path) -> None:
        req, emit = job.request, job.emit
        owner, name = req.repo.split("/")

        installation_id = req.installation_id or self.github.repo_installation_id(owner, name)
        token = self.github.installation_token(installation_id)
        base = req.base_branch or self.github.get_repo(owner, name, token)["default_branch"]

        t0 = time.monotonic()
        workdir = tmp / "repo"
        git_ops.clone(self.github.clone_url(owner, name), workdir, token=token, branch=base,
                      depth=self.clone_depth)
        project = workdir / req.path if req.path else workdir
        if not project.is_dir():
            raise JobFailed(f"folder '{req.path}' doesn't exist in {req.repo}@{base}")
        emit(AgentEvent(stage="clone", message=f"Cloned {req.repo} ({base})" + (f", folder {req.path}" if req.path else ""),
                        latency_ms=round((time.monotonic() - t0) * 1000, 1)))

        inner = self.sandbox_factory()
        # The "before" check runs on a snapshot, in parallel with the AI work, so nobody waits for it.
        snapshot = tmp / "before"
        shutil.copytree(project, snapshot, ignore=IGNORE, symlinks=True)
        emit(AgentEvent(stage="baseline_tests", message="Checking the project as it is today, "
                                                        "in parallel with the research"))
        before_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pf-before")
        before = before_pool.submit(LiveSandbox(inner, emit, stage="baseline_tests").run_tests, snapshot)

        abandoned = threading.Event()  # set if the job fails first: don't report after its error

        def report_before(fut) -> None:
            if fut.exception() is None and not abandoned.is_set():
                r = fut.result()
                emit(AgentEvent(stage="baseline_tests", message=f"Before changes: {_tests_summary(r, icon=False)}",
                                level="info" if r.passed else "warning",
                                data={"failed_tests": r.failed_tests, "duration": r.duration, "checked": r.checked}))
        before.add_done_callback(report_before)

        sandbox = LiveSandbox(inner, emit, stage="tests")
        task = AgentTask(repo=req.repo, instruction=req.instruction, base_branch=base)
        try:
            output = self.agent_factory().run(task, project, sandbox, emit)
            baseline = before.result()
        except BaseException:
            abandoned.set()
            raise
        finally:
            before_pool.shutdown(wait=False)

        changed = git_ops.changed_files(workdir)
        if not changed:
            raise JobFailed("the agent made no changes; nothing to open a PR for")

        emit(AgentEvent(stage="tests", message="Final check on the changed code"))
        final = sandbox.run_tests(project)
        job.checked = final.checked
        emit(AgentEvent(stage="tests", message=f"Final check: {_tests_summary(final, icon=False)}",
                        level="info" if final.passed else "error",
                        data={"failed_tests": final.failed_tests, "duration": final.duration, "checked": final.checked}))
        if not final.passed:
            what = "tests fail" if final.checked == "tests" else f"the {final.checked} check fails"
            raise JobFailed(f"{what} after the patch; refusing to open a PR\n" + final.error_log[-1500:])

        job.branch = branch_name(req.instruction, job.id)
        author_name, author_email = self.git_author()
        git_ops.create_branch(workdir, job.branch)
        sha = git_ops.commit_all(workdir, f"PatchForge: {req.instruction}",
                                 author_name=author_name, author_email=author_email)
        # Fetch the token again: cached if still fresh, renewed if the job outlived it.
        token = self.github.installation_token(installation_id)
        git_ops.push(workdir, job.branch, token=token)
        emit(AgentEvent(stage="push", message=f"Pushed branch {job.branch}",
                        data={"branch": job.branch, "sha": sha, "files": changed}))

        verified = final.checked == "tests"
        pr = self.github.create_pull_request(
            owner, name, token, head=job.branch, base=base,
            title=(f"PatchForge: {req.instruction}" + (f" ({req.path})" if req.path else ""))[:250],
            body=self._pr_body(job, output.summary, changed, baseline, final),
            # Not verified by tests: open it as a draft so nobody merges it without a careful review.
            draft=not verified,
        )
        job.pr_url = pr["html_url"]
        emit(AgentEvent(stage="pr", message=f"Opened {'pull request' if verified else 'draft pull request (not verified by tests)'} "
                                            f"#{pr['number']}",
                        data={"url": pr["html_url"], "number": pr["number"], "verified": verified}))
        emit(AgentEvent(stage="done", message="Finished", data=job.totals()))

    @staticmethod
    def _pr_body(job: Job, summary: str, changed: list[str], baseline: SandboxResult,
                 final: SandboxResult) -> str:
        t = job.totals()
        files = "\n".join(f"- `{f}`" for f in changed[:50])
        return (
            f"{summary}\n\n"
            f"### Files changed\n{files}\n\n"
            + (f"### Tests (run in an isolated sandbox)\n"
               f"- Before: {_tests_summary(baseline)}\n- After: {_tests_summary(final)}\n\n"
               if final.checked == "tests" else
               f"### ⚠️ Not verified by tests\n"
               f"This project has no tests, so PatchForge could only check that the upgraded project "
               f"installs and passes a **{final.checked}** check in an isolated sandbox. Its behaviour was not "
               f"tested; review the changes carefully before merging.\n\n"
               f"- Before: {_tests_summary(baseline)}\n- After: {_tests_summary(final)}\n\n")
            + f"### Run stats\n"
            f"- Model calls: {t['model_calls']} · tokens in/out: {t['tokens_in']:,}/{t['tokens_out']:,}"
            f" · cost: ${t['cost_usd']:.4f}\n\n"
            f"<sub>Opened by PatchForge · job `{job.id}`. Please review before merging.</sub>"
        )
