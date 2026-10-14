"""The job pipeline: clone -> baseline tests -> agent -> final tests -> branch -> push -> PR.

Security boundary: this module holds the GitHub token. The sandbox gets only a
copy of the code, and the agent gets only the working tree; neither ever sees
the token. Pushing and opening the PR happen here, after the tests pass.
"""

from __future__ import annotations

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


class JobFailed(Exception):
    pass


class Job:
    def __init__(self, request: JobRequest):
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.status = "queued"  # queued | running | succeeded | failed
        self.error: str | None = None
        self.pr_url: str | None = None
        self.branch: str | None = None
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

    def snapshot(self) -> dict:
        return {
            "id": self.id, "status": self.status, "repo": self.request.repo,
            "instruction": self.request.instruction, "branch": self.branch, "pr_url": self.pr_url,
            "error": self.error, "created_at": self.created_at, "finished_at": self.finished_at,
            "event_count": len(self.events), "totals": self.totals(),
        }


def branch_name(instruction: str, job_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", instruction.lower()).strip("-")[:40].rstrip("-") or "patch"
    return f"patchforge/{slug}-{job_id[:6]}"


def _tests_summary(r: SandboxResult) -> str:
    if r.passed:
        return f"✅ All tests passed ({r.duration:.1f}s)"
    return f"❌ {len(r.failed_tests)} failing: " + ", ".join(r.failed_tests[:10])


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
    ):
        self.github = github
        self.sandbox_factory = sandbox_factory
        self.agent_factory = agent_factory
        self.workspace_root = workspace_root
        self.git_author = git_author
        self.clone_depth = clone_depth
        self.jobs: dict[str, Job] = {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pf-job")

    def _new_job(self, request: JobRequest) -> Job:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", request.repo):
            raise ValueError("repo must look like 'owner/name'")
        job = Job(request)
        self.jobs[job.id] = job
        job.emit(AgentEvent(stage="job", message=f"queued: {request.instruction} on {request.repo}"))
        return job

    def submit(self, request: JobRequest) -> Job:
        job = self._new_job(request)
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
        emit(AgentEvent(stage="clone", message=f"cloned {req.repo}@{base}",
                        latency_ms=round((time.monotonic() - t0) * 1000, 1)))

        sandbox = self.sandbox_factory()
        baseline = sandbox.run_tests(workdir)
        emit(AgentEvent(stage="baseline_tests", message=_tests_summary(baseline),
                        level="info" if baseline.passed else "warning",
                        data={"failed_tests": baseline.failed_tests, "duration": baseline.duration}))

        task = AgentTask(repo=req.repo, instruction=req.instruction, base_branch=base)
        output = self.agent_factory().run(task, workdir, sandbox, emit)

        changed = git_ops.changed_files(workdir)
        if not changed:
            raise JobFailed("the agent made no changes; nothing to open a PR for")

        final = sandbox.run_tests(workdir)
        emit(AgentEvent(stage="tests", message=f"final check: {_tests_summary(final)}",
                        level="info" if final.passed else "error",
                        data={"failed_tests": final.failed_tests, "duration": final.duration}))
        if not final.passed:
            raise JobFailed("tests fail after the patch; refusing to open a PR\n" + final.error_log[-1500:])

        job.branch = branch_name(req.instruction, job.id)
        author_name, author_email = self.git_author()
        git_ops.create_branch(workdir, job.branch)
        sha = git_ops.commit_all(workdir, f"PatchForge: {req.instruction}",
                                 author_name=author_name, author_email=author_email)
        # Fetch the token again: cached if still fresh, renewed if the job outlived it.
        token = self.github.installation_token(installation_id)
        git_ops.push(workdir, job.branch, token=token)
        emit(AgentEvent(stage="push", message=f"pushed {job.branch} ({sha[:7]})",
                        data={"branch": job.branch, "sha": sha, "files": changed}))

        pr = self.github.create_pull_request(
            owner, name, token, head=job.branch, base=base,
            title=f"PatchForge: {req.instruction}"[:250],
            body=self._pr_body(job, output.summary, changed, baseline, final),
        )
        job.pr_url = pr["html_url"]
        emit(AgentEvent(stage="pr", message=f"opened PR #{pr['number']}",
                        data={"url": pr["html_url"], "number": pr["number"]}))
        emit(AgentEvent(stage="done", message="job finished", data=job.totals()))

    @staticmethod
    def _pr_body(job: Job, summary: str, changed: list[str], baseline: SandboxResult,
                 final: SandboxResult) -> str:
        t = job.totals()
        files = "\n".join(f"- `{f}`" for f in changed[:50])
        return (
            f"{summary}\n\n"
            f"### Files changed\n{files}\n\n"
            f"### Tests (run in an isolated sandbox)\n"
            f"- Before: {_tests_summary(baseline)}\n- After: {_tests_summary(final)}\n\n"
            f"### Run stats\n"
            f"- Model calls: {t['model_calls']} · tokens in/out: {t['tokens_in']:,}/{t['tokens_out']:,}"
            f" · cost: ${t['cost_usd']:.4f}\n\n"
            f"<sub>Opened by PatchForge · job `{job.id}`. Please review before merging.</sub>"
        )
