"""HTTP API: start jobs, read their status, stream their events, receive GitHub webhooks.

Run with:  uvicorn patchforge.api:create_app --factory --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import logging

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import Settings
from .github_app import GitHubApp, verify_webhook_signature
from .jobs import JobRequest, JobRunner

log = logging.getLogger("patchforge")
KEEPALIVE_S = 15


def _make_agent(kind: str):
    if kind == "stub":
        from .agent_stub import StubAgent
        return StubAgent()
    # Person A: add your agent here, e.g. `if kind == "brain": return BrainAgent()`
    raise ValueError(f"unknown PATCHFORGE_AGENT={kind!r}")


def build_runner(settings: Settings) -> JobRunner:
    if not settings.github_configured:
        raise SystemExit("GitHub App isn't configured: set GITHUB_APP_ID and GITHUB_PRIVATE_KEY_PATH (see .env.example)")
    from .sandbox import make_sandbox

    gh = GitHubApp(settings.github_app_id, settings.github_private_key)

    def author() -> tuple[str, str]:
        try:
            return gh.bot_identity()
        except Exception as e:  # noqa: BLE001 - fall back to a generic author
            log.warning("could not look up the App's bot identity, using a generic author: %s", e)
            return ("PatchForge", "patchforge@users.noreply.github.com")

    return JobRunner(
        github=gh,
        sandbox_factory=lambda: make_sandbox(settings.sandbox, image=settings.sandbox_image),
        agent_factory=lambda: _make_agent(settings.agent),
        git_author=author,
    )


def create_app(runner: JobRunner | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    runner = runner or build_runner(settings)
    app = FastAPI(title="PatchForge platform", version="0.1.0")
    app.state.runner = runner
    app.state.installations = {}  # installation_id -> account login, learned from webhooks

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    def require_api_key(authorization: str | None = Header(default=None)) -> None:
        if settings.api_key and authorization != f"Bearer {settings.api_key}":
            raise HTTPException(401, "missing or wrong API key")

    def get_job(job_id: str):
        job = runner.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        return job

    @app.get("/health")
    def health():
        return {"ok": True, "sandbox": settings.sandbox, "agent": settings.agent,
                "github_configured": settings.github_configured}

    @app.post("/jobs", status_code=202, dependencies=[Depends(require_api_key)])
    def create_job(req: JobRequest):
        try:
            job = runner.submit(req)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {**job.snapshot(), "events_url": f"/jobs/{job.id}/events"}

    @app.get("/jobs")
    def list_jobs():
        return [j.snapshot() for j in sorted(runner.jobs.values(), key=lambda j: -j.created_at)]

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        return get_job(job_id).snapshot()

    @app.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, after: int = 0, last_event_id: str | None = Header(default=None)):
        """Server-Sent Events. Replays history first, so a refreshed dashboard misses nothing."""
        job = get_job(job_id)
        start = int(last_event_id) + 1 if last_event_id and last_event_id.isdigit() else after

        async def stream():
            i = start
            while True:
                events = await anyio.to_thread.run_sync(job.wait_for_events, i, KEEPALIVE_S)
                for ev in events:
                    yield f"id: {ev['id']}\ndata: {json.dumps(ev)}\n\n"
                i += len(events)
                if job.done and i >= len(job.events):
                    yield f"event: end\ndata: {json.dumps(job.snapshot())}\n\n"
                    return
                if not events:
                    yield ": keepalive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/webhooks/github")
    async def github_webhook(request: Request, x_github_event: str = Header(default=""),
                             x_hub_signature_256: str | None = Header(default=None)):
        body = await request.body()
        if not verify_webhook_signature(settings.github_webhook_secret, body, x_hub_signature_256):
            raise HTTPException(401, "bad webhook signature")
        payload = json.loads(body or b"{}")
        if x_github_event == "ping":
            return {"ok": True, "pong": True}
        if x_github_event in ("installation", "installation_repositories"):
            inst = payload.get("installation", {})
            inst_id, account = inst.get("id"), inst.get("account", {}).get("login")
            if payload.get("action") == "deleted":
                app.state.installations.pop(inst_id, None)
            elif inst_id:
                app.state.installations[inst_id] = account
            log.info("installation %s (%s): %s", inst_id, account, payload.get("action"))
        return {"ok": True}

    @app.get("/installations")
    def installations():
        return [{"id": k, "account": v} for k, v in app.state.installations.items()]

    return app
