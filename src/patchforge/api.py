"""HTTP API: sign in with GitHub, start jobs, stream their events, receive GitHub webhooks.

Run with:  uvicorn patchforge.api:create_app --factory --host 127.0.0.1 --port 8000

Who can do what:
* Anyone signs in with GitHub. They see and use only repositories where the
  PatchForge App is installed and they have write access, and only their own jobs.
* Scripts can use `Authorization: Bearer $PATCHFORGE_API_KEY` instead; that key
  acts as an operator across every installation, so keep it secret.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from .auth import SESSION_TTL_S, AuthError, GitHubOAuth, Session, SessionStore, parse_repo
from .config import Settings
from .github_app import GitHubApp, GitHubError, verify_webhook_signature
from .jobs import JobRequest, JobRunner

log = logging.getLogger("patchforge")
KEEPALIVE_S = 15
STATIC = Path(__file__).parent / "static"
COOKIE = "pf_session"


def _make_agent(kind: str):
    if kind == "stub":
        from .agent_stub import StubAgent
        return StubAgent()
    if kind == "brain":
        from .brain import BrainAgent
        return BrainAgent()
    raise ValueError(f"unknown PATCHFORGE_AGENT={kind!r}; use brain or stub")


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
        history_dir=Path(os.getenv("PATCHFORGE_DATA_DIR", ".patchforge-data")) / "jobs",
    )


def build_sessions(settings: Settings) -> SessionStore | None:
    if not settings.oauth_configured:
        log.warning("Sign in with GitHub is off: set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET (see README)")
        return None
    oauth = GitHubOAuth(settings.github_client_id, settings.github_client_secret,
                        redirect_uri=settings.public_url.rstrip("/") + "/auth/callback")
    return SessionStore(oauth)


@dataclass
class Principal:
    login: str | None  # None = operator (API key)
    session: Session | None = None

    @property
    def operator(self) -> bool:
        return self.login is None


class NewJob(BaseModel):
    repo: str  # "owner/name" or a github.com URL
    instruction: str
    base_branch: str | None = None
    path: str = ""  # project folder inside the repo; "" = repo root
    installation_id: int | None = None  # operators only; users get it from their verified access


def create_app(runner: JobRunner | None = None, settings: Settings | None = None,
               sessions: SessionStore | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    runner = runner or build_runner(settings)
    sessions = sessions if sessions is not None else build_sessions(settings)
    secure_cookie = settings.public_url.startswith("https://")
    app = FastAPI(title="PatchForge", version="0.2.0")
    app.state.runner = runner

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    # --- who is calling --------------------------------------------------------------

    def current_session(request: Request) -> Session | None:
        return sessions.get(request.cookies.get(COOKIE)) if sessions else None

    def principal(request: Request, authorization: str | None = Header(default=None)) -> Principal:
        if settings.api_key and authorization == f"Bearer {settings.api_key}":
            return Principal(login=None)
        session = current_session(request)
        if session:
            return Principal(login=session.login, session=session)
        raise HTTPException(401, "sign in with GitHub first")

    def user_session(p: Principal = Depends(principal)) -> Session:
        if not p.session:
            raise HTTPException(400, "this endpoint needs a signed-in GitHub user")
        return p.session

    def get_job(job_id: str, p: Principal):
        job = runner.jobs.get(job_id)
        # Someone else's job looks the same as a missing one.
        if not job or not (p.operator or job.owner == p.login):
            raise HTTPException(404, "no such job")
        return job

    def install_url() -> str | None:
        gh = runner.github
        if not hasattr(gh, "install_url"):
            return None
        try:
            return gh.install_url()
        except GitHubError:
            return None

    # --- pages -------------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"ok": True, "sandbox": settings.sandbox, "agent": settings.agent,
                "github_configured": settings.github_configured, "sign_in_enabled": sessions is not None}

    # --- sign in with GitHub -----------------------------------------------------------------

    @app.get("/auth/login", include_in_schema=False)
    def login():
        if not sessions:
            raise HTTPException(503, "Sign in with GitHub isn't configured on this server")
        return RedirectResponse(sessions.oauth.authorize_url(sessions.new_state()))

    @app.get("/auth/callback", include_in_schema=False)
    def callback(code: str | None = None, state: str | None = None, error: str | None = None):
        if not sessions:
            raise HTTPException(503, "Sign in with GitHub isn't configured on this server")
        if error:
            return RedirectResponse(f"/?auth_error={quote(error)}")
        # The state ties this callback to a login this server started (stops login CSRF).
        if not code or not sessions.consume_state(state):
            return RedirectResponse("/?auth_error=expired")
        try:
            session = sessions.login(code)
        except (AuthError, GitHubError) as e:
            log.warning("GitHub sign-in failed: %s", e)
            return RedirectResponse("/?auth_error=failed")
        resp = RedirectResponse("/")
        resp.set_cookie(COOKIE, session.id, max_age=SESSION_TTL_S, httponly=True, samesite="lax",
                        secure=secure_cookie, path="/")
        return resp

    @app.get("/auth/me")
    def me(request: Request):
        session = current_session(request)
        if not session:
            raise HTTPException(401, "not signed in")
        return session.public()

    @app.post("/auth/logout")
    def logout(request: Request):
        if sessions:
            sessions.logout(request.cookies.get(COOKIE))
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE, path="/")
        return resp

    # --- the user's repositories -----------------------------------------------------------

    @app.get("/github")
    def github_info():
        gh = runner.github
        name = None
        if hasattr(gh, "app_info"):
            try:
                name = gh.app_info().get("name")
            except GitHubError:
                pass
        return {"app_name": name, "install_url": install_url()}

    @app.get("/github/repos")
    def github_repos(refresh: bool = False, session: Session = Depends(user_session)):
        """Repos where the App is installed and this user has access; `can_write` marks the usable ones."""
        try:
            return sessions.repos(session, refresh=refresh)
        except AuthError as e:
            raise HTTPException(401, str(e)) from e
        except GitHubError as e:
            raise HTTPException(502, str(e)) from e

    @app.get("/github/projects")
    def github_projects(repo: str, session: Session = Depends(user_session)):
        """The Python/Node projects inside one of the user's repos, so they can pick a folder."""
        full = parse_repo(repo)
        try:
            access = full and next((r for r in sessions.repos(session) if r["full_name"].lower() == full.lower()), None)
        except AuthError as e:
            raise HTTPException(401, str(e)) from e
        if not access or not hasattr(runner.github, "list_projects"):
            return []
        owner, name = access["full_name"].split("/")
        try:
            return runner.github.list_projects(owner, name, access["installation_id"])
        except GitHubError as e:
            raise HTTPException(502, str(e)) from e

    @app.get("/github/setup", include_in_schema=False)
    def github_setup():
        """GitHub sends users here after they install the App or change its repos (the App's Setup URL)."""
        return RedirectResponse("/?installed=1")

    # --- jobs ---------------------------------------------------------------------------

    @app.post("/jobs", status_code=202)
    def create_job(body: NewJob, p: Principal = Depends(principal)):
        repo = parse_repo(body.repo)
        if not repo:
            raise HTTPException(422, "enter a GitHub repository as owner/name or its github.com URL")
        installation_id = body.installation_id if p.operator else None
        if not p.operator:
            try:
                access = sessions.find_repo(p.session, repo)
            except AuthError as e:
                raise HTTPException(401, str(e)) from e
            except GitHubError as e:
                raise HTTPException(502, str(e)) from e
            if not access:
                raise HTTPException(403, {
                    "code": "not_installed", "repo": repo, "install_url": install_url(),
                    "message": f"PatchForge isn't installed on {repo}, or you don't have access to it. "
                               f"Install it and choose the {repo.split('/')[0]} account and that repository, "
                               "then try again."})
            if not access["can_write"]:
                raise HTTPException(403, {"code": "no_write_access", "repo": repo,
                                          "message": f"You need write access to {repo} to open pull requests on it."})
            repo, installation_id = access["full_name"], access["installation_id"]
        try:
            job = runner.submit(JobRequest(repo=repo, instruction=body.instruction, base_branch=body.base_branch,
                                           installation_id=installation_id, path=body.path), owner=p.login)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {**job.snapshot(), "events_url": f"/jobs/{job.id}/events"}

    @app.get("/jobs")
    def list_jobs(p: Principal = Depends(principal)):
        mine = [j for j in runner.jobs.values() if p.operator or j.owner == p.login]
        return [j.snapshot() for j in sorted(mine, key=lambda j: -j.created_at)]

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str, p: Principal = Depends(principal)):
        return get_job(job_id, p).snapshot()

    @app.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, after: int = 0, last_event_id: str | None = Header(default=None),
                         p: Principal = Depends(principal)):
        """Server-Sent Events. Replays history first, so a refreshed dashboard misses nothing."""
        job = get_job(job_id, p)
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

    @app.get("/skills")
    def skills(p: Principal = Depends(principal)):
        from .skills import SkillStore
        store = SkillStore(Path(os.getenv("PATCHFORGE_SKILLS_DIR", "skills")))
        return [{"name": s.path.parent.name, "library": s.library, "from_version": s.from_version,
                 "to_version": s.to_version, "uses": s.uses, "created": s.created,
                 "changes": len(s.spec.get("changes", []))} for s in store.all()]

    # --- webhooks ----------------------------------------------------------------------

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
            log.info("installation %s (%s): %s", inst.get("id"), inst.get("account", {}).get("login"),
                     payload.get("action"))
        return {"ok": True}

    return app
