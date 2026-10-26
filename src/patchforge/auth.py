"""Sign in with GitHub, and per-user access to repositories.

Two credentials are involved, with different jobs:

* The **user token** (OAuth, from "Sign in with GitHub") identifies who is using
  PatchForge and answers "which repositories can this person reach through the
  App?" (`/user/installations`). It never leaves the server: the browser only
  holds an opaque session id in an HttpOnly cookie.
* The **installation token** (github_app.py) is what pushes the branch and
  opens the PR, scoped to the repositories the owner granted the App.

A user can run a job on a repository only if the App is installed on it and
the user has write (push) access to it. Both are checked again on the server
when the job is submitted, not trusted from the browser.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlencode

import httpx

from .github_app import API_URL, GitHubError

WEB_URL = "https://github.com"
SESSION_TTL_S = 7 * 86400
STATE_TTL_S = 10 * 60
REFRESH_MARGIN_S = 5 * 60
REPOS_CACHE_S = 30


class AuthError(RuntimeError):
    pass


@dataclass
class Session:
    id: str
    user_id: int
    login: str
    name: str | None
    avatar_url: str | None
    token: str
    token_expires_at: float | None
    refresh_token: str | None
    created_at: float = field(default_factory=time.time)
    repos: list[dict] | None = None
    repos_at: float = 0.0

    def public(self) -> dict:
        return {"login": self.login, "name": self.name, "avatar_url": self.avatar_url}


def parse_repo(text: str) -> str | None:
    """'owner/name', 'github.com/owner/name' or a full https URL (with .git or extra path) -> 'owner/name'."""
    text = (text or "").strip()
    m = re.match(r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:[/?#].*)?$", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m = re.match(r"^([\w.-]+)/([\w.-]+?)(?:\.git)?$", text)
    return f"{m.group(1)}/{m.group(2)}" if m else None


class GitHubOAuth:
    """GitHub App user authorization (the "Sign in with GitHub" half of the App)."""

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, *,
                 client: httpx.Client | None = None, api_url: str = API_URL, web_url: str = WEB_URL,
                 clock: Callable[[], float] = time.time):
        self.client_id, self.client_secret, self.redirect_uri = client_id, client_secret, redirect_uri
        self._http = client or httpx.Client(timeout=30)
        self.api_url, self.web_url = api_url.rstrip("/"), web_url.rstrip("/")
        self._clock = clock

    def authorize_url(self, state: str) -> str:
        return f"{self.web_url}/login/oauth/authorize?" + urlencode(
            {"client_id": self.client_id, "redirect_uri": self.redirect_uri, "state": state})

    def _token_request(self, **params: str) -> dict:
        try:
            r = self._http.post(f"{self.web_url}/login/oauth/access_token", headers={"Accept": "application/json"},
                                data={"client_id": self.client_id, "client_secret": self.client_secret, **params})
        except httpx.HTTPError as e:
            raise AuthError(f"could not reach GitHub: {e}") from e
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code >= 400 or "access_token" not in data:
            raise AuthError(data.get("error_description") or data.get("error") or f"GitHub returned {r.status_code}")
        return data

    def _tokens(self, data: dict) -> tuple[str, float | None, str | None]:
        expires = self._clock() + int(data["expires_in"]) if data.get("expires_in") else None
        return data["access_token"], expires, data.get("refresh_token")

    def exchange_code(self, code: str) -> tuple[str, float | None, str | None]:
        return self._tokens(self._token_request(code=code, redirect_uri=self.redirect_uri))

    def refresh(self, refresh_token: str) -> tuple[str, float | None, str | None]:
        return self._tokens(self._token_request(grant_type="refresh_token", refresh_token=refresh_token))

    def api(self, token: str, path: str) -> Any:
        try:
            r = self._http.get(self.api_url + path, headers={
                "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "patchforge"})
        except httpx.HTTPError as e:
            raise GitHubError(f"could not reach GitHub: {e}") from e
        if r.status_code == 401:
            raise AuthError("your GitHub sign-in expired; sign in again")
        if r.status_code >= 400:
            raise GitHubError(f"GitHub GET {path} failed with {r.status_code}", status=r.status_code)
        return r.json()

    def user(self, token: str) -> dict:
        return self.api(token, "/user")

    def user_repos(self, token: str) -> list[dict]:
        """Repos the user can reach through this App's installations, with the user's own permissions."""
        repos = []
        installs = self.api(token, "/user/installations?per_page=100")["installations"]
        for inst in installs:
            for page in range(1, 11):
                data = self.api(token, f"/user/installations/{inst['id']}/repositories?per_page=100&page={page}")
                for r in data["repositories"]:
                    perms = r.get("permissions") or {}
                    repos.append({
                        "full_name": r["full_name"], "private": r["private"], "default_branch": r["default_branch"],
                        "installation_id": inst["id"], "account": inst["account"]["login"],
                        "can_write": bool(perms.get("push") or perms.get("admin") or perms.get("maintain")),
                    })
                if len(data["repositories"]) < 100:
                    break
        return sorted(repos, key=lambda r: r["full_name"].lower())


class SessionStore:
    """Server-side sessions. Tokens stay in memory; a server restart signs everyone out."""

    def __init__(self, oauth: GitHubOAuth, clock: Callable[[], float] = time.time):
        self.oauth = oauth
        self._clock = clock
        self._sessions: dict[str, Session] = {}
        self._states: dict[str, float] = {}
        self._lock = threading.Lock()

    # --- login -----------------------------------------------------------------------

    def new_state(self) -> str:
        state = secrets.token_urlsafe(24)
        with self._lock:
            now = self._clock()
            self._states = {s: t for s, t in self._states.items() if now - t < STATE_TTL_S}
            self._states[state] = now
        return state

    def consume_state(self, state: str | None) -> bool:
        with self._lock:
            created = self._states.pop(state or "", None)
        return created is not None and self._clock() - created < STATE_TTL_S

    def login(self, code: str) -> Session:
        token, expires, refresh = self.oauth.exchange_code(code)
        user = self.oauth.user(token)
        session = Session(id=secrets.token_urlsafe(32), user_id=user["id"], login=user["login"],
                          name=user.get("name"), avatar_url=user.get("avatar_url"), token=token,
                          token_expires_at=expires, refresh_token=refresh, created_at=self._clock())
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        with self._lock:
            s = self._sessions.get(session_id)
            if s and self._clock() - s.created_at > SESSION_TTL_S:
                del self._sessions[session_id]
                return None
            return s

    def logout(self, session_id: str | None) -> None:
        with self._lock:
            self._sessions.pop(session_id or "", None)

    # --- tokens and repos ----------------------------------------------------------------

    def token(self, s: Session) -> str:
        """The user's token, refreshed first if it's about to expire (GitHub App user tokens last 8 hours)."""
        if s.token_expires_at and s.token_expires_at - REFRESH_MARGIN_S < self._clock():
            if not s.refresh_token:
                raise AuthError("your GitHub sign-in expired; sign in again")
            s.token, s.token_expires_at, s.refresh_token = self.oauth.refresh(s.refresh_token)
        return s.token

    def repos(self, s: Session, refresh: bool = False) -> list[dict]:
        if refresh or s.repos is None or self._clock() - s.repos_at > REPOS_CACHE_S:
            s.repos, s.repos_at = self.oauth.user_repos(self.token(s)), self._clock()
        return s.repos

    def find_repo(self, s: Session, full_name: str) -> dict | None:
        """Checked live (not from the cache) right before a job starts."""
        wanted = full_name.lower()
        return next((r for r in self.repos(s, refresh=True) if r["full_name"].lower() == wanted), None)
