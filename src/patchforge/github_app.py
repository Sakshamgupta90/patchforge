"""GitHub App authentication and the few REST calls PatchForge needs.

Flow: sign a short-lived JWT with the App's private key -> exchange it for an
installation access token (valid ~1 hour, scoped to the repos the user picked)
-> use that token to push a branch and open a pull request.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from datetime import datetime
from typing import Any, Callable
from urllib.parse import quote

import httpx
import jwt

API_URL = "https://api.github.com"
# Refresh tokens this long before GitHub says they expire.
TOKEN_REFRESH_MARGIN_S = 5 * 60


class GitHubError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, hint: str | None = None):
        self.status = status
        self.hint = hint
        super().__init__(f"{message} (hint: {hint})" if hint else message)


def _hint(status: int, action: str) -> str | None:
    if status == 401:
        return "check GITHUB_APP_ID and the private key, and that this machine's clock is correct"
    if status == 403:
        return ("the App is missing a permission; it needs Contents: Read & write and "
                "Pull requests: Read & write. After changing permissions, accept them on the installation page")
    if status == 404:
        if action == "installation":
            return "the App isn't installed on this repo, or the owner/name is wrong"
        if action == "token":
            return "wrong installation_id, or GITHUB_APP_ID doesn't match the App that was installed"
        return "repo or resource not found, or the App can't see it"
    if status == 422 and action == "pull_request":
        return "a PR for this branch may already exist, or the branch has no new commits"
    return None


class GitHubApp:
    def __init__(
        self,
        app_id: str | int,
        private_key_pem: str,
        *,
        api_url: str = API_URL,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.app_id = str(app_id)
        self._key = private_key_pem
        self.api_url = api_url.rstrip("/")
        self._http = client or httpx.Client(timeout=30)
        self._clock = clock
        self._tokens: dict[int, tuple[str, float]] = {}  # installation_id -> (token, expires_at)
        self._lock = threading.Lock()
        self._bot_identity: tuple[str, str] | None = None
        self._app: dict | None = None

    # --- auth -------------------------------------------------------------------

    def app_jwt(self) -> str:
        now = int(self._clock())
        # iat is backdated 60s to tolerate clock drift; GitHub caps exp at 10 minutes.
        payload = {"iat": now - 60, "exp": now + 9 * 60, "iss": self.app_id}
        return jwt.encode(payload, self._key, algorithm="RS256")

    def installation_token(self, installation_id: int) -> str:
        with self._lock:
            cached = self._tokens.get(installation_id)
            if cached and cached[1] - TOKEN_REFRESH_MARGIN_S > self._clock():
                return cached[0]
            data = self._request(
                "POST", f"/app/installations/{installation_id}/access_tokens",
                auth=f"Bearer {self.app_jwt()}", action="token",
            )
            expires_at = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00")).timestamp()
            self._tokens[installation_id] = (data["token"], expires_at)
            return data["token"]

    def repo_installation_id(self, owner: str, repo: str) -> int:
        data = self._request("GET", f"/repos/{owner}/{repo}/installation",
                             auth=f"Bearer {self.app_jwt()}", action="installation")
        return int(data["id"])

    # --- repo + PR ----------------------------------------------------------------

    def clone_url(self, owner: str, repo: str) -> str:
        # No credentials in the URL; git_ops passes the token as a header instead.
        return f"https://github.com/{owner}/{repo}.git"

    def get_repo(self, owner: str, repo: str, token: str) -> dict:
        return self._request("GET", f"/repos/{owner}/{repo}", auth=f"token {token}", action="repo")

    def create_pull_request(self, owner: str, repo: str, token: str, *, head: str, base: str,
                            title: str, body: str, draft: bool = False) -> dict:
        payload = {"title": title, "head": head, "base": base, "body": body, "maintainer_can_modify": True}
        try:
            return self._request("POST", f"/repos/{owner}/{repo}/pulls", auth=f"token {token}",
                                 action="pull_request", json={**payload, "draft": draft})
        except GitHubError as e:
            # Draft PRs aren't available on every plan (e.g. private repos on GitHub Free).
            if draft and e.status == 422 and "draft" in str(e).lower():
                return self._request("POST", f"/repos/{owner}/{repo}/pulls", auth=f"token {token}",
                                     action="pull_request", json=payload)
            raise

    # --- installs: which repos users have given PatchForge permission for -----------------

    def app_info(self) -> dict:
        if self._app is None:
            self._app = self._request("GET", "/app", auth=f"Bearer {self.app_jwt()}", action="app")
        return self._app

    def list_projects(self, owner: str, repo: str, installation_id: int) -> list[dict]:
        """Folders in the repo that are a Python or Node project, found from their manifest files."""
        token = self.installation_token(installation_id)
        branch = self.get_repo(owner, repo, token)["default_branch"]
        tree = self._request("GET", f"/repos/{owner}/{repo}/git/trees/{quote(branch)}?recursive=1",
                             auth=f"token {token}", action="tree")
        manifests = {"package.json": "node", "requirements.txt": "python", "pyproject.toml": "python",
                     "setup.py": "python"}
        projects: dict[str, str] = {}
        for item in tree.get("tree", []):
            path = item.get("path", "")
            if item.get("type") != "blob" or "node_modules/" in path or path.count("/") > 3:
                continue
            folder, _, name = path.rpartition("/")
            if name in manifests:
                # A folder with both package.json and requirements.txt counts as Node (its build is npm).
                if projects.get(folder) != "node":
                    projects[folder] = manifests[name]
        return [{"path": k, "language": v} for k, v in sorted(projects.items())]

    def install_url(self) -> str:
        """GitHub's own screen where a user installs the App and picks which repos it may touch."""
        return f"https://github.com/apps/{self.app_info()['slug']}/installations/new"

    def bot_identity(self) -> tuple[str, str]:
        """(name, email) that make commits show up as "<app-slug>[bot]" with the App's avatar."""
        if self._bot_identity is None:
            app = self.app_info()
            login = f"{app['slug']}[bot]"
            # A public endpoint: it rejects the App's JWT, so ask without credentials.
            user = self._request("GET", f"/users/{quote(login)}", auth=None, action="user")
            self._bot_identity = (login, f"{user['id']}+{login}@users.noreply.github.com")
        return self._bot_identity

    # --- plumbing -----------------------------------------------------------------

    def _request(self, method: str, path: str, *, auth: str | None, action: str, json: dict | None = None) -> Any:
        headers = {
            **({"Authorization": auth} if auth else {}),
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "patchforge",
        }
        try:
            resp = self._http.request(method, self.api_url + path, headers=headers, json=json)
        except httpx.HTTPError as e:
            raise GitHubError(f"could not reach GitHub for {action}: {e}") from e
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("message", "")
                errors = resp.json().get("errors")
                if errors:
                    detail += f" {errors}"
            except ValueError:
                detail = resp.text[:300]
            raise GitHubError(f"GitHub {method} {path} failed with {resp.status_code}: {detail}",
                              status=resp.status_code, hint=_hint(resp.status_code, action))
        return resp.json()


def verify_webhook_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    """Check the X-Hub-Signature-256 header GitHub sends with every webhook."""
    if not secret or not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)
