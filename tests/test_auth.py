from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from patchforge.agent_stub import StubAgent
from patchforge.api import create_app
from patchforge.auth import AuthError, GitHubOAuth, SessionStore, parse_repo
from patchforge.config import Settings
from patchforge.jobs import JobRunner
from patchforge.sandbox import AlwaysPassSandbox
from tests.conftest import FakeGitHub

USERS = {
    "code-ada": {"id": 1, "login": "ada", "name": "Ada", "avatar_url": "https://avatars/ada"},
    "code-bob": {"id": 2, "login": "bob", "name": None, "avatar_url": None},
}
REPOS = {
    "ada": [{"full_name": "ada/app", "private": True, "default_branch": "main", "installation_id": 42,
             "account": "ada", "can_write": True},
            {"full_name": "acme/docs", "private": False, "default_branch": "main", "installation_id": 7,
             "account": "acme", "can_write": False}],
    "bob": [],
}


class FakeOAuth:
    """Stands in for GitHub's OAuth + user APIs."""

    def __init__(self):
        self.repo_calls = 0

    def authorize_url(self, state):
        return f"https://github.com/login/oauth/authorize?client_id=cid&state={state}"

    def exchange_code(self, code):
        if code not in USERS:
            raise AuthError("bad_verification_code")
        return f"token-{code}", None, None

    def user(self, token):
        return USERS[token.removeprefix("token-")]

    def user_repos(self, token):
        self.repo_calls += 1
        return REPOS[self.user(token)["login"]]


class AppGitHub(FakeGitHub):
    def app_info(self):
        return {"name": "PatchForge", "slug": "patchforge-team"}

    def install_url(self):
        return "https://github.com/apps/patchforge-team/installations/new"


@pytest.fixture
def setup(fake_github):
    gh = AppGitHub(fake_github.bare)
    runner = JobRunner(github=gh, sandbox_factory=AlwaysPassSandbox, agent_factory=lambda: StubAgent(delay_s=0),
                       clone_depth=None)
    oauth = FakeOAuth()
    app = create_app(runner, Settings(github_app_id="1", github_private_key="x", api_key="ops"), SessionStore(oauth))
    return app, runner, oauth


def sign_in(app, code):
    client = TestClient(app)
    r = client.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    r = client.get(f"/auth/callback?code={code}&state={state}", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/"
    return client


def test_parse_repo():
    assert parse_repo("ada/app") == "ada/app"
    assert parse_repo("https://github.com/ada/app") == "ada/app"
    assert parse_repo("https://github.com/ada/app.git") == "ada/app"
    assert parse_repo("github.com/ada/my.app/tree/main/src") == "ada/my.app"
    assert parse_repo("  https://www.github.com/ada/app/  ") == "ada/app"
    assert parse_repo("https://gitlab.com/ada/app") is None
    assert parse_repo("not a repo") is None


def test_sign_in_sets_http_only_cookie(setup):
    app, _, _ = setup
    client = TestClient(app)
    assert client.get("/auth/me").status_code == 401
    r = client.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    r = client.get(f"/auth/callback?code=code-ada&state={state}", follow_redirects=False)
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert "token-code-ada" not in r.headers["set-cookie"]  # the GitHub token never reaches the browser
    assert client.get("/auth/me").json() == {"login": "ada", "name": "Ada", "avatar_url": "https://avatars/ada"}


def test_callback_rejects_unknown_or_reused_state(setup):
    app, _, _ = setup
    client = TestClient(app)
    r = client.get("/auth/callback?code=code-ada&state=forged", follow_redirects=False)
    assert r.headers["location"] == "/?auth_error=expired"
    r = client.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    client.get(f"/auth/callback?code=code-ada&state={state}", follow_redirects=False)
    other = TestClient(app)
    r = other.get(f"/auth/callback?code=code-bob&state={state}", follow_redirects=False)
    assert r.headers["location"] == "/?auth_error=expired"  # a state works once
    assert other.get("/auth/me").status_code == 401


def test_bad_code_and_denied(setup):
    app, _, _ = setup
    client = TestClient(app)
    r = client.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    assert client.get(f"/auth/callback?code=nope&state={state}", follow_redirects=False).headers["location"] \
        == "/?auth_error=failed"
    assert client.get("/auth/callback?error=access_denied", follow_redirects=False).headers["location"] \
        == "/?auth_error=access_denied"


def test_repos_are_per_user(setup):
    app, _, _ = setup
    assert TestClient(app).get("/github/repos").status_code == 401
    ada = sign_in(app, "code-ada")
    assert [r["full_name"] for r in ada.get("/github/repos").json()] == ["ada/app", "acme/docs"]
    bob = sign_in(app, "code-bob")
    assert bob.get("/github/repos").json() == []


def test_job_uses_verified_access_and_url_input(setup):
    app, runner, _ = setup
    ada = sign_in(app, "code-ada")
    r = ada.post("/jobs", json={"repo": "https://github.com/ada/app", "instruction": "Upgrade x",
                                "installation_id": 1})  # a client-supplied installation id is ignored
    assert r.status_code == 202, r.text
    job = runner.jobs[r.json()["id"]]
    assert job.owner == "ada"
    assert job.request.repo == "ada/app" and job.request.installation_id == 42


def test_job_on_repo_without_the_app_offers_install_link(setup):
    app, runner, _ = setup
    ada = sign_in(app, "code-ada")
    r = ada.post("/jobs", json={"repo": "someone/else", "instruction": "x"})
    assert r.status_code == 403
    detail = r.json()["detail"]
    assert detail["code"] == "not_installed"
    assert detail["install_url"] == "https://github.com/apps/patchforge-team/installations/new"
    assert "someone account" in detail["message"]
    assert runner.jobs == {}


def test_job_needs_write_access(setup):
    app, runner, _ = setup
    ada = sign_in(app, "code-ada")
    r = ada.post("/jobs", json={"repo": "acme/docs", "instruction": "x"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "no_write_access"
    assert runner.jobs == {}


def test_jobs_are_private_to_their_owner(setup):
    app, runner, _ = setup
    ada, bob = sign_in(app, "code-ada"), sign_in(app, "code-bob")
    job_id = ada.post("/jobs", json={"repo": "ada/app", "instruction": "x"}).json()["id"]
    assert [j["id"] for j in ada.get("/jobs").json()] == [job_id]
    assert bob.get("/jobs").json() == []
    assert bob.get(f"/jobs/{job_id}").status_code == 404
    assert bob.get(f"/jobs/{job_id}/events").status_code == 404
    # The operator key sees everything.
    assert TestClient(app).get(f"/jobs/{job_id}", headers={"Authorization": "Bearer ops"}).status_code == 200


def test_logout(setup):
    app, _, _ = setup
    ada = sign_in(app, "code-ada")
    assert ada.post("/auth/logout").status_code == 200
    assert ada.get("/auth/me").status_code == 401
    assert ada.get("/jobs").status_code == 401


def test_sign_in_disabled_without_oauth_config(fake_github):
    runner = JobRunner(github=fake_github, sandbox_factory=AlwaysPassSandbox, agent_factory=lambda: StubAgent(delay_s=0))
    client = TestClient(create_app(runner, Settings(github_app_id="1", github_private_key="x")))
    assert client.get("/health").json()["sign_in_enabled"] is False
    assert client.get("/auth/login").status_code == 503


# --- the real GitHub client, against a mocked GitHub ---------------------------------------

def oauth_with(handler, clock=lambda: 1_000_000):
    return GitHubOAuth("cid", "secret", "http://127.0.0.1:8000/auth/callback",
                       client=httpx.Client(transport=httpx.MockTransport(handler)), clock=clock)


def test_exchange_code_and_user_repos():
    def handler(req):
        path = req.url.path
        if path == "/login/oauth/access_token":
            body = parse_qs(req.content.decode())
            assert body["client_secret"] == ["secret"] and body["code"] == ["abc"]
            return httpx.Response(200, json={"access_token": "ghu_x", "expires_in": 28800, "refresh_token": "ghr_y"})
        assert req.headers["Authorization"] == "Bearer ghu_x"
        if path == "/user/installations":
            return httpx.Response(200, json={"installations": [{"id": 42, "account": {"login": "ada"}}]})
        if path == "/user/installations/42/repositories":
            return httpx.Response(200, json={"repositories": [
                {"full_name": "ada/b", "private": False, "default_branch": "main", "permissions": {"push": False, "pull": True}},
                {"full_name": "ada/a", "private": True, "default_branch": "dev", "permissions": {"admin": True}}]})
        return httpx.Response(404)

    oauth = oauth_with(handler)
    assert oauth.exchange_code("abc") == ("ghu_x", 1_000_000 + 28800, "ghr_y")
    repos = oauth.user_repos("ghu_x")
    assert [(r["full_name"], r["can_write"], r["installation_id"]) for r in repos] == \
        [("ada/a", True, 42), ("ada/b", False, 42)]


def test_oauth_error_is_raised():
    oauth = oauth_with(lambda req: httpx.Response(200, json={"error": "bad_verification_code",
                                                           "error_description": "The code is incorrect"}))
    with pytest.raises(AuthError, match="incorrect"):
        oauth.exchange_code("x")


def test_expiring_user_token_is_refreshed():
    now = [1_000_000]
    calls = []

    def handler(req):
        body = parse_qs(req.content.decode())
        calls.append(body.get("grant_type", ["code"])[0])
        if body.get("grant_type") == ["refresh_token"]:
            assert body["refresh_token"] == ["ghr_1"]
            return httpx.Response(200, json={"access_token": "ghu_2", "expires_in": 28800, "refresh_token": "ghr_2"})
        return httpx.Response(200, json={"access_token": "ghu_1", "expires_in": 28800, "refresh_token": "ghr_1"})

    oauth = oauth_with(handler, clock=lambda: now[0])
    store = SessionStore(oauth, clock=lambda: now[0])
    oauth.user = lambda token: {"id": 1, "login": "ada"}
    session = store.login("abc")
    assert store.token(session) == "ghu_1"
    now[0] += 28800 - 60  # inside the refresh margin
    assert store.token(session) == "ghu_2" and session.refresh_token == "ghr_2"
    assert calls == ["code", "refresh_token"]


def test_projects_in_a_repo(setup):
    app, runner, _ = setup
    runner.github.list_projects = lambda owner, name, inst: [
        {"path": "python-flask", "language": "python"}, {"path": "react", "language": "node"}]
    ada = sign_in(app, "code-ada")
    assert [p["path"] for p in ada.get("/github/projects?repo=https://github.com/ada/app").json()] == ["python-flask", "react"]
    assert ada.get("/github/projects?repo=someone/else").json() == []  # no access, nothing listed
    r = ada.post("/jobs", json={"repo": "ada/app", "instruction": "x", "path": "python-flask"})
    assert runner.jobs[r.json()["id"]].request.path == "python-flask"
    assert ada.post("/jobs", json={"repo": "ada/app", "instruction": "x", "path": "../etc"}).status_code == 422
