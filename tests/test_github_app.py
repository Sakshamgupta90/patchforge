import hashlib
import hmac
import json

import httpx
import jwt
import pytest

from patchforge.github_app import GitHubApp, GitHubError, verify_webhook_signature


def make_app(rsa_key, handler, clock=lambda: 1_800_000_000):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GitHubApp("123", rsa_key[0], client=client, clock=clock)


def iso(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_jwt_claims(rsa_key):
    app = make_app(rsa_key, lambda r: httpx.Response(500))
    claims = jwt.decode(app.app_jwt(), rsa_key[1], algorithms=["RS256"],
                        options={"verify_exp": False, "verify_iat": False})
    assert claims["iss"] == "123"
    assert claims["iat"] == 1_800_000_000 - 60
    assert claims["exp"] - 1_800_000_000 <= 600


def test_installation_token_is_cached_and_refreshed(rsa_key):
    now = [1_800_000_000]
    calls = []

    def handler(req):
        calls.append(req)
        assert req.url.path == "/app/installations/7/access_tokens"
        assert req.headers["Authorization"].startswith("Bearer ")
        return httpx.Response(201, json={"token": f"ghs_tok{len(calls)}", "expires_at": iso(now[0] + 3600)})

    app = make_app(rsa_key, handler, clock=lambda: now[0])
    assert app.installation_token(7) == "ghs_tok1"
    assert app.installation_token(7) == "ghs_tok1"
    assert len(calls) == 1
    now[0] += 3600 - 60  # inside the refresh margin
    assert app.installation_token(7) == "ghs_tok2"
    assert len(calls) == 2


@pytest.mark.parametrize("status,needle", [(404, "isn't installed"), (403, "Pull requests"), (401, "GITHUB_APP_ID")])
def test_errors_carry_hints(rsa_key, status, needle):
    app = make_app(rsa_key, lambda r: httpx.Response(status, json={"message": "nope"}))
    with pytest.raises(GitHubError) as e:
        app.repo_installation_id("o", "r") if status == 404 else app.installation_token(1)
    assert e.value.status == status
    assert needle in str(e.value)


def test_create_pull_request(rsa_key):
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["Authorization"]
        seen["body"] = json.loads(req.content)
        assert req.url.path == "/repos/o/r/pulls"
        return httpx.Response(201, json={"number": 5, "html_url": "https://github.com/o/r/pull/5"})

    app = make_app(rsa_key, handler)
    pr = app.create_pull_request("o", "r", "ghs_x", head="patchforge/x", base="main", title="t", body="b")
    assert pr["number"] == 5
    assert seen["auth"] == "token ghs_x"
    assert seen["body"]["head"] == "patchforge/x" and seen["body"]["base"] == "main"


def test_pr_422_hint(rsa_key):
    app = make_app(rsa_key, lambda r: httpx.Response(422, json={"message": "Validation Failed"}))
    with pytest.raises(GitHubError, match="already exist"):
        app.create_pull_request("o", "r", "t", head="h", base="main", title="t", body="b")


def test_bot_identity(rsa_key):
    def handler(req):
        if req.url.path == "/app":
            return httpx.Response(200, json={"slug": "patchforge"})
        assert req.url.path == "/users/patchforge[bot]"
        assert "Authorization" not in req.headers  # public endpoint; it rejects the App JWT
        return httpx.Response(200, json={"id": 999})

    app = make_app(rsa_key, handler)
    assert app.bot_identity() == ("patchforge[bot]", "999+patchforge[bot]@users.noreply.github.com")


def test_clone_url_has_no_credentials(rsa_key):
    app = make_app(rsa_key, lambda r: httpx.Response(500))
    assert app.clone_url("o", "r") == "https://github.com/o/r.git"


def test_webhook_signature():
    body = b'{"zen":"hi"}'
    sig = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
    assert verify_webhook_signature("s3cret", body, sig)
    assert not verify_webhook_signature("wrong", body, sig)
    assert not verify_webhook_signature("s3cret", body + b" ", sig)
    assert not verify_webhook_signature("s3cret", body, None)
    assert not verify_webhook_signature("", body, sig)


def test_install_url(rsa_key):
    app = make_app(rsa_key, lambda r: httpx.Response(200, json={"slug": "patchforge-team", "name": "PatchForge"}))
    assert app.install_url() == "https://github.com/apps/patchforge-team/installations/new"


def test_list_projects(rsa_key):
    def handler(req):
        path = req.url.path
        if path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "ghs_t", "expires_at": iso(1_800_000_000 + 3600)})
        if path == "/repos/me/cases":
            return httpx.Response(200, json={"default_branch": "main"})
        assert path == "/repos/me/cases/git/trees/main"
        return httpx.Response(200, json={"tree": [
            {"path": "README.md", "type": "blob"},
            {"path": "python-flask/requirements.txt", "type": "blob"},
            {"path": "react/package.json", "type": "blob"},
            {"path": "react/node_modules/x/package.json", "type": "blob"},
            {"path": "fullstack/requirements.txt", "type": "blob"},
            {"path": "fullstack/package.json", "type": "blob"},
            {"path": "go/go.mod", "type": "blob"}]})

    projects = make_app(rsa_key, handler).list_projects("me", "cases", 1)
    assert projects == [{"path": "fullstack", "language": "node"}, {"path": "python-flask", "language": "python"},
                        {"path": "react", "language": "node"}]


def test_draft_pr_falls_back_when_drafts_unavailable(rsa_key):
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if body.get("draft"):
            return httpx.Response(422, json={"message": "Draft pull requests are not supported in this repository."})
        return httpx.Response(201, json={"number": 9, "html_url": "https://github.com/o/r/pull/9"})

    pr = make_app(rsa_key, handler).create_pull_request("o", "r", "t", head="h", base="main", title="t", body="b",
                                                        draft=True)
    assert pr["number"] == 9 and bodies[0]["draft"] is True and "draft" not in bodies[1]
