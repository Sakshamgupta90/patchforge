import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from patchforge.agent_stub import StubAgent
from patchforge.api import create_app
from patchforge.config import Settings
from patchforge.jobs import JobRunner
from patchforge.sandbox import AlwaysPassSandbox


@pytest.fixture
def client(fake_github):
    runner = JobRunner(github=fake_github, sandbox_factory=AlwaysPassSandbox,
                       agent_factory=lambda: StubAgent(delay_s=0), clone_depth=None)
    settings = Settings(github_app_id="1", github_private_key="x", github_webhook_secret="s3cret",
                        sandbox="fake", api_key="k")
    return TestClient(create_app(runner, settings))


AUTH = {"Authorization": "Bearer k"}


def read_sse(client, url, headers=None):
    events, end, name = [], None, None
    with client.stream("GET", url, headers=headers or {}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
                if name == "end":
                    end = data
                    break
                events.append(data)
    return events, end


def test_health(client):
    assert client.get("/health").json()["ok"] is True


def test_api_key_required(client):
    assert client.post("/jobs", json={"repo": "me/demo", "instruction": "x"}).status_code == 401
    assert client.post("/jobs", json={"repo": "bad", "instruction": "x"}, headers=AUTH).status_code == 422


def test_job_lifecycle_and_sse(client):
    r = client.post("/jobs", json={"repo": "me/demo", "instruction": "Upgrade x"}, headers=AUTH)
    assert r.status_code == 202
    job = r.json()

    events, end = read_sse(client, job["events_url"])
    assert end["status"] == "succeeded", end["error"]
    assert end["pr_url"].endswith("/pull/1")
    assert [e["id"] for e in events] == list(range(len(events)))
    assert events[0]["stage"] == "job" and events[-1]["stage"] == "done"

    # A refreshed dashboard resumes from Last-Event-ID.
    later, _ = read_sse(client, job["events_url"], headers={"Last-Event-ID": "3"})
    assert later[0]["id"] == 4

    assert client.get(f"/jobs/{job['id']}").json()["status"] == "succeeded"
    assert client.get("/jobs").json()[0]["id"] == job["id"]
    assert client.get("/jobs/nope").status_code == 404


def sign(body: bytes) -> str:
    return "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()


def test_webhook(client):
    body = json.dumps({"action": "created", "installation": {"id": 77, "account": {"login": "me"}}}).encode()
    assert client.post("/webhooks/github", content=body,
                       headers={"X-GitHub-Event": "installation", "X-Hub-Signature-256": "sha256=bad"}).status_code == 401
    r = client.post("/webhooks/github", content=body,
                    headers={"X-GitHub-Event": "installation", "X-Hub-Signature-256": sign(body)})
    assert r.status_code == 200
    assert client.get("/installations").json() == [{"id": 77, "account": "me"}]

    ping = b'{"zen":"hi"}'
    assert client.post("/webhooks/github", content=ping,
                       headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": sign(ping)}).json()["pong"]
