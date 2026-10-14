# PatchForge Platform

The "body" of PatchForge: GitHub App auth, a sandbox that runs a repo's tests, the
job pipeline that turns an agent's edits into a pull request, and a live event
stream for the dashboard. Person A's agent plugs in through `patchforge/contracts.py`.

```
POST /jobs ─► clone (token as header) ─► baseline tests ─► agent edits files ─► final tests
                                          (sandbox)          (Person A)          (sandbox)
                                                                                    │ pass only
            GET /jobs/{id}/events ◄── every step emits an AgentEvent       push branch ─► open PR
```

**Security rule:** the backend holds the GitHub token. The sandbox gets only a copy
of the code and no environment variables, and the agent gets only the working tree.
Untrusted code never touches credentials.

## Layout

| File | What it does |
|---|---|
| `patchforge/contracts.py` | The 3 shared interfaces: `SandboxResult`, `AgentEvent`, `AgentOutput` (+ `Agent`/`Sandbox` protocols) |
| `patchforge/github_app.py` | App JWT, cached 1-hour installation tokens, repo lookup, open PR, webhook signature check, bot identity |
| `patchforge/git_ops.py` | clone / branch / commit / push; token passed as an HTTP header via env, never in URLs, `.git/config` or errors |
| `patchforge/sandbox/docker_runner.py` | Real sandbox: install deps (network on), then run pytest (network **off**), non-root, read-only FS, caps dropped, mem/CPU/pid limits |
| `patchforge/sandbox/fake.py` | Canned results for Person A (fails once, then passes) |
| `patchforge/sandbox/local.py` | Runs pytest on this machine with no isolation; your own code only |
| `patchforge/agent_stub.py` | Stand-in agent: fake model calls with tokens/costs, writes one file so a real PR opens |
| `patchforge/jobs.py` | The pipeline. Refuses to open a PR if final tests fail or nothing changed |
| `patchforge/api.py` | FastAPI app: `/jobs`, `/jobs/{id}`, `/jobs/{id}/events` (SSE), `/webhooks/github`, `/health` |

## Setup

```bash
cd ~/Desktop/patchforge
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                       # 34 passed, 1 skipped (the Docker test skips if Docker isn't running)
```

Try the sandbox on the sample project:

```bash
python scripts/try_sandbox.py examples/sample_project --local   # no Docker needed
python scripts/try_sandbox.py examples/sample_project           # Docker Desktop must be running
```

The first Docker run pulls `python:3.12-slim` and takes a minute.

## Create the GitHub App (one time, ~5 minutes)

1. GitHub → **Settings → Developer settings → GitHub Apps → New GitHub App**.
2. **Name:** e.g. `patchforge-yourteam` (must be globally unique). **Homepage URL:** anything, e.g. your repo URL.
3. **Webhook:** for local dev, uncheck **Active**. (To receive install events later, expose the server with a tunnel such as smee.io or ngrok, set the URL to `https://<tunnel>/webhooks/github`, and set a **Webhook secret**.)
4. **Repository permissions:**
   - Contents: **Read and write**
   - Pull requests: **Read and write**
   - Metadata: Read-only (set automatically)
5. **Where can this app be installed?** "Only on this account" is fine for the hackathon.
6. Create it. On the App page, note the **App ID**, then **Generate a private key**; a `.pem` file downloads.
7. Move the `.pem` into this folder (it's git-ignored), then: `cp .env.example .env` and fill in `GITHUB_APP_ID` and `GITHUB_PRIVATE_KEY_PATH`.
8. **Install App** (left sidebar) → choose your account → **Only select repositories** → pick your demo repo.

## Run it end to end

```bash
uvicorn patchforge.api:create_app --factory --host 127.0.0.1 --port 8000
# in a second terminal:
python scripts/watch_job.py your-name/your-demo-repo "Upgrade requests to 2.33"
```

You should see each stage stream by, ending with a PR link opened by `your-app[bot]`.
The stub agent only adds `PATCHFORGE_STUB.md`; the demo repo needs a pytest suite
that passes, or the pipeline will (correctly) refuse to open the PR.
Without Docker running, set `PATCHFORGE_SANDBOX=local` in `.env`, but only for repos you wrote.

### API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/jobs` | `{"repo": "owner/name", "instruction": "...", "base_branch"?: "main", "installation_id"?: 123}` → `202` with `events_url`. Needs `Authorization: Bearer $PATCHFORGE_API_KEY` if that's set |
| `GET` | `/jobs`, `/jobs/{id}` | Status, PR URL, error, token/cost totals |
| `GET` | `/jobs/{id}/events` | Server-Sent Events. Replays history first; resumes from `Last-Event-ID` or `?after=N`. Ends with `event: end` carrying the final job snapshot |
| `POST` | `/webhooks/github` | Verifies `X-Hub-Signature-256`; tracks installs |
| `GET` | `/health` | Which sandbox/agent are active |

In the browser: `new EventSource("http://127.0.0.1:8000/jobs/<id>/events")`; add your dashboard's origin to `CORS_ORIGINS`.

## Handoff notes for Person A

- Implement `run(task, workdir, sandbox, emit) -> AgentOutput` from `contracts.Agent`. Edit files in `workdir` in place; don't commit or push, the platform does that.
- Call `sandbox.run_tests(workdir)` as often as you like for the retry loop. During development use `FakeSandbox()` (fails once, then passes) or `FakeSandbox([...your results...])`.
- Emit an `AgentEvent` for every model call with `model`, `tokens_in`, `tokens_out`, `cost_usd`, `latency_ms`; the dashboard's cost HUD sums these. **Change from the Day 1 sketch:** `tokens` is split into `tokens_in`/`tokens_out` because input and output are priced differently.
- Use stages `research`, `plan`, `patch`, `retry`, `skill` so the dashboard can group them.
- Register your agent in `_make_agent()` in `api.py` and set `PATCHFORGE_AGENT=<name>`.
- The platform re-runs tests itself and reads changed files from git, so `AgentOutput.changed_files` and `test_results` are informational. `summary` (markdown) goes straight into the PR body.

## Not done yet (Days 6+)

- Nebius Serverless Jobs sandbox (implement the same `run_tests()`; keep Docker as the fallback)
- Dashboard UI and cost/latency HUD (the event stream is ready for it)
- Persistence: jobs live in memory and vanish on restart
- Only Python + pytest repos are supported by the sandbox
