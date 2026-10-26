# PatchForge

AI coding assistants are stuck at their training cutoff: ask one to apply a new
security fix or a fresh breaking release and it confidently writes the old API.
PatchForge reads today's migration guide from the web, patches the repo, proves
the patch in a sandbox, and opens a pull request only when the tests pass. Each
successful migration is saved as a reusable skill, so the next repo with the same
upgrade is faster and cheaper.

```
POST /jobs ─► clone ─► baseline tests ─► identify library ─► Delta Spec ─► bump pin ─► patch files ─► tests ─┐
                       (sandbox)          (architect model)   (Tavily +     (deps.py)   (worker models,  (sandbox)│
                                                               architect,               in parallel)            │
                                                               or saved skill)          ◄── retry with errors ──┘
                                                                                                      │ pass only
   dashboard ◄── GET /jobs/{id}/events (every step, with tokens, cost, latency)       push branch ─► open PR ─► save skill
```

**Security rule:** the backend holds the GitHub token. The sandbox gets only a copy
of the code and no environment variables, and the agent gets only the working tree.
Untrusted code never touches credentials. The agent never edits test files: they
are the referee.

## Layout

| File | What it does |
|---|---|
| `src/patchforge/contracts.py` | The 3 shared interfaces: `SandboxResult`, `AgentEvent`, `AgentOutput` (+ `Agent`/`Sandbox` protocols) |
| `src/patchforge/brain.py` | The agent: identifies the library, builds the Delta Spec from Tavily search results, patches files in parallel, retries with test errors |
| `src/patchforge/skills.py` | Saves a passing run's Delta Spec as `skills/<lib>-<version>/SKILL.md` and reuses it, skipping web research |
| `src/patchforge/deps.py` | Bumps the library's pin in `requirements*.txt` / `pyproject.toml` (deterministic, not the model) |
| `src/patchforge/github_app.py` | App JWT, cached 1-hour installation tokens, repo lookup, open PR, webhook signature check, bot identity |
| `src/patchforge/git_ops.py` | clone / branch / commit / push; token passed as an HTTP header via env, never in URLs, `.git/config` or errors |
| `src/patchforge/sandbox/docker_runner.py` | Install deps (network on), then run pytest (network **off**), non-root, read-only FS, caps dropped, mem/CPU/pid limits |
| `src/patchforge/sandbox/nebius.py` | Runs the tests as a Nebius Serverless AI Job via the `nebius` CLI; Docker is the fallback |
| `src/patchforge/sandbox/local.py` | Runs pytest on this machine with the repo's pinned deps; no isolation, your own code only |
| `src/patchforge/sandbox/fake.py` | Canned results for agent development (fails once, then passes) |
| `src/patchforge/agent_stub.py` | Fake agent for testing the platform without API keys |
| `src/patchforge/jobs.py` | The pipeline. Refuses to open a PR if final tests fail or nothing changed. Saves job history |
| `src/patchforge/auth.py` | Sign in with GitHub, sessions, and each user's repositories |
| `src/patchforge/api.py` | FastAPI app: sign-in routes, per-user jobs, dashboard |
| `src/patchforge/static/index.html` | Dashboard: start jobs, live events, pipeline steps, cost/token/latency per model, saved skills |

## Setup

```bash
cd ~/Desktop/patchforge
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env         # then fill it in, see below
pytest                       # 119 passed, 1 skipped (the Docker test skips if Docker isn't running)
```

### Keys (`.env`)

| Variable | Where it comes from |
|---|---|
| `NEBIUS_API_KEY` | Nebius Token Factory → API keys |
| `TAVILY_API_KEY` | tavily.com dashboard |
| `GITHUB_APP_ID`, `GITHUB_PRIVATE_KEY_PATH`, `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | The GitHub App (next section) |

### Models

| Role | Setting | Default | Why |
|---|---|---|---|
| Research: identify the library, read the migration guide, build the Delta Spec | `PATCHFORGE_ARCHITECT_MODEL` | `Qwen/Qwen3-235B-A22B-Instruct-2507` | reliable structured output, fast, cheap |
| Code: write each file's patch | `PATCHFORGE_WORKER_MODEL` | `moonshotai/Kimi-K2.7-Code` | on identical jobs it passed every test case; Qwen3-30B/235B, GLM-5.3 and Nemotron failed Flask and/or Express |
| Fixes after a failed check | `PATCHFORGE_FIXER_MODEL` | the code model | |

A code-model call that takes longer than `PATCHFORGE_MODEL_TIMEOUT` seconds (default 120) is retried on the
research model. Prices for the cost display are in `PRICES` in `brain.py`; a model missing there shows as $0.

### How a job decides it's done

- **Research** searches the library's own docs site first (found through PyPI/npm), reads full pages, and
  quotes the guide's examples verbatim; a general web search fills the gaps.
- **Version pins** come from the registry, never the model, and the project's `.python-version` / `.nvmrc`
  is raised when the new version needs a newer runtime.
- **Retries** (up to 5, while each one changes the outcome) show the model every failing test, the failing
  test's code (read-only), and web results for the error message itself.
- Edits that make checks pass by switching checking off (`"strict": false`, `@ts-ignore`, `# type: ignore`,
  skipped tests…) are rejected.
- **No tests in the project?** It's built, type-checked or compiled instead, and the PR opens as a **draft
  marked "not verified by tests"**.

## Create the GitHub App (one time, ~5 minutes)

One GitHub App does both jobs: it lets users **sign in with GitHub**, and it is what opens pull
requests on the repositories they grant.

1. GitHub → **Settings → Developer settings → GitHub Apps → New GitHub App**.
2. **Name:** e.g. `patchforge-yourteam` (must be globally unique). **Homepage URL:** your PatchForge address.
3. **Identifying and authorizing users:**
   - **Callback URL:** `http://127.0.0.1:8000/auth/callback`
   - Keep **Expire user authorization tokens** ticked (PatchForge refreshes them).
   - Leave **Request user authorization (OAuth) during installation** unticked: users sign in first, then install.
4. **Post installation → Setup URL:** `http://127.0.0.1:8000/github/setup`, and tick **Redirect on update**.
   This brings users back to PatchForge after they grant access.
5. **Webhook:** uncheck **Active** for local use. (Later: point it at `https://<your-host>/webhooks/github` with a secret.)
6. **Repository permissions:** Contents **Read and write**, Pull requests **Read and write**
   (Metadata read-only is added automatically). No account permissions are needed.
7. **Where can this GitHub App be installed?** **Any account**, so anyone can use your PatchForge.
8. Create it. On the App's page:
   - copy the **App ID** and **Client ID**,
   - **Generate a new client secret**,
   - **Generate a private key** (a `.pem` downloads; move it into this folder, it's git-ignored).
9. Put them in `.env`: `GITHUB_APP_ID`, `GITHUB_PRIVATE_KEY_PATH`, `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`.

When you deploy somewhere public, replace `http://127.0.0.1:8000` in steps 3–4 and in
`PATCHFORGE_PUBLIC_URL` with your HTTPS address (cookies are then marked `Secure` automatically).

### How users connect their repositories

1. They open PatchForge and click **Sign in with GitHub**.
2. They click **Install it on a repository**. GitHub shows its own screen where they pick an account and
   **only the repositories** PatchForge may open pull requests on. GitHub sends them back afterwards.
3. They pick one of those repositories (or paste any `owner/name` or GitHub URL) and describe the upgrade.
   If PatchForge isn't installed on a pasted repository, they're told which account to pick and get a
   button that opens GitHub's install screen.

### Who can do what

- A user can run a job only on a repository where the App is installed **and** they have write access.
  The server checks this live with the user's own GitHub token when the job is submitted; nothing the
  browser sends (like an installation id) is trusted.
- Users see only their own jobs. Someone else's job id returns 404.
- The user's GitHub token stays on the server. The browser holds only a random session id in an
  `HttpOnly`, `SameSite=Lax` cookie. Sessions live in memory, so a server restart signs everyone out.
- Pushing and opening the PR use a separate installation token that expires after an hour and covers
  only the granted repositories. Owners can revoke access any time under GitHub → Settings → Applications.
- `PATCHFORGE_API_KEY` (for `scripts/watch_job.py`) acts across every installation. Keep it secret, or leave it empty.

## Run the demo

1. Push `tests/fixtures/demo-repo` to its own GitHub repo and install the App on it:
   ```bash
   cp -R tests/fixtures/demo-repo ~/Desktop/patchforge-demo && cd ~/Desktop/patchforge-demo
   git init -b main && git add -A && git commit -m "Demo service on pydantic 1.10"
   git remote add origin https://github.com/<you>/patchforge-demo.git && git push -u origin main
   ```
2. Start the server and open the dashboard at http://127.0.0.1:8000:
   ```bash
   uvicorn patchforge.api:create_app --factory --host 127.0.0.1 --port 8000
   ```
3. Sign in, install the App on `<you>/patchforge-demo`, then start a job with the instruction
   `Upgrade pydantic to v2`. You'll get a PR from `your-app[bot]`.
4. **The wow moment:** run the same instruction on a second copy of the demo repo. The dashboard shows
   "Research (skill)": no web search, fewer model calls, lower cost.

Terminal alternative (uses `PATCHFORGE_API_KEY` instead of sign-in): `python scripts/watch_job.py <you>/patchforge-demo "Upgrade pydantic to v2"`.

### Sandboxes (`PATCHFORGE_SANDBOX`)

| Value | When |
|---|---|
| `docker` (default) | Strongest isolation. Docker Desktop must be running; first run pulls `python:3.12-slim` |
| `nebius` | Tests run as a Nebius Serverless AI Job. Needs the `nebius` CLI logged in, plus `NEBIUS_JOB_PLATFORM` and `NEBIUS_JOB_PRESET` set to a **CPU** platform from your project (Nebius defaults to an H100 GPU). Repo must pack to ≤64 KiB. Falls back to Docker if Nebius can't run. Jobs take a few minutes to start |
| `local` | No Docker: runs on this machine, your own code only. Old libraries often need an older Python: set `PATCHFORGE_LOCAL_PYTHON=python3.12` (pydantic 1.10 doesn't run on 3.14) |
| `fake` | Always passes. Platform testing only |

### API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/` | Dashboard |
| `POST` | `/jobs` | `{"repo": "owner/name or GitHub URL", "instruction": "...", "base_branch"?: "main"}` → `202` with `events_url`. Signed-in user (or the API key) |
| `GET` | `/jobs`, `/jobs/{id}` | The caller's own jobs: status, PR URL, error, token/cost totals. History survives restarts |
| `GET` | `/jobs/{id}/events` | Server-Sent Events. Replays history first; resumes from `Last-Event-ID` or `?after=N`. Ends with `event: end` |
| `GET` | `/auth/login`, `/auth/callback` | Sign in with GitHub |
| `GET` / `POST` | `/auth/me`, `/auth/logout` | The signed-in user; sign out |
| `GET` | `/github/repos` | The signed-in user's repositories that have the App installed (`can_write` marks usable ones) |
| `GET` | `/github`, `/github/setup` | App install link; where GitHub sends users back after granting access |
| `GET` | `/skills` | Saved skills and how often each was reused |
| `POST` | `/webhooks/github` | Verifies `X-Hub-Signature-256`; tracks installs |
| `GET` | `/health` | Which sandbox/agent are active |

## Limits

- Python (pip + pytest) and Node.js/TypeScript (npm) projects. One library (plus its lockstep siblings) per job.
- Verified with real models on 8 cases in [upgrade-test-cases](https://github.com/Sakshamgupta90/upgrade-test-cases):
  pydantic, Flask, Django, pandas, Express, node-security, TypeScript and React all pass their tests after the upgrade.
- The Nebius sandbox hasn't been run against a live Nebius project yet (Docker is the default).
- Jobs run in-process (2 at a time); history is JSON files, not a database.
