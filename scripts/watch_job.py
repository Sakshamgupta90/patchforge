"""Start a job on a running PatchForge server and print its live events.

    python scripts/watch_job.py your-name/your-demo-repo "Upgrade requests to 2.33"
"""

import argparse
import json
import os

import httpx

p = argparse.ArgumentParser()
p.add_argument("repo", help="owner/name of a repo the GitHub App is installed on")
p.add_argument("instruction")
p.add_argument("--server", default="http://127.0.0.1:8000")
p.add_argument("--base", default=None, help="base branch (default: the repo's default branch)")
args = p.parse_args()

headers = {}
if os.getenv("PATCHFORGE_API_KEY"):
    headers["Authorization"] = f"Bearer {os.environ['PATCHFORGE_API_KEY']}"

with httpx.Client(base_url=args.server, headers=headers, timeout=None) as client:
    r = client.post("/jobs", json={"repo": args.repo, "instruction": args.instruction, "base_branch": args.base})
    if r.status_code >= 400:
        raise SystemExit(f"could not start job: {r.status_code} {r.text}")
    job = r.json()
    print(f"job {job['id']} started; streaming events\n")

    event_name = None
    with client.stream("GET", job["events_url"]) as stream:
        for line in stream.iter_lines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
                if event_name == "end":
                    print(f"\n{data['status'].upper()}", end="")
                    print(f": {data['pr_url']}" if data["pr_url"] else f": {data['error']}")
                    t = data["totals"]
                    print(f"tokens in/out {t['tokens_in']:,}/{t['tokens_out']:,}, cost ${t['cost_usd']:.4f}")
                    break
                cost = f" ${data['cost_usd']:.4f}" if data["model"] else ""
                model = f" [{data['model']}]" if data["model"] else ""
                print(f"{data['stage']:>15} | {data['message'].splitlines()[0]}{model}{cost}")
                event_name = None
