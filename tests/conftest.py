import shutil
import subprocess
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_project"
FAKE_TOKEN = "ghs_" + "A" * 36


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture(scope="session")
def rsa_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    pub = key.public_key().public_bytes(serialization.Encoding.PEM,
                                        serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return pem, pub


@pytest.fixture
def remote_repo(tmp_path):
    """A bare repo on disk standing in for GitHub, seeded with the sample project on `main`."""
    seed = tmp_path / "seed"
    shutil.copytree(SAMPLE, seed)
    git("init", "-q", "-b", "main", cwd=seed)
    git("add", ".", cwd=seed)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=seed)
    bare = tmp_path / "remote.git"
    git("clone", "-q", "--bare", str(seed), str(bare))
    return bare


class FakeGitHub:
    """Implements jobs.GitHubLike against a local bare repo."""

    def __init__(self, bare: Path):
        self.bare = bare
        self.prs = []
        self.token_calls = 0

    def repo_installation_id(self, owner, repo):
        return 42

    def installation_token(self, installation_id):
        self.token_calls += 1
        return FAKE_TOKEN

    def get_repo(self, owner, repo, token):
        return {"default_branch": "main"}

    def clone_url(self, owner, repo):
        return f"file://{self.bare}"

    def create_pull_request(self, owner, repo, token, *, head, base, title, body):
        self.prs.append({"head": head, "base": base, "title": title, "body": body, "token": token})
        n = len(self.prs)
        return {"number": n, "html_url": f"https://github.com/{owner}/{repo}/pull/{n}"}


@pytest.fixture
def fake_github(remote_repo):
    return FakeGitHub(remote_repo)
