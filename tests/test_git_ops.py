import pytest

from patchforge import git_ops
from tests.conftest import FAKE_TOKEN, git


def test_clone_change_commit_push(remote_repo, tmp_path):
    work = git_ops.clone(f"file://{remote_repo}", tmp_path / "w", branch="main", depth=1)
    assert git_ops.changed_files(work) == []
    (work / "calc.py").write_text((work / "calc.py").read_text() + "\n# changed\n")
    (work / "new dir").mkdir()
    (work / "new dir" / "x.py").write_text("x = 1\n")
    assert git_ops.changed_files(work) == ["calc.py", "new dir/x.py"]

    git_ops.create_branch(work, "patchforge/test")
    sha = git_ops.commit_all(work, "msg", author_name="bot[bot]", author_email="1+bot[bot]@users.noreply.github.com")
    git_ops.push(work, "patchforge/test", token=FAKE_TOKEN)  # shallow clone push

    assert git("rev-parse", "patchforge/test", cwd=remote_repo).strip() == sha
    assert "bot[bot]" in git("log", "-1", "--format=%an", "patchforge/test", cwd=remote_repo)
    # The token is never written into the repo's config.
    assert FAKE_TOKEN not in (work / ".git" / "config").read_text()


def test_errors_never_contain_the_token(tmp_path):
    with pytest.raises(git_ops.GitError) as e:
        git_ops.clone(f"file://{tmp_path}/missing-{FAKE_TOKEN}.git", tmp_path / "w", token=FAKE_TOKEN)
    assert FAKE_TOKEN not in str(e.value)
    assert "***" in str(e.value)


def test_auth_env_uses_header_not_url():
    env = git_ops._auth_env("ghs_abc")
    assert env["GIT_CONFIG_KEY_0"] == "http.extraheader"
    assert env["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")
    assert "ghs_abc" not in env["GIT_CONFIG_VALUE_0"]  # base64-encoded
    assert env["GIT_TERMINAL_PROMPT"] == "0"
