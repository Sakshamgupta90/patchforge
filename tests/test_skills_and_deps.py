import time

from patchforge import deps
from patchforge.skills import SkillStore

SPEC = {"library": "pydantic", "from_version": "1.10", "to_version": "2.9.2",
        "changes": [{"old": "a", "new": "b", "note": "n", "source": "https://x"}]}


def test_save_and_find(tmp_path):
    store = SkillStore(tmp_path)
    skill = store.save(SPEC, library="pydantic", from_version="1.10", to_version="latest")
    assert skill.path == tmp_path / "pydantic-2.9.2" / "SKILL.md"
    assert skill.spec == SPEC
    assert store.find("Pydantic", "latest").spec == SPEC
    assert store.find("pydantic", "2").to_version == "2.9.2"
    assert store.find("pydantic", "2.9").to_version == "2.9.2"
    assert store.find("pydantic", "3") is None
    assert store.find("requests", "latest") is None


def test_latest_ignores_old_skills(tmp_path):
    store = SkillStore(tmp_path, max_age_days=30)
    skill = store.save(SPEC, library="pydantic", from_version="1.10", to_version="2.9.2")
    text = skill.path.read_text().replace(f"created: {skill.created:.0f}", f"created: {time.time() - 40 * 86400:.0f}")
    skill.path.write_text(text)
    assert store.find("pydantic", "latest") is None
    assert store.find("pydantic", "2.9.2") is not None  # explicit versions don't expire


def test_record_use(tmp_path):
    store = SkillStore(tmp_path)
    skill = store.save(SPEC, library="pydantic", from_version="1.10", to_version="2.9.2")
    store.record_use(skill)
    store.record_use(skill)
    assert store.all()[0].uses == 2


def test_corrupt_skill_is_skipped(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text("not a skill")
    assert SkillStore(tmp_path).all() == []


def test_version_spec():
    assert deps.version_spec("2.9.2") == "==2.9.2"
    assert deps.version_spec("v2.9") == "==2.9"
    assert deps.version_spec("2") == ">=2,<3"
    assert deps.version_spec("latest") == ""


def test_bump_requirements(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "# deps\nfastapi==0.100\npydantic[email]==1.10.13 ; python_version >= '3.8'  # pinned\n"
        "Pydantic_Settings==1.0\npydantic-core==1\n")
    assert deps.bump(tmp_path, "pydantic", "2.9.2") == ["requirements.txt"]
    assert (tmp_path / "requirements.txt").read_text() == (
        "# deps\nfastapi==0.100\npydantic[email]==2.9.2 ; python_version >= '3.8'  # pinned\n"
        "Pydantic_Settings==1.0\npydantic-core==1\n")


def test_bump_to_latest_unpins(tmp_path):
    (tmp_path / "requirements.txt").write_text("pydantic>=1.8,<2\n")
    deps.bump(tmp_path, "pydantic", "latest")
    assert (tmp_path / "requirements.txt").read_text() == "pydantic\n"


def test_bump_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "pydantic-app"\nrequires-python = ">=3.9"\n'
        'dependencies = [\n  "pydantic>=1.10,<2",\n  "httpx",\n]\n')
    assert deps.bump(tmp_path, "pydantic", "2.9.2") == ["pyproject.toml"]
    text = (tmp_path / "pyproject.toml").read_text()
    assert '"pydantic==2.9.2"' in text and 'name = "pydantic-app"' in text and '"httpx"' in text


def test_bump_nothing_to_do(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests==2.0\n")
    assert deps.bump(tmp_path, "pydantic", "2.9.2") == []
