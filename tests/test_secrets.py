"""Real credentials must never be committed (see scripts/check_secrets.py)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_secrets", Path(__file__).resolve().parent.parent / "scripts" / "check_secrets.py")
check_secrets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_secrets)


def test_no_secrets_in_the_repository():
    assert check_secrets.problems() == []


def test_patterns_catch_real_looking_keys():
    samples = {"tvly-" + "a" * 30: "Tavily", "ghs_" + "A" * 36: "GitHub token", "v1." + "x" * 80: "Nebius",
               "-----BEGIN RSA PRIVATE KEY-----": "private key"}
    for text, what in samples.items():
        assert any(p.search(text) for p, _ in check_secrets.PATTERNS), what
