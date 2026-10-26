import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.service import from_record, load_user, load_user_json, rename, to_dict, to_json

ADA = {"id": 1, "name": "  Ada  ", "email": "ADA@Example.com", "address": {"street": "1 Main St", "city": "London", "postcode": "12345"}}


def test_load_cleans_fields():
    user = load_user(ADA)
    assert user.name == "Ada"
    assert user.email == "ada@example.com"
    assert user.address.city == "London"


def test_bad_email_rejected():
    with pytest.raises(ValidationError):
        load_user({**ADA, "email": "not-an-email"})


def test_negative_age_rejected():
    with pytest.raises(ValidationError):
        load_user({**ADA, "age": -1})


def test_bad_postcode_rejected():
    with pytest.raises(ValidationError):
        load_user({**ADA, "address": {**ADA["address"], "postcode": "ABC"}})


def test_dump_skips_none():
    data = to_dict(load_user(ADA))
    assert "age" not in data
    assert data["tags"] == []
    assert json.loads(to_json(load_user(ADA)))["email"] == "ada@example.com"


def test_load_from_json():
    assert load_user_json(json.dumps(ADA)).id == 1


def test_from_record():
    record = SimpleNamespace(id=2, name="Linus", email="l@example.com", age=30, tags=["k"], address=None)
    assert from_record(record).age == 30


def test_rename_returns_copy():
    user = load_user(ADA)
    renamed = rename(user, "Grace")
    assert renamed.name == "Grace" and user.name == "Ada"
