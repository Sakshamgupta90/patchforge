from app.models import User


def load_user(payload: dict) -> User:
    return User.model_validate(payload)


def load_user_json(raw: str) -> User:
    return User.model_validate_json(raw)


def from_record(record) -> User:
    return User.model_validate(record)


def to_dict(user: User) -> dict:
    return user.model_dump(exclude_none=True)


def to_json(user: User) -> str:
    return user.model_dump_json(exclude_none=True)


def rename(user: User, name: str) -> User:
    return user.model_copy(update={"name": name})
