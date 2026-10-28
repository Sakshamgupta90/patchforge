from app.models import User


def load_user(payload: dict) -> User:
    return User.parse_obj(payload)


def load_user_json(raw: str) -> User:
    return User.parse_raw(raw)


def from_record(record) -> User:
    return User.from_orm(record)


def to_dict(user: User) -> dict:
    return user.dict(exclude_none=True)


def to_json(user: User) -> str:
    return user.json(exclude_none=True)


def rename(user: User, name: str) -> User:
    return user.copy(update={"name": name})
