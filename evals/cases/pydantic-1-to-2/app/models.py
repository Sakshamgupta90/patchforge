from typing import List, Optional

from pydantic import BaseModel, Field, root_validator, validator


class Address(BaseModel):
    street: str
    city: str
    postcode: str = Field(..., regex=r"^\d{5}$")


class User(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None
    tags: List[str] = []
    address: Optional[Address] = None

    class Config:
        orm_mode = True
        anystr_strip_whitespace = True

    @validator("email")
    def email_must_contain_at(cls, v):
        if "@" not in v:
            raise ValueError("invalid email")
        return v.lower()

    @root_validator
    def check_age(cls, values):
        age = values.get("age")
        if age is not None and age < 0:
            raise ValueError("age must be positive")
        return values
