from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class UserRegisterRequest(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def username_ok(cls, v: str) -> str:
        v = v.strip()
        if not (3 <= len(v) <= 32):
            raise ValueError("Username must be 3-32 characters")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Username may only contain letters, numbers, - and _")
        return v

    @field_validator("password")
    @classmethod
    def password_ok(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class UserLoginRequest(BaseModel):
    username: str
    password: str


class UserInDB(BaseModel):
    user_id: str
    username: str
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserPublic(BaseModel):
    user_id: str
    username: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
