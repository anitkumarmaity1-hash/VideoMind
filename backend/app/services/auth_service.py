"""
Password hashing (bcrypt) and JWT issuance/verification for user auth.

Uses `bcrypt` directly rather than passlib's bcrypt wrapper — passlib's
bcrypt backend has had version-detection issues against bcrypt>=4.1
(raises on `__about__.__version__` that newer bcrypt releases removed),
which makes it a fragile dependency to pin. Calling bcrypt directly
sidesteps that.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt

from app.config import settings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # Malformed hash (shouldn't happen for hashes we generated ourselves).
        return False


def generate_user_id() -> str:
    return f"user_{uuid.uuid4().hex[:12]}"


def create_access_token(user_id: str, username: str) -> str:
    expire = datetime.now(timezone.utc) + \
        timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": user_id, "username": username, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


class TokenError(Exception):
    pass


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        raise TokenError("Token has expired")
    except jwt.InvalidTokenError:
        raise TokenError("Invalid token")
