import time
import pytest

from app.services.auth_service import (
    hash_password, verify_password, create_access_token, decode_access_token,
    generate_user_id, TokenError,
)


def test_hash_password_is_not_plaintext():
    hashed = hash_password("supersecret123")
    assert hashed != "supersecret123"
    assert hashed.startswith("$2b$")


def test_verify_password_roundtrip():
    hashed = hash_password("supersecret123")
    assert verify_password("supersecret123", hashed) is True
    assert verify_password("wrong-password", hashed) is False


def test_generate_user_id_is_prefixed_and_unique():
    a, b = generate_user_id(), generate_user_id()
    assert a.startswith("user_")
    assert a != b


def test_access_token_roundtrip():
    user_id = generate_user_id()
    token = create_access_token(user_id, "alice")
    payload = decode_access_token(token)
    assert payload["sub"] == user_id
    assert payload["username"] == "alice"


def test_decode_rejects_garbage_token():
    with pytest.raises(TokenError):
        decode_access_token("not-a-real-token")


def test_decode_rejects_token_signed_with_different_secret():
    import jwt as pyjwt
    bad_token = pyjwt.encode(
        {"sub": "user_x", "username": "eve"}, "some-other-secret", algorithm="HS256")
    with pytest.raises(TokenError):
        decode_access_token(bad_token)
