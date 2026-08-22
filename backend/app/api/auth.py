from typing import Optional
from fastapi import APIRouter, HTTPException, Header
from pydantic import ValidationError

from app.database.mongo import users_collection
from app.models.user import (
    UserRegisterRequest, UserLoginRequest, TokenResponse, UserPublic,
)
from app.services.auth_service import (
    hash_password, verify_password, generate_user_id,
    create_access_token, decode_access_token, TokenError,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(request: UserRegisterRequest):
    existing = await users_collection().find_one({"username": request.username})
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")

    user_id = generate_user_id()
    await users_collection().insert_one({
        "user_id": user_id,
        "username": request.username,
        "password_hash": hash_password(request.password),
    })

    token = create_access_token(user_id, request.username)
    return TokenResponse(access_token=token, user=UserPublic(user_id=user_id, username=request.username))


@router.post("/login", response_model=TokenResponse)
async def login(request: UserLoginRequest):
    user = await users_collection().find_one({"username": request.username})
    if not user or not verify_password(request.password, user["password_hash"]):
        # Same message for "no such user" and "wrong password" — don't
        # leak which one it was.
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(user["user_id"], user["username"])
    return TokenResponse(access_token=token, user=UserPublic(user_id=user["user_id"], username=user["username"]))


async def get_current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """Required-auth dependency: raises 401 if there's no valid token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header")
    token = authorization[len("Bearer "):]
    try:
        payload = decode_access_token(token)
    except TokenError as e:
        raise HTTPException(status_code=401, detail=str(e))

    user = await users_collection().find_one({"user_id": payload.get("sub")})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def get_optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    """Best-effort auth dependency for endpoints that should keep working
    for unauthenticated/API callers (e.g. existing integration tests, curl
    usage) but attribute the activity to a user when a valid token is
    present. Never raises."""
    if not authorization:
        return None
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None
