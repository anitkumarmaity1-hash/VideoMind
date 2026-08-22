"""
Integration tests for /api/auth/* and the per-user video history endpoint.
External dependencies (Mongo) are mocked, same pattern as
test_upload_endpoint.py.
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_register_then_login_roundtrip():
    store = {}

    async def fake_find_one(query):
        return store.get(query.get("username")) or store.get(query.get("user_id"))

    async def fake_insert_one(doc):
        store[doc["username"]] = doc
        store[doc["user_id"]] = doc

    mock_users = MagicMock()
    mock_users.find_one = AsyncMock(side_effect=fake_find_one)
    mock_users.insert_one = AsyncMock(side_effect=fake_insert_one)

    with patch("app.database.mongo.ensure_indexes", new=AsyncMock()), \
         patch("app.api.auth.users_collection", return_value=mock_users):
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            reg_resp = await client.post(
                "/api/auth/register",
                json={"username": "alice_test", "password": "supersecret123"},
            )
            assert reg_resp.status_code == 200
            token = reg_resp.json()["access_token"]
            assert reg_resp.json()["user"]["username"] == "alice_test"

            login_resp = await client.post(
                "/api/auth/login",
                json={"username": "alice_test", "password": "supersecret123"},
            )
            assert login_resp.status_code == 200
            assert login_resp.json()["user"]["username"] == "alice_test"

            bad_login = await client.post(
                "/api/auth/login",
                json={"username": "alice_test", "password": "wrong-password"},
            )
            assert bad_login.status_code == 401


@pytest.mark.asyncio
async def test_register_rejects_duplicate_username():
    mock_users = MagicMock()
    mock_users.find_one = AsyncMock(return_value={"username": "taken"})

    with patch("app.database.mongo.ensure_indexes", new=AsyncMock()), \
         patch("app.api.auth.users_collection", return_value=mock_users):
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/auth/register",
                json={"username": "taken", "password": "supersecret123"},
            )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_history_endpoint_requires_auth():
    with patch("app.database.mongo.ensure_indexes", new=AsyncMock()):
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/videos/vid_abc123/history")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_history_endpoint_returns_only_current_user_entries():
    from app.services.auth_service import create_access_token, generate_user_id

    user_id = generate_user_id()
    token = create_access_token(user_id, "bob_test")

    mock_users = MagicMock()
    mock_users.find_one = AsyncMock(
        return_value={"user_id": user_id, "username": "bob_test"})

    mock_questions = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor
    mock_cursor.to_list = AsyncMock(return_value=[
        {"question_id": "q_1", "question": "What is discussed?",
         "timestamp": datetime.utcnow(), "user_id": user_id},
    ])
    mock_questions.find.return_value = mock_cursor

    mock_answers = MagicMock()
    mock_answers.find_one = AsyncMock(
        return_value={"question_id": "q_1", "answer": "It's about testing."})

    with patch("app.database.mongo.ensure_indexes", new=AsyncMock()), \
         patch("app.api.auth.users_collection", return_value=mock_users), \
         patch("app.api.questions.questions_collection", return_value=mock_questions), \
         patch("app.api.questions.answers_collection", return_value=mock_answers):
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/videos/vid_abc123/history",
                headers={"Authorization": f"Bearer {token}"},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["video_id"] == "vid_abc123"
    assert len(body["entries"]) == 1
    assert body["entries"][0]["answer"] == "It's about testing."
