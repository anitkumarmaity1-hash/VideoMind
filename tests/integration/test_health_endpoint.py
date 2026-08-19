"""
Integration test for GET /health.
Mongo connectivity is mocked so this test doesn't require a live database.
"""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_health_endpoint_reports_status():
    with patch("app.database.mongo.ensure_indexes", new=AsyncMock()), \
         patch("app.api.health.ping", new=AsyncMock(return_value=True)):
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mongo_connected"] is True
