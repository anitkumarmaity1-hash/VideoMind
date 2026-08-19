"""
Integration tests for the video upload and status endpoints.
External dependencies (Mongo, storage, background pipeline) are mocked so
the test verifies routing/validation logic without needing real services.
"""
import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_upload_rejects_bad_extension():
    with patch("app.database.mongo.ensure_indexes", new=AsyncMock()):
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("clip.avi", io.BytesIO(b"fake bytes"), "video/avi")}
            response = await client.post("/api/videos/upload", files=files)

    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_accepts_valid_mp4():
    mock_collection = MagicMock()
    mock_collection.insert_one = AsyncMock()

    with patch("app.database.mongo.ensure_indexes", new=AsyncMock()), \
         patch("app.api.videos.videos_collection", return_value=mock_collection), \
         patch("app.api.videos.get_storage_backend") as mock_storage_factory, \
         patch("app.api.videos.run_pipeline", new=AsyncMock()):

        mock_storage = MagicMock()
        mock_storage.save.return_value = "/fake/path/video.mp4"
        mock_storage_factory.return_value = mock_storage

        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("clip.mp4", io.BytesIO(b"fake mp4 bytes"), "video/mp4")}
            response = await client.post("/api/videos/upload", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "clip.mp4"
    assert body["processing_status"] == "uploaded"
    mock_collection.insert_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_status_endpoint_404_for_unknown_video():
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(return_value=None)

    with patch("app.database.mongo.ensure_indexes", new=AsyncMock()), \
         patch("app.api.videos.videos_collection", return_value=mock_collection):
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/videos/vid_doesnotexist/status")

    assert response.status_code == 404
