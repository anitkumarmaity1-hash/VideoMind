"""
MongoDB connection + collection accessors using Motor (async driver).
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    global _db
    if _db is None:
        _db = get_client()[settings.mongo_db_name]
    return _db


# Collection helpers -----------------------------------------------------

def videos_collection():
    return get_db()["videos"]


def processing_jobs_collection():
    return get_db()["processing_jobs"]


def video_segments_collection():
    return get_db()["video_segments"]


def questions_collection():
    return get_db()["questions"]


def answers_collection():
    return get_db()["answers"]


async def ensure_indexes():
    """Create indexes required for efficient queries. Called on startup."""
    await videos_collection().create_index("video_id", unique=True)
    await processing_jobs_collection().create_index("job_id", unique=True)
    await processing_jobs_collection().create_index("video_id")
    await video_segments_collection().create_index([("video_id", 1), ("chunk_id", 1)])
    await questions_collection().create_index("video_id")
    await answers_collection().create_index("question_id")


async def ping() -> bool:
    try:
        await get_client().admin.command("ping")
        return True
    except Exception:
        return False
