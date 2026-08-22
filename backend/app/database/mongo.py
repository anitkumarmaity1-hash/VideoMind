"""
MongoDB connection + collection accessors using Motor (async driver).
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def get_client() -> AsyncIOMotorClient:
    """
    Create the Motor client. Explicitly passes certifi's CA bundle as
    tlsCAFile — this avoids TLSV1_ALERT_INTERNAL_ERROR / SSL handshake
    failures against MongoDB Atlas that occur in some environments
    (notably Google Colab) whose system CA store or OpenSSL build is
    outdated/mismatched relative to what Atlas's TLS termination expects.
    """
    global _client
    if _client is None:
        import certifi
        _client = AsyncIOMotorClient(
            settings.mongo_uri, tlsCAFile=certifi.where())
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


def users_collection():
    return get_db()["users"]


async def ensure_indexes():
    """Create indexes required for efficient queries. Called on startup."""
    await videos_collection().create_index("video_id", unique=True)
    await processing_jobs_collection().create_index("job_id", unique=True)
    await processing_jobs_collection().create_index("video_id")
    await video_segments_collection().create_index([("video_id", 1), ("chunk_id", 1)])
    await questions_collection().create_index("video_id")
    # Powers the per-user, per-video chat history sidebar: newest-first
    # lookups scoped to one user's questions on one video.
    await questions_collection().create_index([("video_id", 1), ("user_id", 1), ("timestamp", -1)])
    await answers_collection().create_index("question_id")
    await users_collection().create_index("user_id", unique=True)
    await users_collection().create_index("username", unique=True)


async def ping() -> bool:
    try:
        await get_client().admin.command("ping")
        return True
    except Exception:
        return False
