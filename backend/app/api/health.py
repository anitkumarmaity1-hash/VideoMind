from fastapi import APIRouter
from app.database.mongo import ping

router = APIRouter()


@router.get("/health")
async def health_check():
    mongo_ok = await ping()
    return {
        "status": "ok" if mongo_ok else "degraded",
        "mongo_connected": mongo_ok,
    }
