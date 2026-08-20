"""
Centralized application configuration.
All values are loaded from environment variables (.env), never hard-coded.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "videomind"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_env: str = "us-east-1"
    pinecone_text_index: str = "videomind-text"
    pinecone_visual_index: str = "videomind-visual"
    text_embedding_dim: int = 384
    visual_embedding_dim: int = 512

    # Groq
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"

    # ASR
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # Embeddings
    text_embedding_model: str = "BAAI/bge-small-en-v1.5"
    visual_embedding_model: str = "ViT-B-32"
    visual_embedding_pretrained: str = "laion2b_s34b_b79k"

    # Chunking / sampling
    chunk_size_seconds: int = 10
    chunk_overlap_seconds: int = 2
    frame_sample_fps: float = 2.0
    temporal_neighbor_window: int = 1

    # Retrieval
    text_score_weight: float = 0.6
    visual_score_weight: float = 0.4
    top_k: int = 5

    # Storage
    storage_backend: str = "local"
    local_data_dir: str = "./data"
    aws_s3_bucket: str = ""
    aws_region: str = ""

    # Upload
    max_upload_size_mb: int = 500
    allowed_video_extensions: str = ".mp4,.mov,.mkv"

    # App
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [e.strip().lower() for e in self.allowed_video_extensions.split(",")]


settings = Settings()
