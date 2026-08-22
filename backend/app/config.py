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
    # Beam search of 5 (faster-whisper's typical default) is noticeably
    # slower than greedy decoding on CPU for marginal accuracy gain on
    # clear speech. Drop to 1 by default for local/CPU dev; bump it back
    # up in .env if running on the GPU worker, where the cost is cheap.
    whisper_beam_size: int = 1

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
    # "List all N things"-style questions need evidence gathered from
    # across the whole video, not just the 5 chunks closest to the
    # question's own wording — otherwise a question like "what are the
    # five ideas discussed" only surfaces whichever 1-2 points happen to
    # be the closest semantic match, and the rest get silently dropped.
    broad_question_top_k: int = 25

    # YouTube / yt-dlp
    # YouTube requires a PO Token (proof-of-origin) for GVS/Player/Subs
    # requests on most clients now — yt-dlp cannot generate one itself.
    # Per yt-dlp's own compatibility table (github.com/yt-dlp/yt-dlp/wiki/
    # PO-Token-Guide), web_embedded and tv currently do NOT require a
    # token with no caveats; android_vr is also listed as exempt but in
    # practice its *https* (progressive) formats have been observed
    # demanding a GVS PO Token anyway ("android_vr client https formats
    # require a GVS PO Token") — kept as a third fallback since its HLS
    # formats can still work, just not tried first.
    # This is deliberately a setting, not a hardcoded constant: YouTube
    # changes which clients are exempt every few months. When downloads
    # start failing again, check that wiki page and update this env var
    # rather than editing code.
    youtube_player_clients: str = "web_embedded,tv,android_vr"

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
