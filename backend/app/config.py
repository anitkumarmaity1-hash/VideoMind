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
    # Groq's on-demand free tier caps `openai/gpt-oss-20b` at a strict
    # 8000 tokens-per-minute *account-wide* limit. Kept comfortably below
    # that (not at 8000) so estimation error and other concurrent traffic
    # against the same key don't tip it over. Raise this if you're on a
    # paid tier with a higher TPM cap.
    groq_tpm_limit: int = 6500
    # How many Groq calls the summarizer fires concurrently. This bounds
    # *concurrency*, not token rate — see llm_service._TokenRateLimiter
    # for the part that actually keeps aggregate usage under the TPM cap.
    # Kept modest so a burst of concurrent requests doesn't itself blow
    # past the per-minute budget before the limiter can react.
    groq_max_concurrent_calls: int = 3

    # ASR
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # Beam search of 5 (faster-whisper's typical default) is noticeably
    # slower than greedy decoding on CPU for marginal accuracy gain on
    # clear speech. Drop to 1 by default for local/CPU dev; bump it back
    # up in .env if running on the GPU worker, where the cost is cheap.
    whisper_beam_size: int = 1
    # 0 = auto (use all logical cores). A single WhisperModel.transcribe()
    # call is otherwise CPU-bound on ctranslate2's internal thread pool,
    # which by default does NOT use every core on the machine — this is
    # the single biggest reason a 20-minute clip could take ~2 hours on
    # CPU. See transcription.py for how this is combined with
    # whisper_parallel_chunks.
    whisper_cpu_threads: int = 0
    # On CPU, split the audio into this many roughly-equal, slightly
    # overlapping segments and transcribe them concurrently (one
    # faster-whisper worker per segment) instead of one long sequential
    # pass. ctranslate2 releases the GIL during inference, so this gives
    # close to linear speedup with core count on a multi-core machine.
    # Set to 1 to disable and fall back to a single sequential pass
    # (used automatically on GPU, where a single pass is already fast and
    # splitting only adds overhead).
    whisper_parallel_chunks: int = 4
    # Below this duration, splitting isn't worth the fixed overhead
    # (model warm-up per thread, ffmpeg segment extraction) — short clips
    # just run as a single pass regardless of whisper_parallel_chunks.
    whisper_parallel_min_duration_seconds: float = 90.0
    # Overlap between adjacent parallel segments, in seconds. Each part is
    # decoded with this many extra seconds of audio *context* on either
    # side so words right at a cut point aren't misheard, but only the
    # segments that start inside the part's own (non-overlapping) time
    # range are kept — so overlap improves boundary accuracy without
    # producing duplicate transcript text.
    whisper_parallel_overlap_seconds: float = 3.0

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

    # Auth
    # MUST be overridden via .env in any real deployment — this default
    # only exists so the app doesn't crash on a fresh clone. Tokens signed
    # with a leaked/default secret can be forged, so treat this the same
    # as a database password.
    jwt_secret_key: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 7 days

    # App
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [e.strip().lower() for e in self.allowed_video_extensions.split(",")]


settings = Settings()
