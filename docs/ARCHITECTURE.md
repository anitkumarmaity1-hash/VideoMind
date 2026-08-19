# VideoMind v2 — Architecture

## High-level flow

```
User
 |
 v
Streamlit Frontend  <---->  FastAPI Backend  <---->  MongoDB (metadata)
                                   |
                                   v
                          Video Processing Pipeline
                          (ffmpeg, faster-whisper,
                           OpenCV, OpenCLIP,
                           Sentence-Transformers)
                                   |
                                   v
                              Pinecone (vectors)
                                   |
                                   v
                              Groq LLM (answers)
```

## Why this shape

- **FastAPI backend** owns all business logic (ingestion, pipeline orchestration, retrieval, answer generation). Streamlit is a thin client that only talks to the backend over HTTP — this keeps the heavy ML code testable and reusable outside the UI (e.g. from the Colab notebook or eval_runner.py).
- **MongoDB** is the source of truth for metadata: video records, processing job status, transcript segments, questions, and answers. It is NOT used for vector similarity search.
- **Pinecone** stores only embeddings + light metadata (video_id, chunk_id, start/end time, modality). Two separate indexes are used for text and visual embeddings because they have different dimensionality (384 for BGE-small text vs 512 for CLIP ViT-B/32).
- **Groq** is the active LLM provider, but is accessed through an `LLMProvider` abstract base class (`app/services/llm_service.py`). Adding OpenAI/Anthropic/Gemini/Ollama later means writing one new subclass and registering it in `get_llm_provider()` — nothing else in the app changes.
- **StorageBackend** abstraction (`app/services/storage.py`) supports local filesystem in development and S3 in production behind the same `save/get_url/delete/exists` interface.

## Processing pipeline stages

1. **Ingestion** — validate file extension/size, generate `video_id`, save via `StorageBackend`, insert a `videos` document with status `uploaded`.
2. **Audio extraction** — `ffmpeg` pulls mono 16kHz WAV audio from the video (`app/pipeline/audio.py`).
3. **Transcription** — `faster-whisper` produces timestamped segments (`app/pipeline/transcription.py`).
4. **Temporal chunking** — transcript segments are grouped into fixed-size overlapping windows (default 10s / 2s overlap) (`app/pipeline/chunking.py`).
5. **Frame extraction** — OpenCV samples frames at a configurable FPS (default 2 FPS), streaming frame-by-frame rather than loading the whole video into memory (`app/pipeline/frames.py`).
6. **Embedding generation** — each chunk's transcript is embedded with Sentence-Transformers (BGE); each chunk's representative frame is embedded with OpenCLIP.
7. **Indexing** — both embedding sets are upserted into their respective Pinecone indexes, along with MongoDB `video_segments` documents.
8. **Ready** — video status flips to `ready`; the frontend can now ask questions and request summaries.

## Retrieval flow (per question)

1. **Question routing** — a lightweight rule-based classifier (`app/pipeline/question_router.py`) labels the question as summary / text / visual / temporal / general. This label is currently informational (returned in the API response); the same hybrid retrieval runs regardless, since transcript-only routing would violate the multimodal requirement.
2. **Text retrieval** — question embedded with BGE, queried against the text Pinecone index, filtered to the target `video_id`.
3. **Visual retrieval** — question embedded with CLIP's text tower (same embedding space as the frame embeddings), queried against the visual Pinecone index.
4. **Score fusion** — `final_score = 0.6 * text_score + 0.4 * visual_score` (configurable via `TEXT_SCORE_WEIGHT` / `VISUAL_SCORE_WEIGHT`).
5. **Temporal reranking** — top fused chunks are expanded with their temporal neighbors (±`TEMPORAL_NEIGHBOR_WINDOW` chunks) and merged into contiguous evidence blocks, removing duplicate/overlapping ranges.
6. **LLM answer generation** — evidence blocks (text + visual, kept separate) are passed to Groq with a strict "use only supplied evidence, cite timestamps, say when evidence is insufficient" system prompt.

## Why VS Code + Google Colab

Heavy model inference (Whisper, OpenCLIP, embedding generation) is far faster on a GPU. The backend code is written to be environment-agnostic: `WHISPER_DEVICE` and CLIP's own CUDA auto-detection mean the exact same `app/pipeline/*` code runs unmodified whether invoked from:
- a local FastAPI process (CPU, for API development in VS Code), or
- the Colab GPU notebook (`notebooks/VideoMind_Colab_Worker.ipynb`), which mounts the repo, installs dependencies, and calls `run_pipeline(video_id)` directly against the same MongoDB Atlas + Pinecone backing services.

This means you can develop/debug API routes and UI locally in VS Code, but hand off actual video processing to Colab's free GPU when you need speed — both write to the same shared database and vector store, so results show up in the same app either way.
