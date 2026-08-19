# Deployment Guide (AWS)

This describes the recommended production topology from the spec. It is a
guide, not a one-click script — fill in your own account details.

## Components

| Component  | Recommended service                       |
|------------|--------------------------------------------|
| Frontend   | Streamlit Community Cloud, or an EC2 instance running the Docker image |
| Backend    | Docker container on EC2 / ECS Fargate      |
| Storage    | S3 bucket (set `STORAGE_BACKEND=s3`)       |
| Database   | MongoDB Atlas (managed)                    |
| Vector DB  | Pinecone (managed, already cloud-hosted)   |
| LLM        | Groq API (already cloud-hosted)            |

## Steps

1. **MongoDB Atlas**
   - Create a free/shared cluster.
   - Create a database user, allow-list your backend's outbound IP (or 0.0.0.0/0 for early testing only).
   - Copy the connection string into `MONGO_URI`.

2. **Pinecone**
   - Create an account and API key.
   - You do NOT need to pre-create indexes — `app/services/vector_store.py` creates them on first use via `_ensure_index()`, using `TEXT_EMBEDDING_DIM` / `VISUAL_EMBEDDING_DIM` from your `.env`.

3. **Groq**
   - Create an API key at console.groq.com.
   - Set `GROQ_API_KEY` and `GROQ_MODEL`.

4. **S3**
   - Create a bucket for video/audio/frame storage.
   - Set `STORAGE_BACKEND=s3`, `AWS_S3_BUCKET`, `AWS_REGION`.
   - Give your backend's IAM role `s3:PutObject`, `s3:GetObject`, `s3:DeleteObject`, `s3:HeadObject` on that bucket.
   - `boto3` is required for S3 mode: `pip install boto3` (not in the default requirements.txt since local dev doesn't need it).

5. **Backend container**
   - Build: `docker build -t videomind-backend ./backend`
   - Push to ECR, run on ECS Fargate or a plain EC2 instance with the container runtime.
   - Ensure the container's environment has all variables from `.env.example` populated (never bake secrets into the image — pass them as environment variables / ECS task secrets).

6. **Frontend**
   - Simplest: Streamlit Community Cloud pointed at `frontend/streamlit_app.py`, with `VIDEOMIND_API_URL` set to your backend's public URL.
   - Alternative: same Docker approach as the backend, on a small EC2 instance.

7. **GPU-heavy processing**
   - The pipeline (Whisper + CLIP) is CPU-compatible but slow. For production-grade latency, run the backend on a GPU-enabled instance (e.g. `g4dn.xlarge`) or offload processing to a separate GPU worker service that calls the same `run_pipeline()` function against the same MongoDB/Pinecone backends — this mirrors the local VS Code + Colab GPU pattern described in `docs/ARCHITECTURE.md`.

## Security checklist before going live

- [ ] `.env` is not committed (confirm `.gitignore` includes it)
- [ ] MongoDB Atlas IP allow-list is scoped, not `0.0.0.0/0`
- [ ] S3 bucket is private; access only via the app's IAM role or presigned URLs
- [ ] CORS `allow_origins` in `app/main.py` is scoped to your actual frontend domain, not `*`
- [ ] Upload size/type validation (`app/utils/validation.py`) is active
