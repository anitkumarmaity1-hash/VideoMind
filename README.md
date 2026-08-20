# VideoMind v2 — Multimodal Video RAG & Temporal Q&A

Upload a video, ask questions about it, get grounded answers with clickable timestamps —
backed by real multimodal retrieval (transcript **and** visual evidence, not transcript-only RAG).

**Stack:** Streamlit · FastAPI · MongoDB · Pinecone · Groq · faster-whisper · OpenCLIP · Sentence-Transformers

---

## Recommended workflow: VS Code (app code) + Google Colab (GPU processing)

Video processing (Whisper transcription + CLIP embeddings) is CPU-slow but GPU-fast.
This project is built so you can:

- **Write and run the FastAPI + Streamlit app locally in VS Code** — fast iteration, full debugging.
- **Run actual video processing on Colab's free GPU** — same code, same MongoDB/Pinecone, just faster.

Both environments point at the **same MongoDB Atlas cluster and same Pinecone project**, so a video
processed in Colab shows up as `ready` in your local Streamlit app immediately.

```
 VS Code (local)                       Google Colab (GPU)
 ┌─────────────────────┐               ┌──────────────────────┐
 │ FastAPI backend      │               │ notebooks/            │
 │ Streamlit frontend   │               │ VideoMind_Colab_      │
 │ (dev/debug/API work) │               │ Worker.ipynb           │
 └──────────┬───────────┘               │ (heavy processing)     │
            │                            └───────────┬───────────┘
            │                                        │
            └───────────────┬────────────────────────┘
                             ▼
              MongoDB Atlas  +  Pinecone  +  Groq
              (shared, cloud-hosted)
```

---

## Prerequisites

- Python 3.11+
- `ffmpeg` installed locally (`sudo apt install ffmpeg` / `brew install ffmpeg`) — only needed if you'll process videos locally instead of via Colab
- A free [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) cluster
- A free [Pinecone](https://www.pinecone.io/) account + API key
- A free [Groq](https://console.groq.com/) API key
- Docker + Docker Compose (only needed for the containerized path in Stage 6 below)
- A Google account (for Colab)

---

## Stage-by-stage setup

### Stage 1 — Clone and configure

```bash
git clone <your-repo-url> VideoMind
cd VideoMind
cp backend/.env.example backend/.env
```

Open `backend/.env` in VS Code and fill in:
- `MONGO_URI` — your Atlas connection string
- `PINECONE_API_KEY`, `PINECONE_ENV`
- `GROQ_API_KEY`

Leave everything else at its default for now.

### Stage 2 — Backend setup (VS Code, local terminal)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the health check:

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` — you should see the Swagger UI with all endpoints, and
`http://localhost:8000/health` should return `{"status": "ok", "mongo_connected": true}`.

If `mongo_connected` is `false`, double check `MONGO_URI` in `.env` and your Atlas IP allow-list.

### Stage 3 — Frontend setup (VS Code, second terminal)

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. You should see the VideoMind dashboard with an upload sidebar.

At this point you have the **full app running locally**. Uploading and processing a video here
works, but will be slow on CPU (Whisper + CLIP). For speed, use the Colab worker instead (Stage 4).

### Stage 4 — GPU processing via Google Colab

1. Push this repo to your own GitHub (or open the notebook standalone and upload the `backend/` folder).
2. Open `notebooks/VideoMind_Colab_Worker.ipynb` in Google Colab.
3. `Runtime → Change runtime type → T4 GPU` (or any available GPU).
4. Run the cells top to bottom:
   - Cell 2: edit `REPO_URL` to your GitHub URL, clone the repo.
   - Cell 3: installs `ffmpeg` + `requirements.txt`.
   - Cell 4: fill in the **same** `MONGO_URI` / `PINECONE_API_KEY` / `GROQ_API_KEY` values as your local `.env`
     (use Colab's Secrets manager — the key icon in the left sidebar — rather than pasting real keys into
     a notebook you might commit).
   - Cell 5: upload a video file via Colab's file picker.
   - Cell 6: registers the video in MongoDB and runs `run_pipeline()` — the **exact same function** the
     FastAPI backend uses — on the GPU. This is the slow part (transcription + embeddings) and is now fast.
   - Cell 7 (optional): sanity-check retrieval directly in the notebook.
5. Copy the printed `video_id`.

### Stage 5 — See it in the local UI

Back in your local Streamlit app (`http://localhost:8501`), paste the `video_id` from Colab into
**"Or load existing video_id"** in the sidebar. The video is already `ready` — ask it a question.

### Stage 6 — (Optional) Run everything in Docker instead

If you'd rather not manage two Python venvs, the whole stack (Mongo + backend + frontend) can run
in containers:

```bash
docker compose up --build
```

- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000/docs`
- Local Mongo runs in a container too (edit `backend/.env`'s `MONGO_URI` isn't needed here —
  `docker-compose.yml` overrides it to point at the `mongo` service). You'd still want Pinecone/Groq
  keys in `backend/.env` since those are cloud services.

This path processes video on whatever CPU Docker has access to — use the Colab notebook for GPU speed
regardless of which path you use for the app itself.

---

## Running tests

```bash
pip install -r backend/requirements.txt pytest pytest-asyncio httpx
pytest tests/unit tests/integration -v
```

All unit tests (chunking, timestamp conversion, score fusion, question routing) and integration
tests (health, upload validation, 404 handling) run with **no external services required** —
Mongo, Pinecone, and Groq calls are mocked. 27 tests, all passing as of this build.

## Running evaluation

```bash
# 1. Process a video (locally or via Colab) and note its video_id
# 2. Copy evaluation/eval_dataset_template.json, fill in real questions/timestamps for that video
# 3. Run:
python evaluation/eval_runner.py evaluation/my_dataset.json
```

This reports Recall@1/5/10, MRR, and Temporal IoU against your own annotated data — no numbers
are fabricated or pre-filled.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR: dependency install, lint (ruff), type check (mypy),
the full test suite (mocked externals, no model inference), and Docker image builds. Push to GitHub
and check the Actions tab.

## Deployment

See `docs/DEPLOYMENT.md` for the AWS deployment guide (S3 storage, MongoDB Atlas, ECS/EC2 backend,
Streamlit Cloud or EC2 frontend).

## More docs

- `docs/ARCHITECTURE.md` — system design, why each piece exists, retrieval flow in detail
- `docs/API_REFERENCE.md` — every endpoint, request/response shapes, error codes

## Troubleshooting

**`ServerSelectionTimeoutError` / `SSL: TLSV1_ALERT_INTERNAL_ERROR` when connecting to MongoDB Atlas
(most common in Google Colab):** Colab's preinstalled `pymongo`/`certifi` versions are often stale
enough to fail Atlas's TLS handshake. Fixed by:
- The notebook now force-upgrades `pymongo` and `certifi` before first use, and `app/database/mongo.py`
  explicitly passes `certifi.where()` as `tlsCAFile` to the Mongo client.
- If you still hit it (e.g. pymongo was already imported earlier in the same kernel session before the
  upgrade), do `Runtime → Restart session` in Colab and re-run all cells from the top.
- Locally, this is rare, but the same fix applies: `pip install --upgrade pymongo certifi`.

## Known limitations (honest, per project rules)

- The rule-based question router labels a question's type but doesn't currently change retrieval
  behavior — hybrid text+visual retrieval always runs, which satisfies the "not transcript-only"
  requirement but means router output is informational rather than gating.
- `get_video_chunks()` in `vector_store.py` is a best-effort approximation (Pinecone isn't built for
  full metadata scans) — MongoDB's `video_segments` collection is the real source of truth for chunk listings.
- WebSocket-based live status push is not implemented; the frontend polls every 3 seconds instead.
- No authentication/authorization layer — this is a single-user portfolio app, not multi-tenant.
- Evaluation numbers are only as good as the dataset you annotate — none are pre-populated.