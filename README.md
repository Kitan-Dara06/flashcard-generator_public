# 🎯 AI Flashcard Generator

Upload a PDF, TXT, DOCX, or PPTX and get an AI-generated deck of flashcards you can flip through and download as JSON.

## Architecture

- **Frontend** — static HTML/CSS/JS in `public/`, served at `/` by Vercel.
- **Vercel functions** (`api/`) — auth, text extraction, job dispatch, status polling, cancellation, and GPU pre-warming. Job state lives in Redis.
- **Modal** (`modal_app.py`) — Qwen 2.5 7B-Instruct (AWQ, served by vLLM) for generation plus BAAI/bge-small-en-v1.5 for semantic chunking and deduplication. No request timeout, so it can process long documents.
- **Upstash Redis** — job text, status, progress, and results. All keys expire after 24 h.

### Job flow

1. The browser uploads a file to `POST /api/generate_flashcards` (requires the bearer token).
2. Vercel extracts text, stores it in Redis under a fresh `job_id`, spawns the Modal worker, and returns `{job_id}` in ~2 s.
3. The Modal worker semantic-chunks the text (~24k chars/chunk), generates flashcards chunk by chunk (guided JSON decoding), dedups near-identical cards (question cosine > 0.92), and writes progress to Redis as it goes.
4. The browser polls `GET /api/status?job_id=…` every 2 s and shows `chunk X of Y` progress.
5. On completion the status endpoint returns the cards; the user can cancel at any time (the worker checks a Redis cancel flag between chunks).

## Cold-start strategy (why latency stays low without 24/7 GPU)

| Layer | What it does |
| --- | --- |
| Qwen 2.5 **7B** AWQ | ~10 GB VRAM; restores on A10G in seconds, unlike a 14B model |
| Modal volume + snapshot | Weights cached on Modal storage; `enable_memory_snapshot=True` + `@modal.enter(snap=True)` restore the loaded engine in ~1–3 s |
| `scaledown_window=1200` | Stays warm 20 min after the last request (20 > 15 min cron gap on purpose, so pings always hit a warm container) |
| Study-hours cron | `keep_warm` pings the container every 15 min, Mon–Fri 08:00–22:00 (`STUDY_TIMEZONE`), then lets it scale to zero overnight (~$0.10–0.20/day) |
| Frontend pre-warm | `/api/warmup` fires on page load, tab focus, and file selection (60 s client cooldown), so the boot happens while the user is still choosing a file |

## Setup

> Python 3.12 is pinned in `.python-version` — Vercel's Python runtime requires >= 3.12, and without a pin the CLI falls back to a default that may not match.

1. **Upstash**: create a Redis database (free tier is plenty) and note the REST URL + token.
2. **Modal** (`pip install -U modal`, `modal token new`):
   ```bash
   # Both keys are required - the app reads the URL *and* the token:
   modal secret create --force upstash \
     UPSTASH_REDIS_REST_URL=https://... \
     UPSTASH_REDIS_REST_TOKEN=...
   modal deploy modal_app.py
   ```
   `STUDY_TIMEZONE` in `modal_app.py` is set to `Africa/Lagos` (WAT, UTC+1) - change it if your audience shifts.

   Optional, so even the first real boot skips the HuggingFace download (1-3 min):
   ```bash
   modal run modal_app.py::download_models   # pre-pulls weights into the volume
   ```
3. **Vercel** env vars (Project → Settings → Environment Variables):

   | Variable | Required | Purpose |
   | --- | --- | --- |
   | `APP_PASSWORD` | yes | Login gate password |
   | `AUTH_SECRET` | recommended | Token signing secret (falls back to `APP_PASSWORD`) |
   | `UPSTASH_REDIS_REST_URL` | yes | Redis REST endpoint |
   | `UPSTASH_REDIS_REST_TOKEN` | yes | Redis REST token |
   | `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | yes | Allow Vercel to spawn Modal jobs |
   | `ALLOWED_ORIGINS` | no | Extra CORS origins, e.g. `http://localhost:3000` |

4. Deploy: push to `main`, or run `vercel --prod`.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/api/verify_password` | Exchange the password for a 24 h bearer token |
| POST | `/api/generate_flashcards` | Submit a document; returns a `job_id` |
| GET | `/api/status?job_id=…` | Poll status / progress / results |
| POST | `/api/cancel` | Cancel a running job (body: `{job_id}`) |
| POST | `/api/tts` | Synthesize one sentence to WAV audio (body: `{text}`) |
| POST | `/api/warmup` | Pre-warm the Modal GPU container |

All endpoints except `verify_password` require `Authorization: Bearer <token>`.

## Text to speech

Each flashcard has a 🔊 button on both faces (plus an **Auto-read** toggle that reads the side you flip to).

- **Engine**: Piper (`piper-tts`), an English neural voice (`en_US-lessac-medium`) running on **CPU inside the same Modal container** as the LLM — no second cold start, and the GPU stays free for vLLM. Only ~500 MB on top of the image.
- **Streaming**: browsers can't reliably play a half-downloaded WAV, so instead the frontend splits text into sentences and **pipelines** the requests — sentence N plays while sentence N+1 is already downloading. Playback typically starts in ~300-500 ms.
- **Caching**: `/api/tts` hashes `voice + text` and caches the WAV in Redis for 7 days, so re-studying a deck is free. The browser also keeps an in-memory cache for the session.
- **Interruptions**: starting a new clip aborts in-flight requests and stops the current audio, so rapid clicks never queue up. Speech also stops when the tab is hidden or a new generation starts.
- **Voice**: change `TTS_VOICE` in `modal_app.py` (and `api/tts.py`, which must match) to any voice from the `rhasspy/piper-voices` repo, then `modal deploy`. The new voice downloads into the volume on first use.

## Full pipeline test

`test_pipeline.py` drives the whole stack over HTTP exactly like the browser does — token → submit → poll → cards → TTS — using only the Python standard library:

```bash
modal deploy modal_app.py     # worker up to date first
vercel dev                    # or point the script elsewhere with --base-url

python3 test_pipeline.py "/path/to/report.docx"
python3 test_pipeline.py report.docx --cancel-after 45   # also exercises the cancel path
python3 test_pipeline.py report.docx --tts-sample 2      # synthesise audio for 2 cards
python3 test_pipeline.py report.docx --base-url https://flashcard-generator-public.vercel.app
```

What it checks and reports:

- **Timings**: cold/warm start (submit → worker started), per-chunk progress, generation time, total.
- **Cards**: count, empty question/answer fields, exact-duplicate questions that survived dedup, answer length min/avg/max, plus a few samples.
- **TTS**: validates the WAV header (rate/channels/duration) and confirms the Redis cache answers a repeat request faster.
- **Cancel** (with `--cancel-after`): asserts the job reports `cancelled`.
- **Failure modes**: a stalled worker, a 401, a 404, or a job `error` all exit non-zero with the server's message.

Exit code is `0` only if the whole pipeline passed.

## Limits & tuning

- Uploads are capped at 3 MB client-side (Vercel request bodies max out at ~4.5 MB).
- `api/generate_flashcards.py`: `MAX_PAGES = 100`, `MAX_TEXT_CHARS = 500_000`.
- `modal_app.py`: `MAX_CHUNK_CHARS = 24_000`, `SIM_BOUNDARY_THRESHOLD`, `DEDUP_THRESHOLD`, `TTS_VOICE`, cron schedule, `STUDY_TIMEZONE`.
- `api/tts.py`: `TTS_MAX_CHARS = 600` per request, `TTS_CACHE_TTL = 7 days`, `TTS_CACHE_MAX_BYTES = 400 KB` (clips larger than this are streamed but not cached, to respect Redis value limits).

## Local development

```bash
pip install -r requirements.txt
npm install -g vercel
vercel link
vercel env pull   # writes .env.local with the project's env vars
vercel dev        # serves public/ and api/ at http://localhost:3000
```

The API calls the *deployed* Modal app, so `modal deploy modal_app.py` must have run at least once.
