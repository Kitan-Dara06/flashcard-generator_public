# 🎯 AI Flashcard Generator

Upload a PDF, TXT, DOCX, or PPTX and get an AI-generated deck of flashcards you can flip through and download as JSON.

## Architecture

- **Frontend** — static HTML/CSS/JS in `public/`, served at `/` by Vercel.
- **Vercel functions** (`api/`) — text extraction, job dispatch, chunk generation, status polling, cancellation, TTS, and GPU pre-warming. Job state lives in Redis. All the shared card rules, chunking, and prompt loading live in `api/generate_flashcards.py`; prompts are data in `api/prompts/*.yaml`.
- **Generation engines**, tried in this order per chunk: **OpenCode Zen** (Gemini, no daily request cap) → **Gemini API key** directly → **Modal** (Qwen 2.5 7B-Instruct AWQ on an A10G, whole document, no timeout).
- **Upstash Redis** — job text, status, progress, and results. All keys expire after 24 h.

### Job flow

1. The browser uploads a file to `POST /api/generate_flashcards`.
2. Vercel extracts text, stores it in Redis under a fresh `job_id`, and replies with a **chunk plan** (`{job_id, chunks: N}`) in ~2 s.
3. The browser then asks for one section group at a time (`{job_id, chunk: i}`). Each request is tens of seconds, inside the serverless limit, so progress is real instead of a spinner.
4. **Each group's cards are written to Redis as they are produced** (`job:{job_id}:chunk:{i}`), and the last one merges everything and dedups across the document. The write makes the call idempotent: asking for a group that is already stored returns it instead of paying for it again. So a failure at group 18 loses group 18 only, a retry costs nothing for groups 1-17, and `status` can report `partial_cards` while the job runs.
5. If a whole engine chain is out of quota, the browser hands the job to the Modal worker (`{job_id, provider: "qwen"}`). It uses the **same** section plan and skips every group already in Redis, so the fallback fills the gap rather than regenerating the document.
5. The browser polls `GET /api/status?job_id=…` every 2 s. On completion the status endpoint returns the cards; the user can cancel at any time, and cancellation is honoured by both engines.

### When an engine fails

Failing every API engine is a reason to try the other one, never a reason to fail the document:

- DNS, connection resets, TLS errors and 408/5xx are classified **transient**; 429 and quota messages as **quota**; 401/403/404 as **fatal** with a message naming the real cause ("this model is not enabled for the key", not "bad key").
- If the chain is exhausted, the kind is the most recoverable one seen - so one blip among three failures does not become a permanent error.
- The chunk endpoint then returns **503 with `fallback: true`** and leaves the job in `processing`. Finished groups stay in Redis. The client hands the job to Modal, which skips those groups.
- A job that stops writing anything for `PROCESSING_STALE_SECONDS` (default 10 min, e.g. a closed tab) is reported as `stalled` **and** `resumable`, with `done_chunks`. The client resumes from the first missing group instead of polling a dead job.

## Cold-start strategy (why latency stays low without 24/7 GPU)

| Layer | What it does |
| --- | --- |
| Qwen 2.5 **7B** AWQ | ~10 GB VRAM; restores on A10G in seconds, unlike a 14B model |
| Modal volume + snapshot | Weights cached on Modal storage; `enable_memory_snapshot=True` + `@modal.enter(snap=True)` restore the loaded engine in ~1–3 s |
| `scaledown_window=1200` | Stays warm 20 min after the last request (20 > 15 min cron gap on purpose, so pings always hit a warm container) |
| Study-hours cron (**off by default**) | `keep_warm` *can* ping every 15 min Mon-Fri 08:00-22:00 (`STUDY_TIMEZONE`), but it ships disabled: with a 20-min `scaledown_window`, a 15-min ping keeps an A10G continuously warm for ~14 h per weekday - the same order as ~$300/month - which dwarfs the ~50 s cold starts it saves on a low-traffic app. Warm on demand instead: `modal run modal_app.py::keep_warm` |
| Frontend pre-warm | `/api/warmup` fires on page load, tab focus, and file selection (60 s client cooldown), so the boot happens while the user is still choosing a file |

## Setup

> Python 3.12 is pinned in `.python-version` — Vercel's Python runtime requires >= 3.12, and without a pin the CLI falls back to a default that may not match.

1. **Upstash**: create a Redis database (free tier is plenty). The code accepts any of these, in order of precedence:
   - `REDIS_URL` — native TLS string, e.g. `rediss://default:<password>@<host>:<port>/0?ssl_cert_reqs=required`
   - `UPSTASH_HOST` + `UPSTASH_PASSWORD` (+ optional `UPSTASH_PORT`, default `6379`) — the fields the Upstash dashboard shows; the code composes the URL
   - `UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN` — the REST pair (no TCP; leanest for serverless)

   **Use the same choice on both sides** (Vercel *and* the Modal secret) — the two client libraries encode values differently, so splitting them corrupts job state.
2. **Modal** (`pip install -U modal`, `modal token new`):
   ```bash
   # one value...
   modal secret create --force upstash \
     REDIS_URL="rediss://default:<password>@<host>:<port>/0?ssl_cert_reqs=required"

   # ...or the REST pair (both keys):
   modal secret create --force upstash \
     UPSTASH_REDIS_REST_URL=https://... \
     UPSTASH_REDIS_REST_TOKEN=...

   modal deploy modal_app.py
   ```
   Verify the worker can reach Redis without burning a full job:
   ```bash
   modal run modal_app.py::diagnose   # first stage checks the secret + does a Redis round trip
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
   | `OPENCODE_API_KEY` | recommended | OpenCode Zen key — the primary engine. No daily request cap |
   | `ZEN_MODEL` | no | Zen model id (default `gemini-3.5-flash-lite`) |
   | `OPENCODE_BASE_URL` | no | Zen endpoint (default `https://opencode.ai/inference/google/v1beta`) |
   | `ZEN_USER_AGENT` | no | Cloudflare in front of Zen 403s urllib's default UA; any explicit value works |
   | `GEMINI_API_KEY` | fallback | Direct Gemini key, used when Zen fails |
   | `DEEPGRAM_API_KEY` | podcast audio | Deepgram Aura key for the two-host podcast. Requests use the `Token` scheme |
   | `GEMINI_MODEL` | no | Direct-model id (default `gemini-2.5-flash`) |
   | `GEMINI_CHUNK_CHARS` | no | Card section-group size in characters (default `6000` — the measured coverage knee; keep a group inside the 60 s limit) |
   | `GEMINI_MAX_CHUNKS` | no | Ceiling on card calls per document (default `100`). Must stay high enough that `len(text)/it` never exceeds `GEMINI_CHUNK_CHARS`, or chunks silently grow past the knee |
   | `EPISODE_CHARS` | no | Podcast episode source size (default `12000`) |
   | `EPISODE_MAX` | no | Ceiling on episodes per document (default `40`) |
   | `EPISODE_MIN_CARDS` | no | Times a term must appear in the deck to be required of an episode (default `2`; `1` re-admits cover-page metadata) |
   | `EPISODE_TERM_CAP` | no | Max terms required per episode (default `35`; measured — lowering it cut coverage without shortening scripts) |
   | `REDIS_URL` | one of three | Native TLS URL, e.g. `rediss://default:...@host:port/0?ssl_cert_reqs=required` |
   | `UPSTASH_HOST` + `UPSTASH_PASSWORD` (optional `UPSTASH_PORT`) | one of three | Composed into `REDIS_URL` automatically |
   | `UPSTASH_REDIS_REST_URL` | one of three | Redis REST endpoint |
   | `UPSTASH_REDIS_REST_TOKEN` | one of three | Redis REST token |
   | `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | yes | Allow Vercel to spawn Modal jobs |
   | `ALLOWED_ORIGINS` | no | Extra CORS origins, e.g. `http://localhost:3000` |

4. Deploy: push to `main`, or run `vercel --prod`.

   `vercel.json` sets `includeFiles` to the single glob `"api/**"`. Every endpoint imports the shared `api/generate_flashcards.py` and reads `api/prompts/*.yaml`, so those files must be in **each** function's bundle - Vercel's per-function auto-detection is not relied on.

   It must be a **string**, not a list. An array is rejected at deploy time:
   `Invalid request: functions.api/*.py.includeFiles should be string`. `test_zen.py` asserts the type so this cannot regress, and checks that every `api/*.py` import resolves to stdlib, another `api/` file, or a package in `requirements.txt`.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/api/verify_password` | Exchange the password for a 24 h bearer token |
| POST | `/api/generate_flashcards` | Submit a document; returns a `job_id` |
| GET | `/api/status?job_id=…` | Poll status / progress / results |
| POST | `/api/cancel` | Cancel a running job (body: `{job_id}`) |
| POST | `/api/tts` | Synthesize one sentence to WAV audio (body: `{text}`) |
| POST | `/api/podcast_script` | Podcast episodes. No `episode` → `{total}` plan only; `{episode: i}` → one script |
| POST | `/api/podcast_audio` | Synthesize one spoken line to WAV (body: `{speaker, text}`) |
| POST | `/api/warmup` | Pre-warm the Modal GPU container |

Auth is disabled (single user, single device, no accounts); the endpoints are public. `ALLOWED_ORIGINS` still gates cross-origin reads.

## Podcast

A deck's podcast is a **list of episodes**, one per heading-aligned section of the document, generated **one episode per request** and synthesized **on demand**.

- **Episodes emerge, they are not chosen.** `episode_sections()` splits on the document's own `## Headings` at up to `EPISODE_CHARS` (12,000) characters, so a course document becomes one episode per week/chapter. Your SIWES report → 10 episodes of ~11k characters.
- **One request per episode**, exactly like card chunks. Generating all N scripts in one function would be N LLM calls against a 60-second limit — a 10-episode deck would time out every time. The client plans first (`{total}`), then asks for each index, saving each script to IndexedDB **as it arrives**, so an interrupted run keeps the episodes already made and pressing again retries only the gaps.
- **The deck is a checklist, not a plan.** `map_cards_to_episodes()` assigns each card to the episode whose source text contains its `source_quote`, then contributes that card's numbers and proper nouns as terms the episode must mention. Episode structure still comes from the document — the podcast and the deck remain two separate products sharing a fact source.
  Three filters keep the checklist honest, all tunable:
  - `EPISODE_MIN_CARDS` (default 2) — a term in one card is *cover-page metadata* (whose report, what month); a term in several is course content. On the real deck this dropped all five cover-page terms while keeping 260 content terms.
  - plural merge — `APIs` folds into `API` **only when the singular actually occurs in the deck**, so `SIWES` is never chopped into `siwe`. Two surface forms of one word cost two cap slots and guarantee a false miss, because a script says the word once.
  - `EPISODE_TERM_CAP` (default 35). The instinct that 35 forces a recitation was measured and did not hold: dropping to 18 gave **lower** coverage (79.6% vs 83.3%) and no shorter scripts (1110 vs 1146 words) — the model just wandered, and one episode at cap 18 failed to mention `SIWES` at all. The checklist behaves as a floor, not a constraint on elegance. See `tune_terms.py`.
- **Coverage is measured, never displayed.** `script_coverage()` returns covered/total/missing as whole-token matches. It is not shown in the UI — it exists so "is this covering enough?" is a number you can tune against, the way chunk size was tuned against cards per 1,000 characters.
- **Voices**: `Alex` is `aura-2-orion-en`, `Sam` is `aura-2-draco-en` (chosen by ear; draco's British baritone against orion's American calm). Mapping lives in `VOICE_BY_SPEAKER` in `api/podcast_audio.py`, overridable with `PODCAST_VOICE_ALEX` / `PODCAST_VOICE_SAM`. Deepgram renders **one voice per request**, so the client sends one line at a time — which also means each line caches and retries on its own.
- **Playback speed is `1.1x`, applied client-side** (`playbackRate` + `preservesPitch`). Not baked into synthesis: Deepgram's own `speed`/`pitch` query params measured unreliable (`speed=1.1` produced *longer* audio), and a client-side rate keeps one cached clip valid at any speed. `PLAYBACK_RATE` is stored in `localStorage` under `podcastRate`.
- **Auth scheme is `Token`, not `Bearer`.** A `Bearer` request returns 401 and looks exactly like an invalid key — worth knowing before debugging a "bad" key.
- **Episode length is bounded by shape, not by word count.** A "700-1100 words" instruction is the one thing this model ignores: measured output ran 480-1976 words with single turns up to **267 words** - a monologue nobody can listen to. What holds is the locally checkable rule: **1-3 sentences, under 70 words per turn**, enforced by `band_problems()` and retried once. Per-episode target length is derived from the checklist size (`EPISODE_WORDS_PER_TERM`), because 35 terms cannot be named in 500 words.

  Two wider clamps were tried, measured, and removed. A **turn ceiling** dropped one episode's coverage from 91% to 80% while producing 71 short turns that were perfectly listenable. A **thinness retry** made the model pad connective filler and cover fewer terms (83% → 71%). Both now only *report*: `in_band` flags real defects, `thin` flags a short-but-valid script without rewriting it. On the real deck after this change: longest turn 267 → 46 words, coverage 75% overall but 91% on a dense technical section and 57% on the front matter - the remaining spread is how discussable each section is, not a defect to clamp. Tune with `tune_terms.py`.
- **Audio pacing**: requests are sequential with a short gap per line and between episodes. Gemini's TTS quota was per-*minute*; Deepgram's is credit-based, but pacing still matters because a full document is several hundred line requests (~55 per episode).

- **Listen while you study.** The podcast is a side **drawer**, not a full-screen modal, and closing it only hides the transcript — playback continues in a persistent mini-player fixed over the deck. (The old modal was `position: fixed; inset: 0`, so the podcast and the flashcards were mutually exclusive by construction.)

## Study modes

**Read & Flip** is the default: flip with a click or space, arrow through the deck, star anything you got wrong, filter to ⭐ Starred and drill those. No forced grading, no timers.

**Typing mode** (⌨️ in the deck header, opt-in, remembered in `localStorage`) is active recall:

1. The question and a textarea. Type your answer, press Enter or **Reveal**.
2. Your answer is shown above the expected one, with **the terms you missed highlighted in your own text** — highlighting the matches in the expected answer only tells you what you already knew.
3. **Again / Hard / Got it** is the grade. It is a thin surface over the *existing* star boolean (Again and Hard star the card, Got it clears it), so active recall adds no new state and no new UI concept.

Deliberately local: `keyWords()` + a stopword set, no percentage, **no LLM call**. A grader call would cost money and latency per attempt, would need the network for a study view that is meant to work offline, and would be a worse judge of "did I get the idea" than the student. A correct answer in different words legitimately shows as misses — which is precisely why the human presses the button.

## Text to speech

Each flashcard has a 🔊 button on both faces (plus an **Auto-read** toggle that reads the side you flip to).

- **Engine**: Piper (`piper-tts`), an English neural voice (`en_US-lessac-medium`) running on **CPU inside the same Modal container** as the LLM — no second cold start, and the GPU stays free for vLLM. Only ~500 MB on top of the image.
- **Streaming**: browsers can't reliably play a half-downloaded WAV, so instead the frontend splits text into sentences and **pipelines** the requests — sentence N plays while sentence N+1 is already downloading. Playback typically starts in ~300-500 ms.
- **Caching**: `/api/tts` hashes `voice + text` and caches the WAV in Redis for 7 days, so re-studying a deck is free. The browser also keeps an in-memory cache for the session.
- **Interruptions**: one voice at a time. Card TTS stops the podcast and vice versa (`stopSpeaking` / `stopPodcast` call each other), so a 🔊 and a playing episode can never talk over one another. Speech also stops when a new generation starts.
- **Voice**: change `TTS_VOICE` in `modal_app.py` (and `api/tts.py`, which must match) to any voice from the `rhasspy/piper-voices` repo, then `modal deploy`. The new voice downloads into the volume on first use.

## Grounding (anti-hallucination)

A small model leaks its own training knowledge into answers — "SIWES was established in 1973 by the ITF" can be correct *and* absent from your document. Three cheap countermeasures are built in:

1. **Prompt** (`api/prompts/card_gemini.yaml` / `card_qwen.yaml`): the text is declared the model's *only* source of truth, and the document's own `## Heading` structure is used as a coverage checklist — every section must produce cards.
2. **`source_quote` is optional.** It was mandatory once, and that was the single biggest cap on answer depth: the model compressed every answer into what one sentence could justify (~23 words). Verbatim quotes are still checked when present.
3. **Gates that drop a card** — exact checks only:
   - a `source_quote` that cannot be recovered from the chunk (verbatim, or as verbatim runs of ≥4 words)
   - a number in the answer that the chunk never mentions (`1,000` and `1000` match; `1973` does not match `11973`)

   An embedding-based semantic pass was built and then removed: it needs a second model, its cosine thresholds are not comparable across embedding providers (so the same number behaves differently on the Modal path than the API path), and on a real 415-card deck the exact checks had nothing left for it to catch. Names missing from a card's own quote are *reported*, never enforced — that check is a weak proxy and once cut a real 80-card deck to 9.

The UI shows difficulty as a chip on both faces and the quote (when there is one) on the answer side.

## Testing the Zen pipeline

`probe_zen.py` calls the *shipped* code (`generate_json` → `zen_json`), not a reimplementation, so a pass means the deployed path works.

```bash
.venv/bin/python probe_zen.py --list-models                          # which models this key can use
.venv/bin/python probe_zen.py                                        # one small generation
.venv/bin/python probe_zen.py --doc report.docx                      # one real 20k-char section group
.venv/bin/python probe_zen.py --doc report.docx --model gemini-3.8-flash
```

It reads `OPENCODE_API_KEY` from the environment or `.env.local` and never prints it.

**Two gotchas this test caught, both now fixed:**

- **Zen requires an explicit `User-Agent`.** Cloudflare answers urllib's default with `403 error code: 1010`. Send any UA (`flashcard-generator/1.0`); a browser UA is not needed.
- **The model list is not under the Gemini path.** It is `GET https://opencode.ai/inference/v1/models`.

The model a key can use is **not** the whole published catalogue. On the current key only `gemini-3.5-flash-lite`, `gemini-3.8-flash` and `gemini-4-argon` resolve; `gemini-3.5-flash` returns `403 Model access is disabled`. On one 19,740-char section of the SIWES report:

| Model | Cards kept | Latency |
| --- | --- | --- |
| `gemini-3.5-flash-lite` | 26 | 8.9 s |
| `gemini-3.8-flash` | 19 | 13.9 s |

The stronger model is more conservative and produced *fewer* cards on this material, so the cheaper/faster default is also the better deck.

### Chunk size is the coverage lever

Measured on the same 19,740-character section, same model, same prompt, varying only how much text went into one call:

| Text per call | Calls | Cards | Cards per 1,000 chars | Time |
| --- | --- | --- | --- | --- |
| 19,740 | 1 | 26 | 1.32 | 8.9 s |
| ~6,600 | 3 | 69 | 3.50 | 18.4 s |
| ~3,900 | 5 | 82 | 4.15 | 24.4 s |

The model stops being exhaustive long before it runs out of context: given 20k characters it summarises, given 6k it enumerates. **`CHUNK_CHARS` now defaults to 6000** (`MAX_CHUNKS = 100`), which turns the SIWES report into 20 sequential calls. That is the single biggest quality change in this project — it recovered the coverage the original 415-card deck had. Smaller still (3.9k) buys another 19% for ~35% more wall-clock, so 6k is the knee.

Two mechanisms used to silently undo this, both now guarded by a test (`every chunk stays at the 6000-char knee up to 500k chars`):

- `section_chunks` raised its budget to `ceil(len(text) / MAX_CHUNKS)`. With `MAX_CHUNKS = 20`, a 500k-character document got 25k chunks — back over the knee. 100 keeps the budget at `CHUNK_CHARS` for anything inside `MAX_TEXT_CHARS`.
- A fold loop merged adjacent chunks whenever the count exceeded the ceiling, pushing them past 6k. `fold_small` now only merges a pair while the result stays within the size budget; if that is impossible it keeps the extra calls and says so, because losing coverage is worse than losing the ceiling.

The Modal/Qwen fallback now uses the same 6k section plan as the API path (it no longer has its own 24k semantic chunker), so both engines produce the same chunk boundaries and the fallback can resume mid-job.

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
- `api/generate_flashcards.py`: `MAX_PAGES = 100`, `MAX_TEXT_CHARS = 500_000`, `CHUNK_CHARS` (env `GEMINI_CHUNK_CHARS`, default `6000`), `MAX_CHUNKS` (env `GEMINI_MAX_CHUNKS`, default `100`).
- `api/status.py`: `QUEUED_STALE_SECONDS = 10 min`, `PROCESSING_STALE_SECONDS` (env, default 10 min).
- Grounding: `GROUNDING_RUN_RATIO = 0.55`, `GROUNDING_COVERAGE = 0.85`. Tune them against a real deck with `audit_cards.py`, which histograms each quote's longest verbatim run.
- **Prompts** are `api/prompts/*.yaml` — `card_gemini.yaml` (Zen + direct Gemini), `card_qwen.yaml` (Modal), `podcast.yaml`. The API reads them from `api/prompts/`; `modal_app.py` copies them into the image at `/root/prompts`. One edit changes both engines.
- `modal_app.py`: `MAX_CHUNK_CHARS = 24_000`, `SIM_BOUNDARY_THRESHOLD`, `DEDUP_THRESHOLD`, cron schedule, `STUDY_TIMEZONE`.
- `TTS_VOICE` lives in `api/generate_flashcards.py` only — both `api/tts.py` (which hashes it into the audio cache key) and the Modal worker read it from there.
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
