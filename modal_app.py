# modal_app.py
"""Modal worker for the flashcard generator.

Runs Qwen 2.5 7B-Instruct (AWQ) via vLLM for generation and
BAAI/bge-small-en-v1.5 for semantic chunking + deduplication.

Deploy with:

    modal deploy modal_app.py

Before deploying:
  * create the Redis secret, e.g.:
      modal secret create upstash \
        UPSTASH_REDIS_REST_URL=... UPSTASH_REDIS_REST_TOKEN=...
  * adjust STUDY_TIMEZONE below to your users' timezone
"""
import json
import os
import re
import struct
import sys

import modal

APP_NAME = "flashcard-generator"

BUILD = "zen-v1"
STUDY_TIMEZONE = "Africa/Lagos"  # <-- set to your users' timezone (Lagos = WAT, UTC+1)
# Where the shared card rules and prompts land inside the container. They are
# mounted from api/ below rather than copied, so there is exactly one definition
# of "what a valid card is" and one copy of each prompt.
SHARED_DIR = "/root/shared"
PROMPT_DIR = "/root/prompts"

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

MAX_CHUNK_CHARS = 24_000        # ~6k tokens per chunk
MIN_CHUNK_CHARS = 6_000         # gather at least this much before honouring a topic break
MAX_CHUNKS = 8                  # ceiling on LLM calls per document (raises min size if needed)
SIM_BOUNDARY_THRESHOLD = 0.5    # adjacent-block similarity below this: topic boundary
DEDUP_THRESHOLD = 0.92          # question similarity above this: duplicate card
REDIS_TTL_SECONDS = 24 * 60 * 60

# Text-to-speech (Piper, runs on CPU in this same container)
TTS_VOICE_REPO = "rhasspy/piper-voices"


def _tts_voice() -> str:
    """The Piper voice, read from the shared module.

    api/tts.py hashes this value into the audio cache key, so the two must agree
    exactly or cached clips get served for the wrong voice. One definition, in
    api/generate_flashcards.py, removes the possibility.
    """
    return _shared().TTS_VOICE

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        # vLLM 0.6.6 pins torch==2.5.1 (and this project's guided-JSON decoding
        # targets its API), so the transformer stack must stay on the same
        # torch generation. Newer transformers calls torch.accelerator.* (torch
        # >= 2.6) and breaks the import on torch 2.5.
        "vllm==0.6.6",
        "transformers==4.46.3",
        "sentence-transformers==3.3.1",
        # vLLM 0.6.6 declares xgrammar>=0.1.6 with NO upper bound, so pip pulls
        # the 0.2.x line, whose TokenizerInfo API changed
        # ("type object 'TokenizerInfo' has no attribute 'from_huggingface'")
        # and breaks guided decoding. 0.1.6 is vLLM 0.6.6's declared minimum,
        # so it is compatible with the calls vLLM makes.
        "xgrammar==0.1.6",
        "piper-tts>=1.2.0",
        "upstash-redis>=1.0.0",
        "redis>=5.0.0",
        # The prompts are YAML (api/prompts/*.yaml), shared with the Vercel side.
        "pyyaml>=6.0",
    )
    .env({"HF_HOME": "/cache"})
    # The card rules, chunking-by-section fallback and prompt loader are the same
    # code the Vercel functions run - mounted, not re-pasted here, so the two
    # engines cannot drift apart.
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "api", "generate_flashcards.py"),
        remote_path=os.path.join(SHARED_DIR, "generate_flashcards.py"),
        copy=True,
    )
    .add_local_dir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "api", "prompts"),
        remote_path=PROMPT_DIR,
        copy=True,
    )
)

hf_cache = modal.Volume.from_name("flashcard-hf-cache", create_if_missing=True)
redis_secret = modal.Secret.from_name("upstash")

_SHARED = None


def _shared():
    """api/generate_flashcards.py, imported from the mounted copy.

    Imported lazily: this module is also executed locally by `modal` to build
    the app, where SHARED_DIR does not exist yet.
    """
    global _SHARED
    if _SHARED is None:
        if SHARED_DIR not in sys.path:
            sys.path.insert(0, SHARED_DIR)
        import generate_flashcards as mod
        _SHARED = mod
    return _SHARED


def _dot(a, b) -> float:
    """Dot product; embeddings are normalized, so this is cosine similarity."""
    return float(sum(x * y for x, y in zip(a, b)))


def _job_access(job_id: str):
    """Redis client plus the job-scoped key/write helpers both workers use."""
    r = _redis_client()

    def key(suffix: str) -> str:
        return f"job:{job_id}:{suffix}"

    def write(suffix: str, value: str) -> None:
        r.set(key(suffix), value, ex=REDIS_TTL_SECONDS)

    return r, key, write


def _redis_url() -> str | None:
    """The Redis connection string, however it was supplied.

    1. REDIS_URL directly, or
    2. composed from the Upstash dashboard fields: UPSTASH_HOST (or
       UPSTASH_ENDPOINT) + UPSTASH_PASSWORD (+ UPSTASH_PORT, default 6379).

    Returns None when neither is available, in which case the caller uses the
    Upstash REST API instead.
    """
    url = os.environ.get("REDIS_URL")
    if url:
        return url

    host = os.environ.get("UPSTASH_HOST") or os.environ.get("UPSTASH_ENDPOINT")
    password = os.environ.get("UPSTASH_PASSWORD")
    if not (host and password):
        return None

    from urllib.parse import quote
    port = os.environ.get("UPSTASH_PORT") or "6379"
    return (f"rediss://default:{quote(password, safe='')}@{host}:{port}/0"
            f"?ssl_cert_reqs=required")


def _redis_client():
    """Redis client for the worker - mirrors the helper in api/*.py.

    Native TLS (redis-py) when a URL can be resolved, otherwise the Upstash REST
    API (URL + token). Writers and readers must use the same transport - the
    clients encode values differently - so keep both sides on the same choice.
    """
    url = _redis_url()
    if url:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "The 'redis' client package is required when connecting with a "
                "rediss:// URL (pinned in requirements.txt and the Modal image). "
                "Install it, or switch to the Upstash REST credentials instead."
            ) from exc
        return redis.from_url(url, decode_responses=True)

    rest_url = os.environ["UPSTASH_REDIS_REST_URL"]
    if not rest_url.startswith(("http://", "https://")):
        raise RuntimeError(
            f"UPSTASH_REDIS_REST_URL={rest_url!r} is missing its scheme - it should "
            f"look like 'https://<host>.upstash.io'. If you meant to use the native "
            f"connection instead, set REDIS_URL (rediss://...) or UPSTASH_HOST + "
            f"UPSTASH_PASSWORD (+ UPSTASH_PORT)."
        )

    from upstash_redis import Redis
    return Redis(
        url=os.environ["UPSTASH_REDIS_REST_URL"],
        token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
    )


def _has_redis_credentials() -> bool:
    if _redis_url():
        return True
    return bool(os.environ.get("UPSTASH_REDIS_REST_URL")
                and os.environ.get("UPSTASH_REDIS_REST_TOKEN"))


def _voice_paths(voice: str) -> tuple[str, str]:
    """Map 'en_US-lessac-medium' to its two files in the piper-voices repo."""
    locale, name, quality = voice.split("-")
    lang = locale.split("_")[0]
    base = f"{lang}/{locale}/{name}/{quality}/{voice}"
    return f"{base}.onnx", f"{base}.onnx.json"


def _wav_bytes(pcm: bytes, sample_rate: int, sample_width: int, channels: int) -> bytes:
    """Wrap raw 16-bit PCM in a minimal WAV container."""
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    return b"".join([
        b"RIFF",
        struct.pack("<I", 36 + len(pcm)),
        b"WAVEfmt ",
        struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, sample_width * 8),
        b"data",
        struct.pack("<I", len(pcm)),
        pcm,
    ])


@app.cls(
    image=image,
    gpu="A10G",
    memory=32768,
    # 20 min idle window > 15 min cron gap, so study-hours pings always
    # hit a warm container (a 15 min window would race the cron).
    scaledown_window=1200,
    min_containers=0,          # scale to zero off-hours; the cron covers study hours
    # Long documents can take much longer than the default 300 s limit.
    timeout=3600,
    # Required for @modal.enter(snap=True) on Modal 1.x; snapshots the
    # loaded engine so later cold starts restore in ~1-3 s.
    enable_memory_snapshot=True,
    volumes={"/cache": hf_cache},
    secrets=[redis_secret],
)
class FlashcardLLM:
    @modal.enter(snap=True)
    def load(self):
        """Load the models generation depends on. Keep this lean: anything that
        fails here takes the whole container down, so optional extras (TTS) are
        loaded lazily on first use instead."""
        from sentence_transformers import SentenceTransformer
        from vllm import LLM

        print(f"[flashcard-worker] build={BUILD} booting")

        self.llm = LLM(
            QWEN_MODEL,
            # vLLM logs that this model can use awq_marlin and that plain awq
            # "is not fully optimized yet. The speed can be slower" - marlin
            # kernels are the faster path on Ampere+ (A10G). Revert to "awq"
            # if it ever misbehaves.
            quantization="awq_marlin",
            gpu_memory_utilization=0.85,
            max_model_len=16384,
            disable_log_stats=True,
        )
        # bge-small runs on CPU: it's tiny, and keeps the GPU free for vLLM.
        self.embedder = SentenceTransformer(EMBED_MODEL)

    # ----- pre-warm -----
    @modal.method()
    def ping(self) -> str:
        return "pong"

    # ----- text to speech -----
    def _get_voice(self):
        """Lazily load the Piper voice (CPU), once per container.

        Deliberately not part of load(): TTS is optional and must never be able
        to stop the container from booting. Raises only when TTS is actually
        used, and the failure surfaces as a 502 from /api/tts.
        """
        voice = getattr(self, "_voice", None)
        if voice is not None:
            return voice

        from huggingface_hub import hf_hub_download
        try:
            from piper.voice import PiperVoice          # piper-tts 1.2.x
        except ImportError:
            from piper import PiperVoice                # piper-tts 1.3.x+

        voice_name = _tts_voice()
        onnx_file, config_file = _voice_paths(voice_name)
        print(f"Loading Piper voice {voice_name} (first TTS request)")
        self._voice = PiperVoice.load(
            hf_hub_download(TTS_VOICE_REPO, onnx_file),
            config_path=hf_hub_download(TTS_VOICE_REPO, config_file),
        )
        return self._voice

    @modal.method()
    def tts(self, text: str) -> bytes:
        """Synthesize text to a complete WAV byte string (16-bit PCM)."""
        text = (text or "").strip()
        if not text:
            return b""

        chunks = list(self._get_voice().synthesize(text))
        if not chunks:
            return b""

        first = chunks[0]
        pcm = b"".join(chunk.audio_int16_bytes for chunk in chunks)
        return _wav_bytes(pcm, first.sample_rate, first.sample_width, first.sample_channels)

    # ----- job worker -----
    @modal.method()
    def process_job(self, job_id: str) -> None:
        if not _has_redis_credentials():
            # We cannot write to Redis without these, so this is the only place
            # the problem can be reported - make it loud and actionable.
            raise RuntimeError(
                "No Redis credentials in the container environment. Set REDIS_URL "
                "(rediss://default:<password>@<host>:<port>/0?ssl_cert_reqs=required) "
                "on the 'upstash' secret, or both UPSTASH_REDIS_REST_URL and "
                "UPSTASH_REDIS_REST_TOKEN. Recreate it with: modal secret create "
                "--force upstash REDIS_URL=..."
            )

        r, key, write = _job_access(job_id)

        try:
            # Cheap fix for abandoned jobs: the cancel flag is honoured before
            # and between chunks, so a cancelled job frees the GPU fast.
            if r.get(key("cancelled")) == "1":
                write("status", "cancelled")
                return

            write("status", "processing")
            # Stamped into logs and into any error message: when a warm container
            # from an older deploy is still serving, this is what tells you which
            # code actually ran.
            print(f"job {job_id}: build={BUILD}")

            text = r.get(key("text"))
            if not text:
                raise ValueError("Job data expired or missing")

            chunks = self._semantic_chunks(text)
            write("progress", f"0/{len(chunks)}")

            cards: list[dict] = []
            failed_chunks = 0
            for i, chunk in enumerate(chunks, 1):
                if r.get(key("cancelled")) == "1":
                    write("status", "cancelled")
                    return
                try:
                    batch = self._generate_cards(chunk)
                except Exception as exc:
                    # One bad chunk must not sink the whole document: log it, score
                    # it as empty, and keep going.
                    failed_chunks += 1
                    print(f"chunk {i}/{len(chunks)} failed: {type(exc).__name__}: {exc}")
                    batch = []
                # Per-chunk count: makes a chunk that yielded nothing (all cards
                # dropped as ungrounded) visible in the log.
                print(f"chunk {i}/{len(chunks)}: {len(batch)} card(s)")
                cards.extend(batch)
                write("progress", f"{i}/{len(chunks)}")

            if not cards:
                raise ValueError(
                    f"No flashcards could be generated from this document "
                    f"({failed_chunks}/{len(chunks)} chunk(s) failed)."
                )

            cards = self._dedup(cards)
            if cards:
                # Shape check: catches any drift between what the model returns,
                # what the filter keeps, and what the UI expects to render.
                first = cards[0]
                quoted = sum(1 for c in cards if (c.get("source_quote") or "").strip())
                print(
                    f"writing {len(cards)} card(s) to redis; "
                    f"keys={sorted(first.keys())} grounded={quoted}/{len(cards)}"
                )
            write("cards", json.dumps(cards))
            write("status", "complete")
        except Exception as exc:
            try:
                write("error", f"[{BUILD}] {type(exc).__name__}: {exc}")
                write("status", "error")
            except Exception as nested:
                # Redis itself is unusable; the Modal log is the only record.
                print("could not record the failure in Redis:", nested)
            raise

    # ----- internals -----
    def _semantic_chunks(self, text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
        """Split text into ~max_chars chunks that respect topic boundaries.

        Blocks (paragraphs) are embedded with bge-small; a chunk boundary is
        placed wherever adjacent-block similarity drops below
        SIM_BOUNDARY_THRESHOLD, or whenever adding a block would exceed
        max_chars.
        """
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        if len(blocks) <= 1:
            blocks = [b.strip() for b in text.split("\n") if b.strip()]

        # hard-split any single block that alone exceeds the budget
        sized: list[str] = []
        for block in blocks:
            while len(block) > max_chars:
                sized.append(block[:max_chars])
                block = block[max_chars:]
            if block:
                sized.append(block)

        if len(sized) <= 1:
            return sized or [text]

        embeddings = list(self.embedder.encode(sized, normalize_embeddings=True))

        chunks: list[str] = []
        current = [sized[0]]
        current_len = len(sized[0])
        # Bound the number of LLM calls: for a long document, require more text
        # per chunk so a report with many short sections can't turn into a
        # dozen-plus generations.
        min_chars = max(MIN_CHUNK_CHARS, len(text) // MAX_CHUNKS)
        for i in range(1, len(sized)):
            similarity = _dot(embeddings[i - 1], embeddings[i])
            block_len = len(sized[i])
            # Only honour a topic break once enough text has accumulated.
            topic_break = (similarity < SIM_BOUNDARY_THRESHOLD
                           and current_len >= min_chars)
            if topic_break or current_len + block_len + 2 > max_chars:
                chunks.append("\n\n".join(current))
                current, current_len = [sized[i]], block_len
            else:
                current.append(sized[i])
                current_len += block_len + 2
        chunks.append("\n\n".join(current))
        return chunks

    def _generate_cards(self, chunk: str) -> list[dict]:
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        shared = _shared()
        sampling = SamplingParams(
            temperature=0.1,
            # Headroom: a 40-card response with question+answer+quote+difficulty is
            # ~3.5-4k tokens. At 4000 the response was cut off mid-string, which
            # (despite guided decoding) yields unparseable JSON.
            max_tokens=7000,
            # require_quote only forces the KEY to be emitted (xgrammar is happier
            # without optional properties); the prompt lets the value stay empty,
            # because a mandatory sentence per card is what capped answers at ~23
            # words.
            guided_decoding=GuidedDecodingParams(json=shared.card_schema(require_quote=True)),
        )
        messages = [
            {"role": "system", "content": shared.prompt_text("card_qwen")},
            {"role": "user", "content": f"Text:\n\n{chunk}"},
        ]
        outputs = self.llm.chat(messages, sampling, use_tqdm=False)

        raw = outputs[0].outputs[0].text
        items = shared.parse_cards(raw)
        if len(raw) >= 7000:
            print(f"response may have hit the token limit - kept {len(items)} card(s)")

        cards, _dropped = shared.clean_cards(items, chunk)
        return cards

    def _dedup(self, cards: list[dict]) -> list[dict]:
        """Drop cards whose question is near-identical to an already-kept one."""
        if len(cards) <= 1:
            return cards
        embeddings = list(
            self.embedder.encode(
                [c["question"] for c in cards], normalize_embeddings=True
            )
        )
        kept: list[int] = []
        for i in range(len(cards)):
            if kept and max(_dot(embeddings[i], embeddings[j]) for j in kept) > DEDUP_THRESHOLD:
                continue
            kept.append(i)
        dropped = len(cards) - len(kept)
        if dropped:
            print(f"dedup: dropped {dropped} near-duplicate card(s)")
        return [cards[i] for i in kept]


# Layer 3 of the cold-start strategy is DISABLED by default.
#
# Why: a ping every 15 min with a 20-minute scaledown_window keeps the A10G
# *continuously* warm for the whole study window - roughly 14 h per weekday,
# which costs far more than the cold starts it avoids. For a low-traffic app
# the frontend pre-warm hook (invoked when a user actually opens the page) is
# the better trade: free when nobody is around.
#
# Warm the worker on demand instead:   modal run modal_app.py::keep_warm
# Or re-enable the schedule by adding back:
#   schedule=modal.Cron("*/15 8-22 * * 1-5", timezone=STUDY_TIMEZONE)
@app.function()
def keep_warm() -> None:
    """Put the LLM container into a warm state without running a job."""
    FlashcardLLM().ping.remote()



@app.function(image=image, volumes={"/cache": hf_cache}, timeout=3600)
def download_models() -> None:
    """Optional one-off: pull the weights into the volume ahead of time so the
    first real container boot only pays the load (not the HuggingFace download).

    Run:  modal run modal_app.py::download_models
    """
    from huggingface_hub import snapshot_download

    snapshot_download(QWEN_MODEL)
    snapshot_download(EMBED_MODEL)
    hf_cache.commit()


@app.function(
    image=image,
    gpu="A10G",
    memory=32768,
    volumes={"/cache": hf_cache},
    secrets=[redis_secret],   # the same secret the worker mounts, so we can verify it
    timeout=1200,
)
def diagnose() -> None:
    """Boot-step probe: checks the worker's environment, Redis access, and models.

    Deliberately a plain function (no class, no memory snapshot), so it isolates
    "the image can't load the models" from "the class container can't boot".
    Imports live inside each stage so one broken dependency can't hide the rest.
    Never prints secret values - only which variables are present.

    Run:  modal run modal_app.py::diagnose
    """
    import traceback

    def stage(label, fn):
        try:
            result = fn()
            print(f"[ok]   {label}")
            return result
        except Exception as exc:
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            print(traceback.format_exc())
            return None

    def check_secret_and_redis():
        expected = ("REDIS_URL", "UPSTASH_HOST", "UPSTASH_ENDPOINT", "UPSTASH_PORT",
                    "UPSTASH_PASSWORD", "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN")
        present = [name for name in expected if os.environ.get(name)]
        lengths = {name: len(os.environ[name]) for name in present}
        print(f"       secret vars present: {present or 'none'} (lengths {lengths})")
        print(f"       transport: {'native rediss:// (redis-py)' if _redis_url() else 'Upstash REST'}")

        if not _has_redis_credentials():
            raise RuntimeError(
                "No Redis credentials in the container. Set ONE of: REDIS_URL "
                "(rediss://default:<password>@<host>:<port>/0?ssl_cert_reqs=required), "
                "UPSTASH_HOST + UPSTASH_PASSWORD (+ optional UPSTASH_PORT), or both "
                "UPSTASH_REDIS_REST_* keys. Recreate the secret, e.g. "
                "modal secret create --force upstash REDIS_URL=..."
            )

        client = _redis_client()
        client.set("diag:ping", "ok", ex=60)
        return client.get("diag:ping")

    def report_versions():
        import importlib.metadata as md
        for name in ("torch", "transformers", "sentence-transformers", "vllm", "piper-tts"):
            try:
                print(f"       {name} {md.version(name)}")
            except Exception:
                print(f"       {name} NOT INSTALLED")
        return True

    def load_vllm():
        from vllm import LLM
        return LLM(
            QWEN_MODEL,
            quantization="awq_marlin",
            gpu_memory_utilization=0.85,
            max_model_len=16384,
            disable_log_stats=True,
        )

    def load_embedder():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBED_MODEL)

    def load_piper():
        from huggingface_hub import hf_hub_download
        try:
            from piper.voice import PiperVoice        # piper-tts 1.2.x
        except ImportError:
            from piper import PiperVoice              # piper-tts 1.3.x+
        onnx_file, config_file = _voice_paths(_tts_voice())
        voice = PiperVoice.load(
            hf_hub_download(TTS_VOICE_REPO, onnx_file),
            config_path=hf_hub_download(TTS_VOICE_REPO, config_file),
        )
        return list(voice.synthesize("Diagnostics test."))

    print("worker environment:")
    if stage("read secret env vars + Redis round trip", check_secret_and_redis) is None:
        print("stopping early - fix the secret above, then re-run")
        return
    print("installed versions:")
    stage("report installed versions", report_versions)
    stage("import + load vLLM (Qwen 2.5 7B Instruct AWQ)", load_vllm)
    stage("import + load bge-small embedder (sentence-transformers)", load_embedder)
    chunks = stage("load Piper voice and synthesize a test phrase", load_piper)
    if chunks:
        pcm = sum(len(chunk.audio_int16_bytes) for chunk in chunks)
        print(f"[ok]   piper returned {len(chunks)} chunk(s), {pcm} PCM bytes")
    print("diagnose finished")
