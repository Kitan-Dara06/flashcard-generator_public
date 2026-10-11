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

import modal

APP_NAME = "flashcard-generator"
STUDY_TIMEZONE = "Africa/Lagos"  # <-- set to your users' timezone (Lagos = WAT, UTC+1)

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

MAX_CHUNK_CHARS = 24_000        # ~6k tokens per chunk
SIM_BOUNDARY_THRESHOLD = 0.5    # adjacent-block similarity below this: topic boundary
DEDUP_THRESHOLD = 0.92          # question similarity above this: duplicate card
REDIS_TTL_SECONDS = 24 * 60 * 60

# Text-to-speech (Piper, runs on CPU in this same container)
TTS_VOICE_REPO = "rhasspy/piper-voices"
TTS_VOICE = "en_US-lessac-medium"   # <-- change to any rhasspy/piper-voices voice

SYSTEM_PROMPT = """You are a flashcard generator for theory-based subjects.
Generate as many flashcards as possible from the user's text (aim for at least 30 if the content allows).
Each flashcard must have:
- "question": a clear, concise question
- "answer": a 2-3 sentence explanatory answer
Stay strictly factual, based only on the provided text.
If the text contains no usable information, return an empty list of flashcards."""

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "vllm==0.6.6",
        "sentence-transformers>=3.0.0",
        "piper-tts>=1.2.0",
        "upstash-redis>=1.0.0",
    )
    .env({"HF_HOME": "/cache"})
)

hf_cache = modal.Volume.from_name("flashcard-hf-cache", create_if_missing=True)
redis_secret = modal.Secret.from_name("upstash")


def _flashcard_schema() -> dict:
    """JSON schema for vLLM's guided decoder, so output is always valid JSON."""
    return {
        "type": "object",
        "properties": {
            "flashcards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                    "required": ["question", "answer"],
                },
            }
        },
        "required": ["flashcards"],
    }


def _dot(a, b) -> float:
    """Dot product; embeddings are normalized, so this is cosine similarity."""
    return float(sum(x * y for x, y in zip(a, b)))


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
        """Load models once; Modal snapshots this state for ~1-3s restores."""
        from sentence_transformers import SentenceTransformer
        from vllm import LLM

        self.llm = LLM(
            QWEN_MODEL,
            quantization="awq",
            gpu_memory_utilization=0.85,
            max_model_len=16384,
            disable_log_stats=True,
        )
        # bge-small runs on CPU: it's tiny, and keeps the GPU free for vLLM.
        self.embedder = SentenceTransformer(EMBED_MODEL)

        # Piper TTS also runs on CPU, co-located so it shares the container
        # (and therefore the warm/snapshot lifecycle) with the LLM.
        from huggingface_hub import hf_hub_download
        from piper.voice import PiperVoice

        onnx_file, config_file = _voice_paths(TTS_VOICE)
        self.tts = PiperVoice.load(
            hf_hub_download(TTS_VOICE_REPO, onnx_file),
            config_path=hf_hub_download(TTS_VOICE_REPO, config_file),
        )

    # ----- pre-warm -----
    @modal.method()
    def ping(self) -> str:
        return "pong"

    # ----- text to speech -----
    @modal.method()
    def tts(self, text: str) -> bytes:
        """Synthesize text to a complete WAV byte string (16-bit PCM)."""
        text = (text or "").strip()
        if not text:
            return b""

        chunks = list(self.tts.synthesize(text))
        if not chunks:
            return b""

        first = chunks[0]
        pcm = b"".join(chunk.audio_int16_bytes for chunk in chunks)
        return _wav_bytes(pcm, first.sample_rate, first.sample_width, first.sample_channels)

    # ----- job worker -----
    @modal.method()
    def process_job(self, job_id: str) -> None:
        from upstash_redis import Redis

        r = Redis(
            url=os.environ["UPSTASH_REDIS_REST_URL"],
            token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
        )

        def key(suffix: str) -> str:
            return f"job:{job_id}:{suffix}"

        def write(suffix: str, value: str) -> None:
            r.set(key(suffix), value, ex=REDIS_TTL_SECONDS)

        # Cheap fix for abandoned jobs: cancel flag is honoured before and
        # between chunks, so a cancelled job releases the GPU almost instantly.
        if r.get(key("cancelled")) == "1":
            write("status", "cancelled")
            return

        write("status", "processing")
        try:
            text = r.get(key("text"))
            if not text:
                raise ValueError("Job data expired or missing")

            chunks = self._semantic_chunks(text)
            write("progress", f"0/{len(chunks)}")

            cards: list[dict] = []
            for i, chunk in enumerate(chunks, 1):
                if r.get(key("cancelled")) == "1":
                    write("status", "cancelled")
                    return
                cards.extend(self._generate_cards(chunk))
                write("progress", f"{i}/{len(chunks)}")

            cards = self._dedup(cards)
            write("cards", json.dumps(cards))
            write("status", "complete")
        except Exception as exc:
            write("error", f"{type(exc).__name__}: {exc}")
            write("status", "error")
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
        for i in range(1, len(sized)):
            similarity = _dot(embeddings[i - 1], embeddings[i])
            block_len = len(sized[i])
            if similarity < SIM_BOUNDARY_THRESHOLD or current_len + block_len + 2 > max_chars:
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

        sampling = SamplingParams(
            temperature=0.1,
            max_tokens=4000,
            guided_decoding=GuidedDecodingParams(json=_flashcard_schema()),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text:\n\n{chunk}"},
        ]
        outputs = self.llm.chat(messages, sampling, use_tqdm=False)

        data = json.loads(outputs[0].outputs[0].text)
        cards = []
        for item in data.get("flashcards", []):
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question and answer:
                cards.append({"question": question, "answer": answer})
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


@app.function(schedule=modal.Cron("*/15 8-22 * * 1-5", timezone=STUDY_TIMEZONE))
def keep_warm() -> None:
    """Tiny call every 15 min during study hours; resets FlashcardLLM's idle timer."""
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
