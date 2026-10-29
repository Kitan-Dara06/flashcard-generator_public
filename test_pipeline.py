#!/usr/bin/env python3
"""End-to-end pipeline test for the flashcard generator.

Drives the real stack over HTTP, exactly like the browser does:

    token -> POST /api/generate_flashcards -> poll /api/status -> cards
                                              -> POST /api/tts (audio check)
                                              -> POST /api/cancel (optional)

Uses only the Python standard library, so it runs anywhere.

Before running:
  1. modal deploy modal_app.py          (worker up to date)
  2. vercel dev                         (or pass --base-url to a deployment)

Examples:
  python3 test_pipeline.py "/home/kitan/Downloads/SIWES_Report_Agboola_Aaliyah (1).docx"
  python3 test_pipeline.py report.docx --cancel-after 45
  python3 test_pipeline.py report.docx --base-url https://flashcard-generator-public.vercel.app
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

MIME_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

MAX_UPLOAD_BYTES = 3 * 1024 * 1024       # frontend cap (base64 inflates ~33%)
MAX_POLL_SECONDS = 20 * 60               # mirrors the frontend ceiling
STALL_AFTER_SECONDS = 6 * 60             # mirrors the frontend stall detector


# ---------------------------------------------------------------- helpers

def log(section, message):
    print(f"[{section:<9}] {message}", flush=True)


def load_env(path=".env.local"):
    """Read .env.local, then let real environment variables win."""
    env = {}
    if os.path.exists(path):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in os.environ.items():
        if key.startswith(("UPSTASH_", "MODAL_", "APP_PASSWORD", "AUTH_SECRET")):
            env[key] = value
    return env


def mint_token(env):
    """Same HMAC scheme as api/verify_password.py."""
    secret = env.get("AUTH_SECRET") or env.get("APP_PASSWORD") or ""
    if not secret:
        sys.exit("FAIL: no APP_PASSWORD/AUTH_SECRET found (.env.local or environment)")
    expires = str(int(time.time()) + 3600)
    signature = hmac.new(secret.encode(), expires.encode(), hashlib.sha256).hexdigest()
    return f"{expires}.{signature}"


def request_json(url, token, payload=None, timeout=180):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": "Bearer " + token}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            try:
                return resp.status, json.loads(body)
            except ValueError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw
    except OSError as exc:
        # connection refused, DNS failure, sandbox/proxy refusal, ...
        reason = getattr(exc, "reason", exc)
        return 0, {"error": f"cannot reach {url}: {reason}"}


def request_bytes(url, token, payload, timeout=180):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": "Bearer " + token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers.get("Content-Type"), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type"), exc.read()
    except OSError as exc:
        reason = getattr(exc, "reason", exc)
        return 0, None, json.dumps({"error": f"cannot reach {url}: {reason}"}).encode()


def wav_info(data):
    """Parse the canonical 44-byte WAV header this project emits."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    channels = struct.unpack("<H", data[22:24])[0]
    rate = struct.unpack("<I", data[24:28])[0]
    bits = struct.unpack("<H", data[34:36])[0]
    data_bytes = struct.unpack("<I", data[40:44])[0]
    byte_rate = rate * channels * (bits // 8)
    return {
        "channels": channels, "rate": rate, "bits": bits, "data_bytes": data_bytes,
        "seconds": round(data_bytes / byte_rate, 2) if byte_rate else None,
    }


def fail(message, extra=None):
    log("FAIL", message)
    if extra is not None:
        try:
            text = json.dumps(extra, indent=2)
        except TypeError:
            text = repr(extra)
        print(text[:2000])
    sys.exit(1)


# ---------------------------------------------------------------- pipeline

def main():
    parser = argparse.ArgumentParser(description="Full pipeline test")
    parser.add_argument("document", help="path to a .docx/.pdf/.txt/.pptx file")
    parser.add_argument("--base-url", default="http://localhost:3000")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--tts-sample", type=int, default=1,
                        help="number of cards to synthesize audio for (0 = skip)")
    parser.add_argument("--cancel-after", type=int, default=0,
                        help="cancel the job this many seconds after submitting (0 = never)")
    parser.add_argument("--provider", choices=["qwen", "gemini"], default=None,
                        help="force an engine (default: let the server pick)")
    parser.add_argument("--chunk-timeout", type=float, default=180,
                        help="timeout for one Gemini section-group request")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    suffix = os.path.splitext(args.document)[1].lower()
    if suffix not in MIME_BY_SUFFIX:
        fail(f"unsupported file type {suffix!r} (expected {', '.join(sorted(MIME_BY_SUFFIX))})")
    if not os.path.exists(args.document):
        fail(f"file not found: {args.document}")

    env = load_env()
    token = mint_token(env)
    raw = open(args.document, "rb").read()
    encoded = base64.b64encode(raw).decode()
    size_mb = len(raw) / 1024 / 1024

    print("=" * 72)
    print(f"Pipeline test: {os.path.basename(args.document)}")
    print(f"  base url : {base}")
    print(f"  file size: {size_mb:.2f} MB (base64 {len(encoded) / 1024 / 1024:.2f} MB)")
    print("=" * 72)
    if len(raw) > MAX_UPLOAD_BYTES:
        fail(f"file exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB upload cap")

    # 1. preflight: verify auth and warm the worker
    started = time.time()

    def warm(tok):
        return request_json(f"{base}/api/warmup", tok, payload={})

    status, body = warm(token)
    if status == 401 and env.get("APP_PASSWORD"):
        # Cross-environment convenience: AUTH_SECRET may differ between this
        # machine's scope and the target one, but APP_PASSWORD usually matches.
        v_status, v_body = request_json(f"{base}/api/verify_password", token,
                                        payload={"password": env["APP_PASSWORD"]})
        if v_status == 200 and isinstance(v_body, dict) and v_body.get("token"):
            token = v_body["token"]
            log("auth", "locally minted token rejected; using one issued by the target server")
            status, body = warm(token)

    if status == 401:
        fail("authentication failed - wrong password/AUTH_SECRET for this environment", body)
    if status == 0:
        fail(f"cannot reach {base} - start the dev server (`vercel dev`) or pass --base-url", body)
    if status != 200:
        fail(f"preflight failed ({status})", body)
    log("preflight", f"ok ({time.time() - started:.1f}s) - worker warmup requested")

    # 2. submit
    t_submit = time.time()
    submit_payload = {"file_content": encoded, "file_type": MIME_BY_SUFFIX[suffix]}
    if args.provider:
        submit_payload["provider"] = args.provider
    status, body = request_json(f"{base}/api/generate_flashcards", token,
                                payload=submit_payload)
    if status == 0:
        fail(f"cannot reach {base}", body)
    if status == 400 or not isinstance(body, dict):
        fail(f"submission rejected ({status})", body)
    if not body.get("job_id"):
        fail(f"unexpected submit response ({status})", body)
    job_id = body["job_id"]
    total_chunks = body.get("chunks") or 0
    log("submit", f"queued job {job_id[:8]} in {time.time() - t_submit:.1f}s"
        + (f" - {total_chunks} section group(s)" if total_chunks else ""))

    # 2b. The API engines are driven one section group at a time (each request
    # stays inside the serverless time limit). The app does this from the browser;
    # the test has to do it too or the job never progresses.
    if total_chunks:
        for i in range(total_chunks):
            c_status, c_body = request_json(f"{base}/api/generate_flashcards", token,
                                            payload={"job_id": job_id, "chunk": i},
                                            timeout=args.chunk_timeout)
            if isinstance(c_body, dict) and (c_body.get("quota") or c_body.get("unavailable")):
                log("engine", "API engines unavailable - falling back to Modal")
                request_json(f"{base}/api/generate_flashcards", token,
                             payload={"job_id": job_id, "provider": "qwen"})
                break
            if c_status != 200:
                fail(f"section group {i + 1}/{total_chunks} failed ({c_status})", c_body)
            log("chunk", f"{i + 1}/{total_chunks} done "
                f"({time.time() - t_submit:.0f}s) - {c_body.get('cards', '?')} card(s)")
            if c_body.get("final"):
                break

    # 3. poll
    last_progress = None
    first_processing = None
    chunk_times = []
    last_activity = time.time()
    current_state = ""
    cancelled = False

    while True:
        elapsed = time.time() - t_submit
        if elapsed > MAX_POLL_SECONDS:
            fail(f"timed out after {elapsed:.0f}s (job {job_id[:8]} may still finish)")
        if current_state == "processing" and time.time() - last_activity > STALL_AFTER_SECONDS:
            fail(f"generation stalled for {STALL_AFTER_SECONDS}s (job {job_id[:8]}) - check the Modal logs")

        if args.cancel_after and not cancelled and elapsed > args.cancel_after:
            c_status, c_body = request_json(f"{base}/api/cancel", token, payload={"job_id": job_id})
            log("cancel", f"requested ({c_status})")
            cancelled = True

        status, data = request_json(f"{base}/api/status?job_id={job_id}", token)
        if status == 401:
            fail("token rejected mid-run (expired or wrong AUTH_SECRET)")
        if status == 404:
            fail("job disappeared from Redis (expired?)")
        if status != 200 or not isinstance(data, dict):
            log("poll", f"transient status {status}, retrying")
            time.sleep(args.poll_seconds)
            continue

        state = data.get("status")
        current_state = state or ""
        progress = data.get("progress") or ""

        if state == "queued":
            if elapsed > 30:
                log("wait", f"worker starting (cold start) - {elapsed:.0f}s elapsed")
        elif state == "processing":
            if first_processing is None:
                first_processing = time.time()
                log("worker", f"started processing {elapsed:.1f}s after submit")
            if progress != last_progress:
                last_progress = progress
                last_activity = time.time()
                chunk_times.append((time.time(), progress))
                log("progress", f"chunk {progress} ({time.time() - t_submit:.0f}s)")
        elif state == "complete":
            total = time.time() - t_submit
            cards = data.get("cards") or []
            log("done", f"complete in {total:.1f}s - {len(cards)} cards")
            break
        elif state == "cancelled":
            log("done", f"job cancelled after {elapsed:.1f}s (cancel flag honoured)")
            print("\nRESULT: cancel path works.")
            return 0
        elif state == "error":
            fail("job failed", data)

        time.sleep(args.poll_seconds)

    # 4. sanity-check the cards
    if not cards:
        fail("job completed but returned zero cards")

    questions = [c.get("question", "") for c in cards]
    normalized = [q.strip().lower() for q in questions]
    duplicates = len(normalized) - len(set(normalized))
    answers = [len(c.get("answer", "")) for c in cards]
    empty = sum(1 for c in cards if not c.get("question") or not c.get("answer"))

    log("cards", f"{len(cards)} cards, {duplicates} exact duplicates, {empty} empty fields")
    log("cards", f"answer length min/avg/max: {min(answers)}/{sum(answers) // len(answers)}/{max(answers)} chars")

    quoted = sum(1 for c in cards if (c.get("source_quote") or "").strip())
    difficulty_mix = {}
    for card in cards:
        key = (card.get("difficulty") or "?").lower()
        difficulty_mix[key] = difficulty_mix.get(key, 0) + 1
    log("cards", f"source quotes: {quoted}/{len(cards)}; difficulty: {difficulty_mix}")
    if quoted < len(cards):
        log("warn", f"{len(cards) - quoted} ungrounded card(s) survived the worker filter - investigate")

    print("\n  sample cards:")
    for card in cards[:3]:
        print(f"    Q: {card['question'][:100]}")
        print(f"    A: {card['answer'][:120]}...")
        if card.get("source_quote"):
            print(f"    📄 {card['source_quote'][:120]}...")

    if empty:
        fail(f"{empty} card(s) have empty question/answer")
    if duplicates:
        log("warn", f"{duplicates} exact-duplicate question(s) survived dedup")

    # 5. TTS
    if args.tts_sample:
        print()
        for card in cards[:args.tts_sample]:
            text = card["question"][:600]
            t0 = time.time()
            status, content_type, audio = request_bytes(f"{base}/api/tts", token, {"text": text})
            if status != 200:
                body = audio.decode("utf-8", "replace")[:400] if isinstance(audio, bytes) else audio
                fail(f"tts returned {status}", body)
            info = wav_info(audio)
            if info is None:
                fail("tts response is not a valid WAV", audio[:16])
            log("tts", f"{content_type} {len(audio) / 1024:.0f} KB, "
                       f"{info['seconds']}s, {info['rate']} Hz, "
                       f"{info['channels']}ch - in {time.time() - t0:.1f}s")
        log("tts", "cache check: re-requesting the same clip")
        t0 = time.time()
        status, _, audio2 = request_bytes(f"{base}/api/tts", token, {"text": cards[0]["question"][:600]})
        log("tts", f"cached response in {time.time() - t0:.2f}s ({len(audio2) / 1024:.0f} KB)")

    print("\n" + "=" * 72)
    print("PIPELINE TEST PASSED")
    if first_processing:
        print(f"  cold/warm start: {first_processing - t_submit:.1f}s")
        print(f"  generation    : {time.time() - first_processing:.1f}s")
    print(f"  total         : {time.time() - t_submit:.1f}s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
