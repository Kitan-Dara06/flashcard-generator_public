# api/podcast_audio.py
# Podcast audio via Deepgram Aura. One spoken line per request, one voice per
# line, so the two hosts alternate by the speaker name the client sends.
#
# Body: { "speaker": "Alex"|"Sam", "text": "..." }  ->  audio/wav
#
# Why not the previous Gemini multi-speaker path: it can render both hosts in a
# single call, but the account is blocked by a *monthly spending cap*, which
# does not clear by retrying. Deepgram is per-character and per-line, which also
# means each line is cached and retried independently.
#
# Auth note: Deepgram uses `Authorization: Token <key>`, NOT `Bearer`. Sending
# Bearer returns 401 and looks exactly like a bad key.
import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_flashcards import json_response, resolve_cors_headers

DEEPGRAM_BASE = os.environ.get("DEEPGRAM_BASE_URL", "https://api.deepgram.com")
SAMPLE_RATE = 24000

# The two hosts. The names must match what podcast_script.py generates and what
# the transcript renderer displays; the models are Deepgram's canonical names.
VOICE_BY_SPEAKER = {
    "Alex": os.environ.get("PODCAST_VOICE_ALEX", "aura-2-orion-en"),
    "Sam": os.environ.get("PODCAST_VOICE_SAM", "aura-2-draco-en"),
}

LINE_MAX_CHARS = 1200       # one turn of dialogue; well inside a spoken sentence pair
DEEPGRAM_TIMEOUT_SECONDS = int(os.environ.get("DEEPGRAM_TIMEOUT_SECONDS", "45"))


class DeepgramError(Exception):
    """A Deepgram failure, already sorted into retryable or not.

    The distinction drives the client: a burst limit clears in seconds, an auth
    or credit failure never will, and retrying the second one only wastes the
    request budget.
    """

    def __init__(self, message, retryable):
        super().__init__(message)
        self.retryable = retryable


def synthesize(text, speaker="Alex"):
    """Returns (wav_bytes, duration_seconds) for one spoken line."""
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise DeepgramError(
            "DEEPGRAM_API_KEY is not set. Add it to this environment's variables "
            "in Vercel and redeploy.", retryable=False)

    model = VOICE_BY_SPEAKER.get(speaker) or VOICE_BY_SPEAKER["Alex"]
    url = (f"{DEEPGRAM_BASE}/v1/speak?model={model}"
           f"&encoding=linear16&sample_rate={SAMPLE_RATE}&container=wav")
    request = urllib.request.Request(
        url,
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={
            # `Token`, not `Bearer` - see the note at the top of this file.
            "Authorization": f"Token {key}",
            "Content-Type": "application/json",
            # Explicit so it is obvious which service a log line refers to.
            "User-Agent": "flashcard-generator/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=DEEPGRAM_TIMEOUT_SECONDS) as resp:
            wav = resp.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise DeepgramError(_describe(exc.code, detail), retryable=_is_transient(exc.code, detail))
    except urllib.error.URLError as exc:
        raise DeepgramError(f"could not reach Deepgram: {exc.reason}", retryable=True)
    except (TimeoutError, OSError) as exc:
        raise DeepgramError(f"Deepgram request failed: {exc}", retryable=True)

    if not wav or len(wav) <= 44:
        raise DeepgramError("Deepgram returned no audio", retryable=True)

    duration = (len(wav) - 44) / (SAMPLE_RATE * 2)   # 16-bit mono, minus the header
    return wav, duration


def _is_transient(status, detail):
    """Retryable = a burst limit or a transport fault, not a credential problem."""
    if status in (408, 429, 500, 502, 503, 504):
        return True
    if status in (401, 403):
        return False
    text = (detail or "").lower()
    return any(marker in text for marker in
               ("timed out", "temporarily", "unavailable", "overloaded", "rate limit"))


def _describe(status, detail):
    """A message a student can act on, because the raw error rarely is."""
    text = (detail or "").lower()
    if status == 401:
        return ("Deepgram rejected the API key (401). Check that DEEPGRAM_API_KEY is "
                "set for this environment - requests must use the Token scheme, not "
                "Bearer.")
    if status in (402, 403) or "insufficient" in text or "balance" in text or "credit" in text:
        return ("Deepgram is out of credit for this account (HTTP "
                f"{status}). Podcast audio needs credit; card read-aloud uses a "
                "different engine and still works.")
    if status == 429:
        return "Deepgram is rate limiting this account (429). Try again shortly."
    return f"Deepgram request failed ({status}): {detail}"


def lambda_handler(event):
    try:
        if event["httpMethod"] != "POST":
            return json_response(405, {"success": False, "error": "Method not allowed"})

        try:
            body = json.loads(event["body"])
            text = str(body.get("text", "")).strip()
            speaker = str(body.get("speaker", "Alex")).strip()
        except (TypeError, ValueError):
            return json_response(400, {"success": False, "error": "Malformed request body"})

        if not text:
            return json_response(400, {"success": False, "error": "Missing text"})
        if len(text) > LINE_MAX_CHARS:
            return json_response(400, {"success": False,
                                       "error": f"Line too long (max {LINE_MAX_CHARS} chars)"})

        try:
            wav, duration = synthesize(text, speaker)
        except DeepgramError as exc:
            print("TTS failed:", exc)
            return json_response(503 if exc.retryable else 502,
                                {"success": False, "retryable": exc.retryable,
                                 "error": str(exc)})
        except Exception as exc:
            print("TTS failed:", repr(exc))
            return json_response(502, {"success": False, "retryable": True,
                                       "error": "TTS failed", "details": str(exc)})

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "audio/wav",
                "Content-Length": str(len(wav)),
                "X-Audio-Duration": "%.2f" % duration,
                "X-Audio-Voice": VOICE_BY_SPEAKER.get(speaker, VOICE_BY_SPEAKER["Alex"]),
            },
            "body": wav,
        }

    except Exception as e:
        import traceback
        print("Unhandled error:", str(e))
        print(traceback.format_exc())
        return json_response(500, {"success": False, "error": str(e)})


class handler(BaseHTTPRequestHandler):
    def _send(self, response):
        headers = response.get("headers", {})
        body = response["body"]
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(response["statusCode"])
        for k, v in headers.items():
            self.send_header(k, v)
        if "Content-Length" not in headers:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        event = {"httpMethod": "POST", "headers": dict(self.headers), "body": body}
        response = lambda_handler(event)
        response.setdefault("headers", {}).update(resolve_cors_headers(event))
        self._send(response)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
