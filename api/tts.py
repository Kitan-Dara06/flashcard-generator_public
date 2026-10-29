# api/tts.py
# Text-to-speech endpoint. Synthesizes one sentence (or short phrase) at a time
# via the co-located Piper voice on Modal, and caches the audio in Redis so
# re-studying a deck doesn't re-pay synthesis.
import base64
import hashlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_flashcards import (
    TTS_VOICE,
    json_response,
    redis_client,
    resolve_cors_headers,
    validate_auth_token,
)

MODAL_APP_NAME = "flashcard-generator"
MODAL_CLASS_NAME = "FlashcardLLM"

TTS_MAX_CHARS = 600                   # per request; the frontend sends one sentence
TTS_CACHE_TTL = 7 * 24 * 60 * 60      # 7 days
TTS_CACHE_MAX_BYTES = 400_000         # don't cache very long clips (Redis value limits)


def synthesize(text):
    """Blocking call to the co-located Piper voice on Modal."""
    import modal
    cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    return cls().tts.remote(text)


def cache_key(text):
    # The voice is part of the key: cached audio must never be served for a
    # different voice than the one that produced it.
    digest = hashlib.sha256((TTS_VOICE + "|" + text).encode("utf-8")).hexdigest()
    return f"tts:{digest}"


def _audio(audio):
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "audio/wav",
            "Content-Length": str(len(audio)),
            "Cache-Control": "private, max-age=86400",
        },
        "body": audio,
    }


def lambda_handler(event):
    try:
        if event["httpMethod"] != "POST":
            return json_response(405, {"success": False, "error": "Method not allowed"})

        if not validate_auth_token(event):
            return json_response(401, {"success": False, "error": "Unauthorized"})

        headers = {k.lower(): v for k, v in event["headers"].items()}
        if "application/json" not in headers.get("content-type", ""):
            return json_response(400, {"success": False,
                                       "error": "Content-Type must be application/json"})

        try:
            body = json.loads(event["body"])
            text = str(body.get("text", "")).strip()
        except (TypeError, ValueError):
            return json_response(400, {"success": False, "error": "Malformed request body"})

        if not text:
            return json_response(400, {"success": False, "error": "Missing text"})
        if len(text) > TTS_MAX_CHARS:
            return json_response(400, {"success": False,
                                       "error": f"Text too long (max {TTS_MAX_CHARS} chars)"})

        r = redis_client()
        key = cache_key(text)

        cached = r.get(key)
        if cached:
            return _audio(base64.b64decode(cached))

        try:
            audio = synthesize(text)
        except Exception as exc:
            print("TTS synthesis failed:", repr(exc))
            return json_response(502, {"success": False,
                                       "error": "Speech synthesis is unavailable. "
                                                "Check that the Modal app is deployed "
                                                "and that MODAL_TOKEN_ID/SECRET are set "
                                                "for this environment.",
                                       "details": str(exc)})

        if not audio:
            return json_response(500, {"success": False, "error": "No audio generated"})

        if len(audio) <= TTS_CACHE_MAX_BYTES:
            r.set(key, base64.b64encode(audio).decode("ascii"), ex=TTS_CACHE_TTL)

        return _audio(audio)

    except Exception as e:
        import traceback
        print("Unhandled error:", str(e))
        print(traceback.format_exc())
        return json_response(500, {"success": False, "error": str(e)})


# ----- Vercel entrypoint -----
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
        response_headers = {
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
        response_headers.update(resolve_cors_headers({"headers": dict(self.headers)}))
        self._send({"statusCode": 204, "headers": response_headers, "body": ""})
