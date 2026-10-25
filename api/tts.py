# api/tts.py
# Text-to-speech endpoint. Synthesizes one sentence (or short phrase) at a time
# via the co-located Piper voice on Modal, and caches the audio in Redis so
# re-studying a deck doesn't re-pay synthesis.
import base64
import hashlib
import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler

MODAL_APP_NAME = "flashcard-generator"
MODAL_CLASS_NAME = "FlashcardLLM"

TTS_VOICE = "en_US-lessac-medium"     # must match modal_app.py
TTS_MAX_CHARS = 600                   # per request; the frontend sends one sentence
TTS_CACHE_TTL = 7 * 24 * 60 * 60      # 7 days
TTS_CACHE_MAX_BYTES = 400_000         # don't cache very long clips (Redis value limits)


# ----- Auth -----
def auth_secret():
    """Secret used to sign auth tokens (falls back to APP_PASSWORD)."""
    return os.environ.get("AUTH_SECRET") or os.environ.get("APP_PASSWORD", "")


def validate_auth_token(event):
    # Auth disabled: single-user app, no accounts. Endpoints are public.
    return True


# ----- CORS -----
def resolve_cors_headers(event):
    """CORS headers for origins listed in ALLOWED_ORIGINS only.

    Same-origin requests don't need CORS headers, so leaving ALLOWED_ORIGINS
    unset keeps the app working while blocking other sites from reading the API.
    """
    headers = {k.lower(): v for k, v in event.get("headers", {}).items()}
    origin = headers.get("origin")
    if not origin:
        return {}
    allowed = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if origin in allowed:
        return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
    return {}


# ----- helpers -----
_REDIS_CLIENT = None


def redis_client():
    """Redis client.

    Credentials are resolved in this order:
      1. REDIS_URL - a full connection string, e.g.
         rediss://default:<password>@<host>:<port>/0?ssl_cert_reqs=required
      2. composed from UPSTASH_HOST + UPSTASH_PASSWORD (+ UPSTASH_PORT, default
         6379) - the fields the Upstash dashboard shows
      3. the Upstash REST pair (UPSTASH_REDIS_REST_URL + _TOKEN), which needs no
         TCP connection and is the leanest option for serverless

    Writers and readers must use the same transport (the clients encode values
    differently), so keep both sides on the same choice.
    """
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    url = os.environ.get("REDIS_URL")
    if not url:
        host = os.environ.get("UPSTASH_HOST") or os.environ.get("UPSTASH_ENDPOINT")
        password = os.environ.get("UPSTASH_PASSWORD")
        if host and password:
            from urllib.parse import quote
            port = os.environ.get("UPSTASH_PORT") or "6379"
            url = (f"rediss://default:{quote(password, safe='')}@{host}:{port}/0"
                   f"?ssl_cert_reqs=required")

    if url:
        import redis
        _REDIS_CLIENT = redis.from_url(url, decode_responses=True)
    else:
        from upstash_redis import Redis
        _REDIS_CLIENT = Redis(
            url=os.environ["UPSTASH_REDIS_REST_URL"],
            token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
        )
    return _REDIS_CLIENT


def synthesize(text: str) -> bytes:
    """Blocking call to the co-located Piper voice on Modal."""
    import modal
    cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    return cls().tts.remote(text)


def cache_key(text: str) -> str:
    digest = hashlib.sha256((TTS_VOICE + "|" + text).encode("utf-8")).hexdigest()
    return f"tts:{digest}"


def _json(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def _audio(audio: bytes):
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
            return _json(405, {"success": False, "error": "Method not allowed"})

        if not validate_auth_token(event):
            return _json(401, {"success": False, "error": "Unauthorized. Please enter the password again."})

        headers = {k.lower(): v for k, v in event["headers"].items()}
        if "application/json" not in headers.get("content-type", ""):
            return _json(400, {"success": False, "error": "Content-Type must be application/json"})

        try:
            body = json.loads(event["body"])
            text = str(body.get("text", "")).strip()
        except (TypeError, ValueError):
            return _json(400, {"success": False, "error": "Malformed request body"})

        if not text:
            return _json(400, {"success": False, "error": "Missing text"})
        if len(text) > TTS_MAX_CHARS:
            return _json(400, {"success": False, "error": f"Text too long (max {TTS_MAX_CHARS} chars)"})

        r = redis_client()
        key = cache_key(text)

        cached = r.get(key)
        if cached:
            return _audio(base64.b64decode(cached))

        try:
            audio = synthesize(text)
        except Exception as exc:
            return _json(502, {"success": False, "error": "TTS worker unavailable", "details": str(exc)})

        if not audio:
            return _json(500, {"success": False, "error": "No audio generated"})

        if len(audio) <= TTS_CACHE_MAX_BYTES:
            r.set(key, base64.b64encode(audio).decode("ascii"), ex=TTS_CACHE_TTL)

        return _audio(audio)

    except Exception as e:
        import traceback
        print("Unhandled error:", str(e))
        print(traceback.format_exc())
        return _json(500, {"success": False, "error": str(e)})


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
        event = {
            "httpMethod": "POST",
            "headers": dict(self.headers),
            "body": body
        }
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
