# api/status.py
# Read-only job status endpoint; answers in ~10 ms from Redis.
import hashlib
import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# A job that has been sitting in "queued" this long never got picked up by the
# worker (boot failure, crash before the first status write, ...). Reported as an
# error so clients stop polling forever - 10 min is comfortably above a slow
# first cold boot (image pull + model load + snapshot).
QUEUED_STALE_SECONDS = 10 * 60


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


def lambda_handler(event):
    try:
        if event["httpMethod"] != "GET":
            return {
                "statusCode": 405,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Method not allowed"})
            }

        if not validate_auth_token(event):
            return {
                "statusCode": 401,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Unauthorized. Please enter the password again."})
            }

        params = event.get("queryStringParameters") or {}
        job_id = params.get("job_id", "")
        if not job_id:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Missing job_id"})
            }

        r = redis_client()
        status = r.get(f"job:{job_id}:status")
        if not status:
            return {
                "statusCode": 404,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Job not found or expired"})
            }

        payload = {
            "success": True,
            "status": status,
            "progress": r.get(f"job:{job_id}:progress") or "",
        }

        age = None
        if status in ("queued", "processing"):
            # Lets the client tell "worker is slow" apart from "worker never started".
            created = r.get(f"job:{job_id}:created")
            try:
                age = max(0, int(time.time()) - int(created))
            except (TypeError, ValueError):
                age = None
            payload["age_seconds"] = age

        if status == "queued" and age is not None and age > QUEUED_STALE_SECONDS:
            # The GPU worker never picked the job up. Report it and let the client
            # decide: it can re-submit the job_id to /api/generate_flashcards to
            # run it on the Gemini fallback instead, which lives in Vercel and so
            # still works when Modal itself is the broken part.
            payload["status"] = "error"
            payload["stalled"] = True
            payload["fallback_available"] = True
            payload["error"] = (
                f"The AI worker never started (queued for {age // 60} min). "
                "It may be cold-starting unusually slowly, or it failed to boot - "
                "check `modal app logs flashcard-generator`."
            )
        if status == "complete":
            payload["cards"] = json.loads(r.get(f"job:{job_id}:cards") or "[]")
            # The source text rides along once so the client can persist it and
            # offer podcast / regenerate without re-uploading.
            payload["text"] = r.get(f"job:{job_id}:text") or ""
        elif status == "error":
            payload["error"] = r.get(f"job:{job_id}:error") or "Generation failed"

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(payload)
        }

    except Exception as e:
        import traceback
        print("Unhandled error:", str(e))
        print(traceback.format_exc())
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"success": False, "error": str(e)})
        }


# ----- Vercel entrypoint -----
class handler(BaseHTTPRequestHandler):
    def _send(self, response):
        self.send_response(response["statusCode"])
        for k, v in response.get("headers", {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(response["body"].encode())

    def do_GET(self):
        query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        event = {
            "httpMethod": "GET",
            "headers": dict(self.headers),
            "queryStringParameters": query,
            "body": ""
        }
        response = lambda_handler(event)
        response.setdefault("headers", {}).update(resolve_cors_headers(event))
        self._send(response)

    def do_OPTIONS(self):
        response_headers = {
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
        response_headers.update(resolve_cors_headers({"headers": dict(self.headers)}))
        self._send({"statusCode": 204, "headers": response_headers, "body": ""})
