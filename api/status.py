# api/status.py
# Read-only job status endpoint; answers in ~10 ms from Redis.
import hashlib
import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


# ----- Auth -----
def auth_secret():
    """Secret used to sign auth tokens (falls back to APP_PASSWORD)."""
    return os.environ.get("AUTH_SECRET") or os.environ.get("APP_PASSWORD", "")


def validate_auth_token(event):
    """Validate the signed bearer token issued by /api/verify_password."""
    secret = auth_secret()
    if not secret:
        return False

    headers = {k.lower(): v for k, v in event.get("headers", {}).items()}
    auth = headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return False

    token = auth[len("bearer "):].strip()
    expires_str, sep, signature = token.partition(".")
    if not sep:
        return False

    try:
        if int(expires_str) < time.time():
            return False
    except ValueError:
        return False

    expected = hmac.new(secret.encode(), expires_str.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


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


def redis_client():
    from upstash_redis import Redis
    return Redis(
        url=os.environ["UPSTASH_REDIS_REST_URL"],
        token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
    )


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
        if status in ("queued", "processing"):
            # Lets the client tell "worker is slow" apart from "worker never started".
            created = r.get(f"job:{job_id}:created")
            try:
                payload["age_seconds"] = max(0, int(time.time()) - int(created))
            except (TypeError, ValueError):
                payload["age_seconds"] = None
        if status == "complete":
            payload["cards"] = json.loads(r.get(f"job:{job_id}:cards") or "[]")
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
