# api/warmup.py
# Best-effort pre-warm: starts the Modal GPU container booting while the user
# is still choosing a file, hiding cold-start latency behind human behaviour.
import hashlib
import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler

MODAL_APP_NAME = "flashcard-generator"
MODAL_CLASS_NAME = "FlashcardLLM"


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


def lambda_handler(event):
    try:
        if event["httpMethod"] != "POST":
            return {
                "statusCode": 405,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Method not allowed"})
            }

        if not validate_auth_token(event):
            return {
                "statusCode": 401,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Unauthorized"})
            }

        try:
            import modal
            cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
            cls().ping.spawn()
        except Exception as exc:
            # Warmup failing must never break the app; the real job still works.
            print("Warmup dispatch failed:", exc)
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Warmup skipped"})
            }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"success": True})
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
