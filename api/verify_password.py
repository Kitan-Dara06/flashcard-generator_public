# api/verify_password.py
import hashlib
import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler

TOKEN_TTL_SECONDS = 24 * 60 * 60


def auth_secret():
    """Secret used to sign auth tokens (falls back to APP_PASSWORD)."""
    return os.environ.get("AUTH_SECRET") or os.environ.get("APP_PASSWORD", "")


def create_auth_token():
    """Create an expiring HMAC-signed token for authenticating API calls."""
    expires = int(time.time()) + TOKEN_TTL_SECONDS
    signature = hmac.new(auth_secret().encode(), str(expires).encode(), hashlib.sha256).hexdigest()
    return f"{expires}.{signature}", expires


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

        correct_password = os.environ.get("APP_PASSWORD")

        if not correct_password:
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Password not configured"})
            }

        # Parse the request body
        body = json.loads(event["body"])
        entered_password = str(body.get("password", ""))

        # Check password (constant-time comparison)
        if hmac.compare_digest(entered_password, correct_password):
            token, expires_at = create_auth_token()
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "success": True,
                    "authenticated": True,
                    "token": token,
                    "expires_at": expires_at
                })
            }
        else:
            return {
                "statusCode": 401,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "authenticated": False})
            }

    except Exception as e:
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
