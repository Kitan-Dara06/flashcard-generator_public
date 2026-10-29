# api/warmup.py
# Best-effort pre-warm: starts the Modal GPU container booting while the user
# is still choosing a file, hiding cold-start latency behind human behaviour.
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_flashcards import (
    MODAL_APP_NAME,
    MODAL_CLASS_NAME,
    json_response,
    resolve_cors_headers,
    validate_auth_token,
)


def lambda_handler(event):
    try:
        if event["httpMethod"] != "POST":
            return json_response(405, {"success": False, "error": "Method not allowed"})

        if not validate_auth_token(event):
            return json_response(401, {"success": False, "error": "Unauthorized"})

        try:
            import modal
            cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
            cls().ping.spawn()
        except Exception as exc:
            # Warmup failing must never break the app; the real job still works.
            print("Warmup dispatch failed:", exc)
            return json_response(200, {"success": False, "error": "Warmup skipped"})

        return json_response(200, {"success": True})

    except Exception as e:
        import traceback
        print("Unhandled error:", str(e))
        print(traceback.format_exc())
        return json_response(500, {"success": False, "error": str(e)})


# ----- Vercel entrypoint -----
class handler(BaseHTTPRequestHandler):
    def _send(self, response):
        body = response["body"]
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(response["statusCode"])
        for k, v in response.get("headers", {}).items():
            self.send_header(k, v)
        if "Content-Length" not in response.get("headers", {}):
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
