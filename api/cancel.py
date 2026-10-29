# api/cancel.py
# Sets a cancel flag that BOTH engines honour: the Modal worker checks it between
# chunks, and /api/generate_flashcards checks it before each section group, so an
# abandoned job stops burning GPU and API credit quickly.
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_flashcards import (
    JOB_TTL_SECONDS,
    json_response,
    redis_client,
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
            body = json.loads(event["body"])
            job_id = str(body["job_id"]).strip()
            if not job_id:
                raise KeyError("job_id")
        except (KeyError, TypeError, ValueError, AttributeError):
            return json_response(400, {"success": False, "error": "Missing job_id"})

        r = redis_client()
        r.set(f"job:{job_id}:cancelled", "1", ex=JOB_TTL_SECONDS)
        # Mirror it into status so a client that is polling sees the job stop
        # instead of watching progress freeze.
        if r.get(f"job:{job_id}:status") in ("queued", "processing"):
            r.set(f"job:{job_id}:status", "cancelled", ex=JOB_TTL_SECONDS)

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
