# api/status.py
# Read-only job status endpoint; answers in ~10 ms from Redis.
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The HTTP + Redis helpers live in one place: the generation module. Imports are
# lazy at module level (nothing heavy runs), so this costs one file read.
from generate_flashcards import (
    json_response,
    redis_client,
    resolve_cors_headers,
    validate_auth_token,
)

# A job that has been sitting in "queued" this long never got picked up by the
# worker (boot failure, crash before the first status write, ...). Reported as an
# error so clients stop polling forever - 10 min is comfortably above a slow
# first cold boot (image pull + model load + snapshot).
QUEUED_STALE_SECONDS = 10 * 60


def lambda_handler(event):
    try:
        if event["httpMethod"] != "GET":
            return json_response(405, {"success": False, "error": "Method not allowed"})

        if not validate_auth_token(event):
            return json_response(401, {"success": False, "error": "Unauthorized"})

        job_id = (event.get("queryStringParameters") or {}).get("job_id", "")
        if not job_id:
            return json_response(400, {"success": False, "error": "Missing job_id"})

        r = redis_client()
        status = r.get(f"job:{job_id}:status")
        if not status:
            return json_response(404, {"success": False, "error": "Job not found or expired"})

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
            # The worker never picked the job up. Report it and let the client
            # decide: it can re-submit the job_id to /api/generate_flashcards to
            # run it on the other engine, which lives in Vercel and so still
            # works when Modal itself is the broken part.
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
            payload["provider"] = r.get(f"job:{job_id}:provider") or ""
            note = r.get(f"job:{job_id}:note")
            if note:
                payload["note"] = note
        elif status == "error":
            payload["error"] = r.get(f"job:{job_id}:error") or "Generation failed"

        return json_response(200, payload)

    except Exception as e:
        import traceback
        print("Unhandled error:", str(e))
        print(traceback.format_exc())
        return json_response(500, {"success": False, "error": str(e)})


# ----- Vercel entrypoint -----
class handler(__import__("http.server", fromlist=["BaseHTTPRequestHandler"]).BaseHTTPRequestHandler):
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

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        event = {"httpMethod": "GET", "headers": dict(self.headers),
                 "queryStringParameters": query, "body": ""}
        response = lambda_handler(event)
        response.setdefault("headers", {}).update(resolve_cors_headers(event))
        self._send(response)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
