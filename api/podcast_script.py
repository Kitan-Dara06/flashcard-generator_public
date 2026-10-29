# api/podcast_script.py
# Generates a two-host podcast script from a document's text.
# Body: { "text": "..." }  ->  { "title": "...", "turns": [{"speaker","text"}, ...] }
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The prompt is data (api/prompts/podcast.yaml); generation goes through the same
# engine chain as the cards, so a script still works when the Gemini key is out of
# credit and Zen is carrying the load.
from generate_flashcards import (
    EngineFailure,
    generate_json,
    json_response,
    prompt_text,
    resolve_cors_headers,
)

SCRIPT_MODEL = os.environ.get("PODCAST_SCRIPT_MODEL", "gemini-2.5-flash")


def schema():
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "enum": ["Alex", "Sam"]},
                        "text": {"type": "string"},
                    },
                    "required": ["speaker", "text"],
                },
            },
        },
        "required": ["title", "turns"],
    }


def generate_script(text: str) -> dict:
    _engine, raw = generate_json(prompt_text("podcast"), text, schema(),
                                 temperature=0.7)
    data = json.loads(raw)
    turns = [t for t in data.get("turns", [])
             if t.get("speaker") in ("Alex", "Sam") and str(t.get("text", "")).strip()]
    return {"title": data.get("title", "Podcast"), "turns": turns}


def lambda_handler(event):
    try:
        if event["httpMethod"] != "POST":
            return json_response(405, {"success": False, "error": "Method not allowed"})

        try:
            body = json.loads(event["body"])
            text = str(body.get("text", "")).strip()
        except (TypeError, ValueError):
            return json_response(400, {"success": False, "error": "Malformed request body"})

        if not text:
            return json_response(400, {"success": False, "error": "Missing text"})
        if len(text) > 500_000:
            return json_response(400, {"success": False, "error": "Text too long"})

        try:
            result = generate_script(text)
        except EngineFailure as exc:
            print("Script generation failed:", repr(exc))
            status = 503 if exc.kind in ("quota", "transient") else 502
            return json_response(status, {"success": False,
                                          "error": "Script generation failed",
                                          "details": str(exc)})
        except Exception as exc:
            print("Script generation failed:", repr(exc))
            return json_response(502, {"success": False,
                                       "error": "Script generation failed",
                                       "details": str(exc)})

        if not result["turns"]:
            return json_response(502, {"success": False, "error": "No turns generated"})

        return json_response(200, {"success": True, **result})

    except Exception as e:
        import traceback
        print("Unhandled error:", str(e))
        print(traceback.format_exc())
        return json_response(500, {"success": False, "error": str(e)})


class handler(BaseHTTPRequestHandler):
    def _send(self, response):
        body = response["body"]
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(response["statusCode"])
        for k, v in response.get("headers", {}).items():
            self.send_header(k, v)
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
