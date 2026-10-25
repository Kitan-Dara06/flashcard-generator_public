# api/podcast_script.py
# Generates a two-host podcast script from a document's text.
# Body: { "text": "..." }  ->  { "title": "...", "turns": [{"speaker","text"}, ...] }
import json
import os
from http.server import BaseHTTPRequestHandler

SCRIPT_MODEL = os.environ.get("PODCAST_SCRIPT_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """You are writing a podcast conversation between two hosts, Alex and Sam, about the document provided.

Rules:
- Cover only the key concepts: the main ideas a student must remember. Skip tables of contents, bibliographies, headers, and low-value detail.
- Aim for a conversation that runs about 5-8 minutes when read aloud (roughly 700-1100 words).
- Tone: warm, clear and curious, like two knowledgeable hosts explaining the material to a student. Not a dry summary - actually discuss the ideas.
- Ground every claim in the document. Do not add outside facts, dates, names or context. If the document does not say it, do not say it.
- Alternate speakers naturally. Alex introduces and summarises; Sam asks the clarifying questions a student would ask and draws out the "why".
- Open with a one-sentence hook and close with a short "if you remember three things" wrap-up.
- Return JSON only."""


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
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=schema(),
        temperature=0.7,
        max_output_tokens=8192,
    )
    response = client.models.generate_content(
        model=SCRIPT_MODEL, contents=text, config=config)
    data = json.loads(response.text)
    turns = [t for t in data.get("turns", []) if t.get("speaker") in ("Alex", "Sam") and str(t.get("text", "")).strip()]
    return {"title": data.get("title", "Podcast"), "turns": turns}


def _json(status, payload):
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(payload)}


def lambda_handler(event):
    try:
        if event["httpMethod"] != "POST":
            return _json(405, {"success": False, "error": "Method not allowed"})

        try:
            body = json.loads(event["body"])
            text = str(body.get("text", "")).strip()
        except (TypeError, ValueError):
            return _json(400, {"success": False, "error": "Malformed request body"})

        if not text:
            return _json(400, {"success": False, "error": "Missing text"})
        if len(text) > 500_000:
            return _json(400, {"success": False, "error": "Text too long"})

        try:
            result = generate_script(text)
        except Exception as exc:
            print("Script generation failed:", repr(exc))
            return _json(502, {"success": False, "error": "Script generation failed", "details": str(exc)})

        if not result["turns"]:
            return _json(502, {"success": False, "error": "No turns generated"})

        return _json(200, {"success": True, **result})

    except Exception as e:
        import traceback
        print("Unhandled error:", str(e))
        print(traceback.format_exc())
        return _json(500, {"success": False, "error": str(e)})


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
        response = lambda_handler({"httpMethod": "POST", "headers": dict(self.headers), "body": body})
        self._send(response)
