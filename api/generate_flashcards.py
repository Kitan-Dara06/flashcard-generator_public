# api/generate_flashcards.py
# Async dispatcher: extracts text, stores the job in Redis, spawns the
# Modal worker, and returns a job_id immediately.
import base64
import hashlib
import hmac
import io
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler

import pdfplumber

MAX_PAGES = 100
MAX_TEXT_CHARS = 500_000
JOB_TTL_SECONDS = 24 * 60 * 60
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


# ----- Redis + Modal -----
def redis_client():
    from upstash_redis import Redis
    return Redis(
        url=os.environ["UPSTASH_REDIS_REST_URL"],
        token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
    )


def dispatch_job(job_id: str):
    """Fire-and-forget the Modal worker; returns immediately."""
    import modal
    cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    cls().process_job.spawn(job_id)


# ----- Text extraction -----
def extract_text_from_pdf(file_content, max_pages=MAX_PAGES):
    try:
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            total_pages = len(pdf.pages)
            if total_pages > max_pages:
                return {
                    "error": "document_too_long",
                    "message": f"Your document has {total_pages} pages. Max allowed: {max_pages}.",
                    "page_count": total_pages,
                    "max_allowed": max_pages
                }
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        return {
            "error": "pdf_extraction_failed",
            "message": f"Error reading PDF: {str(e)}"
        }


def extract_text_from_docx(file_content):
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_content))
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        return {
            "error": "docx_extraction_failed",
            "message": f"Error reading DOCX: {str(e)}"
        }


def extract_text_from_pptx(file_content):
    try:
        from pptx import Presentation
        prs = Presentation(io.BytesIO(file_content))
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
        return "\n".join(text)
    except Exception as e:
        return {
            "error": "pptx_extraction_failed",
            "message": f"Error reading PPTX: {str(e)}"
        }


# ----- Core handler logic -----
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
                "body": json.dumps({"success": False, "error": "Unauthorized. Please enter the password again."})
            }

        headers = {k.lower(): v for k, v in event["headers"].items()}
        content_type = headers.get("content-type", "")
        if "application/json" not in content_type:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Content-Type must be application/json"})
            }

        # Parse JSON body
        try:
            body = json.loads(event["body"])
            file_content = base64.b64decode(body["file_content"])
            file_type = body["file_type"]
        except (KeyError, TypeError, ValueError):
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Malformed request body"})
            }

        # Extract text
        if file_type == "application/pdf":
            text = extract_text_from_pdf(file_content)
        elif file_type == "text/plain":
            try:
                text = file_content.decode("utf-8")
            except UnicodeDecodeError:
                return {
                    "statusCode": 400,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"success": False, "error": "Text file must be UTF-8 encoded"})
                }
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(file_content)
        elif file_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            text = extract_text_from_pptx(file_content)
        else:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Unsupported file type"})
            }

        if isinstance(text, dict):  # extraction helpers return an error dict on failure
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, **text})
            }

        if not text.strip():
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Could not extract text from file"})
            }

        if len(text) > MAX_TEXT_CHARS:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "success": False,
                    "error": "document_too_long",
                    "message": f"Extracted text is too long ({len(text):,} chars). Limit: {MAX_TEXT_CHARS:,}. Try splitting the document."
                })
            }

        # Queue the job
        job_id = uuid.uuid4().hex
        r = redis_client()
        r.set(f"job:{job_id}:text", text, ex=JOB_TTL_SECONDS)
        r.set(f"job:{job_id}:status", "queued", ex=JOB_TTL_SECONDS)
        r.set(f"job:{job_id}:progress", "0/0", ex=JOB_TTL_SECONDS)

        try:
            dispatch_job(job_id)
        except Exception as exc:
            r.set(f"job:{job_id}:error", f"Dispatch failed: {exc}", ex=JOB_TTL_SECONDS)
            r.set(f"job:{job_id}:status", "error", ex=JOB_TTL_SECONDS)
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "success": False,
                    "error": "Could not start the AI worker. Please try again.",
                    "details": str(exc)
                })
            }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"success": True, "job_id": job_id, "status": "queued"})
        }

    except Exception as e:
        import traceback
        print("Unhandled error:", str(e))
        print(traceback.format_exc())
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "success": False,
                "error": str(e),
                "trace": traceback.format_exc()
            })
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
