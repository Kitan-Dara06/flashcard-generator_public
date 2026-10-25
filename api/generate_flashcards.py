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


# ----- Redis + Modal -----
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


def dispatch_job(job_id: str):
    """Fire-and-forget the Modal worker; returns immediately."""
    import modal
    cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    try:
        cls().process_job.spawn(job_id)
    except modal.exception.NotFoundError as exc:
        # The usual cause is MODAL_TOKEN_ID/SECRET belonging to a different
        # workspace than the one this app was deployed to.
        raise RuntimeError(
            f"{exc}. Check that MODAL_TOKEN_ID/MODAL_TOKEN_SECRET belong to the "
            f"workspace where you ran 'modal deploy {MODAL_APP_NAME}', that they are "
            f"set for the Production environment, and that you redeployed afterwards "
            f"(environment variables only take effect on new deployments)."
        ) from exc


# ----- Fallback engine (Gemini, called from Vercel) -----
#
# Why this runs here and not in Modal: it exists for "the Modal worker is not
# available" - out of credit, bad token, GPU class refusing to boot. Anything
# hosted in Modal dies together with Modal, so the fallback has to live outside
# it.
#
# Trade-off: a Vercel function is bounded (60 s on Hobby, 300 s on Pro), so this
# path refuses documents above GEMINI_MAX_CHARS and stops itself at
# GEMINI_TIME_BUDGET rather than being killed mid-answer.

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MAX_CHARS = int(os.environ.get("GEMINI_MAX_CHARS", "120000"))
GEMINI_CHUNK_CHARS = 60_000
GEMINI_TIME_BUDGET = float(os.environ.get("GEMINI_TIME_BUDGET_SECONDS", "240"))

# Keep in sync with SYSTEM_PROMPT in modal_app.py (they are separate deploy
# targets, so the text is duplicated rather than imported).
SYSTEM_PROMPT = """You are a flashcard generator with no world knowledge. Treat the text below as your ONLY source of truth.

Rules you must never break:
1. Never add background, definitions, examples, or context from your own knowledge - even if you are certain it is correct. If the text does not state it, it does not exist.
2. If the text does not explicitly answer a question, do not write that card at all. There is no partial credit for a plausible guess.
3. Every factual claim in the answer must be supported by the source_quote. If the answer needs a fact the quote lacks, remove the fact rather than supplying it yourself.
4. The source_quote MUST be copied character-for-character from the text. Do not paraphrase it, summarise it, fix its spelling, or join sentences from different places.
5. You must include source_quote on every card, plus a difficulty of easy, medium or hard.

Produce between 15 and 40 flashcards depending on how much the text supports - quality over quantity, and never pad the list with restatements of the same fact.
If the text contains no usable information, return an empty list of flashcards.
Do not explain, apologise, or return any text outside the JSON object."""


def flashcard_schema():
    return {
        "type": "object",
        "properties": {
            "flashcards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "source_quote": {"type": "string"},
                        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    },
                    "required": ["question", "answer", "source_quote", "difficulty"],
                },
            }
        },
        "required": ["flashcards"],
    }


def norm(text):
    import re
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def quote_is_grounded(quote, chunk_norm, threshold=0.85):
    """Is this quote really in the source text?

    An invented quote is the tell for parametric leakage: a plausible sentence
    attributed to a document that never said it.
    """
    quote_norm = norm(quote)
    if len(quote_norm) < 12:
        return False
    if quote_norm in chunk_norm:
        return True
    quote_words = quote_norm.split()
    if not quote_words:
        return False
    chunk_words = set(chunk_norm.split())
    coverage = sum(1 for w in quote_words if w in chunk_words) / len(quote_words)
    return coverage >= threshold


def answer_numbers_grounded(answer, chunk_norm):
    """Every number in the answer must appear in the source text."""
    import re
    for token in re.findall(r"\d[\d.,]*", answer):
        token = token.rstrip(".,")
        if token and token not in chunk_norm:
            return False
    return True


# Words too common to say anything about whether the answer is supported.
STOPWORDS = frozenset("""
about above after again also although among any are because been before being
below between both but can could did does doing down during each either few for
from further had has have having here how into its itself just more most much
not now off once only other our out over own same she should some such than that
the their them then there these they this those through too under until very was
were what when where which while who whom why will with would you your
""".split())

# How much of an answer's vocabulary must appear in its own source_quote.
# Deliberately low: audit_cards.py shows real answers cluster around 0.9, so a high
# cut-off drops healthy cards. This is a floor for gross mismatch only.
ANSWER_QUOTE_MIN_COVERAGE = float(os.environ.get("ANSWER_QUOTE_MIN_COVERAGE", "0.50"))


def capitalized_tokens(text):
    """Capitalised words that are not merely capitalised for sentence position."""
    import re
    for match in re.finditer(r"[A-Za-z][A-Za-z'’\-]*", text or ""):
        word = match.group(0)
        if not word[0].isupper():
            continue
        before = (text[:match.start()] or "").rstrip()
        if not before or before[-1] in ".:;!?-\n(\"“'":
            continue      # sentence-initial or list item: not evidence
        yield word


def ungrounded_names(answer, quote):
    """Names in the answer that its own quote never mentions.

    ADVISORY ONLY - this is reported, never used to drop a card.

    Capitalisation is a proxy for "named entity" at best: it misfires on
    abbreviations, adjectives derived from names ("Nigerian" vs "Nigeria"),
    and sentences the model starts mid-string. Dropping cards on it silently
    shrinks the deck, so the real grounding gates remain the verbatim-quote
    check and the number check.
    """
    quote_norm = norm(quote)
    quote_words = set(content_words(quote))
    missing = []
    for word in capitalized_tokens(answer):
        lower = word.lower()
        if len(lower) < 3 or lower in STOPWORDS:
            continue
        if lower in quote_norm:
            continue
        # "Nigerian" should count as covered by "Nigeria".
        if len(lower) >= 5 and any(w.startswith(lower[:5]) for w in quote_words
                                   if len(w) >= 5):
            continue
        missing.append(word)
    return missing


def content_words(text):
    import re
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in STOPWORDS]


def answer_coverage(answer, quote):
    """Fraction of the answer's content words that its own quote contains.

    Catches the remaining leak: a quote that is genuinely in the document, but
    an answer that quietly adds claims the quote never made.
    """
    answer_words = content_words(answer)
    if not answer_words:
        return 1.0
    quote_words = set(content_words(quote))
    if not quote_words:
        return 0.0
    return sum(1 for w in answer_words if w in quote_words) / len(answer_words)


def clean_cards(items, chunk=""):
    """Normalise cards and drop anything not grounded in `chunk`. Mirrors
    _clean_cards in modal_app.py so both engines enforce the same rule."""
    chunk_norm = norm(chunk) if chunk else ""
    cards = []
    dropped_ungrounded = 0
    dropped_leaky = 0
    dropped_drifted = 0
    flagged_names = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        quote = str(item.get("source_quote", "")).strip()
        difficulty = str(item.get("difficulty", "")).strip().lower()
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "medium"
        if not (question and answer):
            continue
        if not quote:
            dropped_ungrounded += 1
            continue
        if chunk_norm and not quote_is_grounded(quote, chunk_norm):
            dropped_leaky += 1
            continue
        if chunk_norm and not answer_numbers_grounded(answer, chunk_norm):
            dropped_leaky += 1
            continue
        if chunk_norm:
            # Reported, never enforced - see ungrounded_names().
            names = ungrounded_names(answer, quote)
            if names:
                flagged_names += 1
                if flagged_names <= 5:
                    print(f"  advisory: name(s) not in this card's quote: {names[:4]}")
        # Short answers cannot drift far; only judge once there is enough of an
        # answer to be worth judging.
        if len(content_words(answer)) > 6 and answer_coverage(answer, quote) < ANSWER_QUOTE_MIN_COVERAGE:
            dropped_drifted += 1
            continue
        cards.append({
            "question": question,
            "answer": answer,
            "source_quote": quote,
            "difficulty": difficulty,
        })
    if dropped_ungrounded:
        print(f"dropped {dropped_ungrounded} card(s) with no source_quote")
    if dropped_leaky:
        print(f"dropped {dropped_leaky} card(s) whose quote/numbers are not in the source")
    if dropped_drifted:
        print(f"dropped {dropped_drifted} card(s) whose answer drifts beyond its quote")
    if flagged_names:
        print(f"advisory: {flagged_names} card(s) mention a name absent from their quote "
              f"(kept - review manually)")
    return cards, dropped_ungrounded + dropped_leaky + dropped_drifted


def parse_cards(raw):
    """Parse the model's JSON, salvaging complete cards from a truncated reply."""
    import re
    try:
        return list(json.loads(raw).get("flashcards") or [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    decoder = json.JSONDecoder()
    pos = raw.find("[")
    if pos < 0:
        return []
    pos += 1
    cards = []
    while True:
        brace = raw.find("{", pos)
        if brace < 0:
            break
        try:
            obj, end = decoder.raw_decode(raw, brace)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            cards.append(obj)
        pos = end
    return cards


def paragraph_chunks(text, max_chars=GEMINI_CHUNK_CHARS, max_chunks=6):
    """Split on paragraph boundaries into at most max_chunks pieces.

    The char budget is raised when capping at max_chunks would otherwise force a
    chunk over it: with Gemini's wide window, fewer/larger chunks are cheaper and
    faster than many small calls, so max_chunks wins and max_chars is the floor.
    """
    import re
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return []

    budget = max(max_chars, (len(text) + max_chunks - 1) // max_chunks)

    sized = []
    for block in blocks:
        while len(block) > budget:
            cut = block.rfind(" ", 0, budget)
            cut = cut if cut > budget // 2 else budget
            sized.append(block[:cut])
            block = block[cut:].strip()
        if block:
            sized.append(block)

    chunks = []
    current = []
    current_len = 0
    for block in sized:
        block_len = len(block)
        if current and current_len + block_len + 2 > budget:
            chunks.append("\n\n".join(current))
            current, current_len = [block], block_len
        else:
            current.append(block)
            current_len += block_len + 2
    if current:
        chunks.append("\n\n".join(current))

    # Pathological input (thousands of tiny paragraphs) can still exceed the cap;
    # fold neighbours together rather than dropping text.
    while len(chunks) > max_chunks:
        merged = []
        for i in range(0, len(chunks), 2):
            merged.append("\n\n".join(chunks[i:i + 2]))
        chunks = merged
    return chunks


def text_dedup(cards, threshold=0.75, answer_threshold=0.85):
    """Drop near-duplicate cards.

    Two cards are duplicates if their questions are near-identical, or if their
    answers are: a document that repeats one fact gets asked about it from
    several angles, and question-only matching misses those.
    """
    import re
    if len(cards) <= 1:
        return cards

    def words(text):
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}

    kept = []
    kept_q = []
    kept_a = []
    for card in cards:
        q = words(card["question"])
        a = words(card.get("answer", ""))
        if any(q and other and len(q & other) / len(q | other) >= threshold
               for other in kept_q):
            continue
        if any(a and other and len(a & other) / len(a | other) >= answer_threshold
               for other in kept_a):
            continue
        kept.append(card)
        kept_q.append(q)
        kept_a.append(a)
    dropped = len(cards) - len(kept)
    if dropped:
        print(f"dedup: dropped {dropped} near-duplicate card(s)")
    return kept


def gemini_generate_cards(text, on_progress=None):
    """Generate grounded cards with Gemini. Blocking; bounded by the time budget."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to the Vercel project's Production "
            "environment variables and redeploy."
        )

    client = genai.Client(api_key=api_key)
    chunks = paragraph_chunks(text)
    started = time.time()
    cards = []

    for i, chunk in enumerate(chunks, 1):
        if time.time() - started > GEMINI_TIME_BUDGET:
            print(f"stopping at chunk {i}/{len(chunks)}: time budget reached")
            break

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            # Native structured output: the API enforces the schema, so
            # source_quote is always present (it can still be empty or invented,
            # which is exactly what clean_cards catches).
            response_mime_type="application/json",
            response_schema=flashcard_schema(),
            temperature=0.1,
            max_output_tokens=8192,
            # "Thinking" is a latency and cost multiplier that buys nothing for
            # extraction from a provided document.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=chunk, config=config)
        except Exception as exc:
            # Not every model/API version accepts thinking_budget=0. Retry once
            # without it rather than failing the whole document.
            if "thinking" not in str(exc).lower():
                raise
            print(f"thinking_config rejected ({exc}); retrying without it")
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=flashcard_schema(),
                temperature=0.1,
                max_output_tokens=8192,
            )
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=chunk, config=config)

        items = parse_cards(getattr(response, "text", "") or "")
        batch, dropped = clean_cards(items, chunk)
        print(f"chunk {i}/{len(chunks)}: {len(batch)} card(s), dropped {dropped}")
        cards.extend(batch)
        if on_progress:
            on_progress(i, len(chunks))

    if not cards:
        raise ValueError("Gemini returned no grounded cards for this document")
    return text_dedup(cards)


def run_gemini_job(job_id, r, text):
    """Run the fallback for job_id and return an HTTP response dict."""
    empty = {"Content-Type": "application/json"}

    if len(text) > GEMINI_MAX_CHARS:
        message = (
            f"This document is too long for the backup engine "
            f"({len(text):,} chars, limit {GEMINI_MAX_CHARS:,}). The AI worker is "
            f"unavailable, so try again later or split the document."
        )
        r.set(f"job:{job_id}:error", message, ex=JOB_TTL_SECONDS)
        r.set(f"job:{job_id}:status", "error", ex=JOB_TTL_SECONDS)
        return {"statusCode": 503, "headers": empty,
                "body": json.dumps({"success": False, "error": message})}

    r.set(f"job:{job_id}:status", "processing", ex=JOB_TTL_SECONDS)
    r.set(f"job:{job_id}:provider", "gemini", ex=JOB_TTL_SECONDS)
    r.set(f"job:{job_id}:progress", "0/0", ex=JOB_TTL_SECONDS)

    def on_progress(done, total):
        r.set(f"job:{job_id}:progress", f"{done}/{total}", ex=JOB_TTL_SECONDS)

    try:
        cards = gemini_generate_cards(text, on_progress)
    except Exception as exc:
        import traceback
        print("Gemini fallback failed:", repr(exc))
        print(traceback.format_exc())
        r.set(f"job:{job_id}:error", f"[gemini/{GEMINI_MODEL}] {type(exc).__name__}: {exc}",
              ex=JOB_TTL_SECONDS)
        r.set(f"job:{job_id}:status", "error", ex=JOB_TTL_SECONDS)
        return {"statusCode": 502, "headers": empty,
                "body": json.dumps({"success": False,
                                    "error": "The backup engine also failed. " + str(exc)})}

    r.set(f"job:{job_id}:cards", json.dumps(cards), ex=JOB_TTL_SECONDS)
    r.set(f"job:{job_id}:status", "complete", ex=JOB_TTL_SECONDS)
    return {"statusCode": 200, "headers": empty,
            "body": json.dumps({"success": True, "job_id": job_id,
                                "status": "complete", "provider": "gemini",
                                "cards": len(cards)})}


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
        except (TypeError, ValueError):
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"success": False, "error": "Malformed request body"})
            }

        # Re-run an existing job on the fallback engine. The frontend calls this
        # when /api/status reports a stalled job, so it carries no file.
        if body.get("job_id"):
            job_id = str(body["job_id"])
            r = redis_client()
            if r.get(f"job:{job_id}:status") == "complete":
                # Already finished (perhaps by the GPU worker while the client
                # was deciding); never regenerate and overwrite good cards.
                return {
                    "statusCode": 200,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"success": True, "job_id": job_id,
                                        "status": "complete", "provider": "existing"})
                }
            text = r.get(f"job:{job_id}:text")
            if not text:
                return {
                    "statusCode": 404,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"success": False,
                                        "error": "Job not found or expired"})
                }
            return run_gemini_job(job_id, r, text)

        try:
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
        r.set(f"job:{job_id}:created", str(int(time.time())), ex=JOB_TTL_SECONDS)

        # Explicit override, e.g. "provider": "gemini" to exercise the fallback.
        requested = str(body.get("provider") or "").lower()

        if requested == "gemini":
            return run_gemini_job(job_id, r, text)

        try:
            dispatch_job(job_id)
        except Exception as exc:
            # The GPU worker could not be reached at all (out of credit, bad
            # token, wrong workspace, app undeployed). Run the fallback here.
            print("Dispatch to Modal GPU worker failed:", repr(exc))
            response = run_gemini_job(job_id, r, text)
            if response["statusCode"] != 200:
                print("Fallback engine also failed for job", job_id)
            return response

        r.set(f"job:{job_id}:provider", "qwen", ex=JOB_TTL_SECONDS)
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
