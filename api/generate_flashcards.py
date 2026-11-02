# api/generate_flashcards.py
"""Flashcard generation: the HTTP handler, and every engine it can use.

Deliberately one file. The generation path is small enough that splitting it
into a package cost more in imports, duplicated constants and repeated prompts
than it saved, so everything shared lives here.

Flow
----
  POST <file>                       -> { job_id, provider, chunks }  (nothing generated yet)
  POST {job_id, chunk: i}           -> cards for section group i
                                       (the last group merges and dedups)
  POST {job_id, provider: "qwen"}   -> hand the whole job to the Modal worker

Driving one section group per request keeps every call inside the serverless
time limit and makes progress real instead of a spinner.

Engines, tried in this order
----------------------------
  1. zen     - Gemini via OpenCode Zen (no daily request cap)
  2. gemini  - the Gemini API key directly (google-genai SDK)
  3. qwen    - Qwen 2.5 7B on an A10G, whole-document, see modal_app.py

A quota or outage on one falls through to the next. When the whole chain fails
the client is told to hand the job to the Modal worker.
"""
import base64
import io
import json
import os
import re
import time
import uuid
from urllib import error as urlerror
from urllib import request as urlrequest

MAX_TEXT_CHARS = 500_000
JOB_TTL_SECONDS = 24 * 60 * 60
MODAL_APP_NAME = "flashcard-generator"
MODAL_CLASS_NAME = "FlashcardLLM"

ZEN_BASE_URL = os.environ.get(
    "OPENCODE_BASE_URL", "https://opencode.ai/inference/google/v1beta").rstrip("/")
ZEN_MODEL = os.environ.get("ZEN_MODEL", "gemini-3.5-flash-lite")
ZEN_TIMEOUT_SECONDS = int(os.environ.get("ZEN_TIMEOUT_SECONDS", "55"))
# Cloudflare sits in front of Zen and rejects urllib's default User-Agent with a
# 403 ("error code: 1010") - verified: a bare request is blocked, any explicit UA
# is accepted, so this only has to be honest and non-empty.
ZEN_USER_AGENT = os.environ.get("ZEN_USER_AGENT", "flashcard-generator/1.0")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MAX_OUTPUT = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "65535"))

# The Piper voice. Defined here, once: api/tts.py hashes it into the audio cache
# key and modal_app.py loads it, so a change in one place cannot silently serve
# cached audio from the other.
TTS_VOICE = os.environ.get("TTS_VOICE", "en_US-lessac-medium")


# =====================================================================
# HTTP helpers
# =====================================================================
def auth_secret():
    """Secret used to sign auth tokens (falls back to APP_PASSWORD)."""
    return os.environ.get("AUTH_SECRET") or os.environ.get("APP_PASSWORD", "")


def validate_auth_token(event):
    """Auth is disabled for this single-user app; endpoints are public."""
    return True


def resolve_cors_headers(event):
    """CORS headers for origins listed in ALLOWED_ORIGINS only.

    Same-origin requests don't need CORS headers, so leaving ALLOWED_ORIGINS
    unset keeps the app working while blocking other sites from reading the API.
    """
    headers = {k.lower(): v for k, v in event.get("headers", {}).items()}
    origin = headers.get("origin", "")
    allowed = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if origin in allowed:
        return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
    return {}


def json_response(status, payload, extra_headers=None):
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    return {"statusCode": status, "headers": headers, "body": json.dumps(payload)}


# =====================================================================
# Job store (Upstash, over the REST API or native rediss://)
# =====================================================================
_REDIS_CLIENT = None


def redis_client():
    """Redis client, shared by every endpoint and the Modal worker.

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


def job_access(job_id, ttl=JOB_TTL_SECONDS):
    """Return (client, key, write) for a job's Redis keys.

    Both engines use this, so a job's bookkeeping is identical whichever produced
    it - including `updated`, which /api/status uses to notice an abandoned job.
    """
    r = redis_client()

    def key(suffix):
        return f"job:{job_id}:{suffix}"

    def write(suffix, value):
        r.set(key(suffix), value, ex=ttl)
        # Externally visible touch. Bookkeeping keys are excluded so stamping
        # cannot extend a job's own TTL or feed back into the idle clock.
        if suffix not in ("updated", "text", "created", "total_chunks"):
            r.set(key("updated"), str(int(time.time())), ex=ttl)

    return r, key, write


# =====================================================================
# Prompts (api/prompts/*.yaml)
# =====================================================================
_PROMPT_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts"),
    "/root/prompts",
]
_PROMPT_CACHE = {}


def prompt_text(name):
    """Render a prompt YAML into one string, cached per process.

    Sections are emitted in order: strings as paragraphs, lists as bullets,
    nested mappings as "- key: value" lines. `version`/`applies_to` are metadata
    and are skipped.
    """
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]

    path = next((os.path.join(d, name + ".yaml") for d in _PROMPT_DIRS
                 if os.path.exists(os.path.join(d, name + ".yaml"))), None)
    if path is None:
        raise RuntimeError(f"Prompt '{name}.yaml' not found in: {', '.join(_PROMPT_DIRS)}")

    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "PyYAML is required to load prompts. Add 'pyyaml>=6.0' to "
            "requirements.txt and to the Modal image's pip_install()."
        )

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    lines = []
    for section, value in data.items():
        if section in ("version", "applies_to"):
            continue
        if isinstance(value, str):
            lines.append(value.strip())
        elif isinstance(value, list):
            lines.extend("- " + str(item).strip() for item in value)
        elif isinstance(value, dict):
            lines.append(f"## {section}")
            for sub, sub_value in value.items():
                if isinstance(sub_value, list):
                    lines.extend(f"- {sub}: {item}" for item in sub_value)
                else:
                    lines.append(f"- {sub}: {sub_value}")
    _PROMPT_CACHE[name] = "\n\n".join(lines)
    return _PROMPT_CACHE[name]


def card_schema(require_quote=False):
    """JSON schema for the model's structured output.

    source_quote is OPTIONAL by default. When it was mandatory the model
    compressed every answer into what one sentence could justify, which capped
    answer depth at ~23 words. Grounding is enforced separately by clean_cards.

    `require_quote` exists for vLLM's guided decoder (xgrammar 0.1.6), which is
    happiest with no optional properties: it forces the key to be emitted, and
    the model is still told it may leave the value empty.
    """
    required = ["question", "answer", "difficulty"]
    if require_quote:
        required.insert(2, "source_quote")
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
                    "required": required,
                },
            }
        },
        "required": ["flashcards"],
    }


def rest_schema(schema):
    """Gemini's REST API wants uppercase type names; the SDK wants lowercase.

    One schema definition, converted at the edge, so the two can't drift apart.
    """
    if isinstance(schema, dict):
        return {k: (v.upper() if k == "type" and isinstance(v, str) else rest_schema(v))
                for k, v in schema.items()}
    if isinstance(schema, list):
        return [rest_schema(item) for item in schema]
    return schema


# =====================================================================
# Text extraction
# =====================================================================
MAX_PAGES = 100

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


def extract_text_from_pdf(file_content, max_pages=MAX_PAGES):
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            total_pages = len(pdf.pages)
            if total_pages > max_pages:
                return {
                    "error": "document_too_long",
                    "message": f"Your document has {total_pages} pages. Max allowed: {max_pages}.",
                    "page_count": total_pages,
                    "max_allowed": max_pages,
                }
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        return {"error": "pdf_extraction_failed", "message": f"Error reading PDF: {str(e)}"}


def extract_text_from_docx(file_content):
    """Keep the document's heading structure.

    Word headings are the document's own section map, and the generator uses
    them to guarantee coverage. Joining paragraphs blindly throws that away.
    """
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_content))
        lines = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "").lower() if para.style is not None else ""
            if style.startswith("heading") or style in ("title", "subtitle"):
                digits = "".join(c for c in style if c.isdigit())
                level = int(digits) if digits else 1
                lines.append("\n" + "#" * min(level + 1, 6) + " " + text)
            else:
                lines.append(text)
        # Tables carry facts too (dates, figures) and are easy to miss.
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    lines.append(" | ".join(cells))
        return "\n".join(lines)
    except Exception as e:
        return {"error": "docx_extraction_failed", "message": f"Error reading DOCX: {str(e)}"}


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
        return {"error": "pptx_extraction_failed", "message": f"Error reading PPTX: {str(e)}"}


def extract_text(file_content, file_type):
    """Dispatch on MIME type. Returns text, or an error dict."""
    if file_type == "application/pdf":
        return extract_text_from_pdf(file_content)
    if file_type == "text/plain":
        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            return {"error": "text_not_utf8", "message": "Text file must be UTF-8 encoded"}
    if file_type == DOCX_MIME:
        return extract_text_from_docx(file_content)
    if file_type == PPTX_MIME:
        return extract_text_from_pptx(file_content)
    return {"error": "unsupported_file_type", "message": "Unsupported file type"}


# =====================================================================
# Chunking
# =====================================================================
# Measured on the SIWES report (19,740-char section, same model and prompt):
#   20k chars/call -> 26 cards (1.32 per 1,000 chars)
#    6.6k chars/call -> 69 cards (3.50)
#    3.9k chars/call -> 82 cards (4.15)
# The model stops being exhaustive long before it runs out of context: given 20k
# characters it summarises, given 6k it enumerates. 6k is the knee - the extra
# 19% from 3.9k costs ~35% more sequential latency, and each call is its own
# function invocation.
CHUNK_CHARS = int(os.environ.get("GEMINI_CHUNK_CHARS", "6000"))
# A ceiling, not a target. It used to be 8, then 20: because section_chunks()
# raises its budget to ceil(len(text) / max_chunks), a low ceiling silently made
# chunks grow past the 6k knee - a 500k-character document got 25k chunks and
# went back to summarising. 100 keeps the budget at CHUNK_CHARS for every
# document up to MAX_TEXT_CHARS (100 * 6,000 = 600,000 > 500,000), so the ceiling
# can no longer undo the coverage.
MAX_CHUNKS = int(os.environ.get("GEMINI_MAX_CHUNKS", "100"))


def fold_small(chunks, max_chars, max_chunks):
    """Merge adjacent chunks to reduce the call count, never past the knee.

    Folding exists for documents with many tiny sections (300 headings of 200
    chars each should not become 300 calls). It must not become a way to exceed
    CHUNK_CHARS: merging 5k + 4k into a 9k chunk puts the model back over the
    size where it summarises instead of enumerates, which is exactly the failure
    the chunk size was chosen to avoid. When merging would do that, the extra
    calls are kept and the situation is reported.
    """
    while len(chunks) > max_chunks:
        merged = []
        for i in range(0, len(chunks), 2):
            pair = chunks[i:i + 2]
            if len(pair) == 2 and len(pair[0]) + len(pair[1]) + 2 <= max_chars:
                merged.append("\n\n".join(pair))
            else:
                merged.extend(pair)
        if len(merged) >= len(chunks):
            print(f"{len(chunks)} sections exceed the {max_chunks}-call ceiling; "
                  f"keeping every chunk at <= {max_chars} chars rather than "
                  f"merging past it")
            return merged
        chunks = merged
    return chunks


# A heading line as the extractor writes it: two to six '#' then a space.
# The two-or-more floor matters. A single '#' is not a heading this app ever
# produces - the DOCX extractor emits `## Heading` at minimum - so a lone '#'
# line is something else entirely, typically C preprocessor output in a technical
# report. `#include <stdio.h>` was rendering as an episode title until this was
# tightened to match the splitter, which already required two or more.
HEADING_RE = re.compile(r"^(#{2,6}) +(.+)$")


def heading_title(line):
    """The text of a heading line, or None if the line is not a heading."""
    match = HEADING_RE.match((line or "").strip())
    return match.group(2).strip() if match else None


def section_chunks(text, max_chars=CHUNK_CHARS, max_chunks=MAX_CHUNKS):
    """Chunk on the document's own section boundaries.

    The extractor preserves Word headings as `## Heading`, so chunks follow the
    author's structure instead of an arbitrary paragraph count. Each chunk then
    holds whole sections, which is what makes "cover every section" checkable
    and what stops a chunk from slicing a topic in half.
    """
    parts = re.split(r"\n(?=#{2,6} )", "\n" + text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return paragraph_chunks(text, max_chars, max_chunks)

    budget = max(max_chars, (len(text) + max_chunks - 1) // max_chunks)

    chunks = []
    current = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0

    for part in parts:
        heading = HEADING_RE.match(part.split("\n", 1)[0].strip())
        heading = heading.group(0) if heading else ""
        # Continuation fragments keep the heading so the model still knows which
        # section it is writing cards for. Without this, a hard-split section
        # produces context-free orphans.
        prefix = (heading + " (continued)\n") if heading else ""
        limit = max(500, budget - len(prefix))

        while len(part) > budget:
            cut = part.rfind("\n", 0, limit)
            cut = cut if cut > limit // 2 else limit
            flush()
            piece, part = part[:cut].rstrip(), part[cut:].lstrip()
            if piece:
                chunks.append(piece)
            if not part:
                break
            part = prefix + part
        if not part:
            continue

        if current and current_len + len(part) + 2 > budget:
            flush()
            current, current_len = [part], len(part)
        else:
            current.append(part)
            current_len += len(part) + 2
    flush()

    # Fold neighbours only where that keeps chunks inside the size budget.
    return fold_small(chunks, max_chars, max_chunks)


def paragraph_chunks(text, max_chars=CHUNK_CHARS, max_chunks=MAX_CHUNKS):
    """Split on paragraph boundaries into at most max_chunks pieces.

    Fallback for documents with no heading structure (plain text, PDFs).
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return []

    # The char budget is raised when capping at max_chunks would otherwise force
    # a chunk over it: fewer/larger chunks are cheaper and faster than many small
    # calls, so max_chunks wins and max_chars is the floor.
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

    return fold_small(chunks, max_chars, max_chunks)


# =====================================================================
# Card rules: cleaning, grounding, parsing, deduplication
# =====================================================================
# Words too common to say anything about whether an answer is supported.
STOPWORDS = frozenset("""
about above after again also although among any are because been before being
below between both but can could did does doing down during each either few for
from further had has have having here how into its itself just more most much
not now off once only other our out over own same she should some such than that
the their them then there these they this those through too under until very was
were what when where which while who whom why will with would you your
""".split())

# A quote is grounded when EITHER criterion holds:
#   * its longest run of consecutive words found verbatim covers >= 55% of it
#     (a real quote with a word or two edited keeps a long run), or
#   * 85% of its words sit inside some verbatim run (a real quote with several
#     small edits keeps many short runs).
# Scattered single words satisfy neither. A bag-of-words check scored 1.0 on
# exactly that failure: "The Ford Foundation was established in 1973" assembled
# out of seven unrelated sentences.
GROUNDING_COVERAGE = float(os.environ.get("GROUNDING_COVERAGE", "0.85"))
GROUNDING_RUN_RATIO = float(os.environ.get("GROUNDING_RUN_RATIO", "0.55"))
_GROUND_RUN = 4


def norm(text):
    """Lowercase, collapse everything non-alphanumeric into single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def content_words(text):
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in STOPWORDS]


_NUM_RE = re.compile(r"\d{1,3}(?:[,\s]\d{3})+(?!\d)|\d+(?:\.\d+)?")


def numbers_in(text):
    """Every number in `text`, canonicalised so formatting cannot matter.

    "1,000", "1 000" and "1000" are the same number, so a source that writes one
    and an answer that writes another still agree. Matching whole tokens (rather
    than substrings) means "1973" can never be satisfied by "11973".
    """
    return {re.sub(r"[,\s]", "", token) for token in _NUM_RE.findall(text or "")}


class Source:
    """One chunk, prepared once for all of its grounding checks.

    The n-gram set is built per request and shared by every card in the chunk,
    rather than rebuilt per card.
    """
    __slots__ = ("raw", "norm", "grams", "numbers")

    def __init__(self, raw):
        self.raw = raw or ""
        self.norm = norm(self.raw)
        words = self.norm.split()
        self.grams = {tuple(words[i:i + _GROUND_RUN])
                      for i in range(len(words) - _GROUND_RUN + 1)}
        self.numbers = numbers_in(self.raw)


def quote_support(quote, source):
    """How much of `quote` is recoverable from the source, measured two ways.

    Returns (longest_run_ratio, coverage):
      * longest_run_ratio - the longest stretch of consecutive words found
        verbatim, as a fraction of the quote. A real quote with a phrase edited
        out of the middle still keeps one long run; a sentence assembled from
        scattered words does not.
      * coverage - the fraction of the quote's words that sit inside *any*
        verbatim run of _GROUND_RUN words. A real quote with several small
        edits keeps many short runs.
    """
    quote_norm = norm(quote)
    if len(quote_norm) < 12:
        return 0.0, 0.0
    if quote_norm in source.norm:
        return 1.0, 1.0

    words = quote_norm.split()
    if len(words) < _GROUND_RUN:
        return 0.0, 0.0

    # ok[i] is true when words[i:i+_GROUND_RUN] appears verbatim in the source.
    ok = [tuple(words[i:i + _GROUND_RUN]) in source.grams
          for i in range(len(words) - _GROUND_RUN + 1)]

    covered = bytearray(len(words))
    run = longest = 0
    for i, flag in enumerate(ok):
        if flag:
            covered[i:i + _GROUND_RUN] = b"\x01" * _GROUND_RUN
            run += 1
            longest = max(longest, run)
        else:
            run = 0

    # A run of L matching n-grams spans L + n - 1 words.
    longest_words = longest + _GROUND_RUN - 1 if longest else 0
    return longest_words / len(words), sum(covered) / len(words)


def quote_is_grounded(quote, source):
    """Does this quote actually appear in the source text?

    An invented quote is the tell for parametric leakage: a plausible sentence
    attributed to a document that never said it. A verbatim copy passes
    instantly; otherwise see quote_support for the two criteria.
    """
    longest, coverage = quote_support(quote, source)
    return longest >= GROUNDING_RUN_RATIO or coverage >= GROUNDING_COVERAGE


def answer_numbers_grounded(answer, source):
    """Every number in the answer must appear in the source text.

    Dates and figures are the highest-value hallucination: a model that invents
    "1974" when the text says "1973" passes every other check.
    """
    return numbers_in(answer) <= source.numbers


def capitalized_tokens(text):
    """Capitalised words that are not merely capitalised for sentence position."""
    for match in re.finditer(r"[A-Za-z][A-Za-z'\u2019\-]*", text or ""):
        word = match.group(0)
        if not word[0].isupper():
            continue
        before = (text[:match.start()] or "").rstrip()
        if not before or before[-1] in ".:;!?-\n(\"\u201c'":
            continue
        yield word


def ungrounded_names(answer, quote):
    """Names in the answer that its own quote never mentions.

    ADVISORY ONLY - reported, never used to drop a card. Capitalisation is a
    weak proxy for "named entity": it misfires on abbreviations, on adjectives
    derived from names ("Nigerian" vs "Nigeria"), and on sentences the model
    starts mid-string.
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


def clean_cards(items, chunk="", require_quote=False):
    """Normalise the model's cards and run the checks that are exact.

    Dropped:
      * no source_quote, when `require_quote` is set
      * a source_quote that cannot be recovered from the chunk
      * a number in the answer that the chunk never mentions

    Reported, never enforced: names in the answer absent from its own quote.
    Returns (cards, dropped_count).
    """
    source = Source(chunk) if chunk else None
    cards = []
    dropped_no_quote = 0
    dropped_ungrounded = 0
    dropped_numbers = 0
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
            if require_quote:
                dropped_no_quote += 1
                continue
        else:
            if source and not quote_is_grounded(quote, source):
                dropped_ungrounded += 1
                continue
            if ungrounded_names(answer, quote):
                flagged_names += 1

        # Numbers are checked whether or not the card carries a quote: a wrong
        # date is a wrong date.
        if source and not answer_numbers_grounded(answer, source):
            dropped_numbers += 1
            continue

        cards.append({"question": question, "answer": answer,
                      "source_quote": quote, "difficulty": difficulty})

    if dropped_no_quote:
        print(f"dropped {dropped_no_quote} card(s) with no source_quote")
    if dropped_ungrounded:
        print(f"dropped {dropped_ungrounded} card(s) whose quote is not in the source")
    if dropped_numbers:
        print(f"dropped {dropped_numbers} card(s) with a number absent from the source")
    if flagged_names:
        print(f"advisory: {flagged_names} card(s) mention a name absent from their quote "
              f"(kept - review manually)")
    return cards, dropped_no_quote + dropped_ungrounded + dropped_numbers


_CARDS_KEY_RE = re.compile(r'"flashcards"\s*:\s*\[')


def parse_cards(raw):
    """Parse the model's JSON, salvaging complete cards from a truncated reply.

    Guided decoding guarantees valid JSON *unless* the response is cut off by
    max_tokens mid-string, which leaves an unterminated string and makes
    json.loads raise. Rather than lose the whole chunk, walk the array and keep
    every complete card object before the cut-off point.

    The array is located after the "flashcards" key, not by the first "[" in the
    document: a question like "What is [X]?" would otherwise anchor the salvage
    inside a string and lose the whole chunk's cards.
    """
    try:
        return list(json.loads(raw).get("flashcards") or [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    match = _CARDS_KEY_RE.search(raw)
    if match:
        pos = match.end()
    else:
        pos = raw.find("[")
        if pos < 0:
            return []
        pos += 1

    decoder = json.JSONDecoder()
    cards = []
    while True:
        brace = raw.find("{", pos)
        if brace < 0:
            break
        try:
            obj, end = decoder.raw_decode(raw, brace)
        except json.JSONDecodeError:
            break        # this is the card the token limit cut in half
        if isinstance(obj, dict):
            cards.append(obj)
        pos = end
    return cards


def text_dedup(cards, threshold=0.75, answer_threshold=0.85):
    """Drop near-duplicate cards.

    Two cards are duplicates if their questions are near-identical, or if their
    answers are: a document that repeats one fact gets asked about it from
    several angles, and question-only matching misses those.
    """
    if len(cards) <= 1:
        return cards

    def words(text):
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}

    kept = []
    kept_q = []
    kept_a = []
    for card in cards:
        q = words(card.get("question", ""))
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


# =====================================================================
# Podcast episodes: document sections, plus a per-episode checklist from the deck
# =====================================================================
# An episode is a bigger unit than a card group. It reuses section_chunks so the
# two share boundaries, but with a larger budget (an episode has room for prose a
# card list does not). 12k is still safely below the size where the model stops
# enumerating and starts summarising (the knee we measured for cards).
EPISODE_CHARS = int(os.environ.get("EPISODE_CHARS", "12000"))
EPISODE_MAX = int(os.environ.get("EPISODE_MAX", "40"))


def episode_sections(text):
    """Split the document into heading-aligned, episode-sized pieces.

    Returns a list of {index, title, text}. The title is the first heading in the
    piece, or a fallback "Part N" for documents with no heading structure. The
    episode count therefore emerges from the document rather than being chosen
    up front.
    """
    chunks = section_chunks(text, max_chars=EPISODE_CHARS, max_chunks=EPISODE_MAX)
    episodes = []
    for i, chunk in enumerate(chunks, 1):
        title = next((heading_title(line) for line in chunk.split("\n")
                      if heading_title(line)), None)
        episodes.append({"index": i - 1, "title": title or f"Part {i}", "text": chunk})
    return episodes


def card_key_terms(card):
    """The terms a podcast episode should insist on for one card.

    Numbers and proper nouns are the two things the grounding gates already
    guarantee are in the source, so they are safe to require of the episode
    without an extra model call. They are a checklist, not an extractive
    summary, so imprecision here only means a term is occasionally missed.
    """
    terms = set(numbers_in(str(card.get("answer", ""))))
    for field in ("answer", "question"):
        for word in capitalized_tokens(str(card.get(field, ""))):
            # "Univasa's" can never match a script token: the article is
            # separate, so the possessive would be a permanent false miss.
            word = re.sub(r"['\u2019](s|S)?$", "", word).strip()
            lower = word.lower()
            if len(lower) < 3 or lower in STOPWORDS:
                continue
            terms.add(word)
    return terms


# Two surface forms of one word ("API" and "APIs") cost two cap slots and
# guarantee a false miss, because a script says the word once. Canonicalising
# the checklist is better than stemming while matching: it frees the slot, keeps
# the comparison exact, and needs no fuzzy logic at read time.
#
# The merge is deliberately conservative: a plural is folded into its singular
# ONLY when that singular actually occurs elsewhere in the deck. Stripping "s"
# unconditionally looks harmless until it hits "SIWES" -> "siwe" (or "News",
# "Status", "Glass"): four letters survive the length guard, so the rule below
# needs evidence rather than a suffix.
def _singular_candidates(key):
    if not key.endswith("s"):
        return []
    out = [key[:-1]]                      # APIs -> API
    if key.endswith("es") and len(key) > 3:
        out.append(key[:-2])              # Boxes -> Box
    return [c for c in out if len(c) >= 3 and c != key]


def merge_plural_terms(counter):
    """Fold plural keys into their singular, in place, only where the singular exists.

    Takes and returns {norm_key: count}. "APIs" joins "API" because some card
    writes "API"; "SIWES" stays alone because no card writes "SIWE".
    """
    for key in list(counter):
        for singular in _singular_candidates(key):
            if singular in counter:
                counter[singular] += counter.pop(key)
                break
    return counter


def deck_term_frequency(cards):
    """How many cards mention each key term, with plurals folded into singulars.

    This is the filter that keeps the checklist honest. Cover-page metadata
    (the student's name, the submission month, "DEPARTMENT OF COMPUTING") is
    real and grounded, so it survives every other check - but it appears in
    one or two cards, while a subject worth a lecture appears across many.
    Frequency across the deck is what separates "a fact in the document" from
    "a fact about this course".

    Keys are normalized terms after the plural merge, so "apis" never appears
    here on its own when "api" does.
    """
    counts = {}
    for card in cards:
        for term in card_key_terms(card):
            counts[norm(term)] = counts.get(norm(term), 0) + 1
    return merge_plural_terms(counts)


def canonical_key(term, counts):
    """Where a term's count lives after the plural merge."""
    key = norm(term)
    if key in counts:
        return key
    for singular in _singular_candidates(key):
        if singular in counts:
            return singular
    return key


def deck_term_surface(cards, counts):
    """The wording to ask a script for, per canonical key: the shortest form.

    After "APIs" folds into "API", the checklist must demand "API" - asking for
    the plural would recreate the miss the merge exists to remove.
    """
    forms = {}
    for card in cards:
        for term in card_key_terms(card):
            forms.setdefault(canonical_key(term, counts), set()).add(term)
    return {key: min(group, key=lambda s: (len(s), s.lower()))
            for key, group in forms.items()}


# Tunables. min_cards=2 is what removes cover-page metadata (a student's name,
# the submission month appears in one or two cards); raising it thins the list
# further, lowering it to 1 keeps every one-off figure but reintroduces the
# metadata noise. Coverage is printed to the logs so this can be tuned against
# real output rather than guessed.
EPISODE_MIN_CARDS = int(os.environ.get("EPISODE_MIN_CARDS", "2"))
EPISODE_TERM_CAP = int(os.environ.get("EPISODE_TERM_CAP", "35"))


def map_cards_to_episodes(cards, episodes, min_cards=None, cap=None):
    """Assign each card's terms to an episode by where its quote appears.

    The mapping is by quote location in the document, not by card index: the
    episode is owned by the heading that contains the quote, so the podcast
    stays driven by the document's structure while the deck contributes only
    a per-episode list of terms that must appear. Cards with no (or unmatched)
    quote are skipped, as are terms the deck mentions fewer than `min_cards`
    times.

    `cap` bounds the checklist per episode. It was lowered to 18 on the theory
    that 35 forced a recitation; measured on this document that was not what
    happened - coverage was flat (82.9% vs 81.5%) and the shorter list let the
    model ramble 40% longer and drop the deck's most frequent term entirely. The
    real waste was duplicates like API/APIs, which canonicalisation removes.
    """
    min_cards = EPISODE_MIN_CARDS if min_cards is None else min_cards
    cap = EPISODE_TERM_CAP if cap is None else cap
    counts = deck_term_frequency(cards)
    surface = deck_term_surface(cards, counts)
    sources = [Source(e["text"]) for e in episodes]
    buckets = [set() for _ in episodes]
    for card in cards:
        quote = str(card.get("source_quote", "")).strip()
        if not quote:
            continue
        for i, source in enumerate(sources):
            if quote_is_grounded(quote, source):
                for term in card_key_terms(card):
                    key = canonical_key(term, counts)
                    if counts.get(key, 0) >= min_cards:
                        buckets[i].add(key)
                break
    out = []
    for bucket in buckets:
        ranked = sorted(bucket, key=lambda k: (-counts.get(k, 0), k))
        out.append(sorted({surface.get(k, k) for k in ranked[:cap]}, key=str.lower))
    return out


def script_coverage(terms, turns):
    """How many of the must-include terms actually appear in a generated script.

    Returns {covered, total, missing}. Not shown to the end user; it exists to
    make "is the podcast covering enough?" a number that can be tuned against,
    the way chunk size was tuned against cards per 1,000 characters.

    Compared on token boundaries, not raw substrings: the same class of false
    positive that let "1973" pass inside "11973" in the number gate. Padding
    the normalized text and the term keeps whole-word matching while still
    allowing a multi-token term, because norm() turns a hyphenated name like
    "Igho-Iggue" into the two-word sequence "igho iggue".
    """
    script_padded = " " + norm(" ".join(t.get("text", "") for t in turns)) + " "
    missing = [term for term in terms if " " + norm(term) + " " not in script_padded]
    return {"covered": len(terms) - len(missing),
            "total": len(terms), "missing": missing}


# =====================================================================
# Engines
# =====================================================================
class EngineFailure(Exception):
    """A generation attempt failed, classified by what to do next.

    kind:
      quota      - out of credit / over the rate limit: try the next engine
      transient  - overloaded or timed out: try the next engine
      fatal      - this request will never succeed (bad input, bad key)
    """

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def provider_chain(preferred=None):
    """The engines to try, in order, starting at `preferred` when it is usable.

    A container that just lost its quota never retries that engine first: the
    working one is remembered per job, so chunk 6 doesn't repeat chunk 5's
    failure.
    """
    chain = []
    if os.environ.get("OPENCODE_API_KEY"):
        chain.append("zen")
    if os.environ.get("GEMINI_API_KEY"):
        chain.append("gemini")
    if preferred in chain:
        chain = chain[chain.index(preferred):]
    return chain


def _quota_markers():
    return ("resource_exhausted", "429", "spending cap", "quota", "rate limit",
            "billing", "exceeded your current quota", "insufficient")


# Network-level failures are never the document's fault and always worth another
# engine: DNS, connection resets and TLS errors arrive as exceptions with no HTTP
# status at all, which used to fall through to "fatal".
_NETWORK_MARKERS = ("name resolution", "temporary failure in name", "tunnel", "connection",
                    "connect", "remotedisconnected", "ssl", "certificate", "eai_",
                    "getaddrinfo", "network", "timed out", "timeout", "reset by peer",
                    "proxyerror", "unreadable response", "failed to establish")


def _network_error(text):
    return any(marker in text for marker in _NETWORK_MARKERS)


def classify(status, detail):
    """Map an HTTP status / provider error string onto an EngineFailure kind.

    The messages are what ends up in the job error and in the UI, so they name
    the actual problem: a 403 from Zen is usually "this model is not enabled for
    your key", not "your key is wrong", and saying the wrong one sends you
    hunting for a broken secret that is fine.
    """
    text = (detail or "").lower()
    if status == 401:
        return EngineFailure("fatal", f"API key rejected (401): {detail}")
    if status == 403:
        if any(marker in text for marker in
               ("model access", "disabled", "not enabled", "not available",
                "unsupported", "permission")):
            return EngineFailure(
                "fatal",
                f"this model is not enabled for the key (403): {detail}. "
                f"Check which models the account can use, or change ZEN_MODEL.")
        return EngineFailure("fatal", f"forbidden (403): {detail}")
    if status == 404:
        return EngineFailure("fatal",
                             f"model or endpoint not found (404): {detail}. "
                             f"Check ZEN_MODEL against the provider's model list.")
    if status == 429 or any(marker in text for marker in _quota_markers()):
        return EngineFailure("quota", f"quota/rate limit ({status}): {detail}")
    if _network_error(text):
        return EngineFailure("transient", f"network error reaching the provider: {detail}")
    if status in (408, 500, 502, 503, 504) or any(
            marker in text for marker in
            ("unavailable", "overloaded", "high demand", "deadline exceeded", "timed out")):
        return EngineFailure("transient", f"temporarily unavailable ({status}): {detail}")
    return EngineFailure("fatal", f"request failed ({status}): {detail}")


# ----- engine 1: OpenCode Zen (Gemini, no daily request cap) -----
def _zen_payload(prompt, contents, schema, temperature=0.1, with_thinking_config=True):
    config = {
        "temperature": temperature,
        "maxOutputTokens": GEMINI_MAX_OUTPUT,
        "responseMimeType": "application/json",
        "responseSchema": rest_schema(schema),
    }
    if with_thinking_config:
        # Extraction from a supplied document gains nothing from a reasoning
        # pass; it only costs latency and tokens.
        config["thinkingConfig"] = {"thinkingBudget": 0}
    return {
        "systemInstruction": {"parts": [{"text": prompt}]},
        "contents": [{"role": "user", "parts": [{"text": contents}]}],
        "generationConfig": config,
    }


def _zen_post(payload):
    url = f"{ZEN_BASE_URL}/models/{ZEN_MODEL}:generateContent"
    request = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": ZEN_USER_AGENT,
            # Zen authenticates with a bearer token; the Google SDK's own header
            # (x-goog-api-key) is not accepted, which is why this is a plain POST.
            "Authorization": f"Bearer {os.environ['OPENCODE_API_KEY']}",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=ZEN_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:600]
        except Exception:
            pass
        raise classify(exc.code, detail)
    except urlerror.URLError as exc:
        raise EngineFailure("transient", f"could not reach Zen: {exc.reason}")
    except (TimeoutError, OSError) as exc:
        raise EngineFailure("transient", f"Zen timed out: {exc}")


def _response_text(body):
    """Pull the text out of a Gemini-shaped generateContent response."""
    for candidate in body.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if text:
                return text
    reason = ((body.get("candidates") or [{}])[0].get("finishReason")
              or (body.get("promptFeedback") or {}).get("blockReason") or "unknown")
    raise EngineFailure("fatal", f"empty response (finishReason={reason})")


def zen_json(prompt, contents, schema, temperature=0.1):
    try:
        body = _zen_post(_zen_payload(prompt, contents, schema, temperature))
    except EngineFailure as exc:
        # thinkingConfig is a 2.5-and-later field; an older deployment of the
        # proxy rejects it with a 400 rather than ignoring it.
        if exc.kind != "fatal" or "thinking" not in str(exc).lower():
            raise
        body = _zen_post(_zen_payload(prompt, contents, schema, temperature,
                                      with_thinking_config=False))
    return _response_text(body)


# ----- engine 2: the Gemini API key directly -----
def gemini_json(prompt, contents, schema, temperature=0.1):
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise EngineFailure("fatal", f"google-genai is not installed: {exc}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EngineFailure("fatal", "GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=prompt,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        max_output_tokens=GEMINI_MAX_OUTPUT,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=contents, config=config)
    except Exception as exc:
        raise classify(getattr(exc, "code", 0) or 0, f"{type(exc).__name__}: {exc}")
    return getattr(response, "text", "") or ""


JSON_ENGINES = {"zen": zen_json, "gemini": gemini_json}


def generate_json(prompt, contents, schema, preferred=None, temperature=0.1):
    """One JSON-mode generation through the engine chain.

    Returns (engine_name, raw_text). When no engine worked, raises an EngineFailure
    whose kind is the most recoverable of the ones seen: "the whole chain failed"
    is a reason to try the other engine (Modal), not a reason to fail the document.
    Reporting the last error instead turned a DNS blip into a dead job.
    """
    chain = provider_chain(preferred)
    if not chain:
        raise EngineFailure("fatal",
                            "No generation engine is configured (set OPENCODE_API_KEY "
                            "or GEMINI_API_KEY, or use the Modal worker).")

    last = None
    kinds = []
    for name in chain:
        try:
            raw = JSON_ENGINES[name](prompt, contents, schema, temperature)
        except EngineFailure as exc:
            last = exc
            kinds.append(exc.kind)
            print(f"engine {name} failed ({exc.kind}): {exc}")
            continue
        except Exception as exc:                 # never let an engine crash the request
            last = EngineFailure("fatal", f"{name}: {type(exc).__name__}: {exc}")
            kinds.append("fatal")
            print(f"engine {name} crashed: {last}")
            continue
        return name, raw

    if "transient" in kinds:
        raise EngineFailure("transient", f"every engine failed: {last}")
    if "quota" in kinds:
        raise EngineFailure("quota", f"every engine is out of credit: {last}")
    raise last


def generate_cards(chunk, preferred=None):
    """Cards for one chunk: generate through the engine chain, then verify.

    Verification is exact only - a quote that cannot be recovered from the chunk,
    or a number in the answer the chunk never mentions. An embedding-based
    semantic pass was tried and dropped: it needs a second model, its thresholds
    are not comparable across embedding providers, and on a real 415-card deck the
    exact checks already found nothing for it to catch.

    Returns (cards, dropped, engine_used).
    """
    engine, raw = generate_json(prompt_text("card_gemini"), chunk,
                                card_schema(), preferred)
    cards, dropped = clean_cards(parse_cards(raw), chunk)
    print(f"engine {engine}: {len(cards)} card(s), dropped {dropped}")
    return cards, dropped, engine


# ----- engine 3: the Modal GPU worker (whole document) -----
def dispatch_job(job_id):
    """Fire-and-forget the Modal worker; returns immediately."""
    import modal
    cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    try:
        cls().process_job.spawn(job_id)
    except modal.exception.NotFoundError as exc:
        raise RuntimeError(
            f"{exc}. Check that MODAL_TOKEN_ID/MODAL_TOKEN_SECRET belong to the "
            f"workspace where you ran 'modal deploy {MODAL_APP_NAME}', that they are "
            f"set for the Production environment, and that you redeployed afterwards "
            f"(environment variables only take effect on new deployments)."
        ) from exc


def dispatch_to_modal(job_id, r):
    """Queue the job on the Modal GPU worker and return the HTTP response."""
    try:
        dispatch_job(job_id)
    except Exception as exc:
        print("Dispatch to Modal GPU worker failed:", repr(exc))
        message = (
            "No engine could take this document: the AI APIs are unavailable and "
            "the Modal worker could not be reached. " + str(exc)
        )
        r.set(f"job:{job_id}:error", message, ex=JOB_TTL_SECONDS)
        r.set(f"job:{job_id}:status", "error", ex=JOB_TTL_SECONDS)
        return json_response(503, {"success": False, "error": message})

    r.set(f"job:{job_id}:provider", "qwen", ex=JOB_TTL_SECONDS)
    return json_response(200, {"success": True, "job_id": job_id,
                               "status": "queued", "provider": "qwen"})


# =====================================================================
# Chunk handler
# =====================================================================
def _get_many(r, keys):
    """Values for many keys, in ONE call where the client supports it.

    On the Upstash REST transport every GET is a separate HTTPS round trip, so a
    per-key scan of a 20-chunk job costs ~400 requests before any generation
    happens. mget collapses that to one; clients without it fall back.
    """
    if not keys:
        return []
    try:
        values = r.mget(keys)
        if isinstance(values, list) and len(values) == len(keys):
            return values
    except Exception:
        pass
    return [r.get(k) for k in keys]


def stored_chunks(r, key, total):
    """Chunk results already in Redis, as {index: [cards]}.

    A corrupt value is treated as "not done" so a half-written chunk gets
    regenerated rather than poisoning the merge.
    """
    raw_values = _get_many(r, [key(f"chunk:{i}") for i in range(total)])
    done = {}
    for index, raw in enumerate(raw_values):
        if raw is None:
            continue
        try:
            done[index] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            print(f"ignoring unreadable chunk {index}")
    return done


def merge_and_publish(r, write, key, total, done=None):
    """Merge every stored chunk, dedup once across the document, publish."""
    done = stored_chunks(r, key, total) if done is None else done
    all_cards = []
    for i in range(total):
        all_cards.extend(done.get(i) or [])
    cards = text_dedup(all_cards)
    write("cards", json.dumps(cards))
    write("status", "complete")
    write("note", f"{len(cards)} cards across {total} sections")
    return cards


def run_chunk(job_id, r, key, write, text, index):
    """Generate ONE section group; finalise once every group is stored.

    Each group's cards go to job:<id>:chunk:<idx> the moment they exist, so a
    failure at group 18 loses group 18, not 1-17. The write also makes the call
    idempotent: a group already in Redis is returned as-is instead of being paid
    for again, which is what makes retry and resume cheap.
    """
    chunks = section_chunks(text)
    total = len(chunks)

    if index < 0 or index >= total:
        return json_response(400, {"success": False,
                                   "error": f"chunk {index} out of range 0..{total - 1}"})

    # Honour /api/cancel on this path too. The browser stops asking for chunks,
    # but a tab left open mid-generation would otherwise keep paying for them.
    if r.get(key("cancelled")) == "1":
        write("status", "cancelled")
        return json_response(200, {"success": True, "cancelled": True,
                                   "done": index, "total": total})

    write("status", "processing")
    write("total_chunks", str(total))

    done = stored_chunks(r, key, total)
    reused = index in done

    if reused:
        batch = done[index]
        dropped = 0
        engine = r.get(key("engine")) or ""
        print(f"job {job_id} chunk {index + 1}/{total}: reusing {len(batch)} stored card(s)")
    else:
        try:
            batch, dropped, engine = generate_cards(chunks[index], r.get(key("engine")))
        except EngineFailure as exc:
            import traceback
            print(f"job {job_id} chunk {index} failed ({exc.kind}): {exc}")
            print(traceback.format_exc())
            # Never mark the job errored: the groups that already landed stay in
            # Redis, and a 503 tells the client to hand the remainder to the Modal
            # worker, which resumes from exactly these indices. Failing the job
            # here would discard finished work over one bad API response.
            if exc.kind == "quota":
                return json_response(503, {"success": False, "quota": True,
                                           "fallback": True,
                                           "error": f"API quota exhausted: {exc}"})
            return json_response(503, {"success": False, "unavailable": True,
                                       "fallback": True,
                                       "error": f"AI provider unavailable: {exc}"})

        # Remember the engine that worked so later chunks don't re-pay a failure.
        write("engine", engine)
        write("provider", engine)
        write(f"chunk:{index}", json.dumps(batch))
        done[index] = batch
        print(f"job {job_id} chunk {index + 1}/{total}: {len(batch)} card(s), dropped {dropped}")

    write("progress", f"{len(done)}/{total}")
    write("partial_cards", str(sum(len(v) for v in done.values())))

    if len(done) < total:
        return json_response(200, {"success": True, "done": len(done), "total": total,
                                   "cards": len(batch), "final": False,
                                   "reused": reused, "provider": engine})

    cards = merge_and_publish(r, write, key, total, done)
    return json_response(200, {"success": True, "done": total, "total": total,
                               "cards": len(cards), "final": True,
                               "reused": reused, "provider": engine})


# =====================================================================
# Handler
# =====================================================================
def lambda_handler(event):
    try:
        if event["httpMethod"] != "POST":
            return json_response(405, {"success": False, "error": "Method not allowed"})

        if not validate_auth_token(event):
            return json_response(401, {"success": False, "error": "Unauthorized"})

        headers = {k.lower(): v for k, v in event["headers"].items()}
        if "application/json" not in headers.get("content-type", ""):
            return json_response(400, {"success": False,
                                       "error": "Content-Type must be application/json"})

        try:
            body = json.loads(event["body"])
        except (TypeError, ValueError):
            return json_response(400, {"success": False, "error": "Malformed request body"})

        # Continue an existing job: one section group per request.
        if body.get("job_id"):
            return continue_job(body)

        try:
            file_content = base64.b64decode(body["file_content"])
            file_type = body["file_type"]
        except (KeyError, TypeError, ValueError):
            return json_response(400, {"success": False, "error": "Malformed request body"})

        text = extract_text(file_content, file_type)
        if isinstance(text, dict):     # extraction helpers return an error dict
            return json_response(400, {"success": False, **text})
        if not text.strip():
            return json_response(400, {"success": False,
                                       "error": "Could not extract text from file"})
        if len(text) > MAX_TEXT_CHARS:
            return json_response(400, {"success": False, "error": "document_too_long",
                                       "message": f"Extracted text is too long "
                                                  f"({len(text):,} chars). Limit: "
                                                  f"{MAX_TEXT_CHARS:,}. Try splitting it."})

        job_id = uuid.uuid4().hex
        r = redis_client()
        r.set(f"job:{job_id}:text", text, ex=JOB_TTL_SECONDS)
        r.set(f"job:{job_id}:status", "queued", ex=JOB_TTL_SECONDS)
        r.set(f"job:{job_id}:progress", "0/0", ex=JOB_TTL_SECONDS)
        r.set(f"job:{job_id}:created", str(int(time.time())), ex=JOB_TTL_SECONDS)

        requested = str(body.get("provider") or "").lower()
        chain = provider_chain()
        if requested == "qwen" or not chain:
            return dispatch_to_modal(job_id, r)

        chunk_count = len(section_chunks(text))
        r.set(f"job:{job_id}:provider", chain[0], ex=JOB_TTL_SECONDS)
        r.set(f"job:{job_id}:total_chunks", str(chunk_count), ex=JOB_TTL_SECONDS)
        return json_response(200, {"success": True, "job_id": job_id,
                                   "status": "queued", "provider": "api",
                                   "chunks": chunk_count})

    except Exception as e:
        import traceback
        print("Unhandled error:", str(e))
        print(traceback.format_exc())
        return json_response(500, {"success": False, "error": str(e),
                                   "trace": traceback.format_exc()})


def continue_job(body):
    """Handle {job_id} plus either {chunk: i}, {provider: "qwen"}, or nothing."""
    job_id = str(body["job_id"])
    r = redis_client()
    _, key, write = job_access(job_id)

    if r.get(key("status")) == "complete":
        # Already finished; never regenerate and overwrite good cards.
        return json_response(200, {"success": True, "job_id": job_id,
                                   "status": "complete", "provider": "existing"})

    text = r.get(key("text"))
    if not text:
        return json_response(404, {"success": False, "error": "Job not found or expired"})

    requested = str(body.get("provider") or "").lower()

    # Hand the whole job to Modal (Gemini out of credit, or the worker never booted).
    if requested == "qwen":
        return dispatch_to_modal(job_id, r)

    # Resume on the API engines after a stalled worker: reply with the chunk plan
    # so the client can drive it the normal way.
    if requested in ("gemini", "zen", "api"):
        chunks = section_chunks(text)
        write("status", "processing")
        write("provider", requested)
        return json_response(200, {"success": True, "provider": requested,
                                   "chunks": len(chunks), "resumed": True})

    if body.get("chunk") is None:
        return json_response(400, {"success": False, "error": "Missing chunk index"})

    try:
        index = int(body["chunk"])
    except (TypeError, ValueError):
        return json_response(400, {"success": False, "error": "chunk must be an integer"})

    return run_chunk(job_id, r, key, write, text, index)


# =====================================================================
# Vercel entrypoint (also the local dev server)
# =====================================================================
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

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        event = {"httpMethod": "POST", "headers": dict(self.headers), "body": body}
        response = lambda_handler(event)
        response.setdefault("headers", {}).update(resolve_cors_headers(event))
        self._send(response)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
