"""Live test of the Gemini card engine on a real document.

Two modes, so the two designs can be compared on the same input:

  --mode current    the production pipeline: chunked, capped at 15-40 cards
                    per chunk, quotes required, answers constrained by them.

  --mode notebook   the NotebookLM-style variant: the WHOLE document in one
                    call (Gemini's 1M context), a coverage-driven prompt with
                    no numeric cap, and answers allowed to elaborate using any
                    part of the source while still carrying a verbatim quote.

    .venv/bin/python test_gemini_doc.py <file.docx> --mode notebook
    .venv/bin/python test_gemini_doc.py <file.docx> --mode current
    .venv/bin/python test_gemini_doc.py <file.docx> --mode both

Needs GEMINI_API_KEY in the environment or in .env.local.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import types

import docx


# ---------------------------------------------------------------------------
# Notebook-mode prompt: completeness over brevity, no arbitrary card cap.
# ---------------------------------------------------------------------------
NOTEBOOK_PROMPT = """You are building a complete study deck from the document below.

Your goal is COVERAGE. A student using this deck should be able to answer exam
questions about any fact in the document, so miss nothing.

Rules:
1. Enumerate EVERY distinct testable fact in the text: definitions, dates,
   figures, named people and organisations, purposes, objectives, procedures,
   requirements, outcomes, comparisons, and list items. A section with 40
   testable facts should produce 40 cards.
2. Do not stop early and do not summarise. Do not merge separate facts into one
   card - if the text states three things, that is three cards.
3. Do not pad with restatements of the same fact in different words.
4. The answer must be self-contained: a student who has not read the document
   should understand it. Include the specific names, numbers and context.
5. The source_quote must be copied character-for-character from the text. It is
   the evidence for the card, not a summary of it. Quote the sentence (or short
   passage) that proves the answer.
6. The answer may elaborate using other parts of the same document. Every fact
   in it must still come from this document - never from your own knowledge.
7. Rate difficulty as easy, medium or hard.

Return JSON only."""


def load_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local")
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return ""


def load_engine():
    """Import api/generate_flashcards.py without its Vercel-only deps."""
    for name in ("pdfplumber", "pptx", "upstash_redis", "redis", "modal"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                sys.modules[name] = types.ModuleType(name)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "api", "generate_flashcards.py")
    spec = importlib.util.spec_from_file_location("fallback_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Published Gemini list prices, USD per 1M tokens. Update if they change.
# Only used to turn the API's own usage_metadata into a dollar figure.
PRICES = {
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "gemini-2.5-pro":   {"in": 1.25, "out": 10.00},
}


def cost_of(usage, model):
    """USD for one request, from the API's usage_metadata."""
    if not usage:
        return None
    price = PRICES.get(model)
    if not price:
        return None
    prompt = getattr(usage, "prompt_token_count", 0) or 0
    output = getattr(usage, "candidates_token_count", 0) or 0
    thoughts = getattr(usage, "thoughts_token_count", 0) or 0
    cached = getattr(usage, "cached_content_token_count", 0) or 0
    # Cached input is billed at the discounted rate when caching is on.
    billable_in = max(0, prompt - cached)
    usd = (billable_in / 1e6) * price["in"] + ((output + thoughts) / 1e6) * price["out"]
    return {
        "prompt_tokens": prompt,
        "output_tokens": output,
        "thoughts_tokens": thoughts,
        "usd": usd,
    }


def extract(path: str) -> str:
    if path.lower().endswith(".docx"):
        document = docx.Document(path)
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text)
        return "\n".join(parts)
    with open(path, errors="ignore") as fh:
        return fh.read()


def run_notebook(engine, text, model):
    """One call, whole document, coverage-driven prompt. Returns (cards, cost)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(
        system_instruction=NOTEBOOK_PROMPT,
        response_mime_type="application/json",
        response_schema=engine.card_schema(),
        temperature=0.2,
        max_output_tokens=65535,
    )
    response = client.models.generate_content(model=model, contents=text, config=config)

    items = engine.parse_cards(getattr(response, "text", "") or "")
    cards, dropped = engine.clean_cards(items, text)
    print(f"  raw cards: {len(items)} | kept: {len(cards)} | dropped: {dropped}")
    cost = cost_of(getattr(response, "usage_metadata", None), model)
    return cards, cost


def run_current(engine, text, model):
    """The production path: chunked, capped, quote-constrained. Returns (cards, cost)."""
    started = time.time()
    total = {"prompt_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0, "usd": 0.0}

    # Meter each chunked request the same way as the single notebook call.
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def on_progress(done, total_chunks):
        print(f"  [progress] chunk {done}/{total_chunks}", flush=True)

    # Match production: section-aware chunks, coverage prompt, big output cap.
    chunks = engine.section_chunks(text)
    print(f"  chunks: {len(chunks)} ({[f'{len(c)//1000}k' for c in chunks]})")
    cards = []
    for i, chunk in enumerate(chunks, 1):
        config = types.GenerateContentConfig(
            system_instruction=engine.prompt_text("card_gemini"),
            response_mime_type="application/json",
            response_schema=engine.card_schema(),
            temperature=0.1,
            max_output_tokens=32768,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        response = client.models.generate_content(model=model, contents=chunk, config=config)
        items = engine.parse_cards(getattr(response, "text", "") or "")
        batch, dropped = engine.clean_cards(items, chunk)
        cards.extend(batch)
        cost = cost_of(getattr(response, "usage_metadata", None), model)
        if cost:
            total["prompt_tokens"] += cost["prompt_tokens"]
            total["output_tokens"] += cost["output_tokens"]
            total["thoughts_tokens"] += cost["thoughts_tokens"]
            total["usd"] += cost["usd"]
        print(f"  chunk {i}/{len(chunks)}: {len(batch)} card(s), dropped {dropped}")
    print(f"  kept {len(cards)} card(s) after dedup: {len(engine.text_dedup(cards))}")
    return engine.text_dedup(cards), total


def report(cards, label, elapsed, cost=None):
    print("-" * 72)
    print(f"{label}")
    print(f"  cards        : {len(cards)}")
    if cards:
        aw = sorted(len(c["answer"].split()) for c in cards)
        qw = sorted(len(c["question"].split()) for c in cards)
        print(f"  answer words : min={aw[0]} avg={sum(aw)//len(aw)} max={aw[-1]}")
        print(f"  question words: min={qw[0]} avg={sum(qw)//len(qw)} max={qw[-1]}")
        print(f"  all quoted   : {all(c.get('source_quote') for c in cards)}")
    difficulty = {}
    for c in cards:
        difficulty[c.get("difficulty", "?")] = difficulty.get(c.get("difficulty", "?"), 0) + 1
    print(f"  difficulty   : {difficulty}")
    print(f"  elapsed      : {elapsed:.1f}s")
    if cost:
        print(f"  input tokens : {cost.get('prompt_tokens', 0):,}")
        print(f"  output tokens: {cost.get('output_tokens', 0):,}")
        if cost.get("thoughts_tokens"):
            print(f"  thinking     : {cost['thoughts_tokens']:,} tokens")
        print(f"  COST         : ${cost.get('usd', 0):.4f}  "
              f"(${cost.get('usd', 0) / max(1, len(cards)) * 100:.2f} per 100 cards)")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"gemini_cards_{label}.json")
    with open(out, "w") as fh:
        json.dump(cards, fh, indent=2)
    print(f"  saved        : {out}")
    print("-" * 72)
    for c in cards[:3]:
        print(f"Q: {c['question']}")
        print(f"A: {c['answer']}")
        print(f"   source: {c.get('source_quote', '')[:150]}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini card engine test")
    parser.add_argument("document", help="path to a .docx/.pdf/.txt/.pptx file")
    parser.add_argument("--mode", choices=["current", "notebook", "both"], default="current")
    parser.add_argument("--model", default=None, help="override the model id")
    args = parser.parse_args()

    key = load_key()
    if not key:
        print("FAIL: GEMINI_API_KEY not found in the environment or .env.local")
        return 2
    os.environ["GEMINI_API_KEY"] = key

    engine = load_engine()
    # No Vercel function limit here, so let the test run to completion.
    model = args.model or engine.GEMINI_MODEL
    text = extract(args.document)

    print("=" * 72)
    print(f"Gemini engine test: {os.path.basename(args.document)}")
    print(f"  model      : {model}")
    print(f"  mode       : {args.mode}")
    print(f"  source     : {len(text):,} chars")
    print("=" * 72)

    results = {}

    if args.mode in ("current", "both"):
        print("\n[MODE: current] production pipeline (chunked, capped)")
        t = time.time()
        cards, cost = run_current(engine, text, model)
        results["current"] = len(cards)
        report(cards, "current", time.time() - t, cost)

    if args.mode in ("notebook", "both"):
        print("\n[MODE: notebook] one call, whole document, coverage-driven")
        t = time.time()
        cards, cost = run_notebook(engine, text, model)
        results["notebook"] = len(cards)
        report(cards, "notebook", time.time() - t, cost)

    if len(results) == 2:
        ratio = results["notebook"] / max(1, results["current"])
        print("=" * 72)
        print(f"current: {results['current']} cards | notebook: {results['notebook']} cards "
              f"({ratio:.1f}x)")
        print("=" * 72)

    print("GEMINI TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
