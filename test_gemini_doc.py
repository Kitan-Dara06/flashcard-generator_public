"""Live test of the Gemini fallback engine on a real document.

Runs the exact code path the API uses (gemini_generate_cards from
api/generate_flashcards.py) against a .docx and reports what came back:
card count, what the grounding gates rejected, difficulty mix, and samples.

    .venv/bin/python test_gemini_doc.py "path/to/file.docx"

Needs GEMINI_API_KEY in the environment or in .env.local.
"""
import importlib.util
import json
import os
import sys
import time
import types

import docx


def load_key() -> str:
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


def extract(path: str) -> str:
    document = docx.Document(path)
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)
    return "\n".join(parts)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: test_gemini_doc.py <file.docx>")
        return 2

    key = load_key()
    if not key:
        print("FAIL: GEMINI_API_KEY not found in the environment or .env.local")
        return 2
    os.environ["GEMINI_API_KEY"] = key

    engine = load_engine()
    path = sys.argv[1]
    text = extract(path)

    print("=" * 72)
    print(f"Gemini engine test: {os.path.basename(path)}")
    print(f"  model      : {engine.GEMINI_MODEL}")
    print(f"  thinking   : disabled (thinking_budget=0)")
    print(f"  source     : {len(text):,} chars")
    print(f"  chunks     : {len(engine.paragraph_chunks(text))}")
    print(f"  max chars  : {engine.GEMINI_MAX_CHARS:,} | budget {engine.GEMINI_TIME_BUDGET:.0f}s")
    print("=" * 72)

    started = time.time()
    try:
        cards = engine.gemini_generate_cards(
            text, lambda done, total: print(f"  [progress] chunk {done}/{total}", flush=True))
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.time() - started

    if not cards:
        print("FAIL: no cards returned")
        return 1

    difficulty = {"easy": 0, "medium": 0, "hard": 0, "?": 0}
    for c in cards:
        difficulty[c.get("difficulty", "?")] = difficulty.get(c.get("difficulty", "?"), 0) + 1

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_cards.json")
    with open(out, "w") as fh:
        json.dump(cards, fh, indent=2)

    print("-" * 72)
    print(f"cards        : {len(cards)}")
    print(f"difficulty   : {difficulty}")
    print(f"elapsed      : {elapsed:.1f}s")
    print(f"all have quote: {all(c.get('source_quote') for c in cards)}")
    print(f"saved        : {out}")
    print("-" * 72)

    for c in cards[:4]:
        print(f"Q: {c['question']}")
        print(f"A: {c['answer']}")
        print(f"   source: {c['source_quote'][:160]}")
        print(f"   difficulty: {c['difficulty']}")
        print()

    print("=" * 72)
    print("GEMINI TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
