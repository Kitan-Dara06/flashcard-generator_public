"""Measure: parallel chunk requests vs one streaming request.

Answers a design question - is SSE worth it, or is firing the section groups
concurrently enough? Real timing, real API, real document.

    .venv/bin/python bench_parallel.py <file.docx>
"""
import importlib.util
import json
import os
import sys
import time
import types


def load_engine():
    for name in ("modal", "redis", "upstash_redis", "pdfplumber", "pptx"):
        if name not in sys.modules:
            fake = types.ModuleType(name)
            if name == "modal":
                fake.exception = types.SimpleNamespace(
                    NotFoundError=type("NF", (Exception,), {}))
                fake.Cls = types.SimpleNamespace(from_name=lambda *a, **k: None)
            sys.modules[name] = fake
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "api", "generate_flashcards.py")
    spec = importlib.util.spec_from_file_location("engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_key():
    for line in open(".env.local"):
        if line.strip().startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("GEMINI_API_KEY", "")


def main():
    path = sys.argv[1]
    os.environ["GEMINI_API_KEY"] = load_key()
    engine = load_engine()

    raw = open(path, "rb").read()
    text = engine.extract_text_from_docx(raw)
    chunks = engine.section_chunks(text)
    print(f"document : {len(text):,} chars -> {len(chunks)} section group(s)")
    print(f"sizes    : {[f'{len(c)//1000}k' for c in chunks]}")
    print()

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def one(chunk, streaming=False):
        config = types.GenerateContentConfig(
            system_instruction=engine.prompt_text("card_gemini"),
            response_mime_type="application/json",
            response_schema=engine.card_schema(),
            temperature=0.1,
            max_output_tokens=32768,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        # Retry transient overloads: Gemini returns 503 under demand spikes.
        last = None
        for attempt in range(1, 4):
            try:
                if streaming:
                    first = None
                    text_parts = []
                    for part in client.models.generate_content_stream(
                            model=engine.GEMINI_MODEL, contents=chunk, config=config):
                        if first is None:
                            first = time.time()
                        text_parts.append(part.text or "")
                    return time.time() - first, sum(len(t) for t in text_parts), "".join(text_parts)
                t0 = time.time()
                resp = client.models.generate_content(
                    model=engine.GEMINI_MODEL, contents=chunk, config=config)
                return time.time() - t0, len(resp.text or ""), resp.text or ""
            except Exception as exc:
                last = exc
                if attempt == 3 or "503" not in str(exc):
                    raise
                wait = 2 ** attempt
                print(f"    (503, retrying in {wait}s)")
                time.sleep(wait)
        raise last

    # --- serial baseline (what the app does now) ---
    print("serial (current app behaviour):")
    t0 = time.time()
    for i, c in enumerate(chunks, 1):
        took, size, _ = one(c)
        print(f"  chunk {i}/{len(chunks)}: {took:5.1f}s")
    serial = time.time() - t0
    print(f"  TOTAL: {serial:.1f}s")
    print()

    # --- parallel ---
    print("parallel (all chunks at once):")
    import concurrent.futures
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(one, c) for c in chunks]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            took, size, _ = f.result()
            print(f"  chunk finished {i}/{len(chunks)}: {took:5.1f}s")
    parallel = time.time() - t0
    print(f"  TOTAL: {parallel:.1f}s")
    print()

    # --- streaming (SSE) on one chunk ---
    print("streaming, one chunk (time to first byte vs total):")
    took, size, _ = one(chunks[0], streaming=True)
    print(f"  chunk 1: first byte at {took:.1f}s, {size:,} chars streamed")
    print()

    print("=" * 62)
    print(f"serial   : {serial:6.1f}s")
    print(f"parallel : {parallel:6.1f}s  ({serial / parallel:.1f}x faster)")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
