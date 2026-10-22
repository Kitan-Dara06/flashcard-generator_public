"""Offline audit of generated cards against their source document.

Answers "is the answer actually supported by its own source_quote?" with data,
so the ANSWER_QUOTE_MIN_COVERAGE cut-off can be tuned from real output instead
of guessed. Makes no API calls.

    .venv/bin/python audit_cards.py gemini_cards.json "path/to/file.docx"

Reads cards saved by test_gemini_doc.py (a JSON list of card dicts).
"""
import importlib.util
import json
import os
import sys
import types
from itertools import combinations

import docx


def load_engine():
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


def extract_text(path: str) -> str:
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


def jaccard(a, b) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def bucket(value: float) -> str:
    low = min(int(value * 10) / 10, 0.9)
    return f"{low:.1f}-{low + 0.1:.1f}"


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: audit_cards.py <cards.json> <source.docx|txt>")
        return 2

    cards = json.load(open(sys.argv[1]))
    source = extract_text(sys.argv[2])
    engine = load_engine()
    source_norm = engine.norm(source)

    print("=" * 74)
    print(f"Audit: {len(cards)} cards vs {len(source):,} chars of source")
    print("=" * 74)

    # ---- 1. quote authenticity ----
    exact = fuzzy = missing = 0
    for c in cards:
        q = engine.norm(c.get("source_quote", ""))
        if q and q in source_norm:
            exact += 1
        elif engine.quote_is_grounded(c.get("source_quote", ""), source_norm):
            fuzzy += 1
        else:
            missing += 1
    print("\n1. Source quotes")
    print(f"   exact substring of source : {exact}/{len(cards)}")
    print(f"   re-flowed but present     : {fuzzy}/{len(cards)}")
    print(f"   not in source             : {missing}/{len(cards)}")

    # ---- 2. answer containment ----
    scores = [engine.answer_coverage(c.get("answer", ""), c.get("source_quote", ""))
              for c in cards]
    print("\n2. Answer coverage by its own quote (content words)")
    buckets = {}
    for s in scores:
        buckets[bucket(s)] = buckets.get(bucket(s), 0) + 1
    for key in sorted(buckets):
        bar = "#" * buckets[key]
        print(f"   {key}  {buckets[key]:3d}  {bar}")
    mean = sum(scores) / len(scores) if scores else 0
    print(f"   mean {mean:.2f} | min {min(scores):.2f} | max {max(scores):.2f}")

    print("\n   Cards dropped at each threshold:")
    for t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70):
        dropped = sum(1 for c, s in zip(cards, scores)
                      if len(engine.content_words(c.get("answer", ""))) > 6 and s < t)
        pct = 100 * dropped / len(cards) if cards else 0
        marker = "  <-- current" if abs(t - engine.ANSWER_QUOTE_MIN_COVERAGE) < 1e-9 else ""
        print(f"     {t:.2f}  drops {dropped:3d} ({pct:4.1f}%){marker}")

    # ---- 3. duplicates ----
    print("\n3. Near-duplicate pairs")
    word_sets = [(set(engine.content_words(c.get("question", ""))),
                  set(engine.content_words(c.get("answer", "")))) for c in cards]
    for label, idx, thresholds in (("question", 0, (0.5, 0.6, 0.7, 0.8)),
                                   ("answer", 1, (0.7, 0.8, 0.85, 0.9))):
        counts = {t: 0 for t in thresholds}
        for (i, (q1, a1)), (j, (q2, a2)) in combinations(list(enumerate(word_sets)), 2):
            value = jaccard((q1, a1)[idx], (q2, a2)[idx])
            for t in thresholds:
                if value >= t:
                    counts[t] += 1
        line = " | ".join(f">{t}: {counts[t]}" for t in thresholds)
        print(f"   {label} pairs: {line}")

    # ---- 4. numbers and names ----
    bad_numbers = sum(1 for c in cards
                      if not engine.answer_numbers_grounded(
                          c.get("answer", ""), source_norm))
    flagged = [c for c in cards
               if engine.ungrounded_names(c.get("answer", ""), c.get("source_quote", ""))]
    print(f"\n4. Answers with a number absent from the source: {bad_numbers}/{len(cards)}")
    print(f"   ADVISORY - answers naming something their quote omits: {len(flagged)}/{len(cards)}")
    print("   (reported for review, not dropped)")
    if flagged:
        print("   Examples:")
        for c in flagged[:3]:
            names = engine.ungrounded_names(c.get("answer", ""), c.get("source_quote", ""))
            print(f"     - {c.get('question','')[:60]}")
            print(f"       names not in quote: {names[:5]}")

    # ---- 5. recommendation ----
    print("\n" + "=" * 74)
    target = 0.10 * len(cards)
    suggestion = engine.ANSWER_QUOTE_MIN_COVERAGE
    for t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        dropped = sum(1 for c, s in zip(cards, scores)
                      if len(engine.content_words(c.get("answer", ""))) > 6 and s < t)
        if dropped <= target:
            suggestion = t
    print(f"Highest threshold dropping <=10% of cards: {suggestion:.2f}")
    print(f"Set it with ANSWER_QUOTE_MIN_COVERAGE={suggestion:.2f}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
