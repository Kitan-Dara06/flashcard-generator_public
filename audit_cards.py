"""Offline audit of generated cards against their source document.

Answers "is this quote actually in the source?", "does any answer use a number
the source never mentions?", and "how close is each quote to verbatim?" with
data, so the grounding thresholds can be tuned from real output instead of
guessed. Makes no API calls.

    .venv/bin/python audit_cards.py gemini_cards_current.json "path/to/file.docx"

Reads a JSON list of card dicts (what the API returns / test_gemini_doc.py
writes).
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
    spec = importlib.util.spec_from_file_location("flashcards", path)
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


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: audit_cards.py <cards.json> <source.docx|txt>")
        return 2

    cards = json.load(open(sys.argv[1]))
    source = extract_text(sys.argv[2])
    fc = load_engine()
    src = fc.Source(source)

    print("=" * 74)
    print(f"Audit: {len(cards)} cards vs {len(source):,} chars of source")
    print("=" * 74)

    # ---- 1. quotes: exact, recoverable, missing ----
    exact = recoverable = missing = no_quote = 0
    longest_ratios = []
    for card in cards:
        quote = card.get("source_quote", "")
        if not quote.strip():
            no_quote += 1
            continue
        if fc.norm(quote) in src.norm:
            exact += 1
            longest_ratios.append(1.0)
            continue
        longest, _coverage = fc.quote_support(quote, src)
        longest_ratios.append(longest)
        if fc.quote_is_grounded(quote, src):
            recoverable += 1
        else:
            missing += 1

    judged = exact + recoverable + missing
    print("\n1. Source quotes")
    print(f"   verbatim                    : {exact}/{len(cards)}")
    print(f"   edited but recoverable      : {recoverable}/{len(cards)}")
    print(f"   not in the source (dropped) : {missing}/{len(cards)}")
    if no_quote:
        print(f"   no quote (allowed)          : {no_quote}/{len(cards)}")

    if longest_ratios:
        print("\n   Longest verbatim run, as a fraction of the quote:")
        buckets = {}
        for value in longest_ratios:
            key = min(int(value * 10) / 10, 0.9)
            buckets[key] = buckets.get(key, 0) + 1
        for key in sorted(buckets):
            print(f"     {key:.1f}-{key + 0.1:.1f}  {buckets[key]:3d}  "
                  + "#" * buckets[key])
        print(f"\n   Cards dropped by each GROUNDING_RUN_RATIO:")
        for threshold in (0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8):
            dropped = sum(1 for value in longest_ratios if value < threshold)
            pct = 100 * dropped / len(longest_ratios)
            marker = "  <-- current" if abs(threshold - fc.GROUNDING_RUN_RATIO) < 1e-9 else ""
            print(f"     {threshold:.2f}  drops {dropped:3d} ({pct:4.1f}%){marker}")

    # ---- 2. numbers ----
    bad_numbers = [c for c in cards if not fc.answer_numbers_grounded(
        c.get("answer", ""), src)]
    print(f"\n2. Answers using a number absent from the source: "
          f"{len(bad_numbers)}/{len(cards)}   (dropped by the pipeline)")
    for card in bad_numbers[:5]:
        used = fc.numbers_in(card.get("answer", "")) - src.numbers
        print(f"     - {card.get('question', '')[:60]}")
        print(f"       numbers not in source: {sorted(used)}")

    # ---- 3. duplicates ----
    print("\n3. Near-duplicate pairs")
    word_sets = [(set(fc.content_words(c.get("question", ""))),
                  set(fc.content_words(c.get("answer", "")))) for c in cards]
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

    # ---- 4. advisory: names absent from a card's own quote ----
    flagged = [c for c in cards
               if fc.ungrounded_names(c.get("answer", ""), c.get("source_quote", ""))]
    print(f"\n4. ADVISORY - answers naming something their quote omits: "
          f"{len(flagged)}/{len(cards)}")
    print("   (reported for review, never dropped)")
    for card in flagged[:3]:
        names = fc.ungrounded_names(card.get("answer", ""), card.get("source_quote", ""))
        print(f"     - {card.get('question', '')[:60]}")
        print(f"       names not in quote: {names[:5]}")

    print("\n" + "=" * 74)
    print(f"Current gates: GROUNDING_RUN_RATIO={fc.GROUNDING_RUN_RATIO}, "
          f"GROUNDING_COVERAGE={fc.GROUNDING_COVERAGE}")
    print("Every gate is exact, so this audit reproduces generation exactly.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
