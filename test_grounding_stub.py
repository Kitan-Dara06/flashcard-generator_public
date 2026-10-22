"""Stub tests for the worker's card parsing.

Covers two failures seen in production:
  * cards that arrive without a `source_quote` must be dropped (grounding)
  * a response cut off by max_tokens must still yield its complete cards

Fakes the vLLM imports so there is no GPU and no network.

Run:  python3 test_grounding_stub.py
"""
import json
import sys
import types
from unittest import mock


def _install_vllm_stubs(response_json: dict):
    """Inject fake `vllm` (+ submodules) whose .chat() returns response_json."""
    fake_output = types.SimpleNamespace(
        outputs=[types.SimpleNamespace(text=json.dumps(response_json))]
    )

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

        def chat(self, messages, sampling, use_tqdm=False):
            return [fake_output]

    class FakeSamplingParams:
        def __init__(self, *a, **k):
            pass

    class FakeGuidedDecodingParams:
        def __init__(self, *a, **k):
            pass

    vllm = types.ModuleType("vllm")
    vllm.LLM = FakeLLM
    vllm.SamplingParams = FakeSamplingParams

    vllm_sp = types.ModuleType("vllm.sampling_params")
    vllm_sp.GuidedDecodingParams = FakeGuidedDecodingParams

    vllm.sampling_params = vllm_sp
    vllm.outputs = None

    sys.modules["vllm"] = vllm
    sys.modules["vllm.sampling_params"] = vllm_sp


def test_parse_cards() -> bool:
    import modal_app

    ok = True

    # 1. Well-formed response parses normally.
    good = json.dumps({"flashcards": [
        {"question": "q1", "answer": "a1", "source_quote": "s1", "difficulty": "easy"},
        {"question": "q2", "answer": "a2", "source_quote": "s2", "difficulty": "hard"},
    ]})
    cards, truncated = modal_app._parse_cards(good)
    if truncated or len(cards) != 2:
        print(f"FAIL: valid JSON should parse cleanly, got {cards} truncated={truncated}")
        ok = False

    # 2. Response cut off mid-string (the production JSONDecodeError): the two
    #    complete cards survive, the half-written third one is discarded.
    cut = good[:-25]   # slice inside the last card's strings
    cards, truncated = modal_app._parse_cards(cut)
    if not truncated:
        print("FAIL: truncated response was not flagged")
        ok = False
    if len(cards) != 1:
        print(f"FAIL: expected 1 salvaged card, got {len(cards)}: {cards}")
        ok = False
    if cards and cards[0].get("source_quote") != "s1":
        print(f"FAIL: salvaged the wrong card: {cards}")
        ok = False

    # 3. Nothing usable at all.
    cards, truncated = modal_app._parse_cards("totally not json")
    if cards != [] or not truncated:
        print(f"FAIL: garbage input should salvage nothing, got {cards}")
        ok = False

    if ok:
        print("PASS: _parse_cards handles valid, truncated and garbage output")
    return ok


def _load_fallback_module():
    """Import api/generate_flashcards.py without needing pdfplumber installed."""
    import importlib.util
    import os
    import sys
    import types

    if "pdfplumber" not in sys.modules:
        try:
            import pdfplumber  # noqa: F401
        except ImportError:
            sys.modules["pdfplumber"] = types.ModuleType("pdfplumber")

    path = os.path.join("api", "generate_flashcards.py")
    spec = importlib.util.spec_from_file_location("fallback_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fallback_helpers() -> bool:
    """Chunking, dedup and grounding checks used by the Gemini fallback."""
    import modal_app

    ok = True
    api = _load_fallback_module()

    # Chunking: count is capped, and a small document stays inside the budget.
    small = "\n\n".join(f"Paragraph {i}. " + ("word " * 40) for i in range(12))
    chunks = api.paragraph_chunks(small, max_chars=4000, max_chunks=4)
    if not chunks or len(chunks) > 4:
        print(f"FAIL: expected <=4 chunks, got {len(chunks)}")
        ok = False
    for c in chunks:
        if len(c) > 4000:
            print(f"FAIL: chunk over budget ({len(c)} chars)")
            ok = False
            break
    if len(" ".join(chunks).split()) < len(small.split()) * 0.95:
        print("FAIL: chunking dropped a significant amount of text")
        ok = False

    # A large document trades a bigger per-chunk budget for the chunk-count cap.
    big = "\n\n".join(f"Paragraph {i}. " + ("word " * 400) for i in range(12))
    big_chunks = api.paragraph_chunks(big, max_chars=4000, max_chunks=4)
    if len(big_chunks) > 4:
        print(f"FAIL: large doc exceeded max_chunks ({len(big_chunks)})")
        ok = False
    if len(" ".join(big_chunks).split()) < len(big.split()) * 0.95:
        print("FAIL: large-doc chunking dropped text")
        ok = False

    # One oversized paragraph must still be hard-split.
    blob_chunks = api.paragraph_chunks("word " * 5000, max_chars=4000, max_chunks=4)
    if len(blob_chunks) < 2:
        print("FAIL: oversized single paragraph was not hard-split")
        ok = False

    # Near-duplicate questions collapse; distinct ones survive.
    cards = [
        {"question": "What year was SIWES established in Nigeria?"},
        {"question": "What year was SIWES established in Nigeria"},
        {"question": "Who initially funded the SIWES programme?"},
    ]
    if len(api.text_dedup(cards)) != 2:
        print("FAIL: text_dedup did not collapse the duplicate")
        ok = False

    # ---- grounding gates ----
    source = ("SIWES was established in 1973 by the Industrial Training Fund "
              "to develop manpower in Nigeria.")

    good = {"question": "When was SIWES established?", "answer": "In 1973.",
            "source_quote": "SIWES was established in 1973 by the Industrial Training Fund",
            "difficulty": "easy"}
    # quote invented out of thin air
    invented = {"question": "Who invented SIWES?", "answer": "The Ford Foundation.",
                "source_quote": "SIWES was created by the Ford Foundation in Detroit.",
                "difficulty": "hard"}
    # quote real, but the answer invents a date
    bad_number = {"question": "When did SIWES start?", "answer": "In 1974.",
                  "source_quote": "SIWES was established in 1973",
                  "difficulty": "easy"}

    kept, dropped = api.clean_cards([good, invented, bad_number], source)
    if len(kept) != 1 or kept[0]["question"] != good["question"]:
        print(f"FAIL: grounding gates kept the wrong cards: {kept}")
        ok = False
    if dropped != 2:
        print(f"FAIL: expected 2 drops, got {dropped}")
        ok = False

    # A paraphrased-but-real quote should still pass (fuzzy, not exact).
    reflowed = {"question": "What did SIWES aim to develop?",
                "answer": "Manpower in Nigeria.",
                "source_quote": "established in 1973 by the industrial training fund to develop manpower in Nigeria",
                "difficulty": "medium"}
    kept, _ = api.clean_cards([reflowed], source)
    if len(kept) != 1:
        print("FAIL: a real quote that was re-cased/re-spaced was rejected")
        ok = False

    # Cleaning defaults a nonsense difficulty and drops empty cards.
    cards, dropped = modal_app._clean_cards([
        {"question": "q", "answer": "a", "source_quote": "s", "difficulty": "weird"},
        {"question": "q2", "answer": "a2", "source_quote": ""},
    ])
    if dropped != 1 or len(cards) != 1 or cards[0]["difficulty"] != "medium":
        print(f"FAIL: _clean_cards wrong: dropped={dropped} cards={cards}")
        ok = False

    if ok:
        print("PASS: fallback chunking, dedup and grounding gates behave")
    return ok


def main() -> int:
    response = {
        "flashcards": [
            {
                "question": "What year was SIWES established?",
                "answer": "1973.",
                "source_quote": "SIWES was established in 1973 by the ITF.",
                "difficulty": "easy",
            },
            {  # missing quote -> must be dropped
                "question": "Who invented the transistor?",
                "answer": "Bardeen, Brattain and Shockley.",
                "source_quote": "",
                "difficulty": "hard",
            },
            {  # whitespace-only quote -> must be dropped
                "question": "What is the capital of France?",
                "answer": "Paris.",
                "source_quote": "   ",
                "difficulty": "easy",
            },
            {  # missing difficulty -> defaults to medium, kept
                "question": "What does SIWES stand for?",
                "answer": "Students' Industrial Work Experience Scheme.",
                "source_quote": "The Students' Industrial Work Experience Scheme (SIWES)",
                "difficulty": "",
            },
        ]
    }
    _install_vllm_stubs(response)

    import modal_app

    # @app.cls wraps the class; unwrap it and skip __init__/load.
    user_cls = modal_app.FlashcardLLM._get_user_cls()
    llm = user_cls.__new__(user_cls)
    llm.llm = sys.modules["vllm"].LLM()  # fake engine returning the canned response

    # The chunk must actually contain the quotes the model claims to have copied
    # from it, otherwise the grounding gates (correctly) drop every card.
    chunk = (
        "SIWES was established in 1973 by the ITF. "
        "The Students' Industrial Work Experience Scheme (SIWES) is a programme."
    )
    cards = llm._generate_cards(chunk)

    ok = True
    if len(cards) != 2:
        print(f"FAIL: expected 2 kept cards, got {len(cards)}: {cards}")
        ok = False
    if any(not c.get("source_quote") for c in cards):
        print(f"FAIL: ungrounded card survived: {cards}")
        ok = False
    if cards and cards[-1].get("difficulty") != "medium":
        print(f"FAIL: blank difficulty did not default to medium: {cards}")
        ok = False
    if ok:
        print(f"PASS: kept {len(cards)}/4 cards, dropped 2 ungrounded; "
              f"difficulty default works")
    return 0 if (ok and test_parse_cards() and test_fallback_helpers()) else 1


if __name__ == "__main__":
    sys.exit(main())
