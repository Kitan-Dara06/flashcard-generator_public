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
    cards = llm._generate_cards("irrelevant chunk text")

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
    return 0 if (ok and test_parse_cards()) else 1


if __name__ == "__main__":
    sys.exit(main())
