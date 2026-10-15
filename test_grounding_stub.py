"""Stub test for the grounding drop rule in FlashcardLLM._generate_cards.

Fakes the vLLM imports so we can feed a canned model response and assert that
cards missing `source_quote` are dropped. No GPU, no network.

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
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
