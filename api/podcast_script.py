# api/podcast_script.py
# Generates a podcast as a list of episodes, one per heading-aligned document
# section, driven ONE EPISODE PER REQUEST by the client.
#
#   Body { "text": ..., "cards": [...] }                 -> { "total": N }  (plan)
#   Body { "text": ..., "cards": [...], "episode": i }    -> { "episode": {...} }
#
# The cards are OPTIONAL and used only as a per-episode checklist of terms
# (definitions, names, dates) that must appear. The episode's structure still
# comes from the document's own sections, so the podcast and the deck stay two
# separate products that happen to share a fact source.
#
# One episode per request, not N in a single call: every script is its own LLM
# call, and 10 of them inside one Vercel function would exceed the 60 s limit
# every time. The client keeps what already succeeded and retries only the gaps.
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_flashcards import (
    EngineFailure,
    episode_sections,
    generate_json,
    json_response,
    map_cards_to_episodes,
    prompt_text,
    resolve_cors_headers,
    script_coverage,
)

SCRIPT_MODEL = os.environ.get("PODCAST_SCRIPT_MODEL", "gemini-2.5-flash")
MAX_TEXT_CHARS = 500_000


def schema():
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "enum": ["Alex", "Sam"]},
                        "text": {"type": "string"},
                    },
                    "required": ["speaker", "text"],
                },
            },
        },
        "required": ["title", "turns"],
    }


def episode_prompt(terms):
    """The podcast prompt, plus this episode's checklist and its length budget.

    Both are appended rather than baked into the YAML because they are derived
    per episode from the deck at request time. The budget is stated to the model
    from the same numbers band_problems()/thinness() reason about, so the two
    never disagree about what "too short" means.
    """
    base = prompt_text("podcast")
    if not terms:
        return base
    words, _lo, _hi = episode_budget(len(terms))
    return (base
            + f"\n\nThis section has {len(terms)} terms to cover, so aim for about "
              f"{words} words of dialogue. Cover every term; do not drop one to "
              "save space, and do not shorten the episode to look tidy.\n\n"
              + "These terms appear in this section and MUST appear in the "
              "conversation:\n- "
              + "\n- ".join(terms))


def _clean_turns(data):
    return [t for t in data.get("turns", [])
            if t.get("speaker") in ("Alex", "Sam") and str(t.get("text", "")).strip()]


# Structural bounds. Enforced after generation because a global word count is
# the one instruction this model ignores: measured spreads were 480-1976 words
# with single turns up to 267 words while the prompt asked for 700-1100. Turns
# and per-turn sentences are locally checkable, so they hold.
#
# Length is derived from the checklist instead of fixed. A flat band was tried
# first and cost real coverage (one episode fell 91% -> 65%): 35 terms cannot be
# named in 500 words. Since the stated goal is surface coverage, duration is
# allowed to follow content, and only the rambling-turn defect is clamped down.
WORDS_PER_TERM = int(os.environ.get("EPISODE_WORDS_PER_TERM", "32"))
WORDS_MIN = int(os.environ.get("EPISODE_WORDS_MIN", "450"))
WORDS_MAX = int(os.environ.get("EPISODE_WORDS_MAX", "1600"))
TURN_WORDS_MAX = int(os.environ.get("EPISODE_TURN_WORDS_MAX", "70"))
WORDS_PER_TURN = int(os.environ.get("EPISODE_WORDS_PER_TURN", "35"))
TURN_MIN = int(os.environ.get("EPISODE_TURNS_MIN", "12"))
TURN_MAX = int(os.environ.get("EPISODE_TURNS_MAX", "48"))


def episode_budget(term_count):
    """(words, min_turns, max_turns) sized to this episode's checklist.

    Terms drive words, words drive turns. The bands are deliberately wide: the
    aim is to stop 267-word monologues and 480-word nothing-burgers, not to make
    every episode the same length.
    """
    words = min(WORDS_MAX, max(WORDS_MIN, term_count * WORDS_PER_TERM))
    turns = words // WORDS_PER_TURN
    return words, max(TURN_MIN, int(turns * 0.6)), min(TURN_MAX, int(turns * 1.7))


def _word_count(turns):
    return sum(len(str(t.get("text", "")).split()) for t in turns)


def band_problems(turns, term_count=0):
    """Defects worth a rewrite, as readable strings. Empty means acceptable.

    Monologue only, deliberately. Both wider clamps were measured and removed:
    a turn ceiling suppressed coverage (an episode with 71 short turns scored
    88% and was flagged as too long), and a thinness retry made the model pad
    filler and cover fewer terms (83% -> 71%). Many short turns are exactly what
    a conversation should sound like; only a turn nobody can listen to is a bug.
    Use thinness() to report being short, which is information, not a defect.
    """
    problems = []
    longest = max((len(str(t.get("text", "")).split()) for t in turns), default=0)
    if longest > TURN_WORDS_MAX:
        problems.append(f"one turn is {longest} words; keep every turn under "
                        f"{TURN_WORDS_MAX} and at most 3 sentences")
    return problems


def thinness(turns, term_count):
    """How far under the checklist-sized target a script landed, for reporting.

    Surfaced in the response so a thin episode is visible and tunable, without
    triggering a rewrite that would trade coverage for filler.
    """
    target_words, _lo, _hi = episode_budget(term_count)
    words = _word_count(turns)
    if not term_count or words >= target_words * 0.6:
        return None
    return {"words": words, "target": target_words, "terms": term_count}


def band_score(turns, term_count):
    """How far this script is from the band, lower being better.

    Only monologues are scored on the shape side. Length is never penalised for
    being short: that would let a briefer, less complete script win the keep,
    which is the failure measured earlier. Extreme total length is discouraged,
    never rewritten.
    """
    words = _word_count(turns)
    longest = max((len(str(t.get("text", "")).split()) for t in turns), default=0)
    return (len(band_problems(turns, term_count)) * 1000
            + max(0, words - WORDS_MAX) * 2
            + max(0, longest - TURN_WORDS_MAX) * 5)


def generate_episode(text, cards, index):
    """One episode's script. `index` is a position in episode_sections(text).

    Retries once, at most, and only when the script is out of band - a second
    full generation costs seconds, and a token cap would instead truncate the
    JSON and fail the whole episode.
    """
    episodes = episode_sections(text)
    if index < 0 or index >= len(episodes):
        return None, len(episodes)
    ep = episodes[index]
    terms = map_cards_to_episodes(cards or [], [ep])[0]
    prompt = episode_prompt(terms)

    engine, raw = generate_json(prompt, ep["text"], schema(), temperature=0.7)
    data = json.loads(raw)
    turns = _clean_turns(data)
    problems = band_problems(turns, len(terms))

    if problems and turns:
        print(f"episode {index + 1} out of band ({', '.join(problems)}); rewriting once")
        correction = ("\n\nYour previous attempt broke the format: "
                      + "; ".join(problems)
                      + ". Rewrite it keeping the same coverage and the same hosts, "
                        "but inside the turn limits.")
        try:
            _engine2, raw2 = generate_json(prompt + correction, ep["text"], schema(),
                                           temperature=0.5)
            data2 = json.loads(raw2)
            turns2 = _clean_turns(data2)
        except Exception as exc:
            print(f"episode {index + 1}: rewrite failed ({exc}); keeping the first script")
            turns2 = []
        # Keep whichever attempt is nearer the band. A rewrite that made things
        # worse must not win just because it exists.
        if turns2 and band_score(turns2, len(terms)) < band_score(turns, len(terms)):
            data, turns = data2, turns2
            print(f"episode {index + 1}: kept the rewrite ({_word_count(turns)} words)")
        elif turns2:
            print(f"episode {index + 1}: rewrite was no better, keeping the first script")

    return {
        "index": index,
        "title": ep["title"],
        "hook": str(data.get("title", "")).strip() or ep["title"],
        "turns": turns,
        "terms": terms,
        "coverage": script_coverage(terms, turns),
        "words": _word_count(turns),
        # Describes the script actually returned, not the first attempt.
        "in_band": not band_problems(turns, len(terms)),
        "thin": thinness(turns, len(terms)),
    }, len(episodes)


def lambda_handler(event):
    try:
        if event["httpMethod"] != "POST":
            return json_response(405, {"success": False, "error": "Method not allowed"})

        try:
            body = json.loads(event["body"])
            text = str(body.get("text", "")).strip()
            cards = body.get("cards") or []
            index = body.get("episode")
        except (TypeError, ValueError):
            return json_response(400, {"success": False, "error": "Malformed request body"})

        if not text:
            return json_response(400, {"success": False, "error": "Missing text"})
        if len(text) > MAX_TEXT_CHARS:
            return json_response(400, {"success": False, "error": "Text too long"})
        if not isinstance(cards, list):
            return json_response(400, {"success": False, "error": "cards must be a list"})

        # Plan only: how many episodes this document will make. The client then
        # asks for one per request, exactly like the card chunks. Generating all
        # N scripts in this one function would be N LLM calls against a 60 s
        # limit - a 10-episode deck would always time out.
        if index is None:
            return json_response(200, {"success": True,
                                       "total": len(episode_sections(text))})

        try:
            index = int(index)
        except (TypeError, ValueError):
            return json_response(400, {"success": False, "error": "episode must be an integer"})

        try:
            episode, total = generate_episode(text, cards, index)
        except EngineFailure as exc:
            print(f"Episode {index} script failed ({exc.kind}):", repr(exc))
            status = 503 if exc.kind in ("quota", "transient") else 502
            return json_response(status, {"success": False, "episode": index,
                                          "retryable": exc.kind != "fatal",
                                          "error": "Script generation failed",
                                          "details": str(exc)})
        except Exception as exc:
            print(f"Episode {index} script failed:", repr(exc))
            return json_response(502, {"success": False, "episode": index,
                                       "error": "Script generation failed",
                                       "details": str(exc)})

        if episode is None:
            return json_response(400, {"success": False,
                                       "error": f"episode {index} out of range 0..{total - 1}"})
        if not episode["turns"]:
            return json_response(502, {"success": False, "episode": index,
                                       "error": "No turns generated"})

        return json_response(200, {"success": True, "total": total, "episode": episode})

    except Exception as e:
        import traceback
        print("Unhandled error:", str(e))
        print(traceback.format_exc())
        return json_response(500, {"success": False, "error": str(e)})


class handler(BaseHTTPRequestHandler):
    def _send(self, response):
        body = response["body"]
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(response["statusCode"])
        for k, v in response.get("headers", {}).items():
            self.send_header(k, v)
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
