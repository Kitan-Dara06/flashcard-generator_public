#!/usr/bin/env python3
"""Audition every English Deepgram voice, sped to the rate you settled on.

Each clip is the voice saying its own name, then one sentence of your actual
content, so the comparison is like-for-like. Clips are also written individually
(voice_<name>.wav) so a favourite can be replayed alone.

    python3 audition_voices.py [--speed 1.1] [--gender masculine|feminine|all]
                               [--text "..."]

Needs DEEPGRAM_API_KEY in the environment or .env.local. Makes one request per
voice (~41 for English), so run it once.
"""
import argparse
import concurrent.futures
import json
import os
import struct
import subprocess
import sys
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

DEFAULT_TEXT = ("Univasa Nigeria Limited is a technology and telecommunications "
                "company, founded by Doctor Ben Adeniyi.")


def api_key():
    env = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if env:
        return env
    path = os.path.join(REPO, ".env.local")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.startswith("DEEPGRAM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def english_voices(key):
    req = urllib.request.Request("https://api.deepgram.com/v1/models",
                                 headers={"Authorization": f"Token {key}"})
    body = json.loads(urllib.request.urlopen(req, timeout=45).read().decode())
    out = []
    for m in body.get("tts") or []:
        if "aura-2" not in str(m.get("architecture", "")):
            continue
        langs = m.get("languages") or []
        if not any(str(l).startswith("en") for l in langs):
            continue
        md = m.get("metadata") or {}
        tags = md.get("tags") or []
        out.append({
            "name": m["name"],
            "model": m["canonical_name"],
            "accent": md.get("accent") or "?",
            "age": md.get("age") or "?",
            "gender": "masculine" if "masculine" in tags else (
                      "feminine" if "feminine" in tags else "?"),
            "tags": [t for t in tags if t not in ("masculine", "feminine")],
        })
    return sorted(out, key=lambda v: (v["gender"], v["name"]))


def speak(key, model, text):
    url = (f"https://api.deepgram.com/v1/speak?model={model}"
           f"&encoding=linear16&sample_rate=24000&container=wav")
    req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(),
                                 headers={"Authorization": f"Token {key}",
                                        "Content-Type": "application/json"},
                                 method="POST")
    return urllib.request.urlopen(req, timeout=90).read()


def set_speed(path, factor):
    """Tempo-preserving speed change, so 1.1x sounds like faster speech and not
    like a chipmunk. Deepgram's own speed/pitch params were measured to be
    unreliable, which is why this is done locally."""
    out = path + ".tmp.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-filter:a", f"atempo={factor}", out], check=True)
    os.replace(out, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=1.1)
    ap.add_argument("--gender", default="all", choices=["all", "masculine", "feminine"])
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--voices", default="",
                    help="comma-separated names to audition (default: all English)")
    args = ap.parse_args()

    key = api_key()
    if not key:
        print("DEEPGRAM_API_KEY not found")
        return 2

    voices = english_voices(key)
    if args.voices:
        want = [v.strip() for v in args.voices.split(",") if v.strip()]
        by_name = {v["name"]: v for v in voices}
        missing = [w for w in want if w not in by_name]
        if missing:
            print(f"unknown voices: {missing}")
        voices = [by_name[w] for w in want if w in by_name]
    elif args.gender != "all":
        voices = [v for v in voices if v["gender"] == args.gender]
    print(f"{len(voices)} English aura-2 voices, {args.speed}x speed\n")
    for v in voices:
        print(f"  {v['name']:12} {v['gender']:10} {v['accent']:16} {v['age']:12} "
              f"{', '.join(v['tags'][:4])}")

    def render(v):
        # The voice introduces itself, so one long file can still be followed.
        audio = speak(key, v["model"], f"{v['name']}. {args.text}")
        path = os.path.join(REPO, f"voice_{v['name']}.wav")
        with open(path, "wb") as fh:
            fh.write(audio)
        if args.speed != 1.0:
            set_speed(path, args.speed)
        rate = struct.unpack("<I", open(path, "rb").read()[24:28])[0]
        size = os.path.getsize(path)
        return v["name"], (size - 44) / (rate * 2)

    done = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, secs in pool.map(render, voices):
            done[name] = secs
            print(f"  rendered {name:12} {secs:4.1f}s")

    # Concatenate in catalogue order into one audition file.
    order = [v for v in voices if v["name"] in done]
    listed = "\n".join(f"file 'voice_{v['name']}.wav'" for v in order)
    with open("voices_list.txt", "w") as fh:
        fh.write(listed)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", "voices_list.txt", "-c", "copy", "deepgram_voices_audition.wav"],
                   check=True)
    os.remove("voices_list.txt")

    total = sum(done.values())
    print(f"\ndeepgram_voices_audition.wav  {total/60:.1f} min, {len(order)} voices")
    print("individual files: voice_<name>.wav")
    print("\nwarm order (orion is one of these):")
    for v in order:
        print(f"  {v['name']:12} {v['gender']:10} {', '.join(v['tags'][:5])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
