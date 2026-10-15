"""Read one job's cards straight from the Upstash REST API.

Prints only shapes/counts, never the token.

Usage: python3 inspect_job.py <job_id>
"""
import json
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))


def load_env(path: str) -> dict:
    env = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


def rest_get(host: str, token: str, key: str) -> str:
    url = f"https://{host}/get/{key}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read()).get("result") or ""


def main() -> int:
    env = load_env(os.path.join(BASE, ".env.local"))
    host = env.get("UPSTASH_REDIS_REST_URL", "")
    token = env.get("UPSTASH_REDIS_REST_TOKEN", "")
    if not host or not token:
        print("missing UPSTASH_REDIS_REST_URL / _TOKEN in .env.local")
        return 1
    host = host.replace("https://", "").replace("http://", "").rstrip("/")

    job_id = sys.argv[1] if len(sys.argv) > 1 else "fa4aea6a"
    for suffix in ("status", "progress", "error", "cards"):
        raw = rest_get(host, token, f"job:{job_id}:{suffix}")
        if suffix != "cards":
            print(f"{suffix:9s}: {raw[:200]}")
            continue
        cards = json.loads(raw) if raw else []
        print(f"cards    : {len(cards)} card(s)")
        if cards:
            print(f"  first card keys: {sorted(cards[0].keys())}")
            print(f"  first card: {json.dumps(cards[0], indent=2)[:800]}")
            with_quote = sum(1 for c in cards if (c.get("source_quote") or "").strip())
            print(f"  cards with source_quote: {with_quote}/{len(cards)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
