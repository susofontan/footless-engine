"""Concurrent requests against a running `footless serve`: speed and sameness.

The same chat prompts go to the server one at a time, then 2, 3 ... N at once.
Each level reports the aggregate decode rate (completion tokens over the wall
time of the whole round) and the slowest request's own rate, and checks that
every answer is the one the prompt got alone: with greedy sampling (or a
pinned seed) a request's tokens must not depend on what else was running.

Usage (the server is started separately, e.g. `footless serve --mtp`):
    python3 tools/serve_concurrency.py --url http://127.0.0.1:8080 \
        [--max 4] [--tokens 256] [--temperature 0] [--thinking]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "Explain how a refrigerator moves heat out of its inside, step by step.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Describe the main differences between TCP and UDP, with an example use for each.",
    "Give a detailed recipe for a vegetable lasagna, with quantities and timings.",
    "Summarize the causes and consequences of the fall of the Western Roman Empire.",
    "Write a Python function that merges overlapping intervals, and explain it.",
    "What should someone consider before adopting a dog? Give a thorough list.",
    "Explain the idea of opportunity cost to a high-school student, with examples.",
]


def ask(url: str, prompt: str, args, seed: int, key: str | None) -> dict:
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.tokens, "temperature": args.temperature, "seed": seed}
    if not args.thinking:
        body["reasoning_effort"] = "none"
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        out = json.loads(resp.read())
    seconds = time.perf_counter() - t0
    message = out["choices"][0]["message"]
    return {"text": (message.get("reasoning") or "") + "|" + (message.get("content") or ""),
            "tokens": out["usage"]["completion_tokens"], "seconds": seconds}


def round_of(url: str, prompts: list[str], args, key,
             first: int = 0) -> tuple[list[dict], float]:
    """`prompts` at once; prompt i is PROMPTS[first + i], and so is its seed."""
    results: list[dict | None] = [None] * len(prompts)
    errors: list[BaseException] = []

    def one(i: int) -> None:
        try:
            results[i] = ask(url, prompts[i], args, 1000 + first + i, key)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--max", type=int, default=4, help="most requests at once")
    ap.add_argument("--tokens", type=int, default=256, help="max_tokens a request")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true", help="leave thinking on")
    ap.add_argument("--api-key")
    args = ap.parse_args()
    prompts = PROMPTS[:args.max]
    req = urllib.request.Request(args.url + "/v1/models")
    if args.api_key:
        req.add_header("Authorization", f"Bearer {args.api_key}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        args.model = json.loads(resp.read())["data"][0]["id"]
    ask(args.url, "Say hello.", args, 1, args.api_key)          # warm-up
    alone = []
    for i, prompt in enumerate(prompts):
        res, _ = round_of(args.url, [prompt], args, args.api_key, first=i)
        alone.append(res[0])
    rate = sum(r["tokens"] for r in alone) / sum(r["seconds"] for r in alone)
    print(f"1 at a time : {rate:5.1f} tok/s  "
          f"({sum(r['tokens'] for r in alone)} tokens)")
    same_all = True
    for n in range(2, args.max + 1):
        res, wall = round_of(args.url, prompts[:n], args, args.api_key)
        tokens = sum(r["tokens"] for r in res)
        slowest = min(r["tokens"] / r["seconds"] for r in res)
        same = [r["text"] == a["text"] for r, a in zip(res, alone)]
        same_all &= all(same)
        print(f"{n} at once   : {tokens / wall:5.1f} tok/s total, slowest request "
              f"{slowest:5.1f} tok/s, answers as alone: {sum(same)}/{n}")
    print("answers identical to the ones alone" if same_all
          else "SOME ANSWERS DIFFER from the ones alone")
    return 0 if same_all else 1


if __name__ == "__main__":
    sys.exit(main())
