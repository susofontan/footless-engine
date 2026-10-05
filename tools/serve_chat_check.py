"""/v1/chat/completions over HTTP (plain and streamed) against the engine alone.

The same chat request (fixed seed, the package's default sampling otherwise)
three ways: `engine.generate_tokens` on the rendered prompt, a non-streaming
POST, and a streaming POST read to the end. Same tokens, so the wall times are
directly comparable: the difference is the serving layer.

    python3 tools/serve_chat_check.py --model models/<model> [--mtp]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from footless import serve as serving                            # noqa: E402
from footless.engine import Batcher, Engine                      # noqa: E402
from footless.sdk import SamplingSpec                            # noqa: E402

MESSAGES = [{"role": "user", "content": "Explain in detail how a hash table works, "
                                        "with examples of collisions and how they are resolved."}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--speculative", "--mtp", dest="mtp", action="store_true")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--reasoning", default=None, metavar="LEVEL",
                    help="reasoning_effort: one of the package's thinking levels, or none")
    args = ap.parse_args()
    eng = Engine.open(args.model)
    eng.speculative(args.mtp)
    front = serving.FrontEnd(eng, eng.model.name, root=args.model)
    httpd = serving.ThreadingHTTPServer(("127.0.0.1", 8011), serving._Handler)
    httpd.daemon_threads = True
    httpd.front = front
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:8011"
    body = {"model": eng.model.name, "messages": MESSAGES, "max_tokens": args.tokens,
            "seed": 7}
    if args.reasoning is not None:
        body["reasoning_effort"] = args.reasoning

    def http(stream: bool):
        b = dict(body, stream=stream)
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(b).encode(),
                                     {"content-type": "application/json"})
        t0 = time.perf_counter()
        first = None
        with urllib.request.urlopen(req, timeout=600) as r:
            if not stream:
                out = json.loads(r.read())
                n = out["usage"]["completion_tokens"]
            else:
                n = 0
                for line in r:
                    if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                        first = first or time.perf_counter() - t0
                        n += 1
        return time.perf_counter() - t0, n, first

    results = {}

    def driver():
        try:
            for rep in range(2):
                for kind in ("engine", "http", "stream"):
                    eng_lock.acquire()
                    if kind == "engine":
                        work.append(None)
                        done.wait(); done.clear()
                        wall, n, first = results.pop("engine")
                    else:
                        eng_lock.release()
                        wall, n, first = http(kind == "stream")
                        eng_lock.acquire()
                    eng_lock.release()
                    print(f"  rep {rep} {kind:<7} {n:4d} {'chunks' if kind == 'stream' else 'tokens'} "
                          f"{wall:6.2f} s  -> {n / wall:5.1f} /s"
                          + (f"  first chunk {first:.2f} s" if first else ""), flush=True)
        finally:
            work.append("stop")
            front.close()

    # the engine runs on this thread (its device context lives here); the
    # direct-engine arm is executed here too, between FrontEnd jobs
    import collections
    work, done, eng_lock = collections.deque(), threading.Event(), threading.Lock()
    thinking = args.reasoning != "none"
    level = args.reasoning if thinking else None
    toks = eng.chat_tokens(MESSAGES, thinking, level)
    spec = replace(eng.facts.default_sampling or SamplingSpec(), seed=7)
    t = threading.Thread(target=driver, daemon=True)
    t.start()
    while True:
        if work:
            item = work.popleft()
            if item == "stop":
                break
            eng.cache.free_all()
            t0 = time.perf_counter()
            r = eng.generate_tokens(toks, args.tokens, sampling=spec)
            results["engine"] = (time.perf_counter() - t0, len(r.tokens), None)
            done.set()
            continue
        try:
            job = front._queue.get(timeout=0.01)
        except Exception:
            continue
        if job is None:
            break
        eng.cache.free_all()            # every arm starts cold, from the same seed
        Batcher(eng).run(front._task(job))
    t.join()
    httpd.shutdown()
    eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
