"""The engine side of serve_check's requests, without HTTP.

serve_check's completion body through `generate_tokens` (with its prefill and
decode split) and its first decision question through `label_logprobs`, each
timed alone. Set beside serve_check's numbers for the same package, the
difference is the serving layer's own cost.

    python3 tools/serve_profile.py --model models/<model> [--mtp]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import serve_check as sc                                         # noqa: E402
from footless import decisions, wire                            # noqa: E402
from footless.engine import Engine                               # noqa: E402
from footless.sdk import LABEL_LOGPROBS                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--speculative", "--mtp", dest="mtp", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    eng = Engine.open(args.model)
    eng.speculative(args.mtp)
    rt = eng.runtime
    spec = eng.facts.default_sampling
    toks = rt.encode(sc.COMPLETION_PROMPT)
    print(f"default sampling: {spec}")
    print(f"completion prompt: {len(toks)} tokens, {sc.COMPLETION_TOKENS} generated")
    for i in range(args.runs + 1):
        eng.cache.free_all()
        t0 = time.perf_counter()
        r = eng.generate_tokens(toks, sc.COMPLETION_TOKENS, sampling=spec)
        wall = time.perf_counter() - t0
        t = r.timing
        if i:
            print(f"  engine: wall {wall:.2f} s  prefill {t.prefill_seconds:.2f} s  decode "
                  f"{t.decode_seconds:.2f} s ({t.decode_tokens / t.decode_seconds:.1f} t/s)  "
                  f"-> {len(r.tokens) / wall:.1f} t/s end to end")

    # one decision question, the way _run_decisions renders it
    if LABEL_LOGPROBS not in eng.facts.capabilities:
        print("decision question: skipped, the package declares no label_logprobs")
    else:
        text = decisions.render_text(sc.DECISIONS_BODY["input"])
        req = wire.parse_decisions(sc.DECISIONS_BODY, {eng.model.name})
        view = wire.decision_view(req.questions[0])
        labels = decisions.default_labels(view)
        content = decisions.render_question(text, view, labels)
        pids = eng.chat_tokens([{"role": "user", "content": content}], thinking=False)
        print(f"decision question: {len(pids)} prompt tokens")
        for i in range(args.runs + 1):
            t0 = time.perf_counter()
            eng.label_logprobs(pids, labels)
            if i:
                print(f"  label_logprobs {time.perf_counter() - t0:.3f} s")
    eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
