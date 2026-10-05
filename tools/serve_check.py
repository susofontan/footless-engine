"""Start `footless serve`, exercise three of its routes, report tps, stop.

One warmup request and --runs measured ones per route, over real HTTP against a
server this script starts: POST /v1/completions (reported as generated tokens
per second), POST /v1/decisions and POST /v1/systemone (reported as scored
input tokens per second -- they generate nothing). Every response is checked
against its wire shape and its answer math, so one run reports correctness and
speed together. The server stops once the results are printed -- always,
including when a check fails.

`--mtp` is the switch `footless chat --mtp` takes, applied here before serving:
the package drafts ahead and verifies its own draft. A package that does not
declare `speculative` is refused with the reason (CONTRACT.md, rule 4) and this
exits 2.

Usage:
    python3 tools/serve_check.py --model models/<model> [--mtp]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from footless import serve as serving
from footless.engine import DEFAULT_BUDGET, Engine
from footless.sdk import LABEL_LOGPROBS, ContractError

# A prompt that spends its budget on content instead of stopping early, so the
# completion tokens are a decode measurement rather than an end-of-answer one.
COMPLETION_PROMPT = (
    "Write a long, detailed history of bread making, from ancient Egypt to "
    "modern industrial bakeries, with many sections and concrete details:\n\n"
)
COMPLETION_TOKENS = 96

# The documented examples (PLAN_DECISIONS.md §2.1): one state and three typed
# questions per route, so the two decision routes score comparable work.
STATE = (
    "I have been trying to connect my Stripe account for 3 days and the "
    "integration keeps failing. I am losing sales."
)
DECISIONS_BODY = {
    "input": STATE,
    "questions": [
        {
            "id": "team",
            "type": "choice",
            "question": "Which team should handle this ticket?",
            "options": [
                {"name": "billing", "description": "Payment or subscription issues"},
                {"name": "technical", "description": "Bugs or integration problems"},
                {"name": "sales", "description": "Pricing or account questions"},
            ],
        },
        {
            "id": "frustration",
            "type": "score",
            "question": "How frustrated is the customer?",
            "levels": ["Calm", "Frustrated but civil", "Very angry"],
        },
        {"id": "urgent", "type": "yes_no", "question": "The customer needs an answer today."},
    ],
}
SYSTEMONE_BODY = {
    "state": STATE,
    # Any name answers, whatever the request calls the model; the response
    # reports the served one (PLAN_DECISIONS.md §3).
    "model": "jev-latest",
    "questions": {
        "urgent": {"type": "noul", "instructions": "The customer needs an answer today."},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": None,
                "technical": "Bugs or integration problems",
                "sales": None,
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated but civil", "Very angry"],
        },
    },
}

TOL = 1e-6


def post(base: str, path: str, body: dict, timeout: float = 600.0):
    """One JSON POST: (status, parsed body, wall seconds)."""
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
            status = response.status
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read())
        status = exc.code
    return status, payload, time.perf_counter() - started


def measure(base: str, path: str, body: dict, runs: int):
    """Warmup plus `runs` measured requests; returns the measured triples.

    The warmup absorbs the clock ramp and any first-call cost, so the measured
    runs are comparable to each other. Per-run numbers stay in the report: this
    machine throttles when hot, and a mean alone hides the drift.
    """
    status, _, _ = post(base, path, body)
    if status != 200:
        raise RuntimeError(f"{path} warmup answered {status}")
    return [post(base, path, body) for _ in range(runs)]


# ---------------------------------------------------------------- checks


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL


def check_completions(payload: dict) -> list[str]:
    """Wire shape of a text completion; failures name what is wrong."""
    bad = []
    if payload.get("object") != "text_completion":
        bad.append(f"object is {payload.get('object')!r}, not 'text_completion'")
    choices = payload.get("choices") or []
    if len(choices) != 1:
        bad.append(f"{len(choices)} choices, expected 1")
    elif not choices[0].get("text"):
        bad.append("choice text is empty")
    elif choices[0].get("finish_reason") not in ("stop", "length"):
        bad.append(f"finish_reason is {choices[0].get('finish_reason')!r}")
    usage = payload.get("usage") or {}
    if usage.get("completion_tokens", 0) < 1:
        bad.append(f"usage.completion_tokens is {usage.get('completion_tokens')!r}")
    return bad


def check_decisions(payload: dict) -> list[str]:
    bad = []
    if payload.get("object") != "decisions":
        bad.append(f"object is {payload.get('object')!r}, not 'decisions'")
    if payload.get("prompt_format_version") != 1:
        bad.append(f"prompt_format_version is {payload.get('prompt_format_version')!r}")
    answers = payload.get("answers") or {}
    if set(answers) != {"team", "frustration", "urgent"}:
        bad.append(f"answers are {sorted(answers)}")
        return bad
    for name, answer in answers.items():
        probs = (answer.get("probabilities") or {}).values()
        if not probs or not _close(sum(probs), 1.0):
            bad.append(f"{name}: probabilities do not sum to 1")
        mass = answer.get("label_mass", -1.0)
        if not (0 < mass <= 1 + TOL):
            bad.append(f"{name}: label_mass is {mass}, expected in (0, 1]")
    if answers["team"].get("choice") not in ("billing", "technical", "sales"):
        bad.append(f"team: choice is {answers['team'].get('choice')!r}")
    score = answers["frustration"].get("score", -1.0)
    if not (0 <= score <= 2):
        bad.append(f"frustration: score is {score}, expected in [0, 2]")
    if set(answers["urgent"].get("probabilities") or {}) != {"yes", "no"}:
        bad.append("urgent: probabilities are not keyed yes/no")
    usage = payload.get("usage") or {}
    if usage.get("completion_tokens") != 0:
        bad.append(f"completion_tokens is {usage.get('completion_tokens')!r}, expected 0")
    return bad


def check_systemone(payload: dict, model_name: str) -> list[str]:
    bad = []
    if payload.get("model") != model_name:
        bad.append(f"model is {payload.get('model')!r}, expected the served {model_name!r}")
    answers = payload.get("answers") or {}
    if set(answers) != {"urgent", "team", "frustration"}:
        bad.append(f"answers are {sorted(answers)}")
        return bad
    for name, answer in answers.items():
        mass = answer.get("x_label_mass", -1.0)
        if not (0 < mass <= 1 + TOL):
            bad.append(f"{name}: x_label_mass is {mass}, expected in (0, 1]")
    noul = answers["urgent"].get("noul", -1.0)
    if answers["urgent"].get("type") != "noul" or not (0 <= noul <= 1):
        bad.append(f"urgent: type/noul wrong: {answers['urgent']}")
    team = answers["team"]
    probs = (team.get("probabilities") or {}).values()
    if team.get("type") != "choice" or team.get("choice") not in ("billing", "technical", "sales"):
        bad.append(f"team: type/choice wrong: {team}")
    if not probs or not _close(sum(probs), 1.0):
        bad.append("team: probabilities do not sum to 1")
    if not (0 <= team.get("confidence", -1) <= 1):
        bad.append(f"team: confidence is {team.get('confidence')}")
    mood = answers["frustration"]
    probs = (mood.get("probabilities") or {}).values()
    if mood.get("type") != "score" or not (0 <= mood.get("score", -1) <= 2):
        bad.append(f"frustration: type/score wrong: {mood}")
    if not probs or not _close(sum(probs), 1.0):
        bad.append("frustration: probabilities do not sum to 1")
    if mood.get("legend") != {"0": "Calm", "1": "Frustrated but civil", "2": "Very angry"}:
        bad.append(f"frustration: legend is {mood.get('legend')!r}")
    usage = payload.get("usage") or {}
    if usage.get("output_tokens") != 0:
        bad.append(f"output_tokens is {usage.get('output_tokens')!r}, expected 0")
    return bad


# ---------------------------------------------------------------- report


def report(route: str, runs: list, tokens, per_second) -> None:
    """One route's rows: wall per run and the tps each run implies."""
    print(f"== {route} ==")
    rates = []
    for index, (_, payload, wall) in enumerate(runs, 1):
        rate = per_second(payload) / wall
        rates.append(rate)
        print(f"  run {index}: {tokens(payload):4d} tokens  {wall:6.2f} s  {rate:6.1f} t/s")
    print(f"  mean {sum(rates) / len(rates):6.1f} t/s   min {min(rates):.1f}   "
          f"max {max(rates):.1f}")


# ---------------------------------------------------------------- driver


def driver(base: str, model_name: str, runs: int, front, scoring: bool = True) -> int:
    """The three routes, on a thread of their own: the engine runs on main.

    `front.close()` is in a finally because it is what unblocks the engine
    thread: a driver that dies without it would hang the shutdown.
    """
    checks: list[tuple[bool, str]] = []
    try:
        try:
            completions = measure(base, "/v1/completions",
                                  {"prompt": COMPLETION_PROMPT,
                                   "max_tokens": COMPLETION_TOKENS}, runs)
            decisions = systemone = []
            if scoring:
                decisions = measure(base, "/v1/decisions", DECISIONS_BODY, runs)
                systemone = measure(base, "/v1/systemone", SYSTEMONE_BODY, runs)
            else:
                print("serve_check: decision routes skipped: the package declares no "
                      "label_logprobs", file=sys.stderr)
        except RuntimeError as exc:
            print(f"serve_check: {exc}", file=sys.stderr)
            checks.append((False, str(exc)))
            completions, decisions, systemone = [], [], []

        for name, measured, checker in (
            ("/v1/completions", completions, check_completions),
            ("/v1/decisions", decisions, check_decisions),
            ("/v1/systemone", systemone, lambda p: check_systemone(p, model_name)),
        ):
            if not measured:
                continue  # the failure is already in checks
            problems = checker(measured[-1][1])
            if problems:
                checks.extend((False, f"{name}: {p}") for p in problems)
            else:
                checks.append((True, f"{name}: shape and math as specified"))

        print()
        if completions:
            # tps here is generated tokens per wall second, prefill included
            report("/v1/completions  (out)", completions,
                   lambda p: p["usage"]["completion_tokens"],
                   lambda p: p["usage"]["completion_tokens"])
        if decisions:
            # tps here is scored input tokens per wall second: no generation
            report("/v1/decisions     (in)", decisions,
                   lambda p: p["usage"]["prompt_tokens"],
                   lambda p: p["usage"]["prompt_tokens"])
        if systemone:
            report("/v1/systemone     (in)", systemone,
                   lambda p: p["usage"]["input_tokens"],
                   lambda p: p["usage"]["input_tokens"])
        print()
        for ok, message in checks:
            print(f"  {'OK  ' if ok else 'FAIL'} {message}")
        failed = any(not ok for ok, _ in checks)
        print(f"serve_check: {'FAIL' if failed else 'PASS'}")
        return 1 if failed else 0
    finally:
        # the results are out (or never will be): end FrontEnd.run() on the
        # engine thread either way
        front.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="serve_check", description="serve three routes, check and time them")
    parser.add_argument("--model", required=True, help="model package directory")
    parser.add_argument("--speculative", "--mtp", dest="mtp", action="store_true",
                        help="ask the package to draft ahead and verify "
                             "(needs its `speculative` capability)")
    parser.add_argument("--backend", default=None, help="hardware folder to load")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="bytes")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runs", type=int, default=3, help="measured requests per route")
    args = parser.parse_args()

    engine = Engine.open(args.model, backend=args.backend, budget=args.budget)
    httpd = None
    code = 1
    try:
        # Before serving, so a package that cannot speculate says so here and
        # not mid-measurement. Refused with the reason when undeclared (rule 4).
        engine.speculative(args.mtp)
        front = serving.FrontEnd(engine, engine.model.name, root=args.model)
        httpd = serving.ThreadingHTTPServer((args.host, args.port), serving._Handler)
        httpd.daemon_threads = True
        httpd.front = front
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        host, port = httpd.server_address[:2]
        base = f"http://{host}:{port}"
        front._log("info", f"serve_check: serving {front.model_name} on {base} "
                           f"(mtp={'on' if args.mtp else 'off'}, "
                           f"capabilities: {', '.join(engine.facts.capabilities) or 'none'})")
        # The engine runs on the thread that opened it (its device context
        # lives here); the measurements run beside it and stop the server.
        outcome = {}
        worker = threading.Thread(
            target=lambda: outcome.setdefault(
                "code", driver(base, front.model_name, args.runs, front,
                               LABEL_LOGPROBS in engine.facts.capabilities)),
            daemon=True)
        worker.start()
        try:
            front.run()
        except KeyboardInterrupt:
            front._log("info", "serve_check: interrupted")
        worker.join()
        code = outcome.get("code", 1)
    except ContractError as exc:
        print(f"serve_check: {exc}", file=sys.stderr)
        code = 2
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        engine.close()
        print("serve_check: server stopped")
    return code


if __name__ == "__main__":
    sys.exit(main())
