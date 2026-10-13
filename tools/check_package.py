"""Does a model package keep the contract? A conformance check through the engine.

    python3 tools/check_package.py --model models/<model> [--backend <folder>]
                                   [--budget BYTES] [--tokens 16] [--prompt TEXT]

Every check drives the package through the contract's verbs only (CONTRACT.md),
greedy, and compares exactly: the same tokens, or it fails. Opening the engine
already compares the manifest with the runtime's facts (rule 3). What follows:

  text and chat      decode(encode(text)) round trip; the template renders, at
                     every declared thinking level
  step basics        a prefill chunk samples nothing; a decode samples; greedy
                     is deterministic; a prompt fed in two chunks equals one
  state verbs        truncate_state (cut back = a fresh state fed the prefix),
                     clone_state, export_state / import_state
  batching           each item of a batch gets what it gets alone; a state twice
                     and a batch past max_batch are refused; a refused claim
                     leaves the batch as it was (max_batch >= 2)
  prefix cache       a repeated prompt reuses the cache and answers the same
  capabilities       label_logprobs, tool_calls and speculative, when declared
                     (speculation must not change one token); concurrent_prefill:
                     a prefill step on another thread while a state decodes
                     changes no token of either; concurrent_decode: two threads
                     decoding at once (one state, and a batch of two) change no
                     token, with speculation on when declared
  accounting         everything freed, the ledger is back to zero
  vision             when declared (last: turning it on may keep a claim):
                     chat_images' tokens and spans, greedy determinism with an
                     image, an image split over two steps = fed in one, the
                     ledger back to where vision left it

Exit status 0 when nothing failed. A WARN is not a contract breach but costs
something (a lossy round trip refuses label scoring; no cache reuse is slow).
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from footless.engine import DEFAULT_BUDGET, Engine, StepContext  # noqa: E402
from footless.sdk import (  # noqa: E402
    CONCURRENT_DECODE, CONCURRENT_PREFILL, LABEL_LOGPROBS, TOOL_CALLS, VISION, Image, ImageSpan, BudgetRefused, ContractError, SamplingSpec, StepItem,
)

GREEDY = SamplingSpec(temperature=0.0, top_k=1, top_p=1.0, seed=1)
PROMPT = ("The history of the bicycle begins in the early nineteenth century, when "
          "inventors across Europe tried to build a machine that a person could ride "
          "without a horse. The first designs had no pedals at all.")
TTY = sys.stdout.isatty()
COLORS = {"PASS": "32", "FAIL": "31", "WARN": "33", "SKIP": "90"}


class Skip(Exception):
    """The check does not apply to this package."""


class Warn(Exception):
    """Not a contract breach, but it costs something."""


class Checker:
    def __init__(self, engine: Engine, tokens: list[int], n: int) -> None:
        self.engine, self.rt = engine, engine.runtime
        self.tokens, self.n = tokens, n
        self.ctx = StepContext()
        self.states: list = []
        self.failed = 0

    # -- helpers -----------------------------------------------------------
    def state(self):
        st = self.rt.new_state()
        self.states.append(st)
        return st

    def free(self, st) -> None:
        self.rt.free_state(st)
        self.states = [s for s in self.states if s is not st]

    def feed(self, st, tokens, sample=True):
        return self.rt.step([StepItem(st, list(tokens), GREEDY if sample else None)],
                            self.ctx)[0]

    def decode(self, st, n=None) -> list[int]:
        out = []
        for _ in range(self.n if n is None else n):
            out.append(self.rt.step([StepItem(st, [], GREEDY)], self.ctx)[0].token)
        return out

    def run_from(self, tokens, n=None) -> list[int]:
        """A fresh state fed `tokens` in one step, then `n` decodes."""
        st = self.state()
        first = self.feed(st, tokens).token
        rest = self.decode(st, n)
        self.free(st)
        return [first] + rest

    def check(self, name: str, fn) -> None:
        t0 = time.perf_counter()
        try:
            detail = fn() or ""
            status = "PASS"
        except Skip as exc:
            status, detail = "SKIP", str(exc)
        except Warn as exc:
            status, detail = "WARN", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc) or "assertion failed"
        except Exception as exc:  # noqa: BLE001 - a crash is a failure of the check
            status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
        if status == "FAIL":
            self.failed += 1
        tag = f"\x1b[{COLORS[status]}m{status}\x1b[0m" if TTY else status
        took = time.perf_counter() - t0
        print(f"  {tag}  {name:<44} {detail}  ({took:.1f} s)" if detail
              else f"  {tag}  {name:<44} ({took:.1f} s)", flush=True)

    # -- the checks ----------------------------------------------------------
    def text_round_trip(self):
        text = self.rt.decode(self.tokens)
        if self.rt.encode(text) != self.tokens:
            raise Warn("encode(decode(ids)) != ids: label scoring refuses such prompts")

    def chat_template(self):
        msgs = [{"role": "user", "content": "Hello there."}]
        for thinking in (True, False):
            ids = self.engine.chat_tokens(msgs, thinking)
            assert ids and all(isinstance(t, int) for t in ids), "chat() returned no ids"
        for level in self.engine.facts.thinking_levels:
            assert self.engine.chat_tokens(msgs, True, level), f"level {level!r} refused"
        levels = self.engine.facts.thinking_levels
        return f"levels: {', '.join(levels)}" if levels else ""

    def step_basics(self):
        st = self.state()
        res = self.feed(st, self.tokens[:-1], sample=False)
        assert res.token is None and not res.finished, "a prefill chunk sampled a token"
        res = self.feed(st, self.tokens[-1:])
        assert isinstance(res.token, int), "a sampled step returned no token"
        self.free(st)
        self.reference = self.run_from(self.tokens)
        again = self.run_from(self.tokens)
        assert again == self.reference, "greedy is not deterministic"

    def chunked_prefill(self):
        half = len(self.tokens) // 2
        st = self.state()
        self.feed(st, self.tokens[:half], sample=False)
        got = [self.feed(st, self.tokens[half:]).token] + self.decode(st)
        self.free(st)
        assert got == self.reference, "a prompt in two chunks answers differently"

    def truncate(self):
        half = len(self.tokens) // 2
        st = self.state()
        self.feed(st, self.tokens[:half] + self.tokens[half:][::-1])  # a wrong tail
        self.decode(st, 2)
        self.rt.truncate_state(st, half)
        got = [self.feed(st, self.tokens[half:]).token] + self.decode(st)
        self.free(st)
        assert got == self.reference, "a truncated state is not a fresh one fed the prefix"

    def clone(self):
        st = self.state()
        first = self.feed(st, self.tokens).token
        twin = self.rt.clone_state(st)
        self.states.append(twin)
        a, b = self.decode(st), self.decode(twin)
        self.free(st)
        self.free(twin)
        assert [first] + a == self.reference, "the original drifted after a clone"
        assert [first] + b == self.reference, "the clone decodes differently"

    def export_import(self):
        st = self.state()
        first = self.feed(st, self.tokens).token
        blob = self.rt.export_state(st)
        assert isinstance(blob, (bytes, bytearray)), "export_state returned no bytes"
        back = self.rt.import_state(bytes(blob))
        self.states.append(back)
        got = self.decode(back)
        self.free(st)
        self.free(back)
        assert [first] + got == self.reference, "an imported state decodes differently"
        return f"{len(blob):,} bytes"

    def _prompts(self, k):
        n = len(self.tokens)
        return [self.tokens[: max(2, n - 3 * i)] for i in range(k)]

    def batch_equivalence(self):
        mb = self.engine.facts.max_batch
        if mb < 2:
            raise Skip("max_batch is 1")
        k = min(mb, 8)                  # the whole batch a package declares, up to 8
        prompts = self._prompts(k)
        alone = [self.run_from(p) for p in prompts]
        states = [self.state() for _ in prompts]
        got = [[] for _ in prompts]
        # the first step: every prompt as a prefill item of one batch
        res = self.rt.step([StepItem(s, list(p), GREEDY) for s, p in zip(states, prompts)],
                           self.ctx)
        for g, r in zip(got, res):
            g.append(r.token)
        for _ in range(self.n):
            res = self.rt.step([StepItem(s, [], GREEDY) for s in states], self.ctx)
            for g, r in zip(got, res):
                g.append(r.token)
        for s in states:
            self.free(s)
        for i, (a, g) in enumerate(zip(alone, got)):
            assert a == g, f"sequence {i} of the batch differs from it alone"
        return f"{k} sequences"

    def batch_refusals(self):
        mb = self.engine.facts.max_batch
        st = self.state()
        try:
            self.rt.step([StepItem(st, list(self.tokens), None)] * (mb + 1), self.ctx)
            raise AssertionError("a batch past max_batch was accepted")
        except ContractError:
            pass
        if mb >= 2:
            try:
                self.rt.step([StepItem(st, self.tokens[:2], None),
                              StepItem(st, self.tokens[2:4], None)], self.ctx)
                raise AssertionError("a state twice in one batch was accepted")
            except ContractError:
                pass
        self.free(st)

    def refused_step(self):
        acc = self.engine.services.accounting
        if not hasattr(acc, "budget"):
            raise Skip("the ledger has no budget to clamp")
        k = min(self.engine.facts.max_batch, 2)
        prompts = self._prompts(k)
        alone = [self.run_from(p) for p in prompts]
        states = [self.state() for _ in prompts]
        got = [[self.feed(s, p).token] for s, p in zip(states, prompts)]
        refused = 0
        for _ in range(self.n):
            budget = acc.budget
            acc.budget = acc.used                # nothing more may be claimed
            try:
                res = self.rt.step([StepItem(s, [], GREEDY) for s in states], self.ctx)
            except BudgetRefused:
                refused += 1
                acc.budget = budget
                res = self.rt.step([StepItem(s, [], GREEDY) for s in states], self.ctx)
            finally:
                acc.budget = budget
            for g, r in zip(got, res):
                g.append(r.token)
        for s in states:
            self.free(s)
        assert got == alone, "a refused step left a state changed"
        if not refused:
            raise Warn("no step needed room here: the refusal path was not exercised")
        return f"{refused} refusals, each retried cleanly"

    def prefix_cache(self):
        self.engine.cache.free_all()
        r1 = self.engine.generate_tokens(self.tokens, self.n, sampling=GREEDY, loop_guard="off")
        r2 = self.engine.generate_tokens(self.tokens, self.n, sampling=GREEDY, loop_guard="off")
        assert r1.tokens == r2.tokens, "the cached prompt answers differently"
        longer = self.tokens + r1.absorbed_tokens + self.tokens[:5]
        r3 = self.engine.generate_tokens(longer, self.n, sampling=GREEDY, loop_guard="off")
        self.engine.cache.free_all()
        r4 = self.engine.generate_tokens(longer, self.n, sampling=GREEDY, loop_guard="off")
        self.engine.cache.free_all()
        assert r3.tokens == r4.tokens, "a continued conversation answers differently from fresh"
        if r3.prompt_tokens_fed >= len(longer):
            raise Warn("a continued conversation re-read its whole prompt")
        return f"continued turn fed {r3.prompt_tokens_fed} of {len(longer)} tokens"

    def label_logprobs(self):
        if LABEL_LOGPROBS not in self.engine.facts.capabilities:
            raise Skip("not declared")
        ids = self.engine.chat_tokens(
            [{"role": "user", "content": "Is the sky blue? Answer yes or no."}], False)
        try:
            one = self.engine.label_logprobs(ids, ["yes", "no"])
        except ValueError as exc:
            raise Skip(f"'yes'/'no' are not single tokens here: {exc}") from None
        assert all(lp <= 1e-6 for _, lp in one), "a log-probability above zero"
        mass = sum(math.exp(lp) for _, lp in one)
        assert mass <= 1 + 1e-3, f"labels hold more than all the mass ({mass:.4f})"
        both = self.engine.label_logprobs_batch([(ids, ["yes", "no"]), (ids, ["no", "yes"])])
        assert both is not None and len(both) == 2, "the batch form returned nothing"
        for (t1, a), (t2, b) in zip(one, both[0]):
            assert t1 == t2 and abs(a - b) < 1e-3, "the batch form differs from one call"
        return f"label mass {mass:.3f}"

    def tool_calls(self):
        if TOOL_CALLS not in self.engine.facts.capabilities:
            raise Skip("not declared")
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {
            "type": "object", "properties": {"q": {"type": "string"}}}}}]
        ids = self.engine.chat_tokens([{"role": "user", "content": "Look up bicycles."}],
                                      True, tools=tools)
        assert ids, "chat() with tools returned no ids"
        content, calls = self.engine.tool_calls("A plain answer.", tools, True)
        assert isinstance(content, str) and isinstance(calls, list), "tool_calls() shape"
        # read as it streams: each result extends the previous one
        text, before = "A plain answer.", ""
        for n in range(1, len(text) + 1):
            content, calls = self.engine.tool_calls(text[:n], tools, n == len(text))
            assert content.startswith(before), f"the content of {text[:n]!r} drops what was read"
            assert not calls, f"a call read in {text[:n]!r}"
            before = content

    def speculative(self):
        if "speculative" not in self.engine.facts.capabilities:
            raise Skip("not declared")
        self.engine.speculative(True)
        got = self.run_from(self.tokens, self.n * 2)
        want = self.reference_long
        assert got == want, "speculation changed the tokens"

    def concurrent_prefill(self):
        if CONCURRENT_PREFILL not in self.engine.facts.capabilities:
            raise Skip("not declared")
        long, short = self._prompts(2)
        alone = [self.run_from(long), self.run_from(short)]
        a, b = self.state(), self.state()
        got_b = [self.feed(b, short).token]
        got_a, failed = [], []

        def prefill():
            try:
                got_a.append(self.feed(a, long).token)
            except BaseException as exc:  # noqa: BLE001 - reported below
                failed.append(exc)
        worker = threading.Thread(target=prefill)
        worker.start()
        beside = 0
        while worker.is_alive() and len(got_b) <= self.n:
            got_b.append(self.decode(b, 1)[0])
            beside += 1
        worker.join()
        if failed:
            raise failed[0]
        got_a += self.decode(a)
        got_b += self.decode(b, self.n + 1 - len(got_b))
        self.free(a)
        self.free(b)
        assert got_a == alone[0], "the prefill beside decode steps answers differently"
        assert got_b == alone[1], "decode steps beside a prefill answer differently"
        return f"{beside} decode steps beside the prefill"

    def concurrent_decode(self):
        if CONCURRENT_DECODE not in self.engine.facts.capabilities:
            raise Skip("not declared")
        if "speculative" in self.engine.facts.capabilities:
            self.engine.speculative(True)        # each thread verifies its own drafts
        prompts = self._prompts(3)
        alone = [self.run_from(p) for p in prompts]
        sts = [self.state() for _ in prompts]
        got = [[self.feed(st, p).token] for st, p in zip(sts, prompts)]
        pair = self.engine.facts.max_batch >= 2
        groups = [[0], [1, 2]] if pair else [[0], [1]]
        failed = []

        def decode(group):
            try:
                for _ in range(self.n):
                    res = self.rt.step([StepItem(sts[k], [], GREEDY) for k in group], self.ctx)
                    for k, r in zip(group, res):
                        got[k].append(r.token)
            except BaseException as exc:  # noqa: BLE001 - reported below
                failed.append(exc)
        workers = [threading.Thread(target=decode, args=(g,)) for g in groups]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        if not pair:
            got[2] += self.decode(sts[2], self.n)
        for st in sts:
            self.free(st)
        if failed:
            raise failed[0]
        assert got == alone, "two decode steps at once answer differently"
        return "a state, and a batch of two, on two threads" if pair else "two states on two threads"

    def vision(self):
        if VISION not in self.engine.facts.capabilities:
            raise Skip("not declared")
        self.engine.vision(True)
        used = getattr(self.engine.services.accounting, "used", 0)
        a, b = Image.of(_png(64, 48, (200, 30, 30))), Image.of(_png(64, 48, (30, 30, 200)))
        prompts = [self.engine.chat_prompt([{"role": "user", "content": [
            {"type": "image", "image": im}, {"type": "text", "text": "What colour is this?"}]}]) for im in (a, b)]
        for p in prompts:
            assert len(p.images) == 1, "chat_images placed one image"
            sp = p.images[0]
            assert 0 <= sp.start and sp.length > 0 and sp.start + sp.length <= len(p.tokens), "span outside the prompt"
        p = prompts[0]
        sp = p.images[0]

        def run(cut):
            st = self.state()
            for lo, hi in ((0, cut), (cut, len(p.tokens))):
                if lo == hi:
                    continue
                spans = [ImageSpan(sp.start - lo, sp.length, sp.image)]
                first = self.rt.step([StepItem(st, p.tokens[lo:hi], GREEDY if hi == len(p.tokens) else None,
                                               spans)], self.ctx)[0].token
            out = [first] + self.decode(st)
            self.free(st)
            return out
        whole = run(len(p.tokens))
        assert run(len(p.tokens)) == whole, "greedy with an image is not deterministic"
        assert run(sp.start + sp.length // 2) == whole, "an image split over two steps differs from one"
        left = getattr(self.engine.services.accounting, "used", 0)
        assert left == used, f"{left - used} bytes still claimed after the states were freed"
        return f"{sp.length} tokens an image"

    def accounting(self):
        for st in list(self.states):
            self.free(st)
        self.engine.cache.free_all()
        used = getattr(self.engine.services.accounting, "used", 0)
        assert used == 0, f"{used} bytes still claimed after every state was freed"


def _png(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    """A small PNG: a colour square on white (stdlib only)."""
    import struct
    import zlib
    rows = b"".join(b"\0" + b"".join(bytes(rgb) if w // 4 <= x < 3 * w // 4 and h // 4 <= y < 3 * h // 4
                                       else b"\xff\xff\xff" for x in range(w)) for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="model package directory")
    ap.add_argument("--backend", default=None, help="hardware folder to load")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="bytes")
    ap.add_argument("--tokens", type=int, default=16, help="tokens each decode check runs")
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    try:
        engine = Engine.open(args.model, backend=args.backend, budget=args.budget)
    except Exception as exc:  # noqa: BLE001 - the first check, and a fatal one
        print(f"  FAIL  open: {type(exc).__name__}: {exc}")
        return 1
    f = engine.facts
    print(f"{engine.model.name}: opened in {time.perf_counter() - t0:.1f} s · "
          f"max_batch {f.max_batch} · max_context {f.max_context} · "
          f"{f.cache_bytes_per_token} B/token · capabilities "
          f"{', '.join(f.capabilities) or 'none'}")
    tokens = engine.runtime.encode(args.prompt)
    if len(tokens) < 8:
        print("  FAIL  the prompt encodes to fewer than 8 tokens")
        return 1
    c = Checker(engine, tokens, args.tokens)
    try:
        c.check("text round trip", c.text_round_trip)
        c.check("chat template", c.chat_template)
        c.check("step: prefill, decode, determinism", c.step_basics)
        if not hasattr(c, "reference"):
            print("  (the remaining checks compare against the reference decode)")
            return 1
        c.reference_long = c.run_from(tokens, args.tokens * 2)
        c.check("step: a prompt in two chunks", c.chunked_prefill)
        c.check("truncate_state", c.truncate)
        c.check("clone_state", c.clone)
        c.check("export_state / import_state", c.export_import)
        c.check("batch: each item as alone", c.batch_equivalence)
        c.check("batch: max_batch and duplicate states", c.batch_refusals)
        c.check("batch: a refused claim moves nothing", c.refused_step)
        c.check("prefix cache", c.prefix_cache)
        c.check("label_logprobs", c.label_logprobs)
        c.check("tool_calls", c.tool_calls)
        c.check("speculative", c.speculative)
        c.check("concurrent_prefill", c.concurrent_prefill)
        c.check("concurrent_decode", c.concurrent_decode)
        c.check("accounting back to zero", c.accounting)
        c.check("vision", c.vision)
    finally:
        engine.close()
    print(f"{'FAILED' if c.failed else 'OK'}: {c.failed} check(s) failed")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
