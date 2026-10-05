"""A scripted mock with thinking blocks, reasoning levels, and per-step logprobs.

It is the mock in `tests/mockup` plus what the serving API needs beyond it:
thinking markers that delimit a block (and render as text in `decode`, the way
a real tokenizer's added tokens do — the front end, not the decoder, keeps them
out of a client's text), the `logprobs` capability (every sampled step reports
a top list headed by the sampled token), and the reasoning levels
`xhigh`, `medium`, `low`, which its template writes into the prompt so a test
can see which level arrived.

The template tail follows the real packages: with thinking on the prompt ends
on an open marker, so generation begins inside a block (`thinking hard`), and
with thinking off it ends on a closed, empty one, so the answer starts outside
it. The script is chosen from the prompt, not from a remembered call — a raw
completion with no template behind it starts outside a block too.
"""

from __future__ import annotations

import json
import random

from footless.sdk import (
    BudgetRefused,
    ContractError,
    LABEL_LOGPROBS,
    Claim,
    Facts,
    SamplingSpec,
    Services,
    State,
    StepContext,
    StepItem,
    StepResult,
)

BPT = 8  # bytes per cached token
MAX_BATCH = 4  # most items one step carries (the manifest's max_batch)
EOS = 256  # end of script
OPEN, CLOSE = 257, 258  # thinking block markers
PIECES = {OPEN: "<think>", CLOSE: "</think>", EOS: "<eos>"}
# Longest first, so a piece is never shadowed by a shorter one's prefix
_PIECE_TOKENS = sorted(
    ((text, token) for token, text in PIECES.items()), key=lambda kv: -len(kv[0])
)
# Labels a decision question scores that are one token in a real vocabulary and
# several bytes in this one ("yes" is three). One id each, so a byte mock can
# answer a yes or no question the way a real package does.
LABEL_TOKENS = {"yes": 300, "no": 301, "AA": 302, "AB": 303, "AC": 304}
LEVELS = ["xhigh", "medium", "low"]
THINKING_SCRIPT = (
    list("thinking hard".encode("utf-8"))
    + [CLOSE]
    + list("the answer".encode("utf-8"))
    + [EOS]
)
PLAIN_SCRIPT = list("the answer".encode("utf-8")) + [EOS]


class _State:
    def __init__(self, tokens: list[int], fed: int, claim: Claim | None) -> None:
        self.tokens = tokens
        self.fed = fed
        self.claim = claim


class MockThinkingRuntime:
    def __init__(self) -> None:
        self.services: Services | None = None
        self._streams: dict = {}
        self.chats: list[dict] = []  # what each chat() call was asked for
        self.scripted_label_logprobs: dict[str, float] = {}  # label -> logprob

    # lifecycle
    def open(self, model_dir: str, services: Services) -> Facts:
        self.services = services
        services.log("info", "mock-thinking runtime open")
        return Facts(
            cache_bytes_per_token=BPT,
            max_batch=MAX_BATCH,
            max_context=256,
            capabilities=["logprobs", LABEL_LOGPROBS],
            stop_tokens=[EOS],
            thinking_markers=[OPEN, CLOSE],
            thinking_open_at_start=True,
            thinking_levels=list(LEVELS),
        )

    def close(self) -> None:
        self._services().log("info", "mock-thinking runtime close")

    # cache state
    def new_state(self) -> State:
        return _State([], 0, None)

    def clone_state(self, state: _State) -> State:
        clone = _State(list(state.tokens), state.fed, None)
        self._reaccount(clone)
        return clone

    def truncate_state(self, state: _State, n_tokens: int) -> None:
        # what is kept is the next request's prompt, all of it: the script
        # (this mock's "model") then answers a cut-back state as a fresh one
        state.tokens = state.tokens[:n_tokens]
        state.fed = len(state.tokens)
        self._reaccount(state)

    def state_nbytes(self, state: _State) -> int:
        return BPT * len(state.tokens)

    def export_state(self, state: _State) -> bytes:
        return json.dumps({"tokens": state.tokens, "fed": state.fed}).encode()

    def import_state(self, blob: bytes) -> State:
        raw = json.loads(blob)
        state = _State(list(raw["tokens"]), raw["fed"], None)
        self._reaccount(state)
        return state

    def free_state(self, state: _State) -> None:
        if state.claim is not None:
            self._services().accounting.release(state.claim)
            state.claim = None

    # text
    def encode(self, text: str) -> list[int]:
        """Bytes for text, marker ids for marker text: decode(encode(text))
        gives the text back and encode(decode(ids)) the ids, which is what the
        label check's round-trip gate needs after a chat() render ends on
        marker ids."""
        out: list[int] = []
        i = 0
        while i < len(text):
            for piece, token in _PIECE_TOKENS:
                if text.startswith(piece, i):
                    out.append(token)
                    i += len(piece)
                    break
            else:
                out.extend(text[i].encode("utf-8"))
                i += 1
        return out

    def decode(self, tokens: list[int]) -> str:
        """Markers as their text and byte runs as utf-8, the way a real
        tokenizer's added tokens and byte pieces decode."""
        out: list[str] = []
        buf = bytearray()

        def flush() -> None:
            if buf:
                out.append(buf.decode("utf-8", "replace"))
                buf.clear()

        for token in tokens:
            if token in PIECES:
                flush()
                out.append(PIECES[token])
            elif 0 <= token < 256:
                buf.append(token)
            else:
                flush()
                out.append(chr(0xFFFD))
        flush()
        return "".join(out)

    def chat(self, messages: list[dict], thinking: bool = True,
             level: str | None = None) -> list[int]:
        self.chats.append({"thinking": thinking, "level": level})
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        # the level goes into the prompt the way a template writes it, so a
        # test can see it arrive; the tail is the real packages' tail
        prompt = text + "assistant: " + (f"[{level}]" if level else "")
        tail = [OPEN] if thinking else [OPEN, CLOSE]
        return self.encode(prompt) + tail

    # one step over a batch of requests
    def _claim_first(self, batch: list[StepItem]) -> None:
        """The batch's own rules (at most max_batch items, a state once), then
        every item's bytes claimed before any state moves: a refusal leaves the
        batch as it was (CONTRACT.md, `step`)."""
        if len(batch) > MAX_BATCH:
            raise ContractError(f"batch of {len(batch)} exceeds max_batch {MAX_BATCH}")
        if len({id(item.state) for item in batch}) != len(batch):
            raise ContractError("a state appears more than once in one batch")
        accounting = self._services().accounting
        grown: list[tuple[_State, int]] = []    # (state, bytes it held)
        try:
            for item in batch:
                state = item.state
                held = state.claim.nbytes if state.claim is not None else 0
                if state.claim is not None:
                    accounting.release(state.claim)
                    state.claim = None
                grown.append((state, held))
                n = len(state.tokens) + len(item.input_tokens) + (item.sampling is not None)
                state.claim = accounting.claim(BPT * n)
        except BudgetRefused:
            for state, held in grown:
                if state.claim is not None:
                    accounting.release(state.claim)
                state.claim = accounting.claim(held) if held else None
            raise

    def step(self, batch: list[StepItem], ctx: StepContext) -> list[StepResult]:
        self._claim_first(batch)
        results: list[StepResult] = []
        for item in batch:
            state = item.state
            if ctx.cancelled():
                results.append(StepResult(None, True, None))
                continue
            state.tokens.extend(item.input_tokens)
            state.fed += len(item.input_tokens)
            pos = len(state.tokens) - state.fed
            if item.sampling is None:
                self._reaccount(state)
                results.append(StepResult(None, False, None))
                continue
            script = THINKING_SCRIPT if self._inside(state) else PLAIN_SCRIPT
            token = script[pos] if pos < len(script) else EOS
            self._draw(item.sampling)
            state.tokens.append(token)
            self._reaccount(state)
            top = self._top(token, item.sampling.logprobs)
            results.append(StepResult(token, token == EOS, top))
        return results

    # the OPTIONAL verb behind capability "label_logprobs" (CONTRACT.md): one
    # prefill, no sampling. The mock runs no model -- the values come from
    # `scripted_label_logprobs`, falling to -(1.0 + index) per labels argument
    # order -- and it touches no state and no draw stream: scoring must leave
    # generation exactly as it found it.
    def label_logprobs(self, input_tokens: list[int],
                       labels: list[str]) -> list[tuple[int, float]]:
        """Resolve each label to the single token it adds at the answer
        position after input_tokens' text, and return (token_id,
        full-vocabulary next-token log-probability) for each, in order. One
        prefill, no sampling, no penalties, no temperature. Raises ValueError
        naming the offending label when it is not exactly one token at the
        answer position, when labels collide on one token id, or when the
        prompt text does not round-trip through decode/encode."""
        text = self.decode(input_tokens)
        base = self.encode(text)
        if base != list(input_tokens):
            raise ValueError("prompt text does not round-trip through decode/encode")
        out: list[tuple[int, float]] = []
        seen: dict[int, str] = {}
        for index, label in enumerate(labels):
            # LABEL_TOKENS first: the label texts a decision question scores
            # are one token each in a real vocabulary, which utf-8 bytes split.
            token = LABEL_TOKENS.get(label)
            if token is None:
                ids = self.encode(text + label)
                if ids[:len(base)] != base or len(ids) != len(base) + 1:
                    raise ValueError(
                        f"label {label!r} is not exactly one token at the "
                        "answer position"
                    )
                token = ids[-1]
            if token in seen:
                raise ValueError(
                    f"label {label!r} collides with {seen[token]!r} "
                    f"on one token id {token}"
                )
            seen[token] = label
            out.append((token,
                        self.scripted_label_logprobs.get(label, -(1.0 + index))))
        return out

    @staticmethod
    def _inside(state: _State) -> bool:
        """Does generation begin inside a block? The prompt says so."""
        return state.fed > 0 and state.tokens[state.fed - 1] == OPEN

    def _top(self, token: int, count: int) -> list[tuple[int, float]] | None:
        """The step's top list, headed by the sampled token."""
        if count <= 0:
            return None
        others = [candidate for candidate in (1, 2, 3) if candidate != token]
        return [(token, -0.25)] + [
            (candidate, -1.5 - index) for index, candidate in enumerate(others)
        ][: count - 1]

    def _draw(self, spec: SamplingSpec) -> float:
        entry = self._streams.get(id(spec))
        if entry is None or entry[0] is not spec:
            entry = [spec, random.Random(spec.seed), []]
            self._streams[id(spec)] = entry
        return entry[1].random()

    def _reaccount(self, state: _State) -> None:
        accounting = self._services().accounting
        if state.claim is not None:
            accounting.release(state.claim)
            state.claim = None
        if state.tokens:
            state.claim = accounting.claim(BPT * len(state.tokens))

    def _services(self) -> Services:
        if self.services is None:
            raise RuntimeError("mock-thinking runtime is not open")
        return self.services


def create() -> MockThinkingRuntime:
    return MockThinkingRuntime()
