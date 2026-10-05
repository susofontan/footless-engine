"""A scripted mock model runtime: fixed output, no architecture, no kernels.

It exists so the engine can be exercised end to end with nothing but stdlib.

Tokens are scripted, but every sampling step also draws from the spec's
per-request stream (`random.Random(spec.seed)`, one draw per sampled token --
the draw the real model makes) and records it in `draw_streams`, one list per
request. The engine's seeding policy -- which seed each request gets, and
that a seed replays one stream -- is the part of sampling a scripted model
can still observe; the real draw is pinned in the model's own tests.
"""

from __future__ import annotations

import json
import random

from footless.sdk import (
    BudgetRefused,
    ContractError,
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
EOS = 256  # the mock model's end-of-script token
SCRIPT = list("the mock model speaks".encode("utf-8")) + [EOS]
# Labels a decision question scores that are one token in a real vocabulary and
# several bytes in this one ("yes" is three). One id each, so a byte mock can
# answer a yes or no question the way a real package does.
LABEL_TOKENS = {"yes": 300, "no": 301, "AA": 302, "AB": 303, "AC": 304}


class _State:
    """One request's cached tokens; fed counts the tokens that came in as input."""

    def __init__(self, tokens: list[int], fed: int, claim: Claim | None) -> None:
        self.tokens = tokens
        self.fed = fed
        self.claim = claim


class MockRuntime:
    def __init__(self) -> None:
        self.services: Services | None = None
        self.draw_streams: list[list[float]] = []
        self.batches: list[int] = []   # each step's batch size, in order
        self._streams: dict = {}
        self.scripted_label_logprobs: dict[str, float] = {}  # label -> logprob

    # lifecycle
    def open(self, model_dir: str, services: Services) -> Facts:
        self.services = services
        services.log("info", "mock runtime open")
        return Facts(
            cache_bytes_per_token=BPT,
            max_batch=MAX_BATCH,
            max_context=256,
            capabilities=[],
            stop_tokens=[EOS],
        )

    def close(self) -> None:
        self._services().log("info", "mock runtime close")

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
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(t for t in tokens if 0 <= t < 256).decode("utf-8", errors="replace")

    # conversation to prompt tokens: "role: content" lines, generation prompt last
    def chat(self, messages: list[dict], thinking: bool = True) -> list[int]:
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        return self.encode(text + "assistant: ")

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
        self.batches.append(len(batch))
        self._claim_first(batch)
        results: list[StepResult] = []
        for item in batch:
            state = item.state
            if ctx.cancelled():
                results.append(StepResult(None, True, None))
                continue
            state.tokens.extend(item.input_tokens)
            state.fed += len(item.input_tokens)
            pos = len(state.tokens) - state.fed  # tokens generated so far
            if item.sampling is None:
                self._reaccount(state)
                results.append(StepResult(None, False, None))
                continue
            token = SCRIPT[pos] if pos < len(SCRIPT) else EOS
            self._draw(item.sampling)
            state.tokens.append(token)
            self._reaccount(state)
            results.append(StepResult(token, token == EOS, None))
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

    # the draw stream a sampling spec drives: one advancing stream per request
    def _draw(self, spec: SamplingSpec) -> float:
        entry = self._streams.get(id(spec))
        if entry is None or entry[0] is not spec:
            entry = [spec, random.Random(spec.seed), []]
            self._streams[id(spec)] = entry
            self.draw_streams.append(entry[2])
        q = entry[1].random()
        entry[2].append(q)
        return q

    # byte accounting: at most one claim, worth BPT * len(tokens)
    def _reaccount(self, state: _State) -> None:
        accounting = self._services().accounting
        if state.claim is not None:
            accounting.release(state.claim)
            state.claim = None
        if state.tokens:
            state.claim = accounting.claim(BPT * len(state.tokens))

    def _services(self) -> Services:
        if self.services is None:
            raise RuntimeError("mock runtime is not open")
        return self.services


def create() -> MockRuntime:
    return MockRuntime()
