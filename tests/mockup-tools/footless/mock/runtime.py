"""A scripted mock with function tools and images: the plain mock plus the
`tool_calls` and `vision` capabilities.

Its template writes the tools it is given into the prompt (and records each
render in `chats`), and it answers with `script`, a text a test sets. A call
in that text is `[[NAME ARGS]]`, ARGS being JSON text; the `tool_calls` verb
reads them back out. The format is the mock's own on purpose: the engine must
not know any package's.

An image in a message takes IMAGE_TOKENS placeholder tokens (IMG); each step
records the images it was given (`steps_images`: (tokens before, start,
length, digest) per span).
"""

from __future__ import annotations

import json

from footless.sdk import (BudgetRefused, ContractError, Claim, Facts, ImageSpan, Services, State, StepContext,
                          StepItem, StepResult)

BPT = 8  # bytes per cached token
MAX_BATCH = 4  # most items one step carries (the manifest's max_batch)
EOS = 256
OPEN, CLOSE = "[[", "]]"
IMG, IMAGE_TOKENS = 300, 3


class _State:
    def __init__(self, tokens: list[int], fed: int, claim: Claim | None) -> None:
        self.tokens = tokens
        self.fed = fed
        self.claim = claim


class MockToolsRuntime:
    def __init__(self) -> None:
        self.services: Services | None = None
        self.chats: list[dict] = []  # what each chat() call was asked for
        self.script = "the mock model speaks"
        self.vision = False
        self.steps_images: list[tuple[int, int, int, str]] = []

    def open(self, model_dir: str, services: Services) -> Facts:
        self.services = services
        return Facts(cache_bytes_per_token=BPT, max_batch=MAX_BATCH, max_context=256,
                     capabilities=["tool_calls", "vision"], stop_tokens=[EOS])

    def close(self) -> None:
        pass

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
            self.services.accounting.release(state.claim)
            state.claim = None

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(t for t in tokens if 0 <= t < 256).decode("utf-8", errors="replace")

    def chat(self, messages: list[dict], thinking: bool = True,
             tools: list | None = None) -> list[int]:
        self.chats.append({"messages": messages, "tools": tools})
        head = "".join(f"tool: {t['function']['name']}\n" for t in tools or ())
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        return self.encode(head + text + "assistant: ")

    # the OPTIONAL verbs behind `vision`
    def set_vision(self, enabled: bool) -> None:
        self.vision = enabled

    def chat_images(self, messages: list[dict], thinking: bool = True,
                    tools: list | None = None) -> tuple[list[int], list[ImageSpan]]:
        if not self.vision:
            raise ContractError("chat_images before set_vision(True)")
        self.chats.append({"messages": messages, "tools": tools})
        tokens = self.encode("".join(f"tool: {t['function']['name']}\n" for t in tools or ()))
        spans = []
        for m in messages:
            tokens += self.encode(f"{m['role']}: ")
            parts = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
            for part in parts:
                if part["type"] == "image":
                    spans.append(ImageSpan(len(tokens), IMAGE_TOKENS, part["image"]))
                    tokens += [IMG] * IMAGE_TOKENS
                else:
                    tokens += self.encode(part["text"])
            tokens += self.encode("\n")
        return tokens + self.encode("assistant: "), spans

    def _claim_first(self, batch: list[StepItem]) -> None:
        """The batch's own rules (at most max_batch items, a state once), then
        every item's bytes claimed before any state moves: a refusal leaves the
        batch as it was (CONTRACT.md, `step`)."""
        if len(batch) > MAX_BATCH:
            raise ContractError(f"batch of {len(batch)} exceeds max_batch {MAX_BATCH}")
        if len({id(item.state) for item in batch}) != len(batch):
            raise ContractError("a state appears more than once in one batch")
        accounting = self.services.accounting
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
        script = list(self.script.encode("utf-8")) + [EOS]
        results = []
        for item in batch:
            state = item.state
            for span in item.images:
                if not self.vision:
                    raise ContractError("images in a step before set_vision(True)")
                self.steps_images.append((len(state.tokens), span.start, span.length, span.image.digest))
            state.tokens.extend(item.input_tokens)
            state.fed += len(item.input_tokens)
            if item.sampling is None:
                self._reaccount(state)
                results.append(StepResult(None, False, None))
                continue
            pos = len(state.tokens) - state.fed
            token = script[pos] if pos < len(script) else EOS
            state.tokens.append(token)
            self._reaccount(state)
            results.append(StepResult(token, token == EOS, None))
        return results

    # the OPTIONAL verb behind `tool_calls`
    def tool_calls(self, text: str, tools: list, final: bool) -> tuple[str, list[dict]]:
        content, calls, pos = [], [], 0
        while True:
            start = text.find(OPEN, pos)
            if start < 0:
                tail = text[pos:]
                if not final and tail.endswith(OPEN[0]):
                    tail = tail[:-1]  # may open a call
                content.append(tail)
                break
            content.append(text[pos:start])
            end = text.find(CLOSE, start)
            if end < 0:
                break  # an open call: nothing after it is known yet
            name, _, args = text[start + len(OPEN):end].partition(" ")
            calls.append({"name": name, "arguments": args or "{}"})
            pos = end + len(CLOSE)
        return "".join(content), calls

    def _reaccount(self, state: _State) -> None:
        accounting = self.services.accounting
        if state.claim is not None:
            accounting.release(state.claim)
            state.claim = None
        if state.tokens:
            state.claim = accounting.claim(BPT * len(state.tokens))


def create() -> MockToolsRuntime:
    return MockToolsRuntime()
