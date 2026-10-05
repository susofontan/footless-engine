"""<my-model> on <hardware>: the footless runtime. A starting point.

Copy `templates/model-package/footless/` into the model's own directory, next to
its weights (rule 6: nothing is moved), rename `example/` to the hardware folder
it targets and list that name in the manifest's `backends`. Then fill in the
verbs below, in the order docs/roadmap.md walks them. Each docstring says what
the engine relies on; CONTRACT.md is the normative text.

The split, in one line: this file owns execution and representation -- the
architecture, kernels, cache layout, weights, tokenizer and every per-hardware
choice. The engine owns policy and bookkeeping, and never looks inside a state.

`python3 tools/check_package.py --model <model dir>` tells you, verb by verb,
whether the package keeps the contract.
"""

from __future__ import annotations

from footless.sdk import ContractError, Facts, StepResult

# Must equal the manifest (the engine compares them at open: rule 3).
MAX_BATCH = 1
MAX_CONTEXT = 4096
CACHE_BYTES_PER_TOKEN = 0
CAPABILITIES: list[str] = []
THINKING_LEVELS: list[str] = []


class State:
    """One sequence's cache. Opaque to the engine: it moves it, counts its
    bytes and hands it back; only this file reads it.

    Positions are derived from the state, never passed in: keep the ids it has
    absorbed (`tokens`), and the device memory that holds its cache.
    """

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.claim = None               # the ledger's claim for what it holds


class Runtime:
    # -- lifecycle -----------------------------------------------------------

    def open(self, model_dir: str, services) -> Facts:
        """Bind the weights and tokenizer where they are; return the facts.

        Read the checkpoint's config and check it against the constants this
        file assumes (a contradiction is a hard error). Upload weights, allocate
        scratch, then hand the ledger the device truth: if the device has less
        room than the engine's budget, `services.accounting.set_budget(bytes)`
        (when it exists) so states are refused as BudgetRefused before the
        allocator fails mid-request.
        """
        self.services = services
        raise NotImplementedError("load the model, then return self.facts()")

    def facts(self) -> Facts:
        """What `open` returns: the manifest's numbers and what the weights say."""
        return Facts(
            cache_bytes_per_token=CACHE_BYTES_PER_TOKEN, max_batch=MAX_BATCH,
            max_context=MAX_CONTEXT, capabilities=list(CAPABILITIES),
            stop_tokens=[],                 # ids that end generation (from the weights)
            thinking_markers=[],            # [open id, close id] of thinking blocks
            thinking_open_at_start=False,   # the thinking template leaves a block open
            thinking_levels=list(THINKING_LEVELS),
            default_sampling=None,          # the generation config, as a SamplingSpec
        )

    def close(self) -> None:
        """Release what open() took. The engine has freed its states first."""

    # -- cache state (opaque to the engine) ------------------------------------

    def new_state(self) -> State:
        """An empty sequence. Claim its bytes from `services.accounting` BEFORE
        allocating, and free what was taken if anything below refuses."""
        raise NotImplementedError

    def clone_state(self, state: State) -> State:
        """An independent copy (it may share storage until written: your call).
        The engine clones a cached prefix for a new request."""
        raise NotImplementedError

    def truncate_state(self, state: State, n_tokens: int) -> None:
        """Keep the first `n_tokens`: the state must then behave exactly as a
        fresh one fed those tokens. A cache that cannot be cut back (a
        recurrence) may zero itself and queue the kept ids as a replay for the
        next step (report the replayed work in StepResult.forwarded_tokens).
        Everything kept becomes the next request's prompt."""
        raise NotImplementedError

    def state_nbytes(self, state: State) -> int:
        """The bytes the state holds."""
        raise NotImplementedError

    def export_state(self, state: State) -> bytes:
        """The state as plain bytes; `import_state` must restore it exactly."""
        raise NotImplementedError

    def import_state(self, blob: bytes) -> State:
        raise NotImplementedError

    def free_state(self, state: State) -> None:
        """Release the state's memory and its claim. Safe to call twice; a
        freed state given to `step` is a ContractError."""
        raise NotImplementedError

    # -- text --------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, tokens: list[int]) -> str:
        """Thinking markers and other special ids decode as the engine asks:
        it never renders the thinking markers as text."""
        raise NotImplementedError

    def chat(self, messages: list[dict], thinking: bool = True) -> list[int]:
        """The conversation as prompt ids, through the model's own template.
        Add `level=` (behind `thinking_levels`) and `tools=` (behind
        `tool_calls`) only when the package declares them."""
        raise NotImplementedError

    # -- the step ------------------------------------------------------------------

    def step(self, batch, ctx) -> list[StepResult]:
        """One step over the batch, one StepResult per item, in order.

        An item's `input_tokens` enter the model this step (empty = decode the
        token sampled last time); `sampling` None is a prefill chunk that
        samples nothing. Claim every state's room before any state moves, so a
        BudgetRefused leaves the batch as it was. Return only when the device
        has finished: the engine times this call.
        """
        if len(batch) > MAX_BATCH:
            raise ContractError(f"batch of {len(batch)} exceeds max_batch {MAX_BATCH}")
        if len({id(item.state) for item in batch}) != len(batch):
            raise ContractError("a state appears more than once in one batch")
        raise NotImplementedError("forward the batch and sample")


def create() -> Runtime:
    return Runtime()
