# The contract

The exact shape of the boundary between the engine and a model package. Both
sides are Python; the types below live in `footless.sdk`, the SDK runtimes are
built against. `README.md` gives the design; this file is normative for what
crosses.

A model package carries a manifest and one folder per hardware target. The
engine detects hardware, picks one backend folder, and loads its runtime:

```
<model>/footless/
├── manifest          # read before any model code runs
└── <backend>/
    └── runtime.py    # create() -> Runtime
```

Three surfaces: the **manifest** (before load), the **runtime** (engine →
model), the **services** (model → engine).

## Manifest

`footless/manifest`, TOML. The single source of truth before load (rule 3):

| key | type | meaning |
| --- | ---- | ------- |
| `name` | string | package identity; keys reused cache state |
| `backends` | list of strings | hardware folders present; must match the disk |
| `max_batch` | integer | most requests one step may carry |
| `max_context` | integer | most tokens one request may carry |
| `cache_bytes_per_token` | integer | cost of cache state, per token |
| `capabilities` | list of strings | what the model offers; undeclared is unavailable and reported (rule 4) |
| `thinking_levels` | list of strings, optional | reasoning levels `chat` accepts by name; absent means one thinking mode, on or off |
| `runtime` | table, optional | the package's own settings, read by its runtime after load; the engine checks only that it is a table |

The capabilities defined so far are `logprobs` (top log-probabilities on step
results), `speculative` (the package offers a drafter), `label_logprobs`
(next-token label scoring at the answer position), `tool_calls` (function
tools rendered by `chat` and read back), `vision` (the package reads images),
`concurrent_prefill` (a prefill step may run beside decode steps) and
`concurrent_decode` (two decode steps may run at once); the middle four are
reached through the optional verbs below, the last two are promises about
`step` (see `step` below).

## Runtime — engine → model

`<backend>/runtime.py` exposes `create() -> Runtime`:

```python
class Runtime(Protocol):
    # lifecycle — binds weights and tokenizer where they already are (rule 6)
    def open(self, model_dir: str, services: "Services") -> "Facts": ...
    def close(self) -> None: ...

    # cache state: opaque to the engine (rule 2) — it moves it and counts
    # its bytes, never looks inside
    def new_state(self) -> "State": ...
    def clone_state(self, state: "State") -> "State": ...
    def truncate_state(self, state: "State", n_tokens: int) -> None: ...
    def state_nbytes(self, state: "State") -> int: ...
    def export_state(self, state: "State") -> bytes | bytearray: ...
    def import_state(self, blob: bytes) -> "State": ...
    def free_state(self, state: "State") -> None: ...

    # text
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...

    # conversation to prompt tokens, via the package's own template
    def chat(self, messages: list[dict], thinking: bool = True) -> list[int]: ...
    # (`level=` and `tools=` are passed only when set, behind `thinking_levels`
    # and `tool_calls`)

    # one step over a batch of requests
    def step(self, batch: list["StepItem"], ctx: "StepContext") -> list["StepResult"]: ...
```

### Optional verbs

Everything above is required: `Engine.open` fails on a runtime missing any of
them. A small number of verbs are **optional**, are not in that list, and the
engine reaches them only behind a declared capability. Today there are six:

| verb | capability | meaning |
| --- | --- | --- |
| `set_speculative(enabled: bool) -> None` | `speculative` | the model may draft ahead and verify its own draft, producing the same tokens it would have produced one at a time |
| `label_logprobs(input_tokens: list[int], labels: list[str]) -> list[tuple[int, float]]` | `label_logprobs` | each label resolved to the single token it adds at the answer position after `input_tokens`' text, returned with its full-vocabulary next-token log-probability, in the labels' order |
| `label_logprobs_batch(items: list[tuple[list[int], list[str]]], state: State \| None = None) -> list[list[tuple[int, float]]]` | `label_logprobs` | `label_logprobs` for each (input_tokens, labels) pair, in order, from `state` when given; optional even behind the capability — without it the engine calls `label_logprobs` per pair |
| `tool_calls(text: str, tools: list[dict], final: bool) -> tuple[str, list[dict]]` | `tool_calls` | the answer `text` as its content and the calls it makes (`{"name", "arguments"}`, arguments as JSON text); with `final` false the text may still grow, so content stops short of what may yet open a call; the last call listed may be one still being written, marked `"partial": True`, its `arguments` so far a prefix of the complete call's (a package may list only complete calls); each result extends the previous one, but the final one lists complete calls only (a call the answer ended inside is dropped) |
| `set_vision(enabled: bool) -> None` | `vision` | loads (or drops) what reading images takes; the operator asks for it (`--vision`), after `open`, so it may claim from `accounting` |
| `chat_images(messages: list[dict], thinking: bool = True) -> tuple[list[int], list[ImageSpan]]` | `vision` | `chat` for messages whose content may hold `{"type": "image", "image": Image}` parts: the prompt ids and where each image sits among them, in order (`level=` and `tools=` as for `chat`) |

- The engine branches on the **capability**, never on the package. A package
  that does not declare `speculative` is refused when the switch is asked for,
  naming the reason — never quietly generated without it (rule 4).
- A package that declares `speculative` but exposes no `set_speculative` is a
  contradiction between the manifest and the code, and is reported as one.
- The engine does not know how a package speculates, or what it costs, or
  whether it is exact. Whatever the verb turns on is the model's business, and
  the model may make it load anything it needs at that point — including
  claiming from `accounting`, since the switch arrives after `open`.
- An optional verb must not change what the model returns. Here that is
  testable and tested: with the switch on, a package's token stream is
  identical to the same decode with it off.
- `label_logprobs` scores labels at the answer position and generates
  nothing: one prefill, no sampling, no penalties, no temperature. The engine
  branches on the **capability** the same way — a package that does not
  declare `label_logprobs` is refused when scoring is asked for, naming the
  reason (rule 4), and one that declares it but exposes no `label_logprobs`
  verb is a contradiction between the manifest and the code: a reported
  `ContractError`, never a silent no-op. The verb must not change what the
  model generates: it reads the next-token distribution and leaves generation
  exactly as it found it.
- Each label must resolve to exactly one token at the answer position, labels
  must not collide on one token id, and the prompt text must round-trip
  through `decode`/`encode`. A violation raises `ValueError` naming the
  offending label — the caller's input at fault, not a contract breach.
- `label_logprobs_batch` may share work between the pairs (a common prompt
  head forwarded once); each result equals what `label_logprobs` returns for
  that pair alone, up to the model's own floating-point rounding. A violation
  raises the same `ValueError`, with `.item` set to the index of the pair.
- A `state` given to `label_logprobs_batch` was fed by `step` (nothing
  pending, no replay queued) and its tokens are a proper prefix of every
  pair's input: the verb scores from it instead of re-reading that prefix, and
  leaves it as it found it, so the engine may keep it in its prefix cache.

- `State` is the model's own object; the engine holds it opaque. `step` and
  `truncate_state` mutate it in place; `new_state`, `clone_state` and
  `import_state` produce new handles. `clone_state` may share storage — the
  model decides.
- `truncate_state` keeps the first `n_tokens`. This is how a reused state is
  fitted to a shorter shared prefix.
- `export_state` / `import_state` are how the engine moves state between tiers
  and across runs; the blob is plain bytes and its length is countable. It may
  be a `bytearray` (built in place: no second copy of a large state); the
  engine only measures and writes it.
- `open` returns the facts below. The engine compares `cache_bytes_per_token`,
  `max_batch` and `max_context` for equality with the manifest and requires
  every declared capability to be present in the facts; a contradiction is a
  hard error (rule 3). Capabilities the runtime reports beyond the manifest are
  simply unused (rule 4).
- `step` runs one step over the batch, one `StepResult` per item, in order. An
  item's `input_tokens` is what enters the model this step (empty = decode); a
  `sampling` spec asks for a token this step, `None` is a prefill chunk that
  samples nothing. Positions are the model's business — they are never passed.
  A batch holds at most `max_batch` items and a state at most once; how its
  items are computed together is the model's business, and each item's result
  is the one it would get in a step of its own. A refused claim
  (`BudgetRefused`) leaves every item of the batch as it was, so the engine can
  run the step again without one of them.
- With `concurrent_prefill` declared, the engine may run a step whose items
  all feed input tokens (a prefill step) on one thread while steps whose items
  feed none (decode steps) run on another: at most those two at once, never
  sharing a state, each returning after the device is done with it. Meanwhile
  it may call the state verbs on other states, and `accounting` and `log` may
  be called from either thread. Each step's results are the ones it would get
  alone. Without the capability the engine calls the runtime from one thread.
- With `concurrent_decode` declared, the engine may run two decode steps on
  two threads at once (and, with `concurrent_prefill`, a prefill step on a
  third), never sharing a state, each returning after the device is done
  with it; the rest is as for `concurrent_prefill`. Each step's results are
  the ones it would get alone, so each group of sequences goes at its own
  pace.
- `chat` renders a conversation with the package's own template and encodes
  the result. A message is a dict with `role` and `content`; any other keys
  reach the template unchanged. `thinking` selects the template's thinking
  mode; `level` names one of the package's `thinking_levels` and means the
  package's own default when it is None. The engine passes `level` only when
  a request sets one, so a runtime written before the parameter keeps serving
  requests that do not name a level.

`Facts`: `name`-independent load-time truths — `cache_bytes_per_token`,
`max_batch`, `max_context`, `capabilities`, `stop_tokens: list[int]`
(generation ends on these ids; model knowledge learned from the weights, so
it is reported rather than declared), `thinking_markers: list[int]` — the ids
that delimit thinking blocks, in order: index 0 opens a block, index 1 closes
one (they are generated like any token but never rendered as text),
`thinking_open_at_start: bool` — in the thinking mode the rendered prompt
leaves a thinking block open, so generation begins inside one (a prompt
rendered with thinking off closes the block, so generation begins outside
one), and `thinking_levels: list[str]` — the reasoning levels this runtime's
`chat` accepts, verified against the manifest. `default_sampling:
SamplingSpec | None` — the package's own sampling defaults, its generation
config: what a request that pins nothing runs under. None leaves them to the
engine.

`StepItem`: `state`, `input_tokens: list[int]`, `sampling: SamplingSpec | None`,
`images: list[ImageSpan]` (empty unless `vision` is on).

`Image`: `data: bytes` — the encoded file as the request sent it, opaque to
the engine — and `digest`, the sha256 of the bytes: its identity.
`ImageSpan`: `start`, `length`, `image` — the image takes `length` tokens
from `start`. `chat_images` places them in the prompt; on a `StepItem` a span
counts from `input_tokens[0]` (negative when the image began in an earlier
step) and carries the whole image's length: every step that feeds some of an
image's tokens carries the image. Its tokens are the model's placeholders,
the same for every image, so the engine keys its prefix cache on the images'
digests there: a cached state is reused only for the same images. Without
`vision` on, the engine never sends one, and a conversation with images is
refused.

`StepResult`: `token: int | None`, `finished: bool`,
`top_logprobs: list[tuple[int, float]] | None` (capability `logprobs`), and
`forwarded: bool = True`.
`finished` carries the model's stop rules — its special tokens; the engine adds
its own (length, policy).

`forwarded` says whether the step RAN the model. A model that verifies a draft
can return a token it had already computed on a step that does no forward at
all; that token and the step's wall time are real and both count toward the
request's totals, but the step is not a sample of how fast a forward runs — at
about five microseconds it books as a rate in the hundreds of thousands and
drags the phase's min, mean and max with it. The default is `True`, so a runtime
that has no such step is unaffected.

`forwarded_tokens: int = 0` counts how many tokens the step ran through the
model. It exists for `truncate_state`: a truncate that cannot rewind for free
queues the kept ids as a replay, and the NEXT step's prefill runs the replay
beside what the request fed. The engine's rates must count the work that ran
rather than the work the request fed — invisible re-forwards would otherwise be
booked against the model as slowness. `0` means "as many as were fed", which is
exact for a runtime whose truncate never queues a replay.

`SamplingSpec`: `temperature`, `top_k`, `top_p`, `min_p`, `logprobs: int`
(0 = off), `seed: int`, and the anti-repetition stack `repetition_penalty:
float = 1.0`, `presence_penalty: float = 0.0`, `frequency_penalty: float =
0.0`. The three penalties are vLLM's semantics verbatim: `repetition_penalty`
is the HF scale over tokens seen in the prompt OR the output (seen: `logit > 0
? logit / p : logit * p`), `presence_penalty` subtracts once per token seen in
the OUTPUT, `frequency_penalty` subtracts its value times the OUTPUT count.
They act on the raw logits, before the temperature divide and the top-k/top-p
cuts, in the order repetition -> frequency -> presence, and they count for
greedy too (the argmax is of the penalized row). The engine's sampling policy produces the spec and owns
randomness; the model executes the spec where its logits live. A `seed`
identifies one per-request draw stream, advanced once per sampled token. The
policy layers: a field a request pins wins, then the package's
`Facts.default_sampling`, and only where the package declares none the spec's
own defaults.

`StepContext`: `cancelled() -> bool`, checked at step boundaries. It covers
the whole batch: the engine leaves a cancelled request out of a step itself.

## Services — model → engine

Offered on `open`; optional — a runtime may use none of them:

```python
class Services(Protocol):
    log: Callable[[str, str], None]   # (level, message)
    accounting: Accounting           # claim(nbytes) -> Claim   (raises when refused)
                                     # release(claim) -> None
                                     # available() -> int
    buffers: Buffers                 # host(nbytes) -> Buffer   (accounted host bytes)
                                     # free(buffer) -> None
```

The budget is the engine's to enforce: it evicts cached state before refusing a
claim, and a refused claim is an error, never silent degradation (rule 4).

## Not covered yet

- The serving API (network): not part of this contract. It is an
  OpenAI-compatible front end, `footless/serve.py` over these same verbs.
- Stop strings and other detokenized stop rules.
- Hardware detection: the CLI selects the backend with `--backend`; automatic
  detection is not built.
