# AGENTS.md — how to work in this repository

Read `README.md`, then `CONTRACT.md`. footless is two parts — an engine that
knows nothing about models, and model packages that know everything about
theirs — and six rules (README, "The rules") that hold between them. Those
rules are not style preferences: a change that breaks one is a design error.

## Engine or model?

Before writing anything, decide which side it belongs to.

| It is about… | It goes in |
| --- | --- |
| requests, scheduling, batching policy, admission, cancellation | the engine |
| the memory budget, byte accounting, prefix reuse and eviction | the engine |
| sampling *policy* (what to sample, defaults, seeds), stop rules | the engine |
| the HTTP API, the CLI, rendering output, decisions semantics | the engine |
| the architecture, the forward pass, kernels, dtypes, devices | the model package |
| the cache layout, how state is grown, truncated, copied | the model package |
| weights, the tokenizer, the chat template, the generation config | the model package |
| executing a sampling spec where the logits live | the model package |

When a piece of logic is ambiguous, it belongs to the model. The engine never
gets a model name, an architecture term, a vendor or device name, or a tensor
shape. A new need of a model becomes a manifest field or a contract method —
never an `if` on the model.

## Map of the engine

| Module | Holds |
| --- | --- |
| `footless/sdk.py` | the contract as Python: `Facts`, `SamplingSpec`, `StepItem`, `StepResult`, errors, `RUNTIME_METHODS` |
| `footless/manifest.py` | reading the manifest and verifying it against the disk and the facts |
| `footless/engine.py` | `Engine` (open, `sequence`, `generate_tokens`, optional-verb gates), `Batcher` (scheduling), `PrefixCache`, `FileCache` (states saved on disk), `Ledger`, the loop guard |
| `footless/serve.py`, `wire.py` | the HTTP front end and the OpenAI/vLLM wire shapes |
| `footless/render.py`, `display.py` | output: thinking/content split, stop strings, tool calls; the terminal view |
| `footless/decisions.py` | decision questions: rendering, labels, probabilities |
| `footless/cli.py`, `bench.py` | the command line and the speed benchmark |

A request is `Engine.sequence`, a generator that yields `Admit`, any `Wait`
(a saved state loading) and then one `Want` (a `StepItem`) per step; `Batcher`
admits sequences and runs one `runtime.step` over all of them. `generate_tokens`
is one sequence run alone.

## Process

- Work granularly: one concern per change, each verifiable on its own. Land
  granular commits; never batch unrelated changes.
- Prefer editing an existing document over creating one. Keep documents short
  (about a hundred lines); decisions live next to what they decide.
- No speculative surface: no field, method, key or command before something
  needs it. Anything unfinished is marked as such in place; never describe an
  intention as if it were built.
- Documentation in English. Neutral names outside a package's hardware folders.
- Every engine change keeps `python3 -m unittest discover -s tests -t .` green
  (stdlib only) and adds a test against the mock packages for what it changes.

## Changing the contract

The contract is the expensive surface: every package is written against it.
A change touches, in one piece of work: `CONTRACT.md` (normative text),
`footless/sdk.py`, the engine, the three mocks in `tests/` (they are the
reference packages and must keep conforming), `tools/check_package.py` (a new
obligation gets a check), `templates/model-package/`, the docs that describe
it, and one dated line under "Decided" below. A new runtime verb is OPTIONAL
and behind a declared capability unless every package must have it.

## Working on a model package

Packages live outside this repository's history (`models/` is local
workspace), next to their weights. Follow `docs/roadmap.md`: its phases end at
gates, and correctness gates come before speed. Then:

- `tools/check_package.py --model <dir>` must pass before a package is served.
- The package keeps its own tests in `footless/tests/` and its harnesses,
  notes and measurement logs beside its runtime (e.g. `<backend>/_build/`).
  Run its tests with `FOOTLESS_PACKAGES=<dir> python3 -m unittest discover -s tests -t .`.
- Optimization discipline (docs/runtime-engineering.md): back up a file before
  changing it, measure A/B paired in one process, log every result including
  failures, keep only measured wins whose quality checks are unchanged.
- Exactness is a promise: speculation and batching return the same tokens as
  the plain path. Prove it bit for bit before declaring the capability or
  raising `max_batch`.

## Known traps

- `runtime.step` must return after the device is done; the engine times it.
- `Engine.speculative(False)` does nothing (the engine only turns it on); a
  test that needs it off calls the runtime's `set_speculative(False)`.
- A chat continued from re-rendered text instead of the absorbed token stream
  misses the prefix cache (`GenerationResult.absorbed_tokens`).
- The mocks' scripted "model" restarts its script at every prompt: tests that
  re-feed output as prompt (a preempted request, a cut-back state) see the
  script again, which a real model would not do.
- A stale compiled kernel measures old code: rebuild before measuring.

## Decided

- The engine and the model runtimes are written in Python; a runtime keeps its
  own logic and kernels and is built against `footless.sdk` (2026-09).
- The contract between the two parts is `CONTRACT.md` (2026-09).
- A sampling `seed` identifies one per-request draw stream, advanced once per
  sampled token (2026-09-24).
- Layout: `footless/` engine, CLI and SDK; `tests/` engine tests and mock
  packages; `tools/` reusable harnesses; `templates/` the package starting
  point; `docs/` guides; `models/` local packages, never tracked (2026-10).
- The serving API is an OpenAI-compatible surface in vLLM's shapes that
  ignores and logs what the engine cannot execute (2026-09-26).
- `POST /v1/decisions` and `/v1/systemone` score typed questions at the answer
  position behind `label_logprobs`, all of a request's questions in one
  `label_logprobs_batch` call from the prefix cache's head state (2026-10-01).
- A package declares its `thinking_levels`; `chat` takes one by name, mapped as
  vLLM maps `reasoning_effort` (2026-09-26).
- `set_speculative` is an optional verb behind `speculative`; `generate`, `chat`
  and `serve` take `--speculative` (alias `--mtp`), process-wide on `serve`
  (2026-10-01).
- Function tools are served behind `tool_calls`; "required" and named
  `tool_choice` are served as "auto" and reported (2026-10-01).
- A manifest may carry a `[runtime]` table read only by its runtime (2026-10-01).
- `serve` batches requests: `Batcher` admits them in arrival order up to the
  package's `max_batch` while the budget has room, and runs one `step` over all
  of them (every decode, at most one prefill chunk); a step the budget refuses
  runs again without its youngest request, which waits in the prefix cache
  (2026-10-05).
- States may be kept on disk in two caches the operator asks for, each with
  its own directory and byte cap and LRU at the cap or a full disk:
  `serve --prompt-cache` (a whole prompt's state, skipping its prefill) and
  `--prefix-cache` (absorbed prefixes, beside the RAM cache), both via
  `export_state` / `import_state` — a sentinel at every cut point a hit was
  truncated to, saves and loads beside the model, and a save whose export
  would not fit the memory left beside its state is not made (2026-10-09).
- The states' budget answers the machine: computed at open against the GPU
  working set and the system's reserve (the log names which one governs),
  and moved within that declaration by the system's memory pressure —
  warning stops the states growing, critical evicts their oldest, a minute
  of calm gives the room back (2026-10-10).
- Memory pressure takes the cache's room only: a lowered budget evicts cached
  states, while live requests are admitted and served against the ceiling
  (the most `set_budget` gave) -- it had refused a 110k-token conversation
  against a pressure budget of 1.7 GB (2026-10-12).
- A disk save's blob claims its room from the states' budget until written,
  evicting cached states, instead of needing that room free -- with the RAM
  cache full, as it is after a while, nearly nothing reached the disk
  (2026-10-12).

- Images cross the contract behind the `vision` capability, on only when the
  operator asks (`serve --vision`): `set_vision`, `chat_images` (prompt ids and
  `ImageSpan`s), `StepItem.images`; an `Image` is the file's bytes and their
  sha256, which keys the prefix cache; the API reads base64 data URLs only
  (2026-10-07).
- A stream carries a tool call as vLLM's does (id and name, then the
  arguments as they grow); `tool_calls` may list the call still being written
  as `partial`, its arguments a prefix of the complete call's (2026-10-07).
- Behind `concurrent_prefill` (a promise about `step`, no verb) the Batcher
  runs one prefill step at a time on a thread of its own while decode steps go
  on; the ledger and the prefix cache take one lock (2026-10-11).
- `export_state` may return a `bytearray`: a large state's blob is built in
  place, never copied again (2026-10-11).
- Behind `concurrent_decode` (a promise about `step`, no verb) the Batcher
  steps an even number of decoding requests as two groups on threads of their
  own, each at its own pace, while no prefill is in flight: two sessions with
  drafting 103 -> 111 tok/s in all, four sampled 127 -> 134, the same tokens
  (2026-10-12).
- A cached state is taken for a prompt only when the shared prefix is worth
  its copy (HIT_MIN 256 tokens, or half its prompt): a fresh prompt had read
  long conversations' states from the disk to keep the template's first
  tokens. A request reading a prompt is its head's producer before its own
  lookups: with the disk caches on, requests arriving together each read
  the shared head (2026-10-13).

## Not yet decided

Automatic hardware detection (the CLI takes `--backend`), audio and video
input, stop strings inside the runtime, the repository layout beyond the above, and anything `CONTRACT.md`
marks as not covered. Do not assume them; when one is decided,
record it above in one line.
