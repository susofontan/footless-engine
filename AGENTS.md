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
| `footless/engine.py` | `Engine` (open, `sequence`, `generate_tokens`, optional-verb gates), `Batcher` (scheduling), `PrefixCache`, `Ledger`, the loop guard |
| `footless/serve.py`, `wire.py` | the HTTP front end and the OpenAI/vLLM wire shapes |
| `footless/render.py`, `display.py` | output: thinking/content split, stop strings, tool calls; the terminal view |
| `footless/decisions.py` | decision questions: rendering, labels, probabilities |
| `footless/cli.py`, `bench.py` | the command line and the speed benchmark |

A request is `Engine.sequence`, a generator that yields `Admit` and then one
`Want` (a `StepItem`) per step; `Batcher` admits sequences and runs one
`runtime.step` over all of them. `generate_tokens` is one sequence run alone.

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

- Images cross the contract behind the `vision` capability, on only when the
  operator asks (`serve --vision`): `set_vision`, `chat_images` (prompt ids and
  `ImageSpan`s), `StepItem.images`; an `Image` is the file's bytes and their
  sha256, which keys the prefix cache; the API reads base64 data URLs only
  (2026-10-07).

## Not yet decided

Automatic hardware detection (the CLI takes `--backend`), audio and video
input, stop strings inside the runtime, the repository layout beyond the above, and anything `CONTRACT.md`
marks as not covered. Do not assume them; when one is decided,
record it above in one line.
