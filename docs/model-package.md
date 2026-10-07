# Writing a model package

A model package is a directory that already holds a model (its weights,
tokenizer and configs, untouched — rule 6) plus a `footless/` folder the engine
reads. `templates/model-package/` is the starting point; the mocks in `tests/`
are the smallest complete packages; CONTRACT.md is the normative text this
guide walks through.

```
<model>/
├── config.json, tokenizer.json, *.safetensors …   # as published
└── footless/
    ├── manifest              # TOML, read before any code runs
    ├── <backend>/            # one folder per hardware target (any name)
    │   ├── runtime.py        # create() -> the runtime object
    │   └── …                 # kernels, bridges, tokenizer: the package's own
    └── tests/                # the package's own tests (optional)
```

## The manifest

The facts the engine needs before loading anything: `name`, `backends` (must
match the folders on disk), `max_batch`, `max_context`,
`cache_bytes_per_token`, `capabilities`, optionally `thinking_levels` and a
`[runtime]` table of the package's own settings. The runtime's `open` reports
the same numbers in its `Facts`; any disagreement is a hard error (rule 3).

`cache_bytes_per_token` is the cost that grows with context (a KV cache's bytes
for one token, every layer). Fixed per-sequence state (a recurrence, a conv
window) is not in it: claim it when the state is made.

## The runtime

`<backend>/runtime.py` defines `create()`. The object it returns has the
required verbs (`sdk.RUNTIME_METHODS`) and, behind declared capabilities, the
optional ones. What the engine relies on, verb by verb:

- **open / close.** Bind weights where they are, check the checkpoint against
  what the code assumes, allocate scratch. If the device holds less than the
  engine's budget, call `services.accounting.set_budget(bytes)` so that states
  are refused as `BudgetRefused` before the allocator fails mid-request.
- **new_state / free_state.** A state is opaque: the engine moves it, counts it
  and hands it back. Claim its bytes from `services.accounting` *before*
  allocating, release on every failure path, and make a freed state an error
  for `step`.
- **truncate_state.** After `truncate_state(s, n)` the state behaves exactly as
  a fresh one fed the first `n` tokens — this is how a cached conversation is
  reused for a request that shares only part of it. A cache that cannot be cut
  back (a recurrence) zeroes and queues the kept ids as a replay for the next
  step; report that work in `StepResult.forwarded_tokens`. Everything kept
  becomes the next request's prompt (for the penalties' prompt/output split).
- **clone_state / export_state / import_state.** Independent copies; plain
  bytes that restore exactly. Never clone a state that would not fit beside the
  original: the engine then hands the original over instead.
- **encode / decode / chat.** The tokenizer and the model's own chat template.
  `chat(messages, thinking)` takes `level=` only with `thinking_levels`, and
  `tools=` only with `tool_calls`. Thinking markers are reported in `Facts`
  and never rendered as text by the engine.
- **step(batch, ctx).** One `StepResult` per item, in order. An item's
  `input_tokens` enter the model now (empty means: forward the token sampled
  last step, then sample); `sampling=None` is a prefill chunk that samples
  nothing. Positions come from the state, never from the engine. Return only
  when the device has finished — the engine times this call — and claim every
  state's room before any state moves, so a refusal leaves the batch intact.

## Sampling

The engine decides the policy (`SamplingSpec`), the runtime executes it where
the logits live: temperature, top-k, top-p, min-p, and the penalties with
vLLM's semantics (repetition over prompt and output, frequency and presence
over output only, applied to the raw logits before temperature, greedy
included). `spec.seed` starts one draw stream per request, advanced once per
sampled token: key it by the spec object the request carries, so a repeated
request replays and two at once never share. Evict old streams one by one,
never all at once.

## Optional verbs and capabilities

Declare a capability only when the verb behind it exists; the engine refuses
an undeclared one with the reason (rule 4) and treats a declared one without
its verb as a contract error.

| Capability | Verb | Obligation |
| --- | --- | --- |
| `speculative` | `set_speculative(enabled)` | the same tokens as without it, bit for bit |
| `label_logprobs` | `label_logprobs`, optionally `label_logprobs_batch` | scores at the answer position, generation untouched |
| `tool_calls` | `tool_calls(text, tools, final)` | reads the calls the template's format writes |
| `vision` | `set_vision(enabled)`, `chat_images(messages, ...)` | places each image's tokens in the prompt; a step reads the images its spans name (a span may begin in an earlier step); a state keeps its images (truncate, clone, export) |
| `logprobs` | (on `StepResult.top_logprobs`) | the top log-probabilities of each sampled step |

A step that hands out a token computed earlier (a verified draft) sets
`StepResult.forwarded = False`, so the engine does not book it as a forward.

## Batching

`max_batch > 1` promises that each item of a batch gets exactly the result it
would get in a step of its own. Start at 1; raise it only when batched rows are
proven equal to rows alone (`tools/check_package.py` checks this, greedy and
token for token). How rows share work is the package's choice: typically the
weight-bound projections run once over all rows, and attention and any
recurrence run per row against each row's own state.

## Testing a package

- `python3 tools/check_package.py --model <dir>` drives every verb through the
  engine and fails on any deviation from the contract.
- The package's own tests live in `footless/tests/` (kernel oracles, golden
  logits, perplexity gates); run them with the engine's suite:
  `FOOTLESS_PACKAGES=<dir> python3 -m unittest discover -s tests -t .`
- Then the real thing: `footless chat`, `footless bench`, `footless serve`, and
  the HTTP tools in `tools/`.

docs/roadmap.md orders this work into phases with an exit gate each.
