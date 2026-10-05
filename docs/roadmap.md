# Roadmap: a new model on new hardware

The order that has worked for bringing a model up as a footless package, from
an empty folder to a served, measured runtime. Each phase ends at a gate: a
check that passes before the next phase starts. Correctness gates come before
speed, always: a fast wrong model costs more than a slow right one.
docs/runtime-engineering.md holds the methods each phase leans on.

## 0. Survey

- The architecture from its reference implementation (layer types, attention
  variants, recurrences, norms, rope, any extra heads), the weight format and
  its quantization, the tokenizer and the chat template, the generation config.
- The hardware's ceilings: memory bandwidth, peak FLOPs per precision, memory
  size, launch cost, what the toolchain targets.
- The arithmetic: weights + per-token cache + fixed per-sequence state against
  device memory gives `max_context`, `cache_bytes_per_token` and how many
  sequences could ever share the card.

**Gate:** a written design note in the package (`<backend>/` or `_build/`):
shapes, memory plan, the expected decode floor (bytes per token / bandwidth).

## 1. Skeleton, tokenizer and template

- Copy `templates/model-package/footless/` next to the weights; fill the
  manifest; implement `encode`, `decode`, `chat` and `open`'s config checks.
- Port the tokenizer and template from the model's own files.

**Gate:** golden tests against the reference tokenizer and template (ids for a
spread of texts, languages, special tokens, every thinking mode and level,
tool renders); `footless` opens the package and the manifest agrees.

## 2. A reference forward

- One forward that is obviously right before it is fast: per-layer outputs
  compared against the reference implementation (or a numpy oracle of it).
- Bisect layer by layer when they disagree; keep the probes as tests.

**Gate:** per-layer agreement within tolerance, the reference's next-token
argmax on fixed prompts, and a perplexity on a standard text that matches the
reference within its own noise. Freeze golden logits now.

## 3. Device kernels

- Port each operation to the hardware folder, one at a time, each with an
  oracle test against the reference forward at real shapes; mutation-check the
  oracles (a deliberately broken kernel must fail them).
- Wire `step` for prefill chunks and single-token decode; sync before returning.

**Gate:** kernel oracles green, golden logits within the drift budget,
perplexity unchanged, a greedy chat that reads well.

## 4. State, cache and accounting

- `new_state`, `free_state`, `truncate_state`, `clone_state`,
  `export_state`/`import_state`; claims before allocations; `set_budget` from
  the device's free memory; growth that never needs a second full copy.

**Gate:** `tools/check_package.py` passes; a long chat reuses its cache turn
after turn (cached tokens close to the prompt); a forced small budget refuses
cleanly instead of crashing.

## 5. Serving end to end

- `footless chat`, `footless generate`, `footless serve` with the HTTP tools in
  `tools/` (`serve_check.py`, `serve_chat_check.py`).

**Gate:** streamed and plain answers equal for a fixed seed; thinking, stop
strings and sampling defaults behave as docs/serving.md says.

## 6. Speed

- Measure first: `footless bench` for prefill and decode by context, a
  per-kernel profile, and the floor from phase 0. Attack the largest share.
- One change at a time, A/B paired in one process, backed up and logged;
  keep only measured wins that leave the quality gates exactly where they
  were.

**Gate:** each landed change has its numbers and its quality check in the
package's log; decode within reach of its floor, or the reason it is not.

## 7. Capabilities

- `speculative` (a draft head, n-gram drafting…), `label_logprobs`,
  `tool_calls`, `logprobs` — each declared only with its verb, each with its
  exactness test (speculation: the same tokens, bit for bit, greedy and seeded).

**Gate:** check_package's capability checks; the serving tools' decision and
tool-call routes answer as documented.

## 8. Batching

- Raise `max_batch` once several sequences' rows share one forward and each row
  is the row it would be alone; claim every state's room before any state moves.

**Gate:** check_package's batch checks; `tools/serve_concurrency.py` reports
every answer identical to the one alone, and the aggregate rate grows with the
number of requests.

## 9. Long context and memory

- Context limits measured, not computed: the longest prompt that serves, with
  and without speculation; cache formats (e.g. lower-precision KV) only with a
  quality check at long context.

**Gate:** the documented `max_context` serves; past it, a clean refusal.

## 10. Hand-over

- The package's README: what it runs on, how to install it, measured speed and
  quality, known limits. Its tests and harnesses stay in the package.

**Gate:** someone else follows the README on a clean machine and gets the same
numbers.
