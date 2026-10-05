# Runtime engineering notes

What building model runtimes for footless has taught, written for the next
one. None of it is specific to one model or one device; the numbers are there
to show orders of magnitude.

## Correctness first

- **Oracles before kernels.** Every kernel gets a test against a simple
  reference (numpy, or the reference implementation) at the model's real
  shapes. Mutation-check the tests: a deliberately broken kernel must fail
  them, or they prove nothing.
- **Golden logits, frozen early.** Top-k ids and values for fixed prompts,
  taken before any optimization; afterwards assert the argmax, the top-k
  overlap and a drift budget. Add a perplexity gate on a standard text.
- **Bisect, don't guess.** When outputs go wrong, compare layer by layer
  against the reference and keep each probe as a test.
- **Exercise the public entry points.** Internal tests missed regressions that
  a single `step` call would have caught; test through the engine too
  (`tools/check_package.py`).
- **Exact paths stay exact.** Speculation and batching promise the same tokens
  as the plain path. Floating-point sums are order-dependent: a batched or
  verifying GEMV must keep the same split and reduction order per row as the
  single-row one, or results drift by a rounding and streams diverge. Test
  with bit-for-bit comparisons of the logits, not with "looks the same".

## State and memory

- **Claim before you allocate**, release on every failure path, and test that
  a refused allocation leaves nothing behind. A leaked claim is not a slow
  leak: it is the next request refused.
- **Count what the device really takes.** Allocators round up (often to
  hundreds of KiB or MiB); booking the requested size left tens of MiB per
  state off the ledger and the card ran out with the ledger "below" budget.
- **Give the ledger the device truth** (`set_budget`) after loading: an engine
  budget larger than the card admits clones until the allocator fails.
- **Never hold two copies of the biggest thing.** Grow caches piece by piece
  (layer by layer), and never clone a state that is most of the device: hand
  the original over.
- **A state is what it absorbed.** The prefix cache keys a state by exactly
  the tokens it has seen. Conversations must be extended as token streams,
  not re-rendered from text: re-tokenizing at template seams breaks the match,
  and a partial match forces a truncate — a full replay for a recurrent model.
- **Recurrences cannot be cut back.** Truncating a recurrent state means
  zeroing it and re-running the kept prefix; make that a replay inside the next
  step and report it (`forwarded_tokens`), never a silent slowdown.

## The step

- **`step` returns when the device is done.** An asynchronous step measures
  the host's enqueue rate (it once reported 2x the real prefill speed).
- **Chunk size is a weight-traffic decision.** Every chunk reads all the
  weights once; tiny chunks turn one prompt into hundreds of weight passes.
- **Hand-outs are not forwards.** A token computed by an earlier step (a
  verified draft) comes back with `forwarded=False`, or the engine books
  microsecond "steps" as absurd speeds.

## Sampling

- Execute the engine's spec exactly; keep a slow, obviously-right host
  sampler and check the fast one against it on real logit rows.
- One draw stream per request (keyed by the request's spec), advanced once per
  sampled token. Never clear the stream cache wholesale: a restarted stream
  repeats its draws, and repeated draws repeat tokens.
- Penalties widen the candidate window by how far they can move a logit, bounded
  by the largest per-token count, not the output length — or the window
  overflows and every token falls to the slow path.

## Measuring

- **Paired A/B in one process**, interleaved, after a warm-up. Clocks change
  with temperature and idle time; back-to-back runs in separate processes
  measure the weather. Report both orders.
- **Watch the clocks.** Sustained load throttles a hot card by ~10% or more;
  a cold one after idle runs slow. Record clock and temperature with results.
- **Rebuild before you measure.** A stale compiled kernel measures the old
  code; key builds on source and flags.
- **Check the arithmetic.** Shares of one shape applied to another, or byte
  counts of half-read operands, manufacture gains that do not exist.
- **Choose shapes in the kernel, not in a microbenchmark.** The same loop runs
  at very different rates alone and inside a large kernel.

## Speed

- **Know the floor.** Decode at batch 1 reads every weight once per token:
  bytes / bandwidth is the floor. Compressed weights move the floor to the
  decode arithmetic: measure instructions per weight, not just bytes.
- **Launches add up.** Hundreds of launches per token cost milliseconds of
  host and gaps; fuse neighbours (norm into the next projection's prologue,
  several projections of one input into one launch, residual adds into norms).
- **Batch rows amortize the weights.** With weights decoded once per k-step,
  a group of rows costs far less than that many single rows: the weight-bound
  projections run once over every row, attention and recurrences per row.
- **Fill the machine.** Small grids leave a partial last wave; split the
  reduction axis (and sum the partials in a fixed order) to fill it.
- **Decode attention is split, prefill attention is tiled.** Partials over the
  key axis with an online-softmax merge for one query; query tiles with keys
  staged in fast memory for many.
- **Keep only measured wins.** Back up before a change, log every result —
  including the failures, so nobody tries them again — and revert what does not
  pay.
