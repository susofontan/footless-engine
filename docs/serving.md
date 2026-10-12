# The serving API

`footless serve --model <dir>` answers an OpenAI-compatible API: the wire shape
of vLLM's OpenAI entrypoints, so the same clients work against both. The wire
lives in `footless/wire.py` (requests parse there, responses build there), the
front end in `footless/serve.py`.

| Route | What it does |
| --- | --- |
| `GET /health` | 200 while the server runs; never needs the API key |
| `GET /v1/models`, `GET /v1/models/<name>` | the one served model, with `max_model_len` |
| `POST /v1/chat/completions` | chat, plain or streamed (SSE) |
| `POST /v1/completions` | raw prompts, plain or streamed (SSE) |
| `POST /v1/decisions` | typed questions scored without generating (behind `label_logprobs`) |
| `POST /v1/systemone` | the same, in the System One 0.2.0 shapes |

`--api-key` makes every route but `/health` ask for `Authorization: Bearer <key>`;
`--served-model-name` changes the name clients send.

## Several requests at once

Requests run together. The engine's scheduler (`engine.Batcher`) admits them in
arrival order, up to the package's `max_batch` at once and while the memory
budget has room for them, and runs one `step` over all of them: every request
that is decoding, and at most one prefill chunk — the request's with the
fewest prompt tokens left, so a short prompt is not held behind a long one's
every chunk (one passed over for 20 s goes first) — in chunks of 1024 tokens while
others decode, so their tokens wait for one chunk at a time. A package that
declares `concurrent_prefill` runs the prefill chunk on a thread of its own
instead, one at a time, while the decode steps go on without waiting for it
(chunks of 2048 beside others, 1024 when another prefill waits). How the rows of
one step are computed is the package's business; each item's result must be
the one it would get alone (CONTRACT.md). A step the budget refuses runs again
without its youngest request: that one waits in the prefix cache and is
admitted again later. The choices of one request (`n`) run one after another,
and decision requests run between steps. A client that disconnects cancels its
request at the next step boundary.

A package with `max_batch = 1` is served one request at a time, by the same
code.

A finished request leaves its state in the prefix cache (RAM, evicted
least-recently-used first). The next turn of a conversation starts from a copy
of it and, once done, replaces it: one entry a conversation, not one a turn.
An entry another conversation only shares a head with (a system prompt) stays.
Requests that arrive together on one long head (agents started at once on
one system prompt and tools) read it once: the later one waits until the
first has read the shared part, starts from a copy of its state there, and
reads only its own tail.

## Saved states on disk

Two optional caches keep states in a directory of their own, each under a
byte cap of its own, so reuse survives a restart and a budget eviction:

| Flags | What it keeps |
| --- | --- |
| `--prompt-cache DIR --prompt-cache-max SIZE` | each prompt's state as it stood before its first token: the whole prompt again skips its prefill |
| `--prefix-cache DIR --prefix-cache-max SIZE` | every absorbed prefix's state: a shared head is reused across runs, beside the RAM prefix cache |

`SIZE` is bytes, or with a K/M/G/T suffix (`2G`). A cache is on only when
both its flags are given. Entries are states as the package exports them
(`export_state`, CONTRACT.md: opaque, countable), keyed on the token ids —
an image's placeholders are its digest — and held least-recently-used first.
Over its cap a cache drops its oldest entries; a write that meets a full
filesystem evicts the same way until it fits, and is dropped with the reason
logged when even an empty cache cannot hold it. A cache never fails a
request.

The prompt cache answers whole prompts only; the prefix cache any common
prefix, and a match saved on disk longer than memory's wins. Without the
flags, neither exists and nothing is written.

A hit shorter than its entry's key is fitted by the package's
`truncate_state`: where a state cannot be cut back cheaply, the kept tail
re-forwards in the next step (reported as `forwarded_tokens`, CONTRACT.md).
A whole-prompt hit starts past its prefill outright. The engine also saves
what a hit was cut back to: sentinel entries at the cut points it sees, so
a later turn starts at one instead of re-forwarding the kept tail. Saves and
loads run beside the model — a request waits only for the state it is about
to use. A save's blob — at a long session's size, gigabytes — claims its
room from the states' budget until it is written, evicting cached states as
any claim does (within the room memory pressure leaves the cache); a save
that cannot get it is not made: a cache never risks the machine it is
speeding up.

## Fields

A field the engine can execute is honoured: prompt and messages, `n`, the
sampling knobs (`temperature`, `top_p`, `top_k`, `min_p`), the
`repetition_penalty`, `frequency_penalty` and `presence_penalty` penalties,
`seed`, `max_tokens`, `stop` and `stop_token_ids`, `echo`,
`truncate_prompt_tokens`, `stream` and `stream_options`, `logprobs`.

A field that demands behaviour the engine cannot execute is accepted and
ignored, as vLLM ignores what it does not serve, and the server log says which;
nothing fails for asking. What fails is a request that is wrong of itself — a
mistyped field, a prompt past the context, an unknown model.

Sampling fields a request omits take the package's own defaults (its
generation config, `Facts.default_sampling`), not OpenAI's; only a package that
declares none falls back to the engine's. A pinned `seed` is one draw stream
for that request: the same request with the same seed answers the same, and two
requests running at once never share a stream. Several choices (`n > 1`) drop
the seed, so they differ.

## Thinking

Thinking blocks surface as the `reasoning` field beside `content` (and as
`reasoning` deltas in a stream), their tokens counted in
`usage.completion_tokens_details.reasoning_tokens`; the markers that delimit
them (`Facts.thinking_markers`) are never rendered as text.

`reasoning_effort` is served as vLLM serves it: `"none"` turns thinking off,
any other value is one of the package's `thinking_levels` — a level the package
does not offer is a 400 that names the ones it does — and a package that
declares no levels never sees the value.

Clients should send each assistant turn back with its `reasoning`: the
conversation then matches the cached state token for token, and a new turn only
processes its new tokens (`usage.prompt_tokens_details.cached_tokens`).

## Tools

Function tools are served as vLLM serves them with a tool-call parser, by a
package that declares `tool_calls`: its template renders `tools`, its
`tool_calls` verb reads the calls back, and they leave as `tool_calls` with
`finish_reason` `"tool_calls"`. A stream carries a call as vLLM's does: a
delta with its id and name (arguments `""`), then deltas of `{"index",
"function": {"arguments"}}` as the package reads them (per parameter, for a
package that lists calls still being written);
`parallel_tool_calls: false` keeps the first. `tool_choice` `"required"` or a
named function needs constrained decoding, which the engine lacks: served as
`"auto"` and reported in the log.

## Images

`serve --vision` turns image input on, for a package that declares `vision`
(refused at start-up otherwise). Chat messages then read `image_url` parts
whose url is a data URL with base64 bytes (`data:image/png;base64,...`); any
other url is a 400 (URLs are not fetched). Without `--vision`, image parts are
dropped and reported, like any part the engine cannot execute. A prompt with
images is not cut by `truncate_prompt_tokens`, and a cached state is reused
only for the same images (by their bytes' digest).

## Decisions

`/v1/decisions` and `/v1/systemone` score typed questions (`choice`, `score`,
`yes_no`) about an input without generating text. Each question is one user
message rendered with thinking off; its answer is the probability of every
option, read from the next-token scores at the answer position
(`label_logprobs`). The questions of one request are scored in one
`label_logprobs_batch` call, from the prefix cache's state of the head they
share, so new questions about the same input skip re-reading it.

A package that does not declare `label_logprobs` is refused with the reason.
Requests wrong of themselves fail — 400, or 422 with the System One `detail`
list on `/v1/systemone` — and nothing else does.

## Logprobs and speculation

`logprobs` against a package that declares no `logprobs` capability come back
in shape but empty, as vLLM reports step data it did not produce.

`serve --speculative` (alias `--mtp`) asks the package to draft ahead and
verify its own draft for every request. The engine does not know how: it asks
through the optional `set_speculative` verb, and a package that does not declare
`speculative` is refused at start-up (rule 4). Speculation must not change one
token of any answer.
