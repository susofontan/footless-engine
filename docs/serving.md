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
that is decoding, and at most one prefill chunk — in chunks of 1024 tokens while
others decode, so their tokens wait for one chunk at a time. How the rows of
one step are computed is the package's business; each item's result must be
the one it would get alone (CONTRACT.md). A step the budget refuses runs again
without its youngest request: that one waits in the prefix cache and is
admitted again later. The choices of one request (`n`) run one after another,
and decision requests run between steps. A client that disconnects cancels its
request at the next step boundary.

A package with `max_batch = 1` is served one request at a time, by the same
code.

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
