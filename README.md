# footless

An inference engine that knows nothing about models. Everything model-specific
— the architecture, the kernels, the cache layout, the weights, the tokenizer —
lives in the model's own folder.

**One sentence:** a new architecture is a folder, not a change to the engine.

```
┌─────────────────────────────────────────────┐
│  footless — the inference engine            │
│  requests · scheduling · memory budget      │
│  prefix reuse · sampling policy · serving   │
└──────────────────┬──────────────────────────┘
                   │  the contract, both directions
┌──────────────────▼──────────────────────────┐
│  model packages — one folder per model      │
│  manifest + per-hardware runtime and        │
│  kernels                                    │
└─────────────────────────────────────────────┘
```

This repository is the engine, its contract, its tests and the tools to build
and check model packages. It ships no model: packages live next to their
weights and are written against `footless.sdk` (see
[docs/model-package.md](docs/model-package.md)).

## The two parts

**The engine** owns policy and bookkeeping: the request lifecycle, scheduling
and batching, the memory budget in bytes, prefix reuse, the sampling policy,
the serving API and the command line. It never knows an architecture, a
kernel, a tensor type, a cache layout or what a device does: it moves opaque
state, counts its bytes and asks the model to use it.

**A model package** owns execution and representation: the forward pass and
its kernels, the cache and how state is addressed, the weights and their
transforms, the tokenizer and template, and every per-hardware choice. A
package may target several devices, one folder each.

They meet at one narrow interface, normative in [CONTRACT.md](CONTRACT.md): a
manifest read before any code runs, a runtime the engine drives (`step` over a
batch, state verbs, text verbs), and services the runtime may use (byte
accounting, buffers, logging).

## Quick start

Python 3.11 or newer; the engine uses the standard library only (a package's
runtime brings its own requirements). The mock packages in `tests/` are real
packages with a scripted "model", enough to try everything:

```bash
python3 -m footless generate --model tests/mockup --prompt "hello"
python3 -m footless chat --model tests/mockup-thinking
python3 -m footless serve --model tests/mockup --port 8000
python3 -m unittest discover -s tests -t .
```

`pip install -e .` adds a `footless` command for the same thing.

## Command line

| Command | What it does |
| --- | --- |
| `footless generate --model DIR --prompt TEXT` | one completion per `--prompt` |
| `footless chat --model DIR` | an interactive, streamed chat |
| `footless bench --model DIR` | prefill and decode rates over a context sweep, as CSV |
| `footless serve --model DIR` | the OpenAI-compatible HTTP API ([docs/serving.md](docs/serving.md)) |

All take `--backend` (the hardware folder, when a package has several) and
`--budget` (bytes for states and the prefix cache). `generate` and `chat` take
the sampling knobs a request may pin (`--temperature`, `--top-p`, `--top-k`,
`--min-p`, the three penalties, `--seed`); what is left unset takes the
package's own generation config. They end a generation whose tail repeats one
span verbatim past all use (`--loop-guard exact`, the default; `--loop-guard`
alone also ends near-copies; `--no-loop-guard` lets it run). `chat` takes
`--no-thinking` and `--reasoning LEVEL`; `generate`, `chat` and `serve` take
`--speculative` (alias `--mtp`) for packages that declare it.

## The rules

These are the invariants of the design. A change that breaks one is a design
error.

1. **The engine never branches on the model.** No architecture names, no model
   checks. A new need means a new manifest field or contract method.
2. **Cached state is opaque to the engine.** It moves it, counts its bytes, and
   asks the model to reuse it. It never looks inside.
3. **The manifest is the single source of truth before load.** It is verified
   against the code; a contradiction is a hard error.
4. **No silent degradation.** A capability the model does not declare is
   unavailable — and reported as such. Nothing falls back quietly.
5. **Hardware specialization lives in the model's hardware folders.** The
   engine only selects a folder. It never behaves differently per device.
6. **Weights and tokenizers stay where they already are.** A `footless/` folder
   is added to an existing model directory; nothing is moved.

## Layout

```
footless/        the engine, CLI, serving API and SDK (footless.sdk)
tests/           engine tests and three mock packages
templates/       model-package/: the starting point for a new package
tools/           harnesses for packages and servers (below)
docs/            guides (below)
models/          local workspace for model packages (not tracked)
```

## Tools

| Tool | What it checks |
| --- | --- |
| `tools/check_package.py --model DIR` | that a package keeps the contract, verb by verb |
| `tools/serve_check.py --model DIR` | starts a server; completions, decisions and System One, shape and speed |
| `tools/serve_chat_check.py --model DIR` | the same chat through the engine, plain HTTP and SSE |
| `tools/serve_profile.py --model DIR` | where a served request's time goes, engine against HTTP |
| `tools/serve_concurrency.py --url URL` | a running server at 1..N requests at once: rates, and identical answers |
| `tools/demo_speed.sh --url URL` | a live terminal demo of prefill and decode at 1..N requests |

## Documentation

- [CONTRACT.md](CONTRACT.md) — the interface between engine and package (normative).
- [docs/model-package.md](docs/model-package.md) — writing a package, verb by verb.
- [docs/roadmap.md](docs/roadmap.md) — bringing a new model up, phase by phase, with gates.
- [docs/runtime-engineering.md](docs/runtime-engineering.md) — what building runtimes has taught.
- [docs/serving.md](docs/serving.md) — the HTTP API.
- [AGENTS.md](AGENTS.md) — how to work in this repository (for people and agents).

## Not built yet

Multimodal input: today a request is text, and the contract carries token ids
only. It is not ruled out. When a model needs images or audio, they will reach
the package through a new contract surface, with the encoders living in the
package like the rest of the model.

## Non-goals

Adapters/LoRA, disaggregated prefill/decode fleets, and a catalog of many
architectures. The claim is the decoupling, not covering every model ever made.
