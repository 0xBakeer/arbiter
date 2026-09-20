# Changelog

What a version means here: this repository is not a library, and nothing imports it. What you
depend on is **the defaults it ships and the measurements taken on them**. A release is
therefore a measurement epoch — the configuration as it stood, and the figures that belong to it.

- **MAJOR** — the measurement basis changes: different hardware, different checkpoints.
- **MINOR** — a shipped default changes, or the server gains a capability. Your numbers move.
- **PATCH** — documentation, corrections, tooling. Your numbers do not move.

Every entry leads with **Defaults that changed**, because that is the part that alters what you
would measure if you ran the recipe yourself. `./run.sh` prints the version it was launched from.

## v0.1.0 — 2026-09-20

First release.

### Defaults that changed

Nothing to change from, but two defaults were *chosen against a faster alternative*, and both
choices are the kind that a later version might reverse. They are recorded here so that a
reversal is visible as a reversal.

| Setting | Shipped | The alternative | Why not the alternative |
|---|---|---|---|
| `ARBITER_DTYPE` | `autocast` | `bf16` parameters, 15–35% faster | moves reported probabilities by up to 2.1e-2 and flips one argmax out of 22 |
| `ARBITER_MODE` | `eager` | `graphs`, 20% faster at one question | 1.5× slower at 50 questions and 14% lower throughput at concurrency 8 |

### Added

- **A Jev-compatible System One server.** `POST /v1/systemone` with TypeSafe's request and
  response shape, so a Jev client moves over by changing the base URL. Plus `/healthz`,
  `/readyz`, `/v1/models` and a Prometheus `/metrics`.
- **All three checkpoints resident, with automatic routing.** English, multilingual and
  typed-decisions load once and stay; the router picks by script and language in microseconds.
  `typed-decisions` is never selected automatically. 5,055 MiB of GPU memory for all three.
- **Cross-request micro-batching.** A worker thread per checkpoint collects question rows from
  concurrent requests into one forward. 135 questions/s at one caller becomes 287 at eight.
- **CUDA-graph mode** (`ARBITER_MODE=graphs`), with bucketed shapes, lazy capture, a shared memory
  pool, and an eager fallback for anything outside a bucket.
- **`tools/equivalence.py`**, which is the reason the defaults are what they are: 22 mixed
  questions through the SDK reference and every dtype/mode combination, reporting max |Δp| and
  argmax agreement. `autocast/eager` comes out identical to the reference at four decimals.
- **Nine runnable examples** in `examples/` — support triage, inbox triage, a tool-call guard,
  a PR risk gate, alert triage, model routing, RAG relevance, moderation and invoice fields —
  over `examples/arbiter_client.py`, a single dependency-free client whose API mirrors the hosted
  SDK. Every question set is data, every threshold is in the caller, and the captured output of
  each one is in [docs/use-cases.md](docs/use-cases.md).
- **Integrations for coding agents.** `integrations/mcp/arbiter_mcp.py` exposes `arbiter_check`,
  `arbiter_classify`, `arbiter_score`, `arbiter_gate` and `arbiter_decide` over MCP; the Claude Code plugin
  adds a skill and a `PreToolUse` hook that judges every `Bash` command in tens of milliseconds
  and fails open when the server is not there. Ready-made configuration for Codex, OpenCode, omp
  and any generic `.mcp.json` client, plus a GitHub Actions job that gates pull requests.
- **A playground.** `GET /` serves `playground/index.html`, one self-contained page on the same
  origin as the API: a state, questions of all three types, and the answers with their
  probabilities, the checkpoint that answered and the latency. `playground/serve_stub.py` serves
  the same page with stub answers for UI work without a GPU.
- **`./run.sh`** — `setup`, `serve`, `stop`, `status`, `smoke`, `examples`, `bench`,
  `equivalence`, and `test` over both suites.

### Measured

- 20.9 ms for one question, 152.4 ms for fifty in one call, against the model card's T4 figures
  of 39.5 ms and 771 ms. Full method in [bench/results.md](bench/results.md).
- Graphs mode was 306 ms on a 50-question call before the bucket ladders were made fine-grained
  and 229 ms after. The marker dimension, which was the obvious suspect, accounted for 0.1 ms of
  that; the sequence ladder accounted for the rest.

### Known limits

- A `choice` question with more than about 125 options cannot fit its markers in the English
  checkpoint's 512-token context. The server answers `422` rather than silently answering a
  truncated question. Jev's own limit is 255.
- The model's own limits — near-chance zero-shot typed decisions, weak ordinal scores, and
  probabilities that ship over-confident — are quoted in the README and are not something a
  serving layer can fix.
