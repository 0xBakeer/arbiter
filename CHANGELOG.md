# Changelog

What a version means here: this repository is not a library, and nothing imports it. What you
depend on is **the defaults it ships and the measurements taken on them**. A release is
therefore a measurement epoch — the configuration as it stood, and the figures that belong to it.

- **MAJOR** — the measurement basis changes: different hardware, different checkpoints.
- **MINOR** — a shipped default changes, or the server gains a capability. Your numbers move.
- **PATCH** — documentation, corrections, tooling. Your numbers do not move.

Every entry leads with **Defaults that changed**, because that is the part that alters what you
would measure if you ran the recipe yourself. `./run.sh` prints the version it was launched from.

## Unreleased

A third engine, opt-in, on a lane that already had two dtypes and one backend — and six games the
model plays in the browser, served from the same origin as the API.

### Defaults that changed

None. `ARBITER_ENGINE` still defaults to `laya`, torch is still what a Mac serves unless it is
told otherwise, and every figure in [bench/results.md](bench/results.md) taken before this still
belongs to the path it was taken on. The showcase is static files and a read-only mount; it adds
no state to the server and holds no engine open.

### Added

- **The showcase, `/showcase/`** — six games (snake, hopper, crossing, paddle, mines, a dungeon
  crawl) that the model plays live. Each tick encodes the board as text, asks typed questions
  about it and plays the answer out of a single `POST /v1/systemone`; nothing is generated. One
  shared harness in [`showcase/_lib/harness.mjs`](showcase/_lib/harness.mjs) runs both the browser
  pages and `measure.mjs`, so the page and the measurement play the same game. A code-owned shield
  may veto an action that ends the episode, and every intervention is counted.
- **Measured scores for all six**, 20 episodes per policy, model against a random and a
  hand-written baseline on the same seeds, plus AUROC against ground truth where the game has it.
  `node showcase/measure.mjs --game all` regenerates `<game>/measurements.json`, the README tables
  and the replay files. Every figure came from a run against a live server; none from a mock.
- **Replays in `showcase/replay/`** — the best model episode per game, so a page opened from disk
  with no server still plays.
- **`GET /showcase/...`** — a read-only static mount in [`server/app.py`](server/app.py), path
  traversal rejected, kept out of the OpenAPI schema like the playground.

- **`laya_mlx` engine (Apple Silicon, opt-in)** — [`engines/laya_mlx/loader.py`](engines/laya_mlx/loader.py)
  serves the same three checkpoints through [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx),
  an independent MLX reimplementation of the ModernBERT encoder and Laya's decision heads, over
  the converted `aac6fef/laya-*-mlx` weights. Same engine interface, same rows, same padded
  batching from the same micro-batcher, same routing — `laya_mlx.router` is upstream's router and
  decides 7/7 equivalence cases identically. Selected with `ARBITER_ENGINE=laya_mlx`; on any
  machine that is not Apple silicon it refuses at startup with one line instead of an ImportError
  out of a wheel that does not exist there.
- **`ARBITER_ENGINE=laya_mlx ./run.sh setup`** — adds `laya-mlx` and `mlx` to the same `.venv`
  (nothing it brings conflicts with torch; `pip check` stays clean) and downloads the three
  converted checkpoints into `models/laya-mlx/`. It remains a superset of the torch setup,
  because the equivalence reference is `laya.Agent` under torch and `ARBITER_ENGINE=laya` has to
  keep working on the same checkout.
- **[`tools/equivalence_mlx.py`](tools/equivalence_mlx.py)** — the gate for the new engine, on the
  same 22 questions and against the same reference, plus a routing comparison. Wired into
  `ARBITER_ENGINE=laya_mlx ./run.sh equivalence`. **fp32 ships** because it is the only one that
  passes: max |Δp| **0.00e+00** across all three checkpoints, where fp16 is 1.69e-02 and bf16
  moves an argmax.
- **`ARBITER_MLX_CACHE_MB`, default 1024** — MLX keeps freed buffers for reuse without a bound,
  which on a server that forms a new shape per batch reached a 23 GB footprint and *cost*
  throughput as callers were added (39 q/s at eight, against 103 with the bound). `0` restores
  MLX's own behaviour.
- **`/readyz` now reports `engine`**, and `bench/bench.py` labels its section with
  `ARBITER_ENGINE`, so a measurement says which backend produced it.

### Measured

One M2 Max, 2026-09-20, with the torch server resident on the same GPU and a paired control taken
minutes later. Against that control: **22.7 ms** for one question (30.5), 59.2 / 102.7 / 448.4 ms
for 5 / 10 / 50 (62.8 / 107.1 / 545.9), and 75 / 103 / 109 q/s at 1 / 8 / 32 callers, which is a
tie. The difference worth switching for is the cold load: **2.1 s to `/readyz`** against 88-101 s.
Full tables, the control and the cache evidence in [bench/results.md](bench/results.md).

### Documentation

- [recipes/apple](recipes/apple/README.md) gains **Option 3: the MLX engine** — install, defaults,
  the measured tables, the equivalence numbers and what the lane still owes. The "what an MLX port
  could buy" section that predicted this is replaced by what it actually bought.
- [CREDITS.md](CREDITS.md) credits the port and the converted checkpoints;
  [engines/README.md](engines/README.md) now says which packages are here and what each runs on.
- [showcase/README.md](showcase/README.md) — the six games with their scores side by side, how the
  harness works, how to measure and how to add a game, and an honest section about what the
  numbers do and do not say. A README per game with its own tables. Linked from the main README
  under a new **arbiter plays** section.

## v0.1.1 — 2026-09-20

Documentation only. The recordings that were made for the write-up now live in the repository,
re-encoded small enough to sit in it, and the article that the project came out of is linked from
the README.

### Defaults that changed

None. Nothing in `server/`, `engines/` or the recipes was touched, so every figure in
[bench/results.md](bench/results.md) still belongs to this version.

### Documentation

- **The article** — [My cat woke me at five on a Sunday, so I built a local
  Jev](https://blog.0xbakeer.com/my-cat-woke-me-at-five-on-a-sunday-so-i-built-a-local-jev-1350583cddc6?sharedUserId=0xbakeer),
  linked under the tagline and from the documentation list.
- **Four GIFs in `docs/media/`** — how it works, the playground, the Claude Code guard, and the
  install on a Mac. 960 px wide, re-encoded from the source captures to 10.2 MB in total so that
  a clone stays cheap.
- **A poster PNG per GIF**, 1280 px wide, for the places where a still frame reads better than a
  loop.
- `docs/media` is in `.dockerignore`, so the recordings do not ride along into the image.

## v0.1.0 — 2026-09-20

First release. Renamed from laya-spark to arbiter before it: the repository is the serving
layer, so it is named after neither the model it serves nor the machine it runs on. Everything
that names the product is `arbiter` (the environment variables are `ARBITER_*`, with no aliases
for the old names, since nothing public depended on them) and everything that names the model
is still `laya` — the checkpoints, the SDK, and the `laya-english` / `laya-multilingual` /
`laya-typed-decisions` model ids a client sends.

The layout follows from the same idea. `server/` is model-agnostic: the HTTP surface, the
cross-request batcher and a four-method engine interface. `engines/laya/` implements that
interface over the SDK and `ARBITER_ENGINE` selects it, so a model trained here later plugs in
without touching the server. `recipes/nvidia/` holds the install-and-serve notes for CUDA and
`recipes/apple/` is the Apple Silicon lane on torch's MPS backend; `bench/results.md` has a
section per machine so the two cannot be confused for each other. Both are measured.

### Defaults that changed

Nothing to change from, but two defaults were *chosen against a faster alternative*, and both
choices are the kind that a later version might reverse. They are recorded here so that a
reversal is visible as a reversal.

| Setting | Shipped | The alternative | Why not the alternative |
|---|---|---|---|
| `ARBITER_DTYPE` | `autocast` | `bf16` parameters, 15–35% faster | moves reported probabilities by up to 2.1e-2 and flips one argmax out of 22 |
| `ARBITER_MODE` | `eager` | `graphs`, 20% faster at one question | 1.5× slower at 50 questions and 14% lower throughput at concurrency 8 |
| `ARBITER_DTYPE` on MPS | `fp32` | `fp16` or `bf16` parameters | 1.26e-2 and 6.06e-2 of probability error against the SDK reference; neither was benchmarked, because a dtype outside the gate does not get a speed number |

There is also one variable that was renamed on the way in and has no old name to be compatible
with, since nothing public depended on it: `DEVICE` is now `ARBITER_DEVICE`, and its default is
no longer `cuda` but the best device present — cuda, else mps, else cpu.

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
  pool, and an eager fallback for anything outside a bucket. On a non-CUDA device it refuses
  rather than quietly running eager.
- **An Apple Silicon lane** (`recipes/apple/`), measured on an M2 Max: torch's MPS wheels from
  plain PyPI, the device detected through `ARBITER_DEVICE`, `supported_dtypes()` deciding per
  device which parameter modes are worth offering, and `fp16`/`bf16` as whole-model dtypes where
  autocast does not apply. `/readyz` now reports the device and the dtype that were actually
  taken, not the ones that were asked for.
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
  and any generic `.mcp.json` client, plus a GitHub Actions job that gates pull requests. The
  hook, the MCP gate and the `examples/tool_call_guard.py` example share one
  `guard_policy.py` — the same questions, the same risk arithmetic and the same thresholds — so
  the three cannot drift apart, and `integrations/claude-code/hooks/eval.py` scores that policy
  against a labelled set.
- **A playground.** `GET /` serves `playground/index.html`, one self-contained page on the same
  origin as the API: a state, questions of all three types, and the answers with their
  probabilities, the checkpoint that answered and the latency. `playground/serve_stub.py` serves
  the same page with stub answers for UI work without a GPU.
- **`./run.sh`** — `setup`, `serve`, `stop`, `status`, `smoke`, `examples`, `bench`,
  `equivalence`, and `test` over both suites.

### Measured

- 20.9 ms for one question, 152.4 ms for fifty in one call on a GB10, against the model card's
  T4 figures of 39.5 ms and 771 ms. Full method in [bench/results.md](bench/results.md).
- 30.3 ms and 462.3 ms for the same two calls on an Apple M2 Max in fp32, with the throughput
  ceiling at eight concurrent callers (106 questions/s) and a cold load of 88-101 s. On that
  machine `ps -o rss` is meaningless — it read 1.6 GB and 230 MB for the same unchanged server —
  because the weights live in unified-memory buffers outside the resident set; `footprint -p`
  reports them, stable at 4,922 MB.
- Next to an LLM decoding at 92 % GPU utilisation on the same GB10, the same calls take 326.6 ms
  and 381.6 ms, and throughput roughly halves — 88 questions/s at eight callers against 287 with
  the card idle, with no errors. Both sets of tables are in [bench/results.md](bench/results.md), labelled.
- The Bash guard was scored on 117 labelled real `PreToolUse` events against the live server:
  100% of the allow class decided allow, 0% of the deny class ever allowed, 90.6% of it decided
  deny, and none of the ask class decided allow. Six events miss, and all six are named in
  `integrations/claude-code/hooks/README.md` rather than averaged away.
- Graphs mode was 306 ms on a 50-question call before the bucket ladders were made fine-grained
  and 229 ms after. The marker dimension, which was the obvious suspect, accounted for 0.1 ms of
  that; the sequence ladder accounted for the rest.

### Known limits

- On MPS, `fp16` and `bf16` parameters miss the equivalence gate by 1.26e-2 and 6.06e-2, so the
  Apple lane ships fp32 and there is no faster dtype to reach for there. CUDA graphs have no
  equivalent on that backend either.
- A `choice` question with more than about 125 options cannot fit its markers in the English
  checkpoint's 512-token context. The server answers `422` rather than silently answering a
  truncated question. Jev's own limit is 255.
- The model's own limits — near-chance zero-shot typed decisions, weak ordinal scores, and
  probabilities that ship over-confident — are quoted in the README and are not something a
  serving layer can fix.
