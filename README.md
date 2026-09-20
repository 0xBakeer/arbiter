# arbiter

Serve typed-decision models — Laya or your own — on your GPU or your Mac, with a Jev-compatible
API.

## Why the name, and what this is

A typed-decision model answers questions about a piece of text in **one forward pass** — no
tokens generated, no sampling, no loop. You hand it a state and a set of questions, and it
returns an answer and a probability distribution for each of them at once. There is no prose to
read back and nothing to argue with: it arbitrates, and your code decides what to do with the
numbers.

This repository is the serving layer around that and deliberately nothing more — a small HTTP
server that speaks TypeSafe's **Jev** API, so a client written against Jev works against this by
changing the base URL; cross-request micro-batching; routing between checkpoints; a playground;
and the measurements that picked every default. The model is a plug: **Laya** today, with its
three checkpoints (English, multilingual, and a typed-decisions fine-tune) resident at once and
automatic routing between them, and whatever is trained here next behind the same interface
([engines/README.md](engines/README.md)). So is the machine: one recipe per accelerator under
[recipes/](recipes), NVIDIA measured and Apple next.

On one GB10 it answers a single question in **20.9 ms** and fifty questions in one call in
**152 ms**, measured end-to-end over HTTP. Realistic states — a support ticket, an email, a
diff — carry more tokens and more questions, and land in the **tens of milliseconds**; the
captured runs are in [docs/use-cases.md](docs/use-cases.md). It uses about 5 GB of GPU memory,
which is little enough to sit next to a large language model on the same card.

![One call, many typed answers](docs/assets/one-call-typed-answers.svg)

---

## Install

You need an NVIDIA GPU with a recent driver, and Python 3.12 or newer. The torch wheels come
from the CUDA 13.0 index, which has both aarch64 and x86_64 builds, so this is not specific to
any one box.

```bash
git clone https://github.com/0xBakeer/arbiter.git
cd arbiter
./run.sh setup      # .venv, torch, the deps, and the three checkpoints (2.3 GB)
./run.sh serve      # http://localhost:8010
```

`setup` prefers an interpreter that ships its development headers, because torch routes a couple
of eager operations through Triton and Triton JIT-compiles a small C extension the first time
one runs. If no such interpreter is available it falls back and sets `TORCH_DISABLE_NATIVE_JIT=1`,
which takes torch's plain kernels instead; it prints a line when it does.

```bash
./run.sh status     # /healthz, /readyz, /v1/models
./run.sh smoke      # the playground page and seven real requests across all three checkpoints
./run.sh examples   # a support ticket through examples/support_triage.py
./run.sh bench      # the numbers below, about ninety seconds
./run.sh test       # both pytest suites; no GPU and no running server needed
./run.sh stop
```

## Playground

`./run.sh serve` also serves a page at `http://localhost:8010/`: paste a state, add questions of
all three types, and read the answers with their probabilities, the checkpoint that answered and
the latency. It is one self-contained file, [`playground/index.html`](playground/index.html),
served from the same origin as the API, so it needs no configuration and the server needs no
CORS headers. To work on the page without a GPU, `python3 playground/serve_stub.py` serves it
with stub answers on port 8011, and `--proxy http://localhost:8010` forwards the calls to a real
server while keeping the page same-origin.

## Ask it something

```bash
curl -s localhost:8010/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for the same subscription this month and the second charge has not been refunded. I have emailed support twice with no reply.",
  "questions": {
    "urgency":     {"type": "score",  "instructions": "How urgent is this message?",
                    "criteria": ["not urgent", "can wait a day", "same day", "immediate"]},
    "category":    {"type": "choice", "instructions": "Which queue should this go to?",
                    "criteria": {"billing": "payments, invoices, refunds",
                                 "technical": "the product does not work",
                                 "account": "login, profile, permissions"}},
    "needs_human": {"type": "noul",   "instructions": "This message needs a human rather than an automated reply."}
  }
}'
```

```json
{
  "model": "laya-english",
  "answers": {
    "urgency":     {"type": "score",  "score": 2.12, "legend": {"0": "not urgent", "...": "..."},
                    "probabilities": {"0": 0.0121, "1": 0.1755, "2": 0.4913, "3": 0.3211},
                    "confidence": 0.18},
    "category":    {"type": "choice", "choice": "billing",
                    "probabilities": {"billing": 0.9841, "technical": 0.0092, "account": 0.0067},
                    "confidence": 0.93},
    "needs_human": {"type": "noul",   "noul": 0.002, "confidence": 0.998}
  },
  "usage": {"input_tokens": 210, "output_tokens": 0},
  "routing": {"model": "english", "reason": "English Latin text"},
  "latency_ms": 15.2
}
```

`routing`, `latency_ms`, `confidence` and `action` are additions; everything else is the Jev
response shape, and a Jev client ignores extra keys.

### The API

| endpoint | what it does |
|---|---|
| `POST /v1/systemone` | the Jev contract. `POST /v1/predict` is an alias |
| `GET /healthz` | the process is up |
| `GET /readyz` | the checkpoints are loaded and warm — this is the one to gate traffic on |
| `GET /v1/models` | the three checkpoints and their aliases, OpenAI-style |
| `GET /metrics` | Prometheus text: requests, questions, latency, batch size, queue depth |

Question types are Jev's: `noul` (a probability that a statement holds, criteria optional),
`choice` (up to 255 named options), `score` (2 to 10 ordered levels). Violations come back as
`422` with a JSON error body. Set `ARBITER_API_KEY` and the server requires
`Authorization: Bearer <key>`, answering `401` otherwise. More question rows in flight than
`ARBITER_MAX_QUEUE` gets `529`.

### Routing

The `model` field decides which checkpoint answers.

| you send | what runs |
|---|---|
| `"jev-latest"`, `"laya"`, `"auto"`, or nothing | the router picks |
| `"laya-english"` / `"english"` / `"en"` | English — ModernBERT-large, 512 tokens |
| `"laya-multilingual"` / `"multilingual"` / `"multi"` | multilingual — mmBERT-base, 1024 tokens, 100+ languages |
| `"laya-typed-decisions"` / `"typed-decisions"` / `"typed"` | the typed-decisions fine-tune, 1024 tokens |

Left to itself the router reads the state's script: anything non-Latin goes to multilingual,
Latin text that does not look English goes to multilingual, the rest goes to English. This is
not a nicety. The English checkpoint does not degrade gently off English, it collapses — 0.100
on 20-option Hindi intent classification against 0.050 for guessing — and it stays confident
while doing it. Detection costs microseconds, so the routing is free.

`typed-decisions` is never selected automatically. Ask for it by name.

## Examples

Nine runnable scripts in [`examples/`](examples), each one a real decision with the thresholds
in the caller and a review band in the middle. They need nothing but a Python 3 and a running
server: [`examples/arbiter_client.py`](examples/arbiter_client.py) is a single dependency-free file
whose API mirrors the hosted SDK, so code written against Jev ports by changing the import and
the base URL.

| Use case | Script | What it decides | Route |
|---|---|---|---|
| Support tickets | `support_triage.py` | department, urgency, frustration, refund, churn | `auto` · `review` · `escalate` |
| Inbox triage | `email_triage.py` | category, phishing, action needed, reply-by | `auto` · `review` · `escalate` · `block` |
| Agent shell commands | `tool_call_guard.py` | destructive, secrets, leaves repo, network, blast radius | `allow` · `ask` · `deny` |
| Pull requests | `pr_risk_gate.py` | credentials, migration, shared infra, rollback, blast radius | `allow` · `review` · `block` |
| Monitoring alerts | `alert_triage.py` | service, root cause, severity, duplicate of an open incident | `suppress` · `ticket` · `page` |
| Model selection | `model_router.py` | complexity, needs tools, needs long context | `fast` · `cascade` · `powerful` |
| RAG passages | `rag_relevance.py` | per-passage relevance, hierarchical past 12 | `keep` · `?` · `drop` |
| Message screening | `moderation.py` | jailbreak, harmful, PII, off-topic, severity | `allow` · `review` · `block` |
| Invoices (typed-decisions) | `invoice_fields.py` | currency, amount band, duplicate, approval, bank change | `pay` · `approve` · `review` |

```bash
python examples/support_triage.py                 # a built-in sample
python examples/email_triage.py --sample de       # watch the router pick multilingual
git diff main | python examples/pr_risk_gate.py   # exit code is the gate: 0/1/2
python examples/moderation.py --state msg.txt --json
```

Each script prints one row per question with a probability bar, the checkpoint that answered and
the latency, then the route its own thresholds chose. Captured output for every one of them is
in [`docs/use-cases.md`](docs/use-cases.md); the shapes they are built from — fan-out,
confidence-gated routing, composite scoring, hierarchical intent, cascade — are in
[`docs/patterns.md`](docs/patterns.md).

## Use it from your coding agent

[`integrations/mcp/arbiter_mcp.py`](integrations/mcp/arbiter_mcp.py) is an MCP server over the same
endpoint: `arbiter_check`, `arbiter_classify`, `arbiter_score`, `arbiter_gate` and `arbiter_decide`, each
returning structured content plus one line of text. Point any MCP client at it:

```bash
pip install "mcp>=2"
claude mcp add arbiter --env ARBITER_URL=http://localhost:8010 -- python3 $PWD/integrations/mcp/arbiter_mcp.py
codex  mcp add arbiter --env ARBITER_URL=http://localhost:8010 -- python3 $PWD/integrations/mcp/arbiter_mcp.py
```

[`integrations/claude-code/`](integrations/claude-code) is a plugin that adds a skill (when to
use which primitive, how to shape state and questions, why thresholds belong in your code) and a
`PreToolUse` hook that judges every `Bash` command before it runs — allow, ask or deny, in tens
of milliseconds, failing open when the server is not there. Ready-made configuration for Codex,
OpenCode, omp and any generic `.mcp.json` client, plus a GitHub Actions job that gates pull
requests, is in [`integrations/`](integrations) and documented in
[`docs/integrations.md`](docs/integrations.md).

## The numbers

One GB10, questions cycling through all three types, measured end-to-end over HTTP.
Full method and the rest of the figures in [bench/results.md](bench/results.md).

| questions in one call | this recipe | model card, T4 english | model card, T4 multilingual |
|---:|---:|---:|---:|
| 1  | **20.9 ms** | 39.5 ms | 32.8 ms |
| 5  | **31.1 ms** | — | — |
| 10 | **40.0 ms** | 158.6 ms | 72.3 ms |
| 50 | **152.4 ms** | 771 ms | 337 ms |

Throughput, 4-question calls: **135 questions/s** at one caller, **287/s** at eight, **301/s**
at thirty-two. The T4 figures are in-process SDK calls on older hardware; they are here for
scale, not as a ranking.

### The two settings that are worth understanding

**`ARBITER_DTYPE`** — `autocast` (default) keeps fp32 parameters and runs the matmuls in bf16, as
the SDK does. `bf16` converts the parameters and drops autocast, which is 15–35% faster:
14.0 ms instead of 20.9 at one question, 380 questions/s instead of 287 at concurrency 8.

It is not the default because of what it costs. Against the SDK as reference, `autocast` is
identical to four decimals on all 22 equivalence questions — batching rows from different
callers into one forward changes nothing whatsoever. `bf16` moves reported probabilities by up
to **2.1e-2** and flips one argmax, on a five-level score question whose own confidence was
0.157. Laya's proposition is calibrated probabilities, and the model card is already explicit
that they ship over-confident and want refitting on your data before you trust them; spending
2e-2 of that budget to save six milliseconds is the wrong trade. It is one environment variable
away if your workload disagrees.

**`ARBITER_MODE`** — `eager` (default) or `graphs`. Graphs mode captures the forward as a CUDA
graph per padded shape bucket and replays it, which removes the cost of launching several
hundred small kernels. It is faster where there is little work to amortise padding over and
slower where there is plenty:

| | 1 question | 5 | 10 | 50 | 8 concurrent |
|---|---:|---:|---:|---:|---:|
| eager | 20.9 ms | 31.1 ms | **40.0 ms** | **152.4 ms** | **287 q/s** |
| graphs | **16.8 ms** | **29.0 ms** | 44.6 ms | 229.4 ms | 248 q/s |

Eager is the default because it wins every shape above five questions and every concurrency
above one. Turn graphs on for a single-caller, small-call deployment — a guardrail in front of a
chat endpoint, say — where it is about 20% better.

Graphs mode also carries a 1.4e-3 deviation from the reference where eager carries none, which
is inside the gate but not nothing.

### Everything you can set

| variable | default | |
|---|---|---|
| `PORT` / `HOST` | `8010` / `0.0.0.0` | |
| `DEVICE` | `cuda` | `cpu` works, and is roughly an order of magnitude slower |
| `ARBITER_MODELS` | all three | comma-separated; a subset saves memory, and the router falls back to what is loaded |
| `ARBITER_MODE` | `eager` | or `graphs` |
| `ARBITER_DTYPE` | `autocast` | or `bf16` |
| `ARBITER_API_KEY` | unset | set it to require `Authorization: Bearer` |
| `ARBITER_BATCH_WAIT_MS` | `2` | how long a batch waits for company |
| `ARBITER_MAX_BATCH` | `64` | question rows per forward |
| `ARBITER_MAX_QUEUE` | `256` | rows in flight before `529` |
| `ARBITER_GRAPH_MAX_MARKERS` | `32` | questions with more options than this take the eager path |
| `MODELS_DIR` | `./models` | |

## Runs next to an LLM

![The cascade next to an LLM](docs/assets/cascade-next-to-an-llm.svg)

Measured with an unrelated LLM already holding 63,871 MiB on the same GPU:

| | |
|---|---|
| Laya, all three checkpoints, `autocast` | **5,055 MiB** of GPU memory |
| the same in `ARBITER_DTYPE=bf16` | about 2,900 MiB |
| host RSS | 3.6 GB |
| host memory before / after starting it | 74 GiB / 82 GiB used of 121 |

The LLM was serving throughout and was unaffected. Three checkpoints is 1.16B parameters, which
is small enough that the decision is about whether you want all three rather than about whether
they fit; `ARBITER_MODELS=english` alone is about 1.9 GB.

The reason all three stay resident is that the SDK's `Router` defaults to keeping one. A server
that alternates between English and German requests would then reload a checkpoint from disk on
every request — seconds, against microseconds to decide. Preloading is the only sensible server
setting, and the router is handed the already-built models so nothing is loaded twice.

## Honest limits

These are the model's, not the server's, and they are quoted from the model card because they
matter more than anything this recipe does.

- **The base checkpoints are near chance on typed decisions, zero-shot** — 0.362 for English and
  0.352 for multilingual, against 0.318 for random and 0.461 for always guessing the majority
  class. The 0.766 figure belongs to the checkpoint fine-tuned on that benchmark's own training
  split. Laya is a fast base to specialise, not a zero-shot decision engine.
- **High-cardinality choice questions run out of tokens.** A sequence is split into an option
  budget (`head_max_len`) and what is left for the state. English gets 512 total with 192 for
  options; multilingual and typed-decisions get 1024 with 256. A 77-option question like
  Banking77 therefore gets about 3–4 tokens per label and accuracy collapses — 0.425, against
  0.870 for Jev. Above about 125 options the English checkpoint cannot fit the markers at all,
  and this server returns `422` rather than silently answering a truncated question. If you need
  50+ options, either raise `head_max_len` and `max_len` in the checkpoint config, or split the
  question into a coarse-to-fine hierarchy.
- **Ordinal `score` questions are the weakest primitive** — 0.372 on SST-5.
- **It ships over-confident.** Refitting one temperature per (question type, option count) moves
  mean ECE from **0.466 to 0.081** for English and **0.314 to 0.106** for multilingual. Do that
  on your own data before you trust the probabilities. It is also why this recipe will not
  spend 2e-2 of probability error on a faster dtype.
- **The root checkpoint is English only.** Use multilingual for anything else — the router does
  this for you, and the reason it is aggressive about it is in the routing section above.

## Documentation

- [docs/use-cases.md](docs/use-cases.md) — the nine examples, their question sets, and the
  output captured from a live server for each one
- [docs/patterns.md](docs/patterns.md) — the five shapes the examples are built from, with a
  pasteable sketch of each
- [docs/integrations.md](docs/integrations.md) — the MCP server, the Claude Code plugin, the
  per-agent configuration and the CI job
- [docs/serving-options.md](docs/serving-options.md) — what was considered, what shipped, and
  what was ruled out with the measurement that ruled it out
- [bench/results.md](bench/results.md) — every figure and how it was taken
- [CHANGELOG.md](CHANGELOG.md) — which defaults changed when
- [CREDITS.md](CREDITS.md) — the model, the encoders, and the API shape are other people's work

MIT licensed. The model weights are Apache-2.0 from Convai Innovations and carry their own terms.
