# Apple Silicon

torch's MPS backend, `ARBITER_DEVICE=mps`, eager forward, fp32 parameters. `./run.sh setup` does
all of it, and no environment variable has to be set: the device is detected and `autocast`,
which is a CUDA mode, resolves to fp32 here.

That is the shipped lane and everything down to ["Which dtype ships, and
why"](#which-dtype-ships-and-why) describes it. There is a second, opt-in one on the same
machine -- [Option 3: the MLX engine](#option-3-the-mlx-engine) -- which swaps torch for the
community MLX port and is measured against these rows there.

## Install

```bash
./run.sh setup      # .venv, torch from PyPI, the deps, the checkpoints (2.2 GB)
./run.sh serve      # http://localhost:8010
./run.sh status     # /healthz, /readyz, /v1/models
```

torch comes from the default PyPI index on this lane — those are the MPS-enabled arm64 wheels —
so `TORCH_INDEX` is empty on Darwin and pip resolves normally. The Triton header question from
[recipes/nvidia](../nvidia/README.md) does not arise: there is no Triton on macOS. `setup` still
prefers a uv-managed CPython if one is installed, then `python3.12`, then `python3`; the run
below was on CPython 3.13.13 and torch 2.14.0.

`stop` and the pid file work the same way, with one difference behind them: `setsid` is a
util-linux program that macOS does not have, so `serve` detaches with `nohup` alone.

## Defaults on this lane

| variable | here | why |
|---|---|---|
| `ARBITER_DEVICE` | `mps` | detected; `cuda` → `mps` → `cpu`, and `/readyz` reports which one was taken |
| `ARBITER_DTYPE` | `fp32` | `autocast` is CUDA-only and resolves to fp32; `fp16` and `bf16` exist here and both fail the equivalence gate — see below |
| `ARBITER_MODE` | `eager` | `graphs` is CUDA graph capture. `./run.sh serve` refuses it on a Mac with one line instead of failing inside uvicorn later |

`ARBITER_DTYPE=autocast` is deliberately *not* mapped onto torch's MPS autocast. That one is
fp16 over a different set of operations, and giving the same name to two arithmetics would make
the two machines' equivalence tables incomparable. On a Mac the honest choices are whole-model
dtypes, and `supported_dtypes()` in [`engines/laya/loader.py`](../../engines/laya/loader.py)
says so per device.

## Measured, one Apple M2 Max

32 GB unified memory, macOS 26.6.2, torch 2.14.0 from PyPI, CPython 3.13.13, 2026-09-20, no
other GPU work running. End-to-end over HTTP, all three checkpoints resident. Method and the
equivalence table in [bench/results.md](../../bench/results.md).

| questions in one call | M2 Max, fp32 (shipped) | GB10, autocast (shipped) |
|---:|---:|---:|
| 1  | **30.3 ms** | 20.9 ms |
| 5  | **63.1 ms** | 31.1 ms |
| 10 | **107.4 ms** | 40.0 ms |
| 50 | **462.3 ms** | 152.4 ms |

| concurrency, 4-question calls | M2 Max, fp32 (shipped) | GB10, autocast (shipped) |
|---:|---:|---:|
| 1  | **71.9 q/s** | 134.7 q/s |
| 8  | **101.7 q/s** | 286.6 q/s |
| 32 | **105.9 q/s** | 301.0 q/s |

p95 tracked p50 within 1.3 ms on every latency row, and there were no errors at any
concurrency. The GB10 column is there for scale; it is a different machine and a different
dtype, and the two were not measured against each other.

The interesting part of the shape is that the Mac is 1.45× behind at one question and 3.0×
behind at fifty. A single question is launch-bound on both machines and the M2 Max holds up
well; fifty questions is arithmetic, and that is where the GB10 is actually a bigger GPU. The
throughput ceiling arrives early and flat — 102 q/s at eight callers and 106 at thirty-two —
because the GPU is already saturated at eight callers and past that the batcher only turns more
concurrency into more queueing: p50 goes from 55 ms at one caller to 1.2 s at thirty-two for the
same work.

### Cold load and footprint

| | |
|---|---|
| `./run.sh serve` to `/readyz` | **88 s and 101 s**, two runs |
| first call at a shape not seen before | 0.2-1.2 s, against 30-63 ms warm |
| `footprint -p` at `/readyz` | 5.4-6.1 GB, of which **4,922 MB** is `IOAccelerator (graphics)` |
| `footprint -p` after a bench run | 6,465 MB, peak 6,767 MB |
| `ps -o rss` | 1.6 GB just after `/readyz`, ~230 MB in a process that has been up a while |

Two macOS-specific notes on that table. **`ps -o rss` is not the memory this uses**, and the two
values in that row are the evidence: the same server reported 1.6 GB shortly after `/readyz` and
about 230 MB once it had been up a while, while what it was actually holding never moved. The
weights live in IOAccelerator-backed buffers on the unified memory, which are not in the
process's resident set. Use `footprint -p <pid>`, whose `phys_footprint` counts them and whose
`IOAccelerator (graphics)` line — 4,922 MB, stable across restarts — is the weights themselves.
There is no per-process Activity-Monitor-style "GPU memory" figure to report separately, because
on unified memory there is no separate pool.

**88-101 s is slow and it is the load, not the download** — the files were already in the page
cache from an earlier run both times. Three checkpoints of fp32 weights are ~4.9 GB to materialise and move
onto the GPU, and the SDK loads them one at a time. `ARBITER_MODELS=english` is about a third of
it if the wait matters more than the routing. The first call at any
sequence-and-batch shape the process has not run yet is Metal compiling kernels for it: the
first three-question call after `/readyz` took 207 ms and the first long five-question one
1.24 s, against 30 and 63 ms once warm. The built-in warm-up covers the shape it warms with,
not every shape, so a production Mac pays a fraction of a second once per new size.

### Which dtype ships, and why

fp32, because it is the only one that passes the gate. Against the SDK reference — `laya.Agent`
on CPU in fp32, which is what the SDK itself runs on a Mac — over the same 22 questions:

| parameters | max abs Δp vs reference | argmax | verdict |
|---|---:|---:|---|
| **fp32** | **1.0e-04** | 22/22 | **passes** the 5e-3 / 100% gate |
| fp16 | 1.26e-02 | 22/22 | fails |
| bf16 | 6.06e-02 | 22/22 | fails |

The 1.0e-04 on fp32 is the rounding floor of the comparison itself (probabilities are compared
as a client sees them, at four decimals) and comes from one multilingual question: MPS's fp32
kernels are not bit-identical to the CPU's, which is expected and is the entire content of that
figure.

fp16 and bf16 are worse here than bf16 is on the GB10 (2.1e-2 there), and bf16 on MPS is worse
by a factor of five. The residual stream runs through 22-28 layers in the parameter dtype with
no autocast keeping the norms and the accumulations in fp32, and bf16's eight mantissa bits do
not survive that on this backend. No argmax moved in any mode, which is better than the GB10's
bf16 managed — but Laya's proposition is calibrated probabilities, the model card is explicit
that they ship over-confident and want refitting, and 6e-2 of probability error is most of that
budget. Both are one environment variable away if a workload disagrees, and the gate will say so
out loud: `./run.sh equivalence --reference-device cpu`.

Neither was benchmarked. A dtype that fails the gate does not get a speed number here, because
publishing one invites the trade the gate exists to refuse.

## Known limits

- **No CUDA graphs.** `ARBITER_MODE=graphs` is refused. Removing launch overhead on this backend
  would mean a compiled or captured Metal graph instead, which was not in scope here; eager is
  the only mode measured.
- **fp16 and bf16 fail equivalence**, so the 2-3× smaller resident size and whatever speed they
  would buy are not available at the shipped default.
- **Throughput flattens at eight concurrent callers** and latency grows linearly past it. If a
  Mac is the deployment, size it by p95 at the concurrency you actually have, not by the 106 q/s
  ceiling.
- **A cold load of a minute and a half**, which makes a restart during a deploy visible in a way
  it is not on the GB10's 20-60 s.

## Option 3: the MLX engine

`ARBITER_ENGINE=laya_mlx` serves the same three checkpoints through
[mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx), an independent MLX reimplementation
of the ModernBERT encoder and Laya's decision heads, over converted weights published as
`aac6fef/laya-mlx`, `aac6fef/laya-multilingual-mlx` and `aac6fef/laya-typed-decisions-mlx`. It is
opt-in, it changes no default on the torch lane, and both engines live in the same checkout and
the same `.venv`.

### Install

```bash
ARBITER_ENGINE=laya_mlx ./run.sh setup       # adds laya-mlx + mlx, and 2.1 GB of converted weights
ARBITER_ENGINE=laya_mlx ./run.sh serve
ARBITER_ENGINE=laya_mlx ./run.sh equivalence # the gate, against the SDK on the CPU
```

`setup` on this path is a superset of the one above: it still installs torch and still fetches
the upstream tree, because `ARBITER_ENGINE=laya` has to keep working on the same checkout and
because the equivalence gate's reference is `laya.Agent` under torch. The MLX side adds one
package (`laya-mlx`, which brings `mlx`, `tokenizers`, `huggingface_hub` and `numpy`) and three
checkpoint directories under `models/laya-mlx/`. Nothing it installs conflicts with torch --
`pip check` is clean afterwards -- so there is no sibling `.venv-mlx`.

What it needs: Apple silicon, macOS 26 or newer for the mlx-metal wheel that MLX 0.32 selects
here, and Python 3.11+. On anything else `run.sh` and the engine both refuse with one line rather
than failing inside a wheel that does not exist for the architecture.

### Defaults on this lane

| variable | here | why |
|---|---|---|
| `ARBITER_ENGINE` | `laya_mlx` | opt-in; the default everywhere else is `laya` |
| `ARBITER_DEVICE` | `gpu` | MLX has `gpu` and `cpu`. `mps` is torch's name for the same hardware and is refused here rather than silently taken |
| `ARBITER_DTYPE` | `fp32` | the only one that passes the gate, exactly as on the torch lane. fp16 is the port's own default and is 1.7e-2 out on the multilingual checkpoint |
| `ARBITER_MODE` | `eager` | there is no graph capture on this engine; `mx.compile` and the port's prefix cache are not wired up |
| `ARBITER_MLX_CACHE_MB` | `1024` | MLX's buffer cache is unbounded by default, which costs 23 GB and most of the throughput on a server. `0` restores it |

`/readyz` reports `"engine": "laya_mlx"` alongside the mode, dtype and device, and `/v1/models`
lists the same three ids as the torch lane -- the checkpoints, the aliases and the routing are
the same; only the arithmetic underneath is different.

### Measured, the same M2 Max

Same method as above, taken 2026-09-20 with the torch server still resident on the same GPU, so
the column to compare against is the paired control that was run minutes later rather than the
uncontended rows further up. Full tables and the control: [bench/results.md](../../bench/results.md).

| questions in one call | `laya_mlx`, fp32 | torch MPS, fp32, paired control | torch MPS, published |
|---:|---:|---:|---:|
| 1  | **22.7 ms** | 30.5 ms | 30.3 ms |
| 5  | **59.2 ms** | 62.8 ms | 63.1 ms |
| 10 | **102.7 ms** | 107.1 ms | 107.4 ms |
| 50 | **448.4 ms** | 545.9 ms | 462.3 ms |

| concurrency, 4-question calls | `laya_mlx`, fp32 | torch MPS control | torch MPS, published |
|---:|---:|---:|---:|
| 1  | **75.4 q/s** | 35.1 q/s | 71.9 q/s |
| 8  | **102.7 q/s** | 103.1 q/s | 101.7 q/s |
| 32 | **108.7 q/s** | 107.4 q/s | 105.9 q/s |

| | `laya_mlx` | torch MPS |
|---|---|---|
| `serve` to `/readyz`, three checkpoints | **2.1 s and 2.2 s** | 88 s and 101 s |
| `footprint -p` at `/readyz` | 5,786 MB (5,501 MB IOAccelerator) | 5,381-6,131 MB (4,922 MB IOAccelerator) |
| after a bench run | 5,937 MB, peak 7,800 MB | 6,465 MB, peak 6,767 MB |

Read that as three findings. **The cold load is forty times faster** and is the reason to run this
engine: two seconds instead of a minute and a half, because MLX maps the converted weights into
unified memory and casts them there instead of materialising fp32 tensors and copying them onto
the GPU. **A single question is 25% faster**, which is dispatch cost and nothing else -- the
advantage shrinks to 5% by ten questions, where the matmuls dominate and both backends are
running the same ones. **Throughput is a tie**: the GPU is saturated at eight callers either way,
and the ceiling is the same 102-109 q/s. Resident footprint is also a tie, since both hold the
same 4.7 GB of fp32 parameters.

### The equivalence numbers

Same 22 questions, same reference (`laya.Agent` on the CPU in fp32), same 5e-3 / 100%-argmax gate:

| parameters | max abs Δp vs reference | argmax | verdict |
|---|---:|---:|---|
| **fp32** | **0.00e+00** | 22/22 | **passes** -- every probability is the reference, digit for digit |
| fp16 | 1.69e-02 | 22/22 | fails |
| bf16 | 2.00e-02 | 21/22 | fails, and one argmax moves |

The fp32 row is cleaner than torch's own fp32 on this machine (1.0e-04), and the reason is worth
knowing: the upstream weights are stored in fp16, and both paths widen them -- the SDK to fp32 on
the CPU, the port to fp32 in MLX -- so there is nothing in the reference for the conversion to
lose. It also means the port's arithmetic is not approximately right, it is exactly right at the
four decimals a client sees.

fp16 fails only on the multilingual checkpoint (the other two are at 1.0e-03 and 8.0e-04), which
is the same checkpoint that fails worst on torch's MPS fp16. Routing was compared as well, since
the port carries its own adapted copy of upstream's router: 7/7 cases decided identically,
including the Devanagari one.

### Known limits of this lane

- **No mx.compile and no prefix cache.** The port offers both, plus `pad_to_multiple`; none is
  wired up here, because the batcher's shapes vary per batch and a shape-specialising compile
  wants the opposite. Unmeasured, therefore unclaimed.
- **The buffer cache has to be bounded.** With MLX's unbounded default the server reached a
  23 GB footprint and *lost* throughput as callers were added -- 39 q/s at eight against 103 with
  the bound. That is what `ARBITER_MLX_CACHE_MB` is for, and the evidence table is in
  bench/results.md.
- **macOS 26+ in practice.** MLX 0.32 publishes macOS 14, 15 and 26 wheels; the installer took
  the 26 one on this machine. Older macOS versions were not tested here.
- **A second implementation to trust.** The weights are converted and the encoder is
  reimplemented, so the equivalence gate is not a formality on this lane the way it is for a
  dtype change. It is the reason the gate is wired into `./run.sh equivalence` for this engine
  too.

## What the MLX port bought, and what is still open

The section this replaces was a prediction, written before the port was wired up: that MLX would
win on the cold load and on the launch-bound end of the latency curve, and that it would owe the
same equivalence table before any of it could be believed. That is how it came out, and the
numbers are in [Option 3](#option-3-the-mlx-engine) above: 40× on the cold load, 25% at one
question, a tie at ten and at every concurrency, and a gate it passes more cleanly than torch
does. The prediction was wrong in one place -- fifty questions was supposed to be where fusion
helped most, and it is 3% against the uncontended torch row.

What is still open on this lane:

- **Quantisation.** 4-bit and 8-bit are first-class in MLX and would cut the 4.7 GB of parameters
  and, plausibly, the fifty-question row. Neither is converted or measured, and on the evidence
  of fp16 at 1.7e-2 the gate is where they would have to be argued.
- **`mx.compile`, `pad_to_multiple` and the prefix cache**, which the port measured at about 6.5%
  on its own Snake demo and which would need the batcher's shape ladder rethought here.
- **A second machine.** Everything above is one M2 Max. The M-series spread is wide and the
  port's own figures are from an M3 Max, where a single English question takes 13.4 ms.
