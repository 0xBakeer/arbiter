# Apple Silicon

torch's MPS backend, `ARBITER_DEVICE=mps`, eager forward, fp32 parameters. `./run.sh setup` does
all of it, and no environment variable has to be set: the device is detected and `autocast`,
which is a CUDA mode, resolves to fp32 here.

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

## What an MLX port could buy, as future work

Everything above is torch's MPS backend, which dispatches the same op graph it dispatches on
CUDA through Metal shaders it did not choose the layout for. MLX is the other direction: arrays
that live in unified memory with no host/device copy at all, kernels written for the M-series,
lazy evaluation that fuses the small elementwise chains an encoder is full of, and quantised
formats (4-bit and 8-bit) that are first-class rather than bolted on. For this workload the
plausible wins are the two numbers that look worst above — the minute-and-a-half cold load,
which is mostly materialising and copying fp32 weights that MLX would mmap into unified memory
directly,
and the 50-question row, which is launch-bound enough that fusion should help more than it does
on a big discrete GPU. The cost is real: `engines/mlx/` would have to reimplement the ModernBERT
encoder and Laya's marker head and act head from the checkpoint tensors, and it would arrive
owing exactly the same equivalence table as this one before any of its numbers could be
believed. That is the order — port, then gate, then measure — and it is why it is future work
and not a footnote to this recipe.
