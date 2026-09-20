# NVIDIA

CUDA 13.0 wheels, `ARBITER_DEVICE=cuda` (detected, not set), eager forward with autocast
bf16. `./run.sh setup` does all of it.

## Install

```bash
./run.sh setup      # .venv, torch from the cu130 index, the deps, the checkpoints (2.3 GB)
./run.sh serve      # http://localhost:8010
./run.sh status     # /healthz, /readyz, /v1/models
```

The torch wheels come from `https://download.pytorch.org/whl/cu130`, which publishes both
aarch64 and x86_64 builds, so a GB10 and a desktop card install the same way. `TORCH_INDEX`
overrides the index if you need a different CUDA.

**The interpreter matters.** torch routes a handful of eager operations — ModernBERT's RoPE
among them — through Triton, and Triton JIT-compiles a small C extension against `Python.h` the
first time one runs. A distribution `python3.12` without `python3-dev` does not have that header
on disk, and the failure lands on the first forward pass as *Python.h: No such file or
directory*. `setup` therefore prefers an interpreter that ships its own headers: a uv-managed
CPython (`~/.local/share/uv/python/cpython-3.12*`) is tried first, then `python3.12`, then
`python3`. `PYTHON=/path/to/python3.12` picks one yourself. If none of them has headers, setup
falls back to the first that exists and exports `TORCH_DISABLE_NATIVE_JIT=1`, which takes
torch's plain kernels instead; it prints a line when it does.

## Serve

`./run.sh serve` starts uvicorn detached with a pid file and a log under `logs/`, and waits for
`/readyz` — the checkpoints take 20-60 s to load cold. `deploy/arbiter.service` is the same
thing as a systemd *user* unit. `./run.sh stop` stops it.

Defaults worth knowing on this lane: `ARBITER_MODE=eager` (graphs mode is faster only for a
single caller asking one or five questions) and `ARBITER_DTYPE=autocast` (bf16 parameters are
15-35% faster and move probabilities by up to 2.1e-2, which is why they are not the default).
The full table is in the [README](../../README.md).

## Measured, one NVIDIA GB10

128 GB unified memory, driver 580, CUDA 13.0, torch 2.14.0+cu130, 2026-09-20, with an unrelated
LLM already resident on the same GPU. End-to-end over HTTP. Method and the rest of the figures
in [bench/results.md](../../bench/results.md).

| questions in one call | eager + autocast (shipped) | graphs + autocast | eager + bf16 params (rejected) |
|---:|---:|---:|---:|
| 1  | **20.9 ms** | 16.8 ms | 14.0 ms |
| 5  | **31.1 ms** | 29.0 ms | 19.9 ms |
| 10 | **40.0 ms** | 44.6 ms | 28.5 ms |
| 50 | **152.4 ms** | 229.4 ms | 131.7 ms |

| concurrency, 4-question calls | eager + autocast (shipped) | graphs + autocast | eager + bf16 params (rejected) |
|---:|---:|---:|---:|
| 1  | **134.7 q/s** | 146.8 q/s | 212.0 q/s |
| 8  | **286.6 q/s** | 247.6 q/s | 380.5 q/s |
| 32 | **301.0 q/s** | 266.2 q/s | 351.4 q/s |

All three checkpoints resident in `autocast` cost 5,055 MiB of GPU memory and 3.6 GB of host
RSS. Reproduce with `./run.sh bench`, and the equivalence gate with `./run.sh equivalence`.
