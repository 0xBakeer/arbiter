# Measurements

One section per machine, because nothing here transfers between them: the wheels differ, the
kernels differ, and a default chosen on one is a guess on the other. Each section says what was
measured, on what, and when. The install and serve notes for a machine live next to it in
[`recipes/`](../recipes).

Every number is end-to-end over HTTP against `./run.sh serve`, not an in-process call, so it
includes JSON parsing, routing, the batcher's wait and the response encode.

The reference column is the model card's published figures for `laya`/`laya-multilingual` on a
T4, which are in-process SDK calls. They are not the same measurement and the comparison is
therefore generous to us in one direction (newer hardware) and harsh in the other (we pay for
HTTP and they do not). It is here for scale, not for a ranking.

# NVIDIA GB10

DGX Spark class, 128 GB unified memory, driver 580, CUDA 13.0, torch 2.14.0+cu130, 2026-09-20,
with an unrelated LLM already resident on the same GPU. Setup notes:
[recipes/nvidia](../recipes/nvidia/README.md).

## Latency, one caller

30 calls per row after 3 warm-up calls, English state, auto-routed to the English checkpoint.
Questions cycle through score / choice / noul so every count mixes all three types.

| questions in the call | eager + autocast (shipped) | graphs + autocast | eager + bf16 params (rejected) | model card, T4 english | model card, T4 multilingual |
|---:|---:|---:|---:|---:|---:|
| 1  | **20.9 ms** | 16.8 ms | 14.0 ms | 39.5 ms | 32.8 ms |
| 5  | **31.1 ms** | 29.0 ms | 19.9 ms | — | — |
| 10 | **40.0 ms** | 44.6 ms | 28.5 ms | 158.6 ms | 72.3 ms |
| 50 | **152.4 ms** | 229.4 ms | 131.7 ms | 771 ms | 337 ms |

p95 tracked p50 within 2 ms on every row.

## Throughput, 4-question calls, 10 s per level

| concurrency | eager + autocast (shipped) | graphs + autocast | eager + bf16 params (rejected) |
|---:|---:|---:|---:|
| 1  | **134.7 q/s** | 146.8 q/s | 212.0 q/s |
| 8  | **286.6 q/s** | 247.6 q/s | 380.5 q/s |
| 32 | **301.0 q/s** | 266.2 q/s | 351.4 q/s |

No errors at any level in any mode.

## Numerical equivalence

22 questions across 7 calls: all three question types, 2 to 12 options, English, German and
Hindi states, a JSON state and a conversation-array state. Each path is compared against
`laya.Agent.system_one` as the SDK ships it. Probabilities are compared as a client sees them,
rounded to four decimals, which puts a 1e-4 floor under every figure.

| checkpoint | path | questions | max abs delta p vs reference | argmax agreement |
|---|---|---:|---:|---:|
| english | autocast/eager | 8 | 0.00e+00 | 8/8 |
| english | autocast/graphs | 8 | 1.00e-04 | 8/8 |
| english | bf16/eager | 8 | 1.75e-02 | 7/8 |
| english | bf16/graphs | 8 | 1.75e-02 | 7/8 |
| english | autocast eager vs graphs | 16 | 1.20e-03 | 16/16 |
| english | bf16 eager vs graphs | 16 | 1.40e-03 | 16/16 |
| multilingual | autocast/eager | 9 | 0.00e+00 | 9/9 |
| multilingual | autocast/graphs | 9 | 2.00e-04 | 9/9 |
| multilingual | bf16/eager | 9 | 1.41e-02 | 9/9 |
| multilingual | bf16/graphs | 9 | 1.46e-02 | 9/9 |
| multilingual | autocast eager vs graphs | 18 | 3.40e-03 | 18/18 |
| multilingual | bf16 eager vs graphs | 18 | 3.50e-03 | 18/18 |
| typed-decisions | autocast/eager | 5 | 0.00e+00 | 5/5 |
| typed-decisions | autocast/graphs | 5 | 1.40e-03 | 5/5 |
| typed-decisions | bf16/eager | 5 | 2.11e-02 | 5/5 |
| typed-decisions | bf16/graphs | 5 | 2.11e-02 | 5/5 |
| typed-decisions | autocast eager vs graphs | 10 | 1.40e-03 | 10/10 |
| typed-decisions | bf16 eager vs graphs | 10 | 2.00e-04 | 10/10 |

Against the 5e-3 / 100%-argmax gate:

- `autocast/eager` — max |Δp| **0.00e+00**, argmax unchanged. **Passes.** Batching rows from
  different callers into one forward changes nothing at all at four decimals.
- `autocast/graphs` — max |Δp| 1.40e-03, argmax unchanged. **Passes.**
- `bf16/eager` and `bf16/graphs` — max |Δp| 2.11e-02, and one argmax moves. **Fails.** The
  question that moves is a five-level `score` whose reported confidence is 0.157, i.e. a nearly
  flat distribution where the top two levels are separated by less than the error introduced.

Reproduce with `./run.sh equivalence --models-dir models/laya --device cuda`.

## Bucket granularity, and why graphs mode nearly lost

Graphs mode was measured three times while the bucket ladders were being chosen. The first two
are kept because they are the evidence for the third.

| 50-question call | batch ladder | sequence ladder | marker dimension | p50 |
|---|---|---|---|---:|
| first attempt | 1,2,4,8,16,32,64 | 128,256,512,1024 | pinned at 32 | 306.5 ms |
| marker bucketing added | 1,2,4,8,16,32,64 | 128,256,512,1024 | 2,4,8,16,32 | 306.4 ms |
| fine ladders (shipped) | 1,2,3,4,6,8,12,16,24,32,48,64 | 64,96,...,512,640,...,1024 | 2,3,4,6,8,12,16,24,32 | 229.4 ms |

The marker dimension was the wrong suspect: bucketing it bought 0.1 ms. The sequence ladder was
the cost. These rows are ~130 tokens long, and a ladder whose next step was 256 doubled the
encoder's work on every one of them. A graph's price is its padded shape, and a coarse ladder is
a large and completely invisible tax.

Even at the fine ladder, graphs only win where there is little work to amortise the padding
over — one or five questions, one caller. See README.md for the resulting default.

# Apple Silicon

Not measured yet. [recipes/apple](../recipes/apple/README.md) says what has to be settled first;
the tables land here under this heading, in the same shape as the GB10 ones, so the two machines
can be read side by side.
