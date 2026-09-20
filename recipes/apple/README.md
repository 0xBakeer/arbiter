# Apple Silicon

Measured next. Nothing here has been run on a Mac yet, so this file carries no numbers and no
install instructions rather than plausible ones.

What is already known and what has to be settled on the way:

- `DEVICE=mps` is the target. `DEVICE=cpu` already works everywhere and is roughly an order of
  magnitude slower than CUDA; it is the fallback while the MPS path is being measured.
- torch comes from PyPI on this lane, not from the CUDA index, and the Triton header question in
  [recipes/nvidia](../nvidia/README.md) does not arise.
- `ARBITER_DTYPE` and `ARBITER_MODE` are CUDA-shaped defaults: autocast bf16 and CUDA graphs.
  Graphs mode is off on anything that is not CUDA, and what the dtype should be here is a
  measurement, not a guess.
- The same equivalence gate decides it as on NVIDIA — `./run.sh equivalence` against the SDK
  reference, max |Δp| under 5e-3 and no argmax moving — and the latency and throughput tables
  go in [bench/results.md](../../bench/results.md) under their own machine heading.
