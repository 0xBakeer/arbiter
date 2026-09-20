# How this gets served, and what was rejected

Laya is not a language model, and almost every reflex from LLM serving is wrong about it. It is
a 22-to-28-layer bidirectional encoder with a small typed head bolted on, it emits no tokens,
and a request is over in one forward pass. What follows is the set of options that were
considered for this recipe and the reason each one is or is not in it.

---

## Not applicable: vLLM, SGLang, TensorRT-LLM, llama.cpp

Every one of these is an *autoregressive decoder* server. Their entire value — paged KV cache,
continuous batching across decode steps, speculative decoding, prefix reuse — is machinery for
the loop that generates token N+1 from tokens 1..N. Laya has no such loop. It also has a custom
decision head (`nn.TransformerEncoder` over the encoder output, a marker gather, a scorer MLP,
an act head) that none of these engines can express, so even the encoder half could not be
hosted without writing a new model definition inside somebody else's runtime.

The one idea worth stealing from them is batching, and that is stolen below.

## The baseline: the SDK behind FastAPI

`laya.Agent.system_one` in a request handler. It works, and it is the reference every number in
this repository is measured against. Two things looked like costs it leaves on the table:

- **fp32 parameters cast on every forward.** The SDK loads the checkpoint into an fp32 model and
  wraps the forward in `torch.autocast(bfloat16)`. The weights were trained in bf16
  (`"amp_dtype": "bf16"` in every `rl_agent_config.json`), so the fp32 copy looks like it holds
  no information the bf16 one does not. This turned out to be wrong, and the section below is
  mostly about why.
- **One request, one forward.** Every question is already an independent row; two callers
  arriving in the same millisecond run two forwards where one would do. This one was real.

## Shipped

**fp32 parameters under autocast — the SDK's arithmetic, kept.** This was meant to be the one
easy win: the checkpoints were trained in bf16 (`"amp_dtype": "bf16"` in every
`rl_agent_config.json`), so converting the parameters to bf16 and dropping autocast should have
been free. It is not free, and the equivalence harness is what said so.

Autocast runs matmuls in bf16 but keeps layer norms, the residual stream and the softmaxes in
fp32. Converting the parameters puts all of that in bf16 too, and across 28 ModernBERT layers it
accumulates: measured against the SDK, bf16 parameters move reported probabilities by up to
**2.1e-2** and flip one argmax out of 22, on a five-level `score` question whose own reported
confidence was 0.157. Autocast, batched, is identical to the reference at four decimals —
**0.00e+00**.

bf16 parameters are 15–35% faster (14.0 ms against 20.9 at one question; 380 questions/s against
287 at concurrency 8). The trade was still refused. Laya's entire proposition is calibrated
probabilities, the model card already warns they ship over-confident and need refitting
(mean ECE 0.466 → 0.081), and six milliseconds off a twenty-millisecond answer is not worth
2e-2 of the calibration budget. It is available as `LAYA_DTYPE=bf16` with the numbers next to it.

One implementation note for anyone who tries it: `act_head` has to stay fp32. It is fed
`torch.cat([h[:, 0].float(), feats])`, so a bf16 weight meets an fp32 activation and the forward
dies with a dtype error the moment autocast is no longer there to hide it.

**SDPA attention.** Already the SDK's default (`build_model` passes
`attn_implementation="sdpa"`), and left alone. `reference_compile` is forced off on every
checkpoint before the first forward — it defaults to `"auto"`, which means torch.compile as soon
as the model is on CUDA, and this path has to stay eager for graphs mode to be capturing what it
thinks it is capturing.

**Dynamic cross-request micro-batching.** A worker thread per checkpoint takes the first
request's rows and then keeps draining the queue for `LAYA_BATCH_WAIT_MS` (default 2 ms) or
until `LAYA_MAX_BATCH` rows are in hand, then runs one forward. Rows are independent — attention
is masked per row, the head's `src_key_padding_mask` is per row, the marker gather is per row —
so this cannot change what any single row computes beyond floating-point reassociation. That was
the assumption, and it is the one the harness checked hardest: batching every question of every
case into one padded forward changes the answers by **0.00e+00**. A thread rather than a
coroutine because the forward holds the GIL in stretches and blocking the event loop would
defeat the point.

It is worth the ninety lines: 135 questions/s at one caller becomes 287 at eight.

**All three checkpoints resident.** Together they are 1.16B parameters, about 5 GB of GPU memory
under autocast. The SDK's `Router` defaults to `max_loaded=1`, which means a server alternating
between English and German requests reloads a checkpoint from disk on every request. Detection
costs microseconds; a load costs seconds. Preloading is the only sane server setting, and the
router is handed the already-built agents via `Router.attach` so nothing is loaded twice.

**CUDA-graph replay — shipped, but not as the default.** At batch 1 the model is launch-bound:
28 ModernBERT layers plus 2 head layers is several hundred small kernels, and the host cannot
issue them as fast as the GPU retires them. Graphs replace the whole issue sequence with one
replay, and at one question that is a 20% win (16.8 ms against 20.9).

It loses everywhere else, and the reason is padding. A graph's cost is its *padded* shape, so
every request pays for the bucket it landed in rather than for itself. At 50 questions eager
takes 152 ms and graphs 229.

The first version of this was much worse — 306 ms — and the instructive part is which fix
worked. The marker dimension was pinned at 32, which meant a two-option question ran the
scorer's 1024×1024 MLP over sixteen times the positions it needed; bucketing it bought **0.1 ms**.
The sequence ladder was the real cost: rows here are about 130 tokens, and a ladder stepping
128 → 256 doubled the encoder's work on nearly every request. Fine ladders (batch to 12 steps,
sequence to 15, markers to 9) took 306 ms to 229. Coarse buckets are a large and completely
invisible tax, and they are easy to mistake for "CUDA graphs do not help here".

Anything outside a bucket — a question with more options than `LAYA_GRAPH_MAX_MARKERS`, a batch
above 64 — takes the eager path, which is always there. Graphs are captured lazily on first use
of a bucket, after a 3-iteration warm-up on a side stream, and they share one memory pool.
Capture is plain `torch.cuda.CUDAGraph`, not `torch.compile(mode="reduce-overhead")`, so nothing
on this path can reach inductor.

Padding has to be *neutral*: a pad row gets one attended token and no live markers, which keeps
it well defined (an all-zero attention mask would soften to NaN), and its results are discarded.

**The gate.** `tools/equivalence.py` runs a fixed set of 22 questions — all three types, 2 to 12
options, English, German and Hindi states, JSON and conversation-array states — through the SDK
reference and through each combination of dtype and mode, and reports the largest absolute
difference in any reported probability and whether any argmax moved. The rule set before the
measurement was: ship it only at max |Δp| < 5e-3 with 100% argmax agreement. `autocast/eager`
(0.00e+00) and `autocast/graphs` (1.4e-3) pass; both bf16 paths (2.1e-2, one argmax moved) do
not. That is the whole reason the defaults are what they are, and the table is in
[../bench/results.md](../bench/results.md).

---

## Considered and deferred

**ONNX Runtime / TensorRT export.** The obvious next step for an encoder this size, and the
reason it is not here is specific rather than lazy. ModernBERT alternates full and sliding-window
attention on a fixed layer schedule with two different RoPE thetas, which exporters handle
unevenly; and the decision head is not a standard classification head — it gathers hidden states
at per-row marker positions and scores them, so the export would have to carry a dynamic gather
alongside dynamic sequence length. That is a day of work whose payoff is mostly the same payoff
CUDA graphs already deliver here, because the bottleneck measured on this hardware is kernel
launch, not kernel efficiency. Worth revisiting if the launch overhead is removed and the
kernels themselves become the floor.

**Triton Inference Server, or NIM.** Both earn their keep with multi-model scheduling, model
repositories, versioned rollouts and ensembles across several GPUs. On one GPU serving three
small checkpoints out of one process, they add a hop and an operational surface and remove
nothing. The batching Triton would give us is the batching implemented here in ninety lines,
against models it would have to be taught about anyway via a Python backend.

**int8 or fp8 weights.** Not attempted, and the bf16 result above is why: plain bf16
parameters already cost 2.1e-2 of probability error on this model. Quantisation noise lands on a
calibration that the model card itself calls the weakest thing about it. Bandwidth is not the
constraint at 421M parameters either.

**CPU serving.** Works — `DEVICE=cpu` takes the same code path with fp32 parameters and no
graphs — and `tests/test_real_model.py` uses it. It is not the target, and it is roughly an order
of magnitude slower.
