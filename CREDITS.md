# Credits

Nearly everything that makes this useful is somebody else's. What is ours is the serving design
— the engine interface, the cross-request micro-batcher, the CUDA-graph bucketing, the routing
surface — the equivalence gate that picked the defaults, the measurements, the playground, the
examples and the integrations. The model, the encoders it is built on, the API contract it
speaks and the vocabulary the examples are written in all come from the people below.

## The model

**[Convai Innovations](https://huggingface.co/convaiinnovations)** and **Nandha Kishor M**
([@NandhaKishorM](https://github.com/NandhaKishorM)) — Laya itself, Apache-2.0: the three
checkpoints, the decision-head architecture, the per-bucket temperature calibration, and the
benchmark numbers this repository measures itself against.

Also theirs, and just as load-bearing here: the **`laya` SDK**. `engines/laya/loader.py` is a
server around it, not a reimplementation of it. `Agent` loads the checkpoints and owns the
forward; `Router` decides which one answers and is handed our already-built agents so it never
loads a second copy; `laya.common` builds the sequence — the option markers, the head budget,
the temperature buckets, the collation — and every row this server batches was shaped by their
code. When `tools/equivalence.py` compares the served path against `laya.Agent.system_one`, it
is checking that we did not break what they wrote.

The model card's "Honest Limits" section is quoted at length in our README rather than
summarised, because it is more useful than anything we could say about when not to use this, and
paraphrasing a limitation is usually a way of softening it.

- Model: https://huggingface.co/convaiinnovations/laya
- SDK: https://github.com/NandhaKishorM/laya · https://pypi.org/project/laya/

## The encoders

**[Answer.AI](https://huggingface.co/answerdotai/ModernBERT-large)** — ModernBERT-large, the
backbone of the English and typed-decisions checkpoints. Its alternating local/global attention
schedule is also the reason an ONNX or TensorRT export of this model is more work than it looks;
see [docs/serving-options.md](docs/serving-options.md).

**[JHU CLSP](https://huggingface.co/jhu-clsp/mmBERT-base)** — mmBERT-base, the backbone of the
multilingual checkpoint, and the reason a non-Latin script gets a real answer instead of a
confident wrong one.

## The API shape and the way of thinking

**[TypeSafe AI](https://typesafe.ai)** — the Jev System One request and response contract, which
this server implements so that code written against Jev runs against a local model by changing
nothing but the base URL and the key. The contract is theirs; this is a compatible
implementation, not an affiliated or endorsed one.

The vocabulary is theirs too, and it is what [docs/patterns.md](docs/patterns.md) is written in:
`noul`, `choice` and `score` as the three primitives; a state plus a set of typed questions as
the unit of work; speculative fan-out — ask everything at once because one question costs what
ten cost; confidence-gated routing, where a middle band goes to a human or a larger model;
composite scoring from several cheap signals; and coarse-to-fine intent routing when the label
set does not fit. Our examples supply the question sets, the thresholds and the measurements.
The shapes they follow are not ours.

## Projects the examples learned from

No code was copied from any of these, and none of them is a dependency. They are credited
because the designs are recognisably theirs and we read them before writing ours.

**[codaaiteam/jev-mcp](https://github.com/codaaiteam/jev-mcp)** and
**[parksjr/typesafe-mcp](https://github.com/parksjr/typesafe-mcp)** — the tool shape that
`integrations/mcp/arbiter_mcp.py` exposes: classify, score, check, gate and decide as five
primitives rather than one generic endpoint, with the raw call kept available underneath. That
factoring is the good idea, and it came from them.

**[shivam2003-dev/typesafe-triage-guard](https://github.com/shivam2003-dev/typesafe-triage-guard)**
— the pipelines behind `examples/support_triage.py`, `examples/alert_triage.py` and
`examples/pr_risk_gate.py`: fan out a fixed question set over the incoming item, then let the
caller's own thresholds pick the route, so the policy stays in code that can be reviewed and
changed without touching the model. Every threshold in `examples/` is a constant at the top of
the file for that reason.

## The toolchain

**[PyTorch](https://pytorch.org)** — including the CUDA 13.0 wheel index that makes the same
`./run.sh setup` work on aarch64 and x86_64.
**[Hugging Face](https://huggingface.co)** — `transformers` for the encoder implementations,
`tokenizers`, `safetensors` for the weights, and `huggingface_hub` for the download.
**[FastAPI](https://fastapi.tiangolo.com)** and **[uvicorn](https://www.uvicorn.org)** — the
HTTP surface.
**[Model Context Protocol](https://modelcontextprotocol.io)** — the `mcp` Python SDK, which
`integrations/mcp/arbiter_mcp.py` is built on and which its tests drive over stdio.
**[Instrument Sans](https://fonts.google.com/specimen/Instrument+Sans)** (Rodrigo Fuenzalida,
Jordan Egstad) and **[JetBrains Mono](https://www.jetbrains.com/lp/mono/)** — the playground
loads both from Google Fonts; both are SIL Open Font License 1.1.

---

MIT licensed, except where noted above. The model weights are Apache-2.0 and carry their own
terms; third-party components keep their own licences.
