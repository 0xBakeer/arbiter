# Credits

Nearly everything that makes this work is somebody else's. What is ours is the serving layer —
the micro-batcher, the CUDA-graph bucketing, the equivalence harness, the routing surface — the
measurements, and the documentation.

## The model

**[Convai Innovations](https://huggingface.co/convaiinnovations)**, and
**Nandha Kishor M** ([@NandhaKishorM](https://github.com/NandhaKishorM)) — Laya itself: the
three checkpoints, the decision-head architecture, the per-bucket temperature calibration, the
`laya` SDK on PyPI, and the benchmark numbers this recipe measures itself against. Apache-2.0.

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

## The API shape

**[TypeSafe](https://typesafe.ai)** — the Jev System One request and response contract, which
this server implements so that code written against Jev runs against a local Laya without
changing anything but the base URL and the key. The contract is theirs; this is a compatible
implementation, not an affiliated or endorsed one.

The vocabulary the examples and [docs/patterns.md](docs/patterns.md) are written in is theirs
too — `noul`, `choice` and `score` as the three primitives, a state plus a set of typed
questions as the unit of work, and speculative fan-out as the way to use them. The examples and
the thresholds in them are ours; the way of thinking about the problem is not.

## The toolchain

**[PyTorch](https://pytorch.org)** — including the CUDA 13.0 wheel index that makes the same
`./run.sh setup` work on aarch64 and x86_64.
**[Hugging Face](https://huggingface.co)** — `transformers`, `tokenizers`, `safetensors`, and
`huggingface_hub` for the download.
**[FastAPI](https://fastapi.tiangolo.com)** and **[uvicorn](https://www.uvicorn.org)** — the
HTTP surface.
**[Model Context Protocol](https://modelcontextprotocol.io)** — the `mcp` Python SDK, which
`integrations/mcp/laya_mcp.py` is built on and which its tests drive over stdio.

---

MIT licensed, except where noted above. The model weights are Apache-2.0 and carry their own
terms; third-party components keep their own licences.
