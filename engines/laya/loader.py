"""The Laya backend: the SDK's checkpoints and router behind the server's engine interface.

Three things make this faster than calling `laya.Agent.system_one` per request:

1. **Half-precision parameters instead of fp32 + autocast.** The checkpoints were trained in bf16
   (`"amp_dtype": "bf16"` in every `rl_agent_config.json`), and the SDK loads fp32 weights and
   casts them on every forward under `torch.autocast`. Converting once at load removes the cast
   and a third of the weight traffic. Logits and softmax stay in fp32, exactly as the SDK does.
2. **One forward per batch, not per request.** Every question is already its own row, so rows
   from different HTTP requests collate into one batch with no change to what each row computes.
3. **CUDA-graph replay** for bucketed shapes, because at these sizes the model is launch-bound:
   the encoder is 22-28 layers of small kernels and the host cannot issue them fast enough.

Correctness note on batching: rows are independent. Attention is masked per row, the head's
`src_key_padding_mask` is per row, and the marker gather is per row, so padding a batch with
neutral rows cannot change a real row's logits beyond floating-point reassociation. The size of
that reassociation is measured, not assumed, by `tools/equivalence.py`.

The batching itself is not here: rows are handed to `server.engine.Batcher`, which knows nothing
about what is in them. This file is everything that is specific to Laya.
"""
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from laya.common import (
    QTYPES,
    build_sequence,
    collate_items,
    confidence_from_probs,
    render_options,
    temp_bucket,
)

from server.engine import Engine
from server.errors import OptionBudgetError

# The three checkpoints bundled in the `convaiinnovations/laya` repo, and where each one lives
# under the downloaded tree.
SUBFOLDER = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
MODEL_NAMES = tuple(SUBFOLDER)


# --------------------------------------------------------------------------- device and dtype

def resolve_device(spec: str = "auto") -> str:
    """`ARBITER_DEVICE`, or the best accelerator this machine actually has.

    The same checkout serves a CUDA box and a Mac, and the only honest default is the one the
    hardware decides. An explicit value is passed straight through, including a bad one, so that
    `ARBITER_DEVICE=cuda` on a machine without CUDA fails loudly instead of quietly running a
    tenth as fast on the CPU.
    """
    if spec and spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def supported_dtypes(device: torch.device) -> Tuple[str, ...]:
    """The parameter modes worth running on this device. Everything else collapses to fp32.

    CUDA gets bf16 from compute capability 8.0 up, where there are bf16 tensor cores, and
    `autocast` because that is the mode the SDK itself runs in. MPS gets whole-model fp16 and
    bf16 -- both are native on Apple silicon -- but not autocast: torch's MPS autocast is fp16
    and covers a different op set, which would make `ARBITER_DTYPE=autocast` mean two different
    arithmetics on the two machines. On CPU every half dtype is slower than fp32 at this size.
    """
    if device.type == "cuda":
        return ("fp32", "autocast", "bf16") if torch.cuda.get_device_capability(device)[0] >= 8 else ("fp32",)
    if device.type == "mps":
        return ("fp32", "fp16", "bf16")
    return ("fp32",)


PARAM_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


# --------------------------------------------------------------------------- request shaping

def to_internal(qdef: Dict[str, Any]) -> Dict[str, Any]:
    """Jev-shaped question definition -> the SDK's internal `{t, ins, crit}` form."""
    import json

    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


# --------------------------------------------------------------------------- CUDA graphs

class GraphRunner:
    """Capture and replay the forward for bucketed (batch, sequence) shapes.

    One graph per (batch, sequence, marker) bucket. The marker dimension is bucketed rather than
    pinned at `max_markers` because the scorer runs on every marker slot: padding a two-option
    question out to 32 slots puts a 1024x1024 MLP over sixteen times the positions it needs, and
    that costs more than the launch overhead the graph is here to remove. A question with more
    options than `max_markers` takes the eager path.

    Graphs share one memory pool and are captured lazily on first use of a bucket, so a server
    that only ever sees single two-option questions pays for exactly one graph.
    """

    # Fine ladders on purpose. A graph's cost is the padded shape, not the real one, so a coarse
    # ladder taxes every request that lands just above a step: 130 real tokens rounded to 256 is
    # a doubling of the encoder's work, which is far more than the launch overhead being saved.
    BATCH_BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    SEQ_BUCKETS = (64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024)
    MARKER_BUCKETS = (2, 3, 4, 6, 8, 12, 16, 24, 32)

    def __init__(self, model, device: torch.device, pad_id: int, cls_id: int,
                 max_len: int, max_markers: int = 32):
        self.model = model
        self.device = device
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.max_markers = max_markers
        self.seq_buckets = tuple(s for s in self.SEQ_BUCKETS if s <= max_len) or (max_len,)
        self.marker_buckets = tuple(k for k in self.MARKER_BUCKETS if k <= max_markers) or (max_markers,)
        self.pool = torch.cuda.graph_pool_handle()
        self._graphs: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def bucket_for(self, n_rows: int, seq_len: int, n_markers: int) -> Optional[Tuple[int, int, int]]:
        b = next((x for x in self.BATCH_BUCKETS if x >= n_rows), None)
        s = next((x for x in self.seq_buckets if x >= seq_len), None)
        k = next((x for x in self.marker_buckets if x >= n_markers), None)
        if b is None or s is None or k is None:
            return None
        return b, s, k

    def _capture(self, key: Tuple[int, int, int]) -> Dict[str, Any]:
        b, s, k = key
        dev = self.device
        static = {
            "input_ids": torch.full((b, s), self.pad_id, dtype=torch.long, device=dev),
            "attention_mask": torch.ones((b, s), dtype=torch.long, device=dev),
            "marker_pos": torch.zeros((b, k), dtype=torch.long, device=dev),
            "marker_mask": torch.zeros((b, k), dtype=torch.bool, device=dev),
            "qtype": torch.zeros((b,), dtype=torch.long, device=dev),
        }
        static["input_ids"][:, 0] = self.cls_id
        static["marker_mask"][:, :2] = True

        # Warm up on a side stream first: capture records whatever the allocator and cuBLAS do
        # on the very first call, and that includes one-off workspace allocations.
        stream = torch.cuda.Stream(device=dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.model(**static)
        torch.cuda.current_stream(dev).wait_stream(stream)
        torch.cuda.synchronize(dev)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out_logits, out_act = self.model(**static)
        return {"graph": graph, "static": static, "logits": out_logits, "act": out_act}

    def run(self, batch: Dict[str, torch.Tensor], key: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._lock:
            entry = self._graphs.get(key)
            if entry is None:
                entry = self._graphs[key] = self._capture(key)
        st = entry["static"]
        n, L = batch["input_ids"].shape
        k = batch["marker_pos"].shape[1]

        # Reset to the neutral padding state, then write the real rows in. A pad row is one
        # token of attention and no live markers, which is well defined everywhere and, unlike
        # an all-zero attention mask, does not produce NaNs.
        st["input_ids"].fill_(self.pad_id)
        st["input_ids"][:, 0] = self.cls_id
        st["attention_mask"].zero_()
        st["attention_mask"][:, 0] = 1
        st["marker_pos"].zero_()
        st["marker_mask"].zero_()
        st["qtype"].zero_()

        st["input_ids"][:n, :L].copy_(batch["input_ids"], non_blocking=True)
        st["attention_mask"][:n, :L].copy_(batch["attention_mask"], non_blocking=True)
        st["marker_pos"][:n, :k].copy_(batch["marker_pos"], non_blocking=True)
        st["marker_mask"][:n, :k].copy_(batch["marker_mask"], non_blocking=True)
        st["qtype"][:n].copy_(batch["qtype"], non_blocking=True)

        entry["graph"].replay()
        return entry["logits"][:n, :k].clone(), entry["act"][:n].clone()

    @property
    def captured(self) -> List[Tuple[int, int]]:
        return sorted(self._graphs)


# --------------------------------------------------------------------------- one checkpoint

class Checkpoint:
    """One loaded Laya checkpoint plus everything needed to run and decode a batch of rows.

    `dtype_mode` picks how half precision is used, and the modes are not equivalent:

      autocast  fp32 parameters under `torch.autocast(bf16)`, which is what the SDK does. Matmuls
                run in bf16; layer norms, the residual stream and the softmaxes stay fp32.
      bf16      bf16 parameters and no autocast. Everything runs in bf16, including the residual
                stream through 28 layers, which is where the two diverge.
      fp16      the same, in fp16: more mantissa than bf16 and far less range, so it is the one
                that can overflow rather than merely round.
      fp32      no half precision anywhere.

    A mode the device cannot use collapses to fp32; `supported_dtypes` says which those are.
    Measured, the difference is not decorative: see the equivalence table in README.md.
    """

    def __init__(self, name: str, models_dir: str, device: str = "auto", mode: str = "eager",
                 max_markers: int = 32, dtype_mode: str = "autocast"):
        from laya.agent import Agent

        device = resolve_device(device)
        # Before the weights, not after: loading three checkpoints to then refuse the mode they
        # were loaded for is a minute of someone's time for a one-line mistake.
        if mode == "graphs" and not device.startswith("cuda"):
            raise ValueError("ARBITER_MODE=graphs is CUDA graph capture and cannot run on %s; "
                             "use ARBITER_MODE=eager" % device)

        self.name = name
        self.agent = Agent(models_dir, device=device, subfolder=SUBFOLDER[name])
        self.device = self.agent.device
        self.tok = self.agent.tok
        self.cfg = self.agent.cfg
        self.model = self.agent.model
        self.max_len = int(self.cfg.get("max_len", 512))
        self.head_max_len = int(self.cfg.get("head_max_len", 192))
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.pad_id = self.tok.pad_token_id
        self.cls_id = self.tok.cls_token_id

        # ModernBERT's `reference_compile` defaults to "auto", which means torch.compile as soon
        # as the model is on CUDA. The SDK already turns it off; set it again here because it is
        # the difference between an eager forward and an inductor build, and this path must stay
        # eager -- graphs mode captures the eager kernels with torch.cuda.CUDAGraph and would
        # otherwise be capturing compiled ones.
        for module in (self.model.encoder, getattr(self.model.encoder, "model", None)):
            cfg = getattr(module, "config", None)
            if cfg is not None:
                cfg.reference_compile = False

        # Half-precision parameters, no autocast, and only where the device has something to
        # run them on. Anything else is fp32, which is what the SDK does on those devices too.
        self.dtype_mode = dtype_mode if dtype_mode in supported_dtypes(self.device) else "fp32"
        self.autocast = self.dtype_mode == "autocast"
        self.param_dtype = PARAM_DTYPES.get(self.dtype_mode, torch.float32)
        if self.param_dtype is not torch.float32:
            self.model.to(self.param_dtype)
            # The act head is the one module that must stay fp32: it is fed
            # `torch.cat([h[:, 0].float(), feats])`, where `feats` is built from the fp32 logits,
            # so a half-precision weight meets an fp32 activation. Autocast hid that by casting
            # the input; without autocast it is a hard dtype error. It is three small Linears,
            # and keeping it fp32 makes the act probability marginally closer to the reference,
            # not further.
            self.model.act_head.to(torch.float32)
        self.model.eval()

        self.graphs: Optional[GraphRunner] = None
        if mode == "graphs":
            self.graphs = GraphRunner(self.run_model, self.device, self.pad_id, self.cls_id,
                                      self.max_len, max_markers)

    def run_model(self, **kw):
        """The forward, under autocast or not, so every path runs the same arithmetic.

        `cache_enabled=False` because autocast's weight cache and CUDA-graph capture do not mix:
        the cache would hand a replay a tensor that belonged to the capture.
        """
        if not self.autocast:
            return self.model(**kw)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, cache_enabled=False):
            return self.model(**kw)

    # -- shaping -----------------------------------------------------------------
    def build_rows(self, state, questions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """One row per question, in the order the ids were given."""
        rows = []
        for qid, qdef in questions.items():
            q = to_internal(qdef)
            seq, markers = build_sequence(self.tok, state, q, self.max_len, self.head_max_len)
            if len(markers) != len(render_options(q)):
                raise OptionBudgetError(
                    "question %r has more options than head_max_len=%d can hold; split it or "
                    "raise head_max_len" % (qid, self.head_max_len))
            rows.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "qid": qid, "q": q})
        return rows

    # -- running -----------------------------------------------------------------
    @torch.no_grad()
    def forward(self, rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, int, str]:
        """Run one collated batch. Returns (logits, act probabilities, input tokens, path)."""
        b = collate_items([rows], self.pad_id)
        n_tokens = int(b["attention_mask"].sum())
        n, L = b["input_ids"].shape
        k = b["marker_pos"].shape[1]

        key = self.graphs.bucket_for(n, L, k) if self.graphs is not None else None
        if key is not None:
            logits, act = self.graphs.run(
                {name: b[name].to(self.device, non_blocking=True)
                 for name in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")},
                key)
            path = "graph:%dx%dx%d" % key
        else:
            logits, act = self.run_model(
                input_ids=b["input_ids"].to(self.device),
                attention_mask=b["attention_mask"].to(self.device),
                marker_pos=b["marker_pos"].to(self.device),
                marker_mask=b["marker_mask"].to(self.device),
                qtype=b["qtype"].to(self.device),
            )
            path = "eager"

        logits = logits.float().cpu().numpy()
        act = torch.softmax(act.float(), -1).cpu().numpy()
        return logits, act, n_tokens, path

    # -- decoding ----------------------------------------------------------------
    def answer(self, row: Dict[str, Any], logits_row: np.ndarray, act_row: np.ndarray) -> Dict[str, Any]:
        """Turn one row of logits into the Jev answer object for that question."""
        q = row["q"]
        k = len(row["markers"])
        qt = row["qtype"]
        t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
        z = logits_row[:k] / max(1e-3, float(t_scale))
        p = np.exp(z - z.max())
        p = p / p.sum()

        conf = round(confidence_from_probs(p, k), 4)
        ext = {"act_probability": round(float(act_row[0]), 4)}

        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            return {
                "type": "choice",
                "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                "confidence": conf,
                "action": ext,
            }
        if q["t"] == "score":
            return {
                "type": "score",
                "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": conf,
                "action": ext,
            }
        return {
            "type": "noul",
            "noul": round(float(p[1]), 4),
            "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
            "action": ext,
        }


# --------------------------------------------------------------------------- the backend

class LayaEngine(Engine):
    """Every selected checkpoint, the SDK's router over them, and the decoding of a batch."""

    def __init__(self, models_dir: str = "models/laya", names: Tuple[str, ...] = MODEL_NAMES,
                 device: str = "auto", mode: str = "eager", max_batch: int = 64,
                 wait_ms: float = 2.0, max_queue: int = 256, max_markers: int = 32,
                 dtype_mode: str = "autocast"):
        from laya.router import Router

        self.models_dir = models_dir
        self.device = resolve_device(device)
        self.mode = mode
        self.dtype_mode = dtype_mode
        self.max_markers = max_markers
        self.ckpts: Dict[str, Checkpoint] = {}
        self.router = Router(device=self.device, max_loaded=len(names) or 1)
        super().__init__(names, max_batch=max_batch, wait_ms=wait_ms, max_queue=max_queue)
        self.default = "english" if "english" in self.ckpts else next(iter(self.ckpts))
        self.router.default = self.default

    # -- the engine interface ----------------------------------------------------
    def load(self, name: str) -> None:
        ck = Checkpoint(name, self.models_dir, device=self.device, mode=self.mode,
                        max_markers=self.max_markers, dtype_mode=self.dtype_mode)
        # What the checkpoint settled on, which is what /readyz should report: a mode the device
        # cannot use collapses to fp32, and a server claiming autocast while running fp32 would
        # put the wrong label on every measurement taken against it.
        self.dtype_mode = ck.dtype_mode
        self.ckpts[name] = ck
        # Hand the already-built Agent to the router so it never loads a second copy.
        self.router.attach(name, ck.agent)

    def build_rows(self, name: str, state, questions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.ckpts[name].build_rows(state, questions)

    def predict_rows(self, name: str, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[int], str]:
        ck = self.ckpts[name]
        logits, act, _, path = ck.forward(rows)
        answers = [ck.answer(row, logits[i], act[i]) for i, row in enumerate(rows)]
        return answers, [len(row["ids"]) for row in rows], path

    def route(self, state, questions, model=None, task=None, lang=None) -> Dict[str, Any]:
        """The router's decision, clamped to what this process actually has loaded.

        `ARBITER_MODELS` may select a subset, and a request that routes to a checkpoint outside
        it should still be answered -- by the default one, saying so -- rather than rejected.
        """
        decision = dict(self.router.route(state, questions, model=model, task=task, lang=lang))
        if decision["model"] not in self.ckpts:
            decision["requested"] = decision["model"]
            decision["reason"] = "%s; %r is not loaded here, falling back to %s" % (
                decision["reason"], decision["model"], self.default)
            decision["model"] = self.default
        return decision


def from_env() -> LayaEngine:
    names = tuple(n.strip() for n in os.environ.get("ARBITER_MODELS", ",".join(MODEL_NAMES)).split(",") if n.strip())
    unknown = [n for n in names if n not in SUBFOLDER]
    if unknown:
        raise ValueError("ARBITER_MODELS contains unknown checkpoints: %s" % unknown)
    return LayaEngine(
        models_dir=os.environ.get("ARBITER_MODELS_DIR", "models/laya"),
        names=names,
        device=os.environ.get("ARBITER_DEVICE", "auto"),
        mode=os.environ.get("ARBITER_MODE", "eager"),
        max_batch=int(os.environ.get("ARBITER_MAX_BATCH", "64")),
        wait_ms=float(os.environ.get("ARBITER_BATCH_WAIT_MS", "2")),
        max_queue=int(os.environ.get("ARBITER_MAX_QUEUE", "256")),
        max_markers=int(os.environ.get("ARBITER_GRAPH_MAX_MARKERS", "32")),
        dtype_mode=os.environ.get("ARBITER_DTYPE", "autocast"),
    )
