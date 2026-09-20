"""The Laya-MLX backend: the same three checkpoints, run by the MLX port instead of torch.

`engines/laya` is torch: on a Mac that is the MPS backend dispatching the op graph it dispatches
on CUDA through Metal shaders it did not pick the layout for. `laya-mlx` is an independent
reimplementation of the ModernBERT encoder, the marker head and the act head in MLX, over
converted copies of the same weights -- arrays in unified memory, no host/device copy, kernels
written for the M-series. What it is not is a different model: the prompt construction, the
per-bucket temperature calibration and the answer schema are the upstream SDK's, and this file
reuses them from the port (`laya_mlx.common`) rather than restating them.

Three things to know about the shape of this backend against `engines/laya`:

1. **Batching is the same.** The port takes a padded batch (`laya_mlx.agent.collate_items`), so
   rows from different HTTP requests collate into one forward exactly as they do on torch. The
   port's own `Agent.system_one` chunks at `batch_size=16`; this file does not use it -- rows
   come from `server.engine.Batcher` and go straight to `Agent.forward`.
2. **There is no autocast and no graph capture.** MLX runs whole-model fp32, fp16 or bf16;
   `ARBITER_DTYPE` names which, and a name that does not exist here (`autocast`, a CUDA mode)
   resolves to the shipped default rather than pretending. Which one ships is an equivalence
   result, not a preference: see `recipes/apple/README.md`.
3. **Routing is the SDK's, through the port's copy of it.** `laya_mlx.router` is upstream's
   router with the loader swapped, so the decision for a given input is the same one
   `engines/laya` makes; `tools/equivalence_mlx.py` checks that rather than assuming it.

Everything MLX is imported late, by `port()`. This module is imported by `engine_from_env` on
whatever machine the server runs on, and "wrong machine" has to be one clear line and not an
ImportError out of the middle of a wheel that only exists for arm64 Macs.
"""
import os
import platform
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

from server.engine import Engine
from server.errors import OptionBudgetError

# The converted checkpoints, and the Hub repo each one is downloaded from. A directory of the
# same name under `models_dir` wins, which is what `./run.sh setup` puts there.
HF_REPOS = {
    "english": "aac6fef/laya-mlx",
    "multilingual": "aac6fef/laya-multilingual-mlx",
    "typed-decisions": "aac6fef/laya-typed-decisions-mlx",
}
MODEL_NAMES = tuple(HF_REPOS)

# The port's dtype names, under the names the rest of the server uses. There is no `autocast`:
# MLX has no per-op cast mode, and inventing one would put the same label on two arithmetics.
PARAM_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}

# How much freed GPU memory MLX may hold for reuse. Its own default is unbounded, which is fine
# for a script and not for a server: the batcher forms a different shape for nearly every batch,
# and over one bench run here the retained buffers reached 12 GB on a 32 GB machine. The bound
# hands the excess back to the OS and costs nothing measurable -- see `recipes/apple/README.md`.
DEFAULT_CACHE_MB = 1024

# fp32 ships, even though fp16 is the port's own default and half the size: measured against the
# SDK on the 22 equivalence questions, fp32 here is identical to the reference at every decimal a
# client can see, and fp16 moves a multilingual probability by 1.7e-2. The table is in
# `bench/results.md`; change this line only with a new one.
DEFAULT_DTYPE = "fp32"

_PORT = None


def require_apple_silicon() -> None:
    """Refuse a machine MLX does not exist for, in one line, before any weight is touched."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "ARBITER_ENGINE=laya_mlx needs MLX, which is Apple silicon only; this is %s/%s -- "
            "use ARBITER_ENGINE=laya" % (platform.system(), platform.machine()))


def port() -> SimpleNamespace:
    """The MLX port's pieces, imported on first use rather than at import time."""
    global _PORT
    if _PORT is None:
        require_apple_silicon()
        import mlx.core as mx

        from laya_mlx import Agent, Router
        from laya_mlx.agent import collate_items
        from laya_mlx.common import (
            QTYPES,
            build_sequence,
            confidence_from_probs,
            render_options,
            temp_bucket,
        )

        _PORT = SimpleNamespace(
            mx=mx, Agent=Agent, Router=Router, collate_items=collate_items, QTYPES=QTYPES,
            build_sequence=build_sequence, confidence_from_probs=confidence_from_probs,
            render_options=render_options, temp_bucket=temp_bucket)
    return _PORT


# --------------------------------------------------------------------------- device and dtype

def resolve_device(spec: str = "auto") -> str:
    """`ARBITER_DEVICE` as an MLX device. MLX has two of them, and `mps` is not one of the names.

    `auto` is the GPU, which is what `mx.default_device()` resolves to on every machine this
    engine will start on at all. An unknown value is refused rather than passed through, because
    the one people will actually type is `mps` -- torch's name for the same hardware -- and
    silently taking it would serve from the wrong backend under the right label.
    """
    if not spec or spec == "auto":
        return "gpu"
    if spec in ("gpu", "metal"):
        return "gpu"
    if spec == "cpu":
        return "cpu"
    raise ValueError("ARBITER_DEVICE=%r is not an MLX device; this engine has gpu and cpu. "
                     "(mps is torch's name for this GPU: ARBITER_ENGINE=laya serves it.)" % spec)


def supported_dtypes() -> Tuple[str, ...]:
    """The parameter modes MLX has. No device split: all three exist on every Apple GPU."""
    return tuple(PARAM_DTYPES)


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


# --------------------------------------------------------------------------- one checkpoint

class Checkpoint:
    """One converted checkpoint loaded into MLX, plus the shaping and decoding around it.

    `dtype_mode` is the dtype every parameter is cast to at load, and the three MLX has are the
    three offered: fp32, fp16 and bf16. A name MLX does not have -- `autocast`, which is what
    the torch lane defaults to -- resolves to `DEFAULT_DTYPE` rather than to an invention, and
    `/readyz` then reports what actually ran.
    """

    def __init__(self, name: str, models_dir: str, device: str = "gpu",
                 dtype_mode: str = DEFAULT_DTYPE):
        p = port()

        self.name = name
        self.device = resolve_device(device)
        self.dtype_mode = dtype_mode if dtype_mode in PARAM_DTYPES else DEFAULT_DTYPE
        local = os.path.join(models_dir, name)
        self.source = local if os.path.isdir(local) else HF_REPOS[name]
        self.agent = p.Agent(self.source, device=self.device,
                             dtype=PARAM_DTYPES[self.dtype_mode])
        self.tok = self.agent.tok
        self.cfg = self.agent.cfg
        self.max_len = int(self.cfg.get("max_len", 512))
        self.head_max_len = int(self.cfg.get("head_max_len", 192))
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.pad_id = self.tok.pad_token_id

    # -- shaping -----------------------------------------------------------------
    def build_rows(self, state, questions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """One row per question, in the order the ids were given."""
        p = port()
        rows = []
        for qid, qdef in questions.items():
            q = to_internal(qdef)
            seq, markers = p.build_sequence(self.tok, state, q, self.max_len, self.head_max_len)
            if len(markers) != len(p.render_options(q)):
                raise OptionBudgetError(
                    "question %r has more options than head_max_len=%d can hold; split it or "
                    "raise head_max_len" % (qid, self.head_max_len))
            rows.append({"ids": seq, "markers": markers, "qtype": p.QTYPES[q["t"]], "qid": qid,
                         "q": q})
        return rows

    # -- running -----------------------------------------------------------------
    def forward(self, rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, int, str]:
        """Run one collated batch. Returns (logits, act probabilities, input tokens, path)."""
        p = port()
        batch = p.collate_items(rows, self.pad_id, max_length=self.max_len)
        logits, act = self.agent.forward(batch)
        logits = np.asarray(logits, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        # fp16 has the range to overflow where fp32 and bf16 do not, and a NaN here would leave
        # the client with a well-formed answer made of nothing. The port guards its own path the
        # same way; this is that guard on ours.
        if not (np.isfinite(logits).all() and np.isfinite(act).all()):
            raise FloatingPointError(
                "non-finite outputs from the %s checkpoint in %s; rerun with ARBITER_DTYPE=fp32"
                % (self.name, self.dtype_mode))
        act = np.exp(act - act.max(-1, keepdims=True))
        act = act / act.sum(-1, keepdims=True)
        return logits, act, int(batch["attention_mask"].sum()), "eager"

    # -- decoding ----------------------------------------------------------------
    def answer(self, row: Dict[str, Any], logits_row: np.ndarray, act_row: np.ndarray) -> Dict[str, Any]:
        """Turn one row of logits into the Jev answer object for that question."""
        p = port()
        q = row["q"]
        k = len(row["markers"])
        qt = row["qtype"]
        t_scale = self.temperature_by_options.get(p.temp_bucket(qt, k), self.temperature[qt])
        z = logits_row[:k] / max(1e-3, float(t_scale))
        prob = np.exp(z - z.max())
        prob = prob / prob.sum()

        conf = round(p.confidence_from_probs(prob, k), 4)
        ext = {"act_probability": round(float(act_row[0]), 4)}

        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            return {
                "type": "choice",
                "choice": keys[int(prob.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, prob)},
                "confidence": conf,
                "action": ext,
            }
        if q["t"] == "score":
            return {
                "type": "score",
                "score": round(float((np.arange(k) * prob).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(prob)},
                "confidence": conf,
                "action": ext,
            }
        return {
            "type": "noul",
            "noul": round(float(prob[1]), 4),
            "confidence": round(max(float(prob[1]), 1.0 - float(prob[1])), 4),
            "action": ext,
        }


# --------------------------------------------------------------------------- the backend

class LayaMlxEngine(Engine):
    """Every selected checkpoint under MLX, the port's router over them, and the decoding."""

    def __init__(self, models_dir: str = "models/laya-mlx", names: Tuple[str, ...] = MODEL_NAMES,
                 device: str = "auto", max_batch: int = 64, wait_ms: float = 2.0,
                 max_queue: int = 256, dtype_mode: str = DEFAULT_DTYPE,
                 cache_mb: int = DEFAULT_CACHE_MB):
        p = port()
        if cache_mb:
            p.mx.set_cache_limit(int(cache_mb) * 1024 * 1024)

        self.models_dir = models_dir
        self.device = resolve_device(device)
        self.dtype_mode = dtype_mode if dtype_mode in PARAM_DTYPES else DEFAULT_DTYPE
        self.ckpts: Dict[str, Checkpoint] = {}
        self.router = p.Router(device=self.device, max_loaded=len(names) or 1,
                               dtype=PARAM_DTYPES[self.dtype_mode])
        super().__init__(names, max_batch=max_batch, wait_ms=wait_ms, max_queue=max_queue)
        self.default = "english" if "english" in self.ckpts else next(iter(self.ckpts))
        self.router.default = self.default

    # -- the engine interface ----------------------------------------------------
    def load(self, name: str) -> None:
        ck = Checkpoint(name, self.models_dir, device=self.device, dtype_mode=self.dtype_mode)
        self.ckpts[name] = ck
        # Hand the already-built Agent to the router so it never loads a second copy, and point
        # its repo field at the converted checkpoint that answered rather than at the upstream
        # bundle the port's table names.
        self.router.attach(name, ck.agent)
        self.router.models[name] = ck.source

    def build_rows(self, name: str, state, questions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.ckpts[name].build_rows(state, questions)

    def predict_rows(self, name: str, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[int], str]:
        ck = self.ckpts[name]
        logits, act, _, path = ck.forward(rows)
        answers = [ck.answer(row, logits[i], act[i]) for i, row in enumerate(rows)]
        return answers, [len(row["ids"]) for row in rows], path

    def route(self, state, questions, model=None, task=None, lang=None) -> Dict[str, Any]:
        """The router's decision, clamped to what this process actually has loaded.

        Identical to `engines/laya`, because it is the same router: the port adapted upstream's
        `laya/router.py` and `laya/lang.py` unchanged apart from formatting.
        """
        decision = dict(self.router.route(state, questions, model=model, task=task, lang=lang))
        if decision["model"] not in self.ckpts:
            decision["requested"] = decision["model"]
            decision["reason"] = "%s; %r is not loaded here, falling back to %s" % (
                decision["reason"], decision["model"], self.default)
            decision["model"] = self.default
        return decision


def from_env() -> LayaMlxEngine:
    port()                                  # the machine check first, before anything slower
    names = tuple(n.strip() for n in os.environ.get("ARBITER_MODELS", ",".join(MODEL_NAMES)).split(",") if n.strip())
    unknown = [n for n in names if n not in HF_REPOS]
    if unknown:
        raise ValueError("ARBITER_MODELS contains unknown checkpoints: %s" % unknown)
    return LayaMlxEngine(
        models_dir=os.environ.get("ARBITER_MODELS_DIR", "models/laya-mlx"),
        names=names,
        device=os.environ.get("ARBITER_DEVICE", "auto"),
        max_batch=int(os.environ.get("ARBITER_MAX_BATCH", "64")),
        wait_ms=float(os.environ.get("ARBITER_BATCH_WAIT_MS", "2")),
        max_queue=int(os.environ.get("ARBITER_MAX_QUEUE", "256")),
        dtype_mode=os.environ.get("ARBITER_DTYPE", DEFAULT_DTYPE),
        cache_mb=int(os.environ.get("ARBITER_MLX_CACHE_MB", DEFAULT_CACHE_MB)),
    )
