"""A stand-in Engine, so the HTTP contract can be tested without a GPU or 2.4 GB of weights.

It answers with fixed probabilities but in exactly the shapes the real engine produces, which is
what the response-shape assertions are checking.
"""
from typing import Any, Dict, List

from server.errors import OptionBudgetError, OverloadedError


def _answer(qdef: Dict[str, Any]) -> Dict[str, Any]:
    t = qdef["type"]
    if t == "noul":
        return {"type": "noul", "noul": 0.75, "confidence": 0.75,
                "action": {"act_probability": 0.9}}
    if t == "choice":
        crit = qdef["criteria"]
        keys = list(crit) if isinstance(crit, dict) else list(crit)
        p = round(1.0 / len(keys), 4)
        return {"type": "choice", "choice": keys[0],
                "probabilities": {k: p for k in keys}, "confidence": 0.5,
                "action": {"act_probability": 0.9}}
    levels = qdef["criteria"]
    p = round(1.0 / len(levels), 4)
    return {"type": "score", "score": 1.0,
            "legend": {str(i): c for i, c in enumerate(levels)},
            "probabilities": {str(i): p for i in range(len(levels))},
            "confidence": 0.5, "action": {"act_probability": 0.9}}


class StubEngine:
    mode = "eager"
    dtype_mode = "autocast"

    def __init__(self, loaded: List[str] = None, fail: str = None):
        self._loaded = loaded or ["english", "multilingual", "typed-decisions"]
        self.fail = fail
        self.queue_depth = 0
        self.calls = []

    def checkpoints(self):
        return list(self._loaded)

    def close(self):
        pass

    def route(self, state, questions, model=None, task=None, lang=None):
        if model is not None:
            name, reason = model, "explicit model=%r" % model
        elif task is not None:
            name, reason = "typed-decisions", "explicit task=%r" % task
        elif lang is not None:
            name = "english" if str(lang).startswith("en") else "multilingual"
            reason = "explicit lang=%r" % lang
        elif not str(state).isascii():
            name, reason = "multilingual", "non-Latin script"
        else:
            name, reason = "english", "English Latin text"
        if name not in self._loaded:
            reason = "%s; %r is not loaded here" % (reason, name)
            name = self._loaded[0]
        return {"model": name, "repo": "stub", "reason": reason, "detection": None, "workflow": None}

    def infer(self, name, state, questions):
        self.calls.append((name, list(questions)))
        if self.fail == "overload":
            raise OverloadedError("queue is full (256 question rows in flight, limit 256)")
        if self.fail == "budget":
            raise OptionBudgetError("question 'x' has more options than head_max_len=192 can hold")
        return {"answers": {qid: _answer(q) for qid, q in questions.items()},
                "input_tokens": 42 * len(questions), "batch_rows": len(questions),
                "path": "eager"}
