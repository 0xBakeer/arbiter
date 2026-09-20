"""Opt-in: the same contract against the real checkpoints.

Skipped unless ARBITER_MODELS_DIR points at a downloaded `convaiinnovations/laya` tree, because it
needs 2.4 GB of weights. Runs on whatever ARBITER_DEVICE resolves to -- cuda, mps or cpu -- so the same suite
covers the accelerator it is run on.
"""
import os

import pytest

MODELS_DIR = os.environ.get("ARBITER_MODELS_DIR")
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR and os.path.isdir(MODELS_DIR)),
    reason="set ARBITER_MODELS_DIR to a downloaded convaiinnovations/laya tree to run this")

QUESTIONS = {
    "angry": {"type": "noul", "instructions": "The customer is angry.", "criteria": None},
    "queue": {"type": "choice", "instructions": "Which queue should this go to?",
              "criteria": {"billing": "money", "technical": "it does not work"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["low", "medium", "high"]},
}
STATE = "I was charged twice for the same subscription and nobody has replied to my emails."


@pytest.fixture(scope="module")
def checkpoint():
    from engines.laya.loader import Checkpoint

    return Checkpoint("english", MODELS_DIR, device=os.environ.get("ARBITER_DEVICE", "auto"), mode="eager")


def test_one_forward_answers_every_question(checkpoint):
    rows = checkpoint.build_rows(STATE, QUESTIONS)
    assert len(rows) == 3
    logits, act, tokens, path = checkpoint.forward(rows)
    assert logits.shape[0] == 3
    assert tokens > 0
    answers = {rows[i]["qid"]: checkpoint.answer(rows[i], logits[i], act[i]) for i in range(3)}
    assert answers["angry"]["type"] == "noul"
    assert answers["queue"]["choice"] in ("billing", "technical")
    assert 0.0 <= answers["urgency"]["score"] <= 2.0
    assert abs(sum(answers["queue"]["probabilities"].values()) - 1.0) < 1e-3


def test_batching_does_not_change_an_answer(checkpoint):
    alone = checkpoint.build_rows(STATE, {"angry": QUESTIONS["angry"]})
    l1, a1, _, _ = checkpoint.forward(alone)
    one = checkpoint.answer(alone[0], l1[0], a1[0])

    padded = checkpoint.build_rows(STATE, QUESTIONS) + checkpoint.build_rows(
        "A completely unrelated and much longer state about shipping delays. " * 5, QUESTIONS)
    l2, a2, _, _ = checkpoint.forward(padded)
    together = checkpoint.answer(padded[0], l2[0], a2[0])

    assert abs(one["noul"] - together["noul"]) < 5e-3


def choice_with(n):
    return {"dept": {"type": "choice", "instructions": "Route it.",
                     "criteria": {"option number %d" % i: "a fairly long description of it"
                                  for i in range(n)}}}


def test_too_many_options_is_refused_rather_than_truncated(checkpoint):
    """Jev allows 255 options; this checkpoint's 512-token context cannot hold their markers.

    `build_sequence` squeezes each option to 4 tokens and then drops the markers that fall past
    `max_len`, which would silently answer a different question. The refusal is the point.
    """
    from server.errors import OptionBudgetError

    with pytest.raises(OptionBudgetError, match="head_max_len"):
        checkpoint.build_rows(STATE, choice_with(255))


def test_the_largest_option_set_that_still_fits(checkpoint):
    rows = checkpoint.build_rows(STATE, choice_with(124))
    assert len(rows[0]["markers"]) == 124
    assert len(rows[0]["ids"]) <= checkpoint.max_len


def test_the_backend_answers_through_the_engine_interface():
    """The other end of the split: `LayaEngine` under the generic batcher, start to finish."""
    from engines.laya.loader import LayaEngine

    engine = LayaEngine(MODELS_DIR, names=("english",), device=os.environ.get("ARBITER_DEVICE", "auto"),
                        wait_ms=0)
    try:
        assert engine.checkpoints() == ["english"]
        assert engine.route(STATE, QUESTIONS)["model"] == "english"
        out = engine.infer("english", STATE, QUESTIONS)
        assert list(out["answers"]) == ["angry", "queue", "urgency"]
        assert out["answers"]["queue"]["choice"] in ("billing", "technical")
        assert out["input_tokens"] > 0
        assert out["batch_rows"] == 3
    finally:
        engine.close()
