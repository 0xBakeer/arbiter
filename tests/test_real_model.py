"""Opt-in: the same contract against the real checkpoints.

Skipped unless LAYA_MODELS_DIR points at a downloaded `convaiinnovations/laya` tree, because it
needs 2.4 GB of weights. Runs on CPU by default; set DEVICE=cuda to exercise the GPU path.
"""
import os

import pytest

MODELS_DIR = os.environ.get("LAYA_MODELS_DIR")
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR and os.path.isdir(MODELS_DIR)),
    reason="set LAYA_MODELS_DIR to a downloaded convaiinnovations/laya tree to run this")

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
    from server.engine import Checkpoint

    return Checkpoint("english", MODELS_DIR, device=os.environ.get("DEVICE", "cpu"), mode="eager")


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


def test_too_many_options_is_refused_rather_than_truncated(checkpoint):
    from server.errors import OptionBudgetError

    huge = {"dept": {"type": "choice", "instructions": "Route it.",
                     "criteria": {"option number %d" % i: "a fairly long description of it"
                                  for i in range(120)}}}
    with pytest.raises(OptionBudgetError):
        checkpoint.build_rows(STATE, huge)
