"""Opt-in: the same contract as `test_real_model.py`, against the converted MLX checkpoints.

Skipped unless MLX is installed and the converted tree is on disk, which is what
`ARBITER_ENGINE=laya_mlx ./run.sh setup` puts in `models/laya-mlx`. On CI, on Linux, and on an
Intel Mac this file is skipped and the dispatch tests in `test_engine.py` are what still run.
"""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("ARBITER_MLX_MODELS_DIR") or os.path.join(ROOT, "models", "laya-mlx")
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None or not os.path.isdir(MODELS_DIR),
    reason="needs MLX and the converted checkpoints: ARBITER_ENGINE=laya_mlx ./run.sh setup")

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
    from engines.laya_mlx.loader import Checkpoint

    return Checkpoint("english", MODELS_DIR)


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
    """The port takes a padded batch, so the server's cross-request batching applies here too;
    this is that claim on the real weights, with the padding coming from unrelated rows."""
    alone = checkpoint.build_rows(STATE, {"angry": QUESTIONS["angry"]})
    l1, a1, _, _ = checkpoint.forward(alone)
    one = checkpoint.answer(alone[0], l1[0], a1[0])

    padded = checkpoint.build_rows(STATE, QUESTIONS) + checkpoint.build_rows(
        "A completely unrelated and much longer state about shipping delays. " * 5, QUESTIONS)
    l2, a2, _, _ = checkpoint.forward(padded)
    together = checkpoint.answer(padded[0], l2[0], a2[0])

    assert abs(one["noul"] - together["noul"]) < 5e-3


def test_too_many_options_is_refused_rather_than_truncated(checkpoint):
    """The same refusal as the torch engine, and it has to be the same exception: the port
    raises a plain ValueError of its own, which the HTTP layer would not know how to answer."""
    from server.errors import OptionBudgetError

    questions = {"dept": {"type": "choice", "instructions": "Route it.",
                          "criteria": {"option number %d" % i: "a fairly long description of it"
                                       for i in range(255)}}}
    with pytest.raises(OptionBudgetError, match="head_max_len"):
        checkpoint.build_rows(STATE, questions)


def test_the_backend_answers_through_the_engine_interface():
    from engines.laya_mlx.loader import DEFAULT_DTYPE, LayaMlxEngine, supported_dtypes

    engine = LayaMlxEngine(MODELS_DIR, names=("english",), wait_ms=0)
    try:
        assert engine.checkpoints() == ["english"]
        assert engine.route(STATE, QUESTIONS)["model"] == "english"
        out = engine.infer("english", STATE, QUESTIONS)
        assert list(out["answers"]) == ["angry", "queue", "urgency"]
        assert out["answers"]["queue"]["choice"] in ("billing", "technical")
        assert out["input_tokens"] > 0
        assert out["batch_rows"] == 3
        # What /readyz reports has to be what ran, here as on the torch lane.
        assert engine.dtype_mode == DEFAULT_DTYPE
        assert engine.dtype_mode in supported_dtypes()
        assert engine.device == "gpu"
    finally:
        engine.close()


def test_a_dtype_mlx_does_not_have_falls_back_to_the_shipped_one():
    """`ARBITER_DTYPE=autocast` is the default on the torch lane and means nothing here, so the
    label has to say what actually ran."""
    from engines.laya_mlx.loader import DEFAULT_DTYPE, Checkpoint

    ck = Checkpoint("english", MODELS_DIR, dtype_mode="autocast")
    assert ck.dtype_mode == DEFAULT_DTYPE
