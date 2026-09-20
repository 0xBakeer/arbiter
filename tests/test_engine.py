"""The generic half of the engine: what a backend gets for free by implementing four methods.

The backend here is the smallest one that satisfies the interface in `engines/README.md`, which
is also the point of the test: loading, batching, the queue cap, naming the answers and the
warm-up are the server's, not the model's.
"""
import pytest

from server.engine import Engine
from server.errors import OverloadedError


class CountingEngine(Engine):
    """Answers every question with the checkpoint that answered it and the row's position."""

    def __init__(self, names=("english", "multilingual"), **kw):
        self.loads = []
        self.batches = []
        super().__init__(names, **kw)

    def load(self, name):
        self.loads.append(name)

    def build_rows(self, name, state, questions):
        return [{"qid": qid, "tokens": 1 + i} for i, qid in enumerate(questions)]

    def predict_rows(self, name, rows):
        self.batches.append((name, len(rows)))
        answers = [{"type": "noul", "noul": 0.5, "where": "%s/%s" % (name, r["qid"])} for r in rows]
        return answers, [r["tokens"] for r in rows], "eager"

    def route(self, state, questions, model=None, task=None, lang=None):
        return {"model": model or "english", "reason": "test"}


@pytest.fixture
def engine():
    eng = CountingEngine(wait_ms=0)
    yield eng
    eng.close()


def test_every_named_checkpoint_is_loaded_once_and_gets_a_batcher(engine):
    assert engine.loads == ["english", "multilingual"]
    assert engine.checkpoints() == ["english", "multilingual"]
    assert [b.thread.is_alive() for b in engine.batchers.values()] == [True, True]


def test_answers_come_back_under_the_question_ids_they_were_asked_with(engine):
    out = engine.infer("multilingual", "state", {"a": None, "b": None, "c": None})
    assert list(out["answers"]) == ["a", "b", "c"]
    assert [a["where"] for a in out["answers"].values()] == [
        "multilingual/a", "multilingual/b", "multilingual/c"]
    assert out["input_tokens"] == 1 + 2 + 3
    assert out["batch_rows"] == 3
    assert engine.batches == [("multilingual", 3)]


def test_a_request_that_would_overflow_the_queue_is_refused(engine):
    engine.max_queue = 2
    with pytest.raises(OverloadedError, match="queue is full"):
        engine.infer("english", "state", {"a": None, "b": None, "c": None})
    # and the refusal costs nothing: no rows were ever handed to the backend
    assert engine.batches == []


def test_the_warm_up_runs_one_question_on_every_checkpoint(engine):
    engine.warm()
    assert engine.batches == [("english", 1), ("multilingual", 1)]


def test_closing_stops_every_worker_thread(engine):
    engine.close()
    assert [b.thread.is_alive() for b in engine.batchers.values()] == [False, False]


def test_the_interface_is_what_a_backend_must_implement():
    class Nothing(Engine):
        pass

    with pytest.raises(NotImplementedError):
        Nothing(("english",))
