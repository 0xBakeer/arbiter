"""The generic half of the engine: what a backend gets for free by implementing four methods.

The backend here is the smallest one that satisfies the interface in `engines/README.md`, which
is also the point of the test: loading, batching, the queue cap, naming the answers and the
warm-up are the server's, not the model's.
"""
import importlib.util

import pytest

from server.engine import Engine, engine_from_env
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


def test_the_configured_engine_is_a_package_under_engines(monkeypatch):
    """The backend is selected by name, not by an import the server carries."""
    assert importlib.util.find_spec("engines.laya.loader") is not None

    monkeypatch.setenv("ARBITER_ENGINE", "nope")
    with pytest.raises(ModuleNotFoundError, match="engines.nope"):
        engine_from_env()


# -- the MLX backend, on a machine that may have no MLX at all -------------------------------
#
# Everything laya-mlx supplies is imported by `port()` on first use, so the module itself loads
# anywhere and these run on CI, on Linux, and in this repo's own test suite on any machine.

def test_the_mlx_engine_is_dispatched_by_name_and_says_so(monkeypatch):
    loader = importlib.import_module("engines.laya_mlx.loader")
    built = {}

    class FakeEngine:
        def __init__(self, **kw):
            built.update(kw)

    monkeypatch.setattr(loader, "port", lambda: None)      # no MLX needed to test the dispatch
    monkeypatch.setattr(loader, "LayaMlxEngine", FakeEngine)
    monkeypatch.setenv("ARBITER_ENGINE", "laya_mlx")
    monkeypatch.setenv("ARBITER_MODELS", "english,multilingual")
    monkeypatch.setenv("ARBITER_DTYPE", "fp32")
    monkeypatch.delenv("ARBITER_MODELS_DIR", raising=False)

    eng = engine_from_env()
    assert isinstance(eng, FakeEngine)
    # what /readyz reports: the package that was selected, not a name a backend claims
    assert eng.engine == "laya_mlx"
    assert built["names"] == ("english", "multilingual")
    assert built["dtype_mode"] == "fp32"
    assert built["models_dir"] == "models/laya-mlx"      # its own weights, not the torch tree


def test_the_mlx_engine_refuses_a_checkpoint_it_has_no_repo_for(monkeypatch):
    loader = importlib.import_module("engines.laya_mlx.loader")
    monkeypatch.setattr(loader, "port", lambda: None)
    monkeypatch.setenv("ARBITER_ENGINE", "laya_mlx")
    monkeypatch.setenv("ARBITER_MODELS", "english,klingon")
    with pytest.raises(ValueError, match="klingon"):
        engine_from_env()


def test_the_mlx_engine_refuses_a_machine_that_is_not_apple_silicon(monkeypatch):
    """One line at startup. The alternative is an ImportError from inside a wheel that is not
    built for this architecture, several frames below anything the operator set."""
    loader = importlib.import_module("engines.laya_mlx.loader")
    monkeypatch.setattr(loader.platform, "system", lambda: "Linux")
    monkeypatch.setattr(loader.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="Apple silicon only"):
        loader.require_apple_silicon()


def test_mps_is_not_one_of_the_mlx_device_names():
    """`mps` is torch's name for the same GPU, and taking it here would serve the wrong backend
    under the right label. `auto` is the GPU, which is the only accelerator MLX has."""
    loader = importlib.import_module("engines.laya_mlx.loader")
    assert loader.resolve_device("auto") == "gpu"
    assert loader.resolve_device("metal") == "gpu"
    assert loader.resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="not an MLX device"):
        loader.resolve_device("mps")
