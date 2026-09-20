"""The micro-batcher: rows from concurrent requests share a forward and still come back right.

This is the part of the serving path with a real chance of silently mixing two callers' answers,
so it is tested against a fake backend whose answers carry the tag of the row they came from.
"""
import threading
import time

from server.engine import Batcher


class FakeEngine:
    """Answers with the row's own tag, so misattribution is detectable rather than plausible."""

    def __init__(self):
        self.forwards = []
        self.lock = threading.Lock()

    def build_rows(self, name, state, questions):
        return [{"tokens": 3, "tag": "%s/%s" % (state, qid)} for qid in questions]

    def predict_rows(self, name, rows):
        with self.lock:
            self.forwards.append([r["tag"] for r in rows])
        answers = [{"type": "noul", "noul": 0.5, "tag": r["tag"]} for r in rows]
        return answers, [r["tokens"] for r in rows], "eager"


def run(batcher, state, qids):
    rows = batcher.engine.build_rows(batcher.name, state, {q: None for q in qids})
    job = batcher.submit(rows)
    job.done.wait(timeout=30)
    assert job.error is None, job.error
    return job.result


def test_a_single_request_comes_back_intact():
    eng = FakeEngine()
    b = Batcher(eng, "fake", max_batch=64, wait_ms=0)
    try:
        out = run(b, "s", ["a", "b", "c"])
        assert [a["tag"] for a in out["answers"]] == ["s/a", "s/b", "s/c"]
        assert out["input_tokens"] == 9
    finally:
        b.stop()


def test_concurrent_requests_share_one_forward_and_stay_separate():
    eng = FakeEngine()
    b = Batcher(eng, "fake", max_batch=64, wait_ms=60)
    results = {}

    def caller(i, n):
        results[i] = run(b, "req%d" % i, ["q%d" % j for j in range(n)])

    try:
        threads = [threading.Thread(target=caller, args=(i, n))
                   for i, n in enumerate([1, 3, 2, 5])]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        for i, n in enumerate([1, 3, 2, 5]):
            answers = results[i]["answers"]
            assert len(answers) == n
            # every answer belongs to the request that asked for it
            assert all(a["tag"] == "req%d/q%d" % (i, j) for j, a in enumerate(answers))
            assert results[i]["input_tokens"] == 3 * n

        # and they really did travel together: one forward carried more than one request's rows
        assert max(len(f) for f in eng.forwards) > 5
    finally:
        b.stop()


def test_a_request_larger_than_the_batch_cap_is_chunked():
    eng = FakeEngine()
    b = Batcher(eng, "fake", max_batch=4, wait_ms=0)
    try:
        out = run(b, "big", ["q%d" % i for i in range(10)])
        assert [a["tag"] for a in out["answers"]] == ["big/q%d" % i for i in range(10)]
        assert [len(f) for f in eng.forwards] == [4, 4, 2]
    finally:
        b.stop()


def test_the_queue_drains_to_zero():
    eng = FakeEngine()
    b = Batcher(eng, "fake", max_batch=8, wait_ms=0)
    try:
        for _ in range(5):
            run(b, "s", ["a", "b"])
        for _ in range(50):
            if b.depth == 0:
                break
            time.sleep(0.02)
        assert b.depth == 0
    finally:
        b.stop()


def test_a_failing_forward_fails_only_its_own_batch():
    class Broken(FakeEngine):
        def predict_rows(self, name, rows):
            raise RuntimeError("cuda is on fire")

    b = Batcher(Broken(), "fake", max_batch=8, wait_ms=0)
    try:
        rows = b.engine.build_rows(b.name, "s", {"a": None})
        job = b.submit(rows)
        job.done.wait(timeout=30)
        assert isinstance(job.error, RuntimeError)
        assert b.depth == 0
        assert b.thread.is_alive()
    finally:
        b.stop()
