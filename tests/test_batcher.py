"""The micro-batcher: rows from concurrent requests share a forward and still come back right.

This is the part of the serving path with a real chance of silently mixing two callers' answers,
so it is tested against a fake checkpoint whose logits encode which row they came from.
"""
import threading
import time

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from server.engine import Batcher  # noqa: E402


class FakeCheckpoint:
    """Answers with the row's own tag, so misattribution is detectable rather than plausible."""

    name = "fake"

    def __init__(self):
        self.forwards = []
        self.lock = threading.Lock()

    def build_rows(self, state, questions):
        return [{"ids": [1, 2, 3], "markers": [1, 2], "qtype": 2, "qid": qid, "q": None,
                 "tag": "%s/%s" % (state, qid)} for qid in questions]

    def forward(self, rows):
        with self.lock:
            self.forwards.append([r["tag"] for r in rows])
        logits = np.zeros((len(rows), 2), dtype=float)
        act = np.zeros((len(rows), 2), dtype=float)
        return logits, act, sum(len(r["ids"]) for r in rows), "eager"

    def answer(self, row, logits_row, act_row):
        return {"type": "noul", "noul": 0.5, "tag": row["tag"]}


def run(batcher, state, qids):
    rows = batcher.ckpt.build_rows(state, {q: None for q in qids})
    job = batcher.submit(rows)
    job.done.wait(timeout=30)
    assert job.error is None, job.error
    return job.result


def test_a_single_request_comes_back_intact():
    ck = FakeCheckpoint()
    b = Batcher(ck, max_batch=64, wait_ms=0)
    try:
        out = run(b, "s", ["a", "b", "c"])
        assert list(out["answers"]) == ["a", "b", "c"]
        assert [a["tag"] for a in out["answers"].values()] == ["s/a", "s/b", "s/c"]
        assert out["input_tokens"] == 9
    finally:
        b.stop()


def test_concurrent_requests_share_one_forward_and_stay_separate():
    ck = FakeCheckpoint()
    b = Batcher(ck, max_batch=64, wait_ms=60)
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
            assert list(answers) == ["q%d" % j for j in range(n)]
            # every answer belongs to the request that asked for it
            assert all(a["tag"] == "req%d/q%d" % (i, j) for j, a in enumerate(answers.values()))
            assert results[i]["input_tokens"] == 3 * n

        # and they really did travel together: one forward carried more than one request's rows
        assert max(len(f) for f in ck.forwards) > 5
    finally:
        b.stop()


def test_a_request_larger_than_the_batch_cap_is_chunked():
    ck = FakeCheckpoint()
    b = Batcher(ck, max_batch=4, wait_ms=0)
    try:
        out = run(b, "big", ["q%d" % i for i in range(10)])
        assert list(out["answers"]) == ["q%d" % i for i in range(10)]
        assert [a["tag"] for a in out["answers"].values()] == ["big/q%d" % i for i in range(10)]
        assert [len(f) for f in ck.forwards] == [4, 4, 2]
    finally:
        b.stop()


def test_the_queue_drains_to_zero():
    ck = FakeCheckpoint()
    b = Batcher(ck, max_batch=8, wait_ms=0)
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
    class Broken(FakeCheckpoint):
        def forward(self, rows):
            raise RuntimeError("cuda is on fire")

    b = Batcher(Broken(), max_batch=8, wait_ms=0)
    try:
        rows = b.ckpt.build_rows("s", {"a": None})
        job = b.submit(rows)
        job.done.wait(timeout=30)
        assert isinstance(job.error, RuntimeError)
        assert b.depth == 0
        assert b.thread.is_alive()
    finally:
        b.stop()
