"""The model-agnostic half of serving: the cross-request batcher and the engine interface.

Nothing in this file knows what a model is. A backend under `engines/<name>/loader.py` supplies
that by subclassing `Engine`, and `engines/README.md` states the contract. The split exists
because the batching, the queue cap and the warm-up are the same whatever answers the rows, and
because Laya is the first model served here rather than the only one that ever will be.
"""
import importlib
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .errors import OverloadedError


# --------------------------------------------------------------------------- micro-batching

class _Job:
    __slots__ = ("rows", "done", "result", "error")

    def __init__(self, rows):
        self.rows = rows
        self.done = threading.Event()
        self.result = None
        self.error = None


class Batcher:
    """A worker thread per checkpoint, collecting rows from concurrent requests into one forward.

    The thread takes the first job that arrives and then keeps draining the queue for up to
    `wait_ms`, or until `max_batch` rows are in hand. Under one caller that wait is the only cost
    the batcher adds; under load it is what turns N small forwards into one.

    A thread rather than a coroutine because the forward holds the GIL in stretches, and
    blocking the event loop would make the concurrency it is meant to serve worse.

    Rows are opaque here: they go to `engine.predict_rows` in the order they arrived and the
    answers come back in that order, which is all the batcher needs to give each caller its own.
    """

    def __init__(self, engine: "Engine", name: str, max_batch: int = 64, wait_ms: float = 2.0):
        self.engine = engine
        self.name = name
        self.max_batch = max(1, int(max_batch))
        self.wait_s = max(0.0, float(wait_ms)) / 1000.0
        self.q: "queue.Queue[Optional[_Job]]" = queue.Queue()
        self.depth = 0
        self._depth_lock = threading.Lock()
        self.batch_sizes: List[int] = []
        self.thread = threading.Thread(target=self._loop, name="arbiter-%s" % name, daemon=True)
        self.thread.start()

    def submit(self, rows: List[Any]) -> _Job:
        job = _Job(rows)
        with self._depth_lock:
            self.depth += len(rows)
        self.q.put(job)
        return job

    def stop(self):
        self.q.put(None)
        self.thread.join(timeout=10)

    def _loop(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            jobs = [job]
            rows = len(job.rows)
            deadline = time.monotonic() + self.wait_s
            while rows < self.max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    nxt = self.q.get(timeout=timeout)
                except queue.Empty:
                    break
                if nxt is None:
                    self.q.put(None)
                    break
                jobs.append(nxt)
                rows += len(nxt.rows)
            try:
                self._run(jobs)
            except Exception as exc:                      # noqa: BLE001 - one bad batch must not kill the worker
                for j in jobs:
                    j.error = exc
                    j.done.set()
            finally:
                with self._depth_lock:
                    self.depth -= sum(len(j.rows) for j in jobs)

    def _run(self, jobs: List[_Job]):
        flat: List[Any] = []
        owner: List[Tuple[int, int]] = []
        for ji, j in enumerate(jobs):
            for ri, row in enumerate(j.rows):
                flat.append(row)
                owner.append((ji, ri))

        results: List[List[Optional[Dict[str, Any]]]] = [[None] * len(j.rows) for j in jobs]
        tokens = [0] * len(jobs)
        paths: List[str] = []

        # A single request can be larger than the batch cap, so chunk the flattened rows rather
        # than the jobs.
        for start in range(0, len(flat), self.max_batch):
            chunk = flat[start:start + self.max_batch]
            answers, row_tokens, path = self.engine.predict_rows(self.name, chunk)
            paths.append(path)
            self.batch_sizes.append(len(chunk))
            for i in range(len(chunk)):
                ji, ri = owner[start + i]
                results[ji][ri] = answers[i]
                # Usage is reported per request, so count each row's real tokens rather than the
                # chunk's padded total.
                tokens[ji] += row_tokens[i]

        for ji, j in enumerate(jobs):
            j.result = {
                "answers": results[ji],
                "input_tokens": tokens[ji],
                "batch_rows": len(flat),
                "path": paths[0] if paths else "eager",
            }
            j.done.set()


# --------------------------------------------------------------------------- the engine

class Engine:
    """What the HTTP layer drives, and the generic half of it.

    A backend implements four things: `load(name)` brings one checkpoint into memory,
    `build_rows(name, state, questions)` turns a request into one opaque row per question,
    `predict_rows(name, rows)` runs a batch of them, and `route(...)` picks the checkpoint.
    Loading, the per-checkpoint worker thread, the queue cap, the warm-up and the shutdown are
    the same whatever the model is and live here.

    `mode` and `dtype_mode` are what `/readyz` reports about how the checkpoints are running; a
    backend that has no such distinction leaves them alone.
    """

    mode = "eager"
    dtype_mode = "autocast"

    def __init__(self, names: Tuple[str, ...], max_batch: int = 64, wait_ms: float = 2.0,
                 max_queue: int = 256):
        self.max_queue = int(max_queue)
        self.batchers: Dict[str, Batcher] = {}
        for name in names:
            self.load(name)
            self.batchers[name] = Batcher(self, name, max_batch=max_batch, wait_ms=wait_ms)

    # -- the backend's half ------------------------------------------------------
    def load(self, name: str) -> None:
        """Bring checkpoint `name` into memory, ready to answer."""
        raise NotImplementedError

    def build_rows(self, name: str, state, questions: Dict[str, Dict[str, Any]]) -> List[Any]:
        """One row per question, in the order the ids were given. Rows are the backend's own."""
        raise NotImplementedError

    def predict_rows(self, name: str, rows: List[Any]) -> Tuple[List[Dict[str, Any]], List[int], str]:
        """Run one batch of rows: an answer and a token count per row, and the path taken."""
        raise NotImplementedError

    def route(self, state, questions, model=None, task=None, lang=None) -> Dict[str, Any]:
        """Which checkpoint should answer this, and why."""
        raise NotImplementedError

    # -- the generic half --------------------------------------------------------
    def checkpoints(self) -> List[str]:
        return list(self.batchers)

    @property
    def queue_depth(self) -> int:
        return sum(b.depth for b in self.batchers.values())

    def warm(self):
        """One real question per checkpoint, so /readyz means 'the first user request is fast'."""
        q = {"warm": {"type": "noul", "instructions": "the text mentions a warm-up"}}
        for name in self.checkpoints():
            self.predict_rows(name, self.build_rows(name, "warm up the serving path", q))

    def close(self):
        for b in self.batchers.values():
            b.stop()

    def infer(self, name: str, state, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Blocking inference on one checkpoint. Call it from a thread, not the event loop."""
        rows = self.build_rows(name, state, questions)
        if self.queue_depth + len(rows) > self.max_queue:
            raise OverloadedError("queue is full (%d question rows in flight, limit %d)"
                                  % (self.queue_depth, self.max_queue))
        job = self.batchers[name].submit(rows)
        job.done.wait()
        if job.error is not None:
            raise job.error
        # One row per question, in the order they were asked, is what `build_rows` promises.
        job.result["answers"] = dict(zip(questions, job.result["answers"]))
        return job.result


def engine_from_env() -> Engine:
    """Build the configured backend. `ARBITER_ENGINE` names a package under `engines/`."""
    name = os.environ.get("ARBITER_ENGINE", "laya")
    return importlib.import_module("engines.%s.loader" % name).from_env()
