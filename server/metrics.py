"""Prometheus text exposition, written by hand so the server keeps its dependency list short."""
import threading
from typing import Dict, List, Tuple

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class Registry:
    """Counters and histograms for the serving path. Cheap enough to update on every request."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests: Dict[Tuple[str, int], int] = {}
        self.questions: Dict[str, int] = {}
        self.latency_counts: List[int] = [0] * len(LATENCY_BUCKETS)
        self.latency_sum = 0.0
        self.latency_total = 0
        self.batch_counts: List[int] = [0] * len(BATCH_BUCKETS)
        self.batch_sum = 0
        self.batch_total = 0

    def observe_request(self, model: str, status: int, n_questions: int, seconds: float):
        with self._lock:
            key = (model, status)
            self.requests[key] = self.requests.get(key, 0) + 1
            if n_questions:
                self.questions[model] = self.questions.get(model, 0) + n_questions
            self.latency_sum += seconds
            self.latency_total += 1
            for i, edge in enumerate(LATENCY_BUCKETS):
                if seconds <= edge:
                    self.latency_counts[i] += 1

    def observe_batch(self, rows: int):
        with self._lock:
            self.batch_sum += rows
            self.batch_total += 1
            for i, edge in enumerate(BATCH_BUCKETS):
                if rows <= edge:
                    self.batch_counts[i] += 1

    def render(self, queue_depth: int) -> str:
        with self._lock:
            out = [
                "# HELP laya_requests_total System One requests, by routed checkpoint and status.",
                "# TYPE laya_requests_total counter",
            ]
            for (model, status), n in sorted(self.requests.items()):
                out.append('laya_requests_total{model="%s",status="%d"} %d' % (_escape(model), status, n))

            out += ["# HELP laya_questions_total Questions answered, by checkpoint.",
                    "# TYPE laya_questions_total counter"]
            for model, n in sorted(self.questions.items()):
                out.append('laya_questions_total{model="%s"} %d' % (_escape(model), n))

            out += ["# HELP laya_request_latency_seconds End-to-end request latency.",
                    "# TYPE laya_request_latency_seconds histogram"]
            for edge, n in zip(LATENCY_BUCKETS, self.latency_counts):
                out.append('laya_request_latency_seconds_bucket{le="%s"} %d' % (edge, n))
            out.append('laya_request_latency_seconds_bucket{le="+Inf"} %d' % self.latency_total)
            out.append("laya_request_latency_seconds_sum %.6f" % self.latency_sum)
            out.append("laya_request_latency_seconds_count %d" % self.latency_total)

            out += ["# HELP laya_batch_rows Question rows per forward pass.",
                    "# TYPE laya_batch_rows histogram"]
            for edge, n in zip(BATCH_BUCKETS, self.batch_counts):
                out.append('laya_batch_rows_bucket{le="%d"} %d' % (edge, n))
            out.append('laya_batch_rows_bucket{le="+Inf"} %d' % self.batch_total)
            out.append("laya_batch_rows_sum %d" % self.batch_sum)
            out.append("laya_batch_rows_count %d" % self.batch_total)

            out += ["# HELP laya_queue_depth Question rows waiting for or inside a forward pass.",
                    "# TYPE laya_queue_depth gauge",
                    "laya_queue_depth %d" % queue_depth]
            return "\n".join(out) + "\n"
