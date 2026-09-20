"""A stand-in `/v1/systemone` over `http.server`, so the examples can be tested without a GPU.

It answers in exactly the shapes `server/app.py` produces, with probabilities derived from a
hash of the question id, so a given question always gets the same numbers and a test can assert
on a route rather than on luck. `scripted` forces specific answers where a test needs a
particular decision; `fail` turns on the three error paths the client has to explain.

Also runnable on its own, which is the quickest way to work on an example's output without a
server anywhere near you:

    python examples/tests/stub_server.py --port 8010
"""
import hashlib
import json
import math
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOADED = ["english", "multilingual", "typed-decisions"]


class _Server(ThreadingHTTPServer):
    """A ThreadingHTTPServer that does not look itself up in DNS.

    `HTTPServer.server_bind` calls `socket.getfqdn()`, which is a reverse lookup on the bind
    address and can block for half a minute on a machine with no reachable resolver. The name it
    produces only ever appears in CGI variables this stub does not generate.
    """

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def _unit(*parts) -> float:
    """A stable float in [0, 1) from the given strings."""
    digest = hashlib.md5("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2 ** 32


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    total = sum(e)
    return [v / total for v in e]


def answer_for(qid: str, qdef: dict) -> dict:
    t = qdef["type"]
    if t == "noul":
        p = round(0.05 + 0.9 * _unit(qid, "noul"), 4)
        return {"type": "noul", "noul": p, "confidence": round(max(p, 1 - p), 4),
                "action": {"act_probability": 1.0}}
    crit = qdef["criteria"]
    if t == "choice":
        keys = list(crit) if isinstance(crit, dict) else list(crit)
        probs = _softmax([4.0 * _unit(qid, k) for k in keys])
        top = keys[max(range(len(keys)), key=lambda i: probs[i])]
        return {"type": "choice", "choice": top,
                "probabilities": {k: round(p, 4) for k, p in zip(keys, probs)},
                "confidence": round(max(probs), 4), "action": {"act_probability": 1.0}}
    probs = _softmax([3.0 * _unit(qid, i) for i in range(len(crit))])
    return {"type": "score",
            "score": round(sum(i * p for i, p in enumerate(probs)), 4),
            "legend": {str(i): c for i, c in enumerate(crit)},
            "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)},
            "confidence": round(max(probs), 4), "action": {"act_probability": 1.0}}


class StubServer:
    """A server on an ephemeral port. Use as a context manager; `url` is the base URL."""

    def __init__(self, api_key=None, fail=None, scripted=None):
        self.api_key = api_key
        self.fail = fail
        self.scripted = scripted or {}
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _error(self, status, kind, message):
                self._send(status, {"type": "error", "error": {"type": kind, "message": message}})

            def do_GET(self):
                if self.path == "/healthz":
                    return self._send(200, {"status": "ok", "version": "stub"})
                if self.path == "/readyz":
                    return self._send(200, {"status": "ready", "models": LOADED,
                                            "mode": "stub", "version": "stub"})
                if self.path == "/v1/models":
                    return self._send(200, {"object": "list", "data": [
                        {"id": "laya-%s" % n, "object": "model"} for n in LOADED]})
                return self._error(404, "not_found_error", "no route %s" % self.path)

            def do_POST(self):
                if self.path not in ("/v1/systemone", "/v1/predict"):
                    return self._error(404, "not_found_error", "no route %s" % self.path)
                raw = self.rfile.read(int(self.headers.get("content-length", 0)))
                if stub.api_key and self.headers.get("authorization") != "Bearer %s" % stub.api_key:
                    return self._error(401, "authentication_error",
                                       "missing or invalid Authorization: Bearer <key>")
                if stub.fail == "overload":
                    return self._error(529, "overloaded_error", "queue is full")
                try:
                    payload = json.loads(raw)
                except ValueError:
                    return self._error(422, "invalid_request_error",
                                       "request body is not valid JSON")
                stub.requests.append(payload)
                questions = payload.get("questions") or {}
                if not questions:
                    return self._error(422, "invalid_request_error",
                                       "questions must contain at least one question")
                if stub.fail == "budget":
                    return self._error(422, "invalid_request_error",
                                       "question has more options than head_max_len=192 can hold")
                state = json.dumps(payload.get("state"), ensure_ascii=False)
                checkpoint = "english" if state.isascii() else "multilingual"
                reason = "English Latin text" if state.isascii() else "non-Latin script"
                answers = {qid: stub.scripted.get(qid) or answer_for(qid, q)
                           for qid, q in questions.items()}
                return self._send(200, {
                    "model": "laya-%s" % checkpoint,
                    "answers": answers,
                    "usage": {"input_tokens": 40 * len(questions), "output_tokens": 0},
                    "routing": {"model": checkpoint, "repo": "stub", "reason": reason},
                    "latency_ms": 12.34,
                })

        self.httpd = _Server(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       kwargs={"poll_interval": 0.02}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()
    stub = StubServer()
    stub.httpd.server_close()
    stub.httpd = _Server(("127.0.0.1", args.port), stub.httpd.RequestHandlerClass)
    print("stub /v1/systemone on http://127.0.0.1:%d" % args.port)
    stub.httpd.serve_forever()
