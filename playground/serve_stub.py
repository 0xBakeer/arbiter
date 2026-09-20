#!/usr/bin/env python3
"""Serve playground/index.html next to a fake `/v1/systemone`, for UI work without a GPU.

    python3 playground/serve_stub.py                    # stub answers on http://localhost:8011
    python3 playground/serve_stub.py --proxy http://localhost:8010
                                                        # same page, API calls forwarded to a real server

The stub answers with stable pseudo-random probabilities in exactly the shapes the real server
produces. With `--proxy` the page is still served from here (same origin), which is the way to
use a bare arbiter server from a page that was not mounted on it.
"""
import hashlib
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

INDEX = Path(__file__).parent / "index.html"
ARGS = sys.argv[1:]
PROXY = ARGS[ARGS.index("--proxy") + 1].rstrip("/") if "--proxy" in ARGS else None
PORT = int(ARGS[ARGS.index("--port") + 1]) if "--port" in ARGS else 8011
MODELS = ["english", "multilingual", "typed-decisions"]


def frac(*parts):
    """A stable number in (0, 1) derived from the text, so the UI looks alive but repeatable."""
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:6], 16) / 0xFFFFFF


def answer(qid, q):
    t = q.get("type")
    if t == "noul":
        p = round(frac(qid, q.get("instructions")), 4)
        return {"type": "noul", "noul": p, "confidence": round(max(p, 1 - p), 4),
                "action": {"act_probability": 1.0}}
    crit = q.get("criteria")
    keys = list(crit) if isinstance(crit, dict) else [str(i) for i in range(len(crit or []))]
    if len(keys) < 2:
        raise ValueError("question %r: %s needs at least 2 %s" % (qid, t, "options" if t == "choice" else "levels"))
    w = [frac(qid, k, q.get("instructions")) ** 3 for k in keys]
    s = sum(w)
    ps = [round(x / s, 4) for x in w]
    conf = round(max(ps) - sorted(ps)[-2], 4)
    if t == "choice":
        return {"type": "choice", "choice": keys[ps.index(max(ps))], "probabilities": dict(zip(keys, ps)),
                "confidence": conf, "action": {"act_probability": 1.0}}
    if t == "score":
        return {"type": "score", "score": round(sum(i * p for i, p in enumerate(ps)), 4),
                "legend": {str(i): c for i, c in enumerate(crit)}, "probabilities": dict(zip(keys, ps)),
                "confidence": conf, "action": {"act_probability": 1.0}}
    raise ValueError("question %r has unknown type %r; expected one of ['noul', 'choice', 'score']" % (qid, t))


def route(state, model):
    if model and model not in ("auto", "jev-latest", "laya", "default", ""):
        name = model.replace("laya-", "")
        return {"model": name, "reason": "explicit model=%r" % model, "detection": None}
    text = json.dumps(state, ensure_ascii=False)
    if not text.isascii():
        return {"model": "multilingual", "reason": "non-ASCII text; the English checkpoint cannot read it",
                "detection": {"script": "unknown", "language": None}}
    return {"model": "english", "reason": "English Latin text", "detection": {"script": "latin", "language": "en"}}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type, authorization")
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self, body=None):
        req = urllib.request.Request(PROXY + self.path, data=body, method=self.command)
        for h in ("content-type", "authorization"):
            if self.headers.get(h):
                req.add_header(h, self.headers[h])
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                self._send(r.status, r.read(), r.headers.get("content-type", "application/json"))
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read(), e.headers.get("content-type", "application/json"))
        except Exception as e:  # noqa: BLE001
            self._send(502, {"type": "error", "error": {"type": "proxy_error", "message": str(e)}})

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")  # re-read: edit and reload
        if PROXY:
            return self._proxy()
        if self.path == "/readyz":
            return self._send(200, {"status": "ready", "models": MODELS, "mode": "stub", "version": "stub"})
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [{"id": "laya-%s" % m, "object": "model"} for m in MODELS]})
        self._send(404, {"type": "error", "error": {"type": "not_found", "message": self.path}})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        if PROXY:
            return self._proxy(body)
        if self.path not in ("/v1/systemone", "/v1/predict"):
            return self._send(404, {"type": "error", "error": {"type": "not_found", "message": self.path}})
        try:
            req = json.loads(body)
            questions = req["questions"]
            if not questions:
                raise ValueError("questions must contain at least one question")
            answers = {qid: answer(qid, q) for qid, q in questions.items()}
        except (ValueError, KeyError, TypeError) as e:
            return self._send(422, {"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}})
        routing = route(req.get("state"), req.get("model"))
        self._send(200, {"model": "laya-" + routing["model"], "answers": answers,
                         "usage": {"input_tokens": 40 + 9 * len(questions), "output_tokens": 0},
                         "routing": routing, "latency_ms": round(8 + 6 * frac(body), 2)})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.command, self.path))


if __name__ == "__main__":
    print("playground on http://localhost:%d  (%s)" % (PORT, "proxy to " + PROXY if PROXY else "stub answers"))
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
