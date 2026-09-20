"""A tiny client for the Jev-compatible `/v1/systemone` endpoint served by this repo.

Standard library only -- `urllib` and `json` -- so an example script is one file and a `python`
that exists everywhere. The names mirror TypeSafe's SDK (`Choice`, `Score`, `Noul`, a client
with one `system_one` call taking `state` and a dict of questions), so code written against the
hosted Jev API ports by changing the import and the base URL.

    from arbiter_client import ArbiterClient, Choice, Noul, Score

    arbiter = ArbiterClient()                       # ARBITER_URL, default http://localhost:8010
    r = arbiter.system_one(ticket, {
        "department": Choice("Which team owns this?", {"billing": "payments", "tech": "bugs"}),
        "urgency":    Score("How urgent is it?", ["no rush", "this week", "today", "now"]),
        "refund":     Noul("The customer is asking for a refund."),
    })
    r.choice("department"), r.score("urgency"), r.noul("refund")

Every question in one call is answered in a single forward pass, so asking ten questions costs
roughly what asking one costs. Ask everything you might need at once and let your code decide
what to do with the answers -- that is the whole trick.
"""
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Union

DEFAULT_URL = "http://localhost:8010"

__all__ = [
    "Choice", "Score", "Noul", "ArbiterClient", "Response",
    "ArbiterError", "AuthError", "RequestError", "OverloadedError", "UnreachableError",
]


# --------------------------------------------------------------------------- questions

class Question:
    """Base for the three primitives. Subclasses only differ in how `criteria` is built."""

    type = ""

    def __init__(self, instructions: Union[str, Mapping, list], criteria: Any = None):
        self.instructions = instructions
        self.criteria = criteria

    def to_json(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body

    def __repr__(self) -> str:
        return "%s(%r)" % (type(self).__name__, self.instructions)


class Noul(Question):
    """A probability that a statement about the state is true.

    The strongest primitive of the three. `criteria` is optional and, when given, says what
    true and what false look like: `Noul("...", {"true": "...", "false": "..."})`.
    """

    type = "noul"


class Choice(Question):
    """Pick one option. `criteria` maps option name -> description (or None).

    Keep it to about a dozen options: the option descriptions share a fixed token budget with
    the state, and accuracy falls off past roughly twenty. For a large taxonomy ask a coarse
    Choice first and a second, narrower Choice for the winning branch -- two forward passes
    still cost less than one LLM call.
    """

    type = "choice"

    def __init__(self, instructions, criteria: Union[Mapping[str, Optional[str]], List[str]]):
        super().__init__(instructions, criteria)


class Score(Question):
    """An ordinal level, given 2-10 level descriptions in order, lowest first.

    The answer is the expectation over the levels, so it comes back as a float between 0 and
    len(levels)-1. This is the weakest of the three primitives; where a hard boundary matters,
    a Noul phrased as the boundary ("this needs a human before it runs") beats a Score plus a
    cutoff.
    """

    type = "score"

    def __init__(self, instructions, levels: List[str]):
        super().__init__(instructions, list(levels))


def encode_questions(questions: Mapping[str, Union[Question, Mapping[str, Any]]]) -> Dict[str, Any]:
    """Accept either the classes above or plain dicts already in the wire shape."""
    return {qid: (q.to_json() if isinstance(q, Question) else dict(q))
            for qid, q in questions.items()}


# --------------------------------------------------------------------------- errors

class ArbiterError(Exception):
    """Anything that stopped a request from producing answers."""


class AuthError(ArbiterError):
    """401: the server wants an API key and did not get a valid one."""


class RequestError(ArbiterError):
    """422: the question set or the state does not satisfy the contract."""


class OverloadedError(ArbiterError):
    """529: too many question rows in flight. Back off and retry."""


class UnreachableError(ArbiterError):
    """The server did not answer at all -- not running, wrong port, tunnel down."""


def _raise_for(status: int, body: str, url: str) -> None:
    try:
        message = json.loads(body)["error"]["message"]
    except Exception:
        message = body.strip()[:400] or "(empty body)"
    if status == 401:
        raise AuthError(
            "%s rejected the key: %s. Set ARBITER_API_KEY to the key the server was started with."
            % (url, message))
    if status == 422:
        raise RequestError("%s rejected the request: %s" % (url, message))
    if status == 529:
        raise OverloadedError("%s is overloaded: %s" % (url, message))
    if status == 503:
        raise UnreachableError(
            "%s is still loading its checkpoints: %s. Poll /readyz until it reports ready."
            % (url, message))
    raise ArbiterError("%s returned HTTP %d: %s" % (url, status, message))


# --------------------------------------------------------------------------- response

class Response(dict):
    """The raw JSON body, with accessors for the three answer shapes.

    It really is the response dict -- `json.dumps(response)` gives you back exactly what the
    server sent -- so `--json` output and the accessors cannot drift apart.
    """

    def answer(self, qid: str) -> Dict[str, Any]:
        try:
            return self["answers"][qid]
        except KeyError:
            raise KeyError("no answer for %r; the server answered %s"
                           % (qid, sorted(self.get("answers", {})))) from None

    def noul(self, qid: str) -> float:
        return float(self.answer(qid)["noul"])

    def choice(self, qid: str) -> str:
        return self.answer(qid)["choice"]

    def score(self, qid: str) -> float:
        return float(self.answer(qid)["score"])

    def probabilities(self, qid: str) -> Dict[str, float]:
        """Choice and score answers carry the full distribution; a noul is its own probability."""
        a = self.answer(qid)
        if a["type"] == "noul":
            return {"true": a["noul"], "false": round(1.0 - a["noul"], 4)}
        return a["probabilities"]

    def confidence(self, qid: str) -> float:
        return float(self.answer(qid)["confidence"])

    def label(self, qid: str) -> str:
        """The level description nearest a score answer, for printing."""
        a = self.answer(qid)
        if a["type"] != "score":
            raise TypeError("label() is for score answers; %r is a %s" % (qid, a["type"]))
        return a["legend"][str(int(round(a["score"])))]

    @property
    def checkpoint(self) -> str:
        """Which of the three checkpoints answered: english, multilingual, typed-decisions."""
        return self.get("routing", {}).get("model", self.get("model", "?"))

    @property
    def routing_reason(self) -> str:
        return self.get("routing", {}).get("reason", "")

    @property
    def latency_ms(self) -> float:
        return float(self.get("latency_ms", 0.0))

    @property
    def input_tokens(self) -> int:
        return int(self.get("usage", {}).get("input_tokens", 0))


# --------------------------------------------------------------------------- client

class ArbiterClient:
    """One HTTP call per `system_one`, no session state, safe to share between threads."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 timeout: float = 30.0, retries: int = 2):
        self.base_url = (base_url or os.environ.get("ARBITER_URL") or DEFAULT_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("ARBITER_API_KEY")
        self.timeout = timeout
        self.retries = max(0, int(retries))

    # -- plumbing ----------------------------------------------------------------
    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        url = self.base_url + path
        data = json.dumps(body).encode() if body is not None else None
        headers = {"accept": "application/json"}
        if data is not None:
            headers["content-type"] = "application/json"
        if self.api_key:
            headers["authorization"] = "Bearer %s" % self.api_key
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        # 529 is the server saying "the queue is full", which is worth waiting out; every other
        # status is a decision the caller has to see immediately.
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                payload = exc.read().decode("utf-8", "replace")
                if exc.code == 529 and attempt < self.retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                _raise_for(exc.code, payload, url)
            except urllib.error.URLError as exc:
                raise UnreachableError(
                    "cannot reach %s (%s). Start the server with ./run.sh, or point ARBITER_URL at "
                    "one that is running." % (url, exc.reason)) from None
            except OSError as exc:
                # A reset or a dropped connection mid-response: the server went away while it
                # was answering. Same advice as above, and worth not showing as a raw traceback.
                raise UnreachableError("lost the connection to %s (%s)" % (url, exc)) from None

    # -- the API -----------------------------------------------------------------
    def system_one(self, state: Union[str, Mapping, list],
                   questions: Mapping[str, Union[Question, Mapping[str, Any]]],
                   model: str = "auto", **extra: Any) -> Response:
        """Ask every question about one state in a single forward pass.

        `model` is "auto" (let the router pick a checkpoint) or one of "laya-english",
        "laya-multilingual", "laya-typed-decisions". `extra` passes the server's two router
        hints, `task=` and `lang=`, for callers that already know the answer.
        """
        body: Dict[str, Any] = {"state": state, "questions": encode_questions(questions),
                                "model": model}
        body.update({k: v for k, v in extra.items() if v is not None})
        return Response(self._request("POST", "/v1/systemone", body))

    def models(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/models")

    def ready(self) -> bool:
        try:
            return self._request("GET", "/readyz").get("status") == "ready"
        except ArbiterError:
            return False
