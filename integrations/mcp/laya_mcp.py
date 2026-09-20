#!/usr/bin/env python3
"""An MCP server that gives a coding agent a System One: five fast, typed decisions.

Agents are good at deciding things and slow at it. Every "is this command safe", "which of these
passages matters", "how risky is this change" costs a full model turn, and the agent pays for it
in latency and in context. These tools answer the same questions against a local Laya server in
tens of milliseconds, return a probability rather than a paragraph, and leave the agent's
context alone.

Tools:

    laya_classify   pick one of up to 12 labelled options
    laya_score      place something on an ordered scale of 2-10 levels
    laya_check      the probability that a statement about the state is true
    laya_gate       is this action safe to run: allow | confirm | block, with the signals
    laya_decide     any set of questions at once, raw -- one forward pass for all of them

Every tool returns structured content (the probabilities) plus one short line of text, so an
agent can act on the numbers and a human reading the transcript can see why.

Run it:  LAYA_URL=http://localhost:8010 python laya_mcp.py
Install: pip install "mcp>=2"   (or `pip install .` in this directory for the `laya-mcp` script)

Configuration comes from the environment: LAYA_URL (default http://localhost:8010),
LAYA_API_KEY (optional), LAYA_MODEL (default "auto" -- let the server route).
"""
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from mcp.server.mcpserver import MCPServer
except ModuleNotFoundError as exc:                       # pragma: no cover - install-time path
    raise SystemExit("laya-mcp needs the official MCP SDK v2 or newer: pip install 'mcp>=2'") from exc

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations

LAYA_URL = os.environ.get("LAYA_URL", "http://localhost:8010").rstrip("/")
LAYA_API_KEY = os.environ.get("LAYA_API_KEY")
LAYA_MODEL = os.environ.get("LAYA_MODEL", "auto")
TIMEOUT = float(os.environ.get("LAYA_TIMEOUT", "30"))

# Option descriptions share a fixed token budget with the state, and accuracy falls off well
# before the server's hard limit, so the tool refuses long option lists rather than answering
# badly. The fix is hierarchical: a coarse classify, then a second one inside the winner.
MAX_OPTIONS = 12
MIN_LEVELS, MAX_LEVELS = 2, 10

# Same question set and the same thresholds as examples/tool_call_guard.py. They are repeated
# here rather than imported because this file is installed on its own, away from the repo.
GATE_QUESTIONS = {
    "is_destructive": {
        "type": "noul",
        "instructions": ("Running this action destroys or irreversibly changes something: "
                         "deleting files, dropping data, force-pushing, overwriting history, "
                         "terminating infrastructure."),
        "criteria": {"true": "data or state is lost and cannot be trivially restored",
                     "false": "it reads, builds, tests, or makes a change that is easy to undo"},
    },
    "touches_secrets": {
        "type": "noul",
        "instructions": ("This action reads, prints, copies or transmits credentials: private "
                         "keys, tokens, passwords, .env files, cloud credentials, keychains."),
    },
    "leaves_repo": {
        "type": "noul",
        "instructions": "This action reads or writes outside the working directory it runs in.",
    },
    "needs_network": {
        "type": "noul",
        "instructions": "This action talks to the network: fetching, uploading, deploying, or calling an API.",
    },
    "blast_radius": {
        "type": "score",
        "instructions": "If this action does the wrong thing, how far does the damage reach?",
        "criteria": ["nothing outside this shell session",
                     "files in this one project",
                     "the whole developer machine",
                     "shared or production systems other people depend on"],
    },
}
GATE_DENY_DESTRUCTIVE = 0.75
GATE_DENY_SECRETS = 0.70
GATE_DENY_BLAST = 2.40
# Well above 0.5: the model is not shy about a mild yes, and `npm test` scores around 0.48 on
# "is this destructive". A gate that stops for that is a gate agents learn to ignore.
GATE_ASK_SIGNAL = 0.55
GATE_ASK_BLAST = 1.50

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True,
                            openWorldHint=True)

server = MCPServer(
    name="laya",
    instructions=("Fast typed decisions from a local Laya System One server. Prefer these tools "
                  "over reasoning about a routine judgement yourself: they answer in tens of "
                  "milliseconds and give you a probability you can threshold. Ask every question "
                  "need in one laya_decide call -- they all run in a single forward pass, so ten "
                  "questions cost what one costs. Never treat a probability as a verdict: pick "
                  "your own thresholds and leave a middle band where you ask the user."),
)


# --------------------------------------------------------------------------- transport

def call_laya(state: Any, questions: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    body = json.dumps({"state": state, "questions": questions,
                       "model": model or LAYA_MODEL}).encode()
    headers = {"content-type": "application/json"}
    if LAYA_API_KEY:
        headers["authorization"] = "Bearer %s" % LAYA_API_KEY
    request = urllib.request.Request(LAYA_URL + "/v1/systemone", data=body, headers=headers,
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except Exception:
            pass
        raise ToolError("the Laya server rejected the request (HTTP %d): %s" % (exc.code, detail))
    except OSError as exc:
        raise ToolError("cannot reach the Laya server at %s (%s). Start it, or set LAYA_URL."
                         % (LAYA_URL, exc))


def result(text: str, structured: Dict[str, Any]) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)],
                          structured_content=structured)


def meta(response: Dict[str, Any]) -> Dict[str, Any]:
    return {"checkpoint": response.get("routing", {}).get("model"),
            "latency_ms": response.get("latency_ms")}


# --------------------------------------------------------------------------- validation

def require_text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ToolError("%s must be a non-empty string, got %r" % (field, value))


def require_state(state: Any) -> None:
    if state is None or (isinstance(state, str) and not state.strip()):
        raise ToolError("state must be a non-empty string, object or array: it is the thing "
                         "being judged, and an empty one has no answer")


def check_options(options: Any) -> Dict[str, Optional[str]]:
    if not isinstance(options, dict) or not options:
        raise ToolError("options must be an object mapping option name -> description")
    if len(options) < 2:
        raise ToolError("a choice needs at least 2 options, got %d" % len(options))
    if len(options) > MAX_OPTIONS:
        raise ToolError(
            "a choice takes at most %d options here, got %d. Option descriptions share a fixed "
            "token budget with the state, so long lists get less accurate, not just slower. Ask "
            "a coarse question first and a second, narrower one inside the winning group."
            % (MAX_OPTIONS, len(options)))
    for name, description in options.items():
        if description is not None and not isinstance(description, str):
            raise ToolError("option %r: the description must be a string or null" % name)
    return options


def check_levels(levels: Any) -> List[str]:
    if not isinstance(levels, list) or not all(isinstance(x, str) and x.strip() for x in levels):
        raise ToolError("levels must be an array of non-empty level descriptions, lowest first")
    if not MIN_LEVELS <= len(levels) <= MAX_LEVELS:
        raise ToolError("a score needs between %d and %d levels, got %d"
                         % (MIN_LEVELS, MAX_LEVELS, len(levels)))
    return levels


def check_questions(questions: Any) -> Dict[str, Any]:
    if not isinstance(questions, dict) or not questions:
        raise ToolError("questions must be a non-empty object of question id -> question")
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in ("noul", "choice", "score"):
            raise ToolError("question %r: type must be one of noul, choice, score" % qid)
        require_text(q.get("instructions"), "question %r: instructions" % qid)
        if q["type"] == "choice":
            check_options(q.get("criteria"))
        elif q["type"] == "score":
            check_levels(q.get("criteria"))
        elif q.get("criteria") is not None:
            extra = sorted(set(q["criteria"]) - {"true", "false"})
            if extra:
                raise ToolError("question %r: noul criteria may only have 'true' and 'false', "
                                 "found %s" % (qid, extra))
    return questions


# --------------------------------------------------------------------------- tools

@server.tool(
    title="Classify",
    description=("Pick one of up to 12 labelled options for a piece of state, with the full "
                 "probability distribution. Use it for routing, intent, category and triage "
                 "decisions instead of reasoning about them yourself."),
    annotations=READ_ONLY, structured_output=False)
def laya_classify(state: Any, instructions: str, options: Dict[str, Optional[str]],
                  model: Optional[str] = None) -> CallToolResult:
    require_state(state)
    require_text(instructions, "instructions")
    check_options(options)
    response = call_laya(state, {"q": {"type": "choice", "instructions": instructions,
                                       "criteria": options}}, model)
    answer = response["answers"]["q"]
    payload = {"choice": answer["choice"], "probabilities": answer["probabilities"],
               "confidence": answer["confidence"]}
    payload.update(meta(response))
    return result("%s (p=%.2f, confidence %.2f)" % (
        answer["choice"], answer["probabilities"][answer["choice"]], answer["confidence"]), payload)


@server.tool(
    title="Score",
    description=("Place a piece of state on an ordered scale of 2-10 levels, lowest first. "
                 "Returns the expected level as a float plus the distribution. Ordinal scales "
                 "are the weakest of the three primitives: where a hard boundary matters, phrase "
                 "the boundary as a laya_check instead."),
    annotations=READ_ONLY, structured_output=False)
def laya_score(state: Any, instructions: str, levels: List[str],
               model: Optional[str] = None) -> CallToolResult:
    require_state(state)
    require_text(instructions, "instructions")
    check_levels(levels)
    response = call_laya(state, {"q": {"type": "score", "instructions": instructions,
                                       "criteria": levels}}, model)
    answer = response["answers"]["q"]
    payload = {"score": answer["score"], "max_score": len(levels) - 1, "legend": answer["legend"],
               "probabilities": answer["probabilities"], "confidence": answer["confidence"]}
    payload.update(meta(response))
    nearest = answer["legend"][str(int(round(answer["score"])))]
    return result("%.2f / %d (%s), confidence %.2f"
                  % (answer["score"], len(levels) - 1, nearest, answer["confidence"]), payload)


@server.tool(
    title="Check",
    description=("The probability that a statement about the state is true. The strongest of the "
                 "three primitives: phrase the thing you actually want to know as a statement "
                 "('this passage answers the question', 'this change needs a migration') and "
                 "threshold the number yourself."),
    annotations=READ_ONLY, structured_output=False)
def laya_check(state: Any, instructions: str, true_desc: Optional[str] = None,
               false_desc: Optional[str] = None, model: Optional[str] = None) -> CallToolResult:
    require_state(state)
    require_text(instructions, "instructions")
    criteria = {k: v for k, v in (("true", true_desc), ("false", false_desc)) if v}
    question: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = criteria
    response = call_laya(state, {"q": question}, model)
    answer = response["answers"]["q"]
    payload = {"probability": answer["noul"], "confidence": answer["confidence"]}
    payload.update(meta(response))
    return result("%.2f (%s)" % (answer["noul"], "likely true" if answer["noul"] >= 0.5
                                 else "likely false"), payload)


@server.tool(
    title="Gate an action",
    description=("Judge whether an action -- a shell command, a deploy, a file write -- is safe "
                 "to run. Returns allow | confirm | block with the five signals behind it. Call "
                 "this before doing anything destructive; it costs tens of milliseconds."),
    annotations=READ_ONLY, structured_output=False)
def laya_gate(action: str, context: Optional[str] = None,
              model: Optional[str] = None) -> CallToolResult:
    require_text(action, "action")
    state: Dict[str, Any] = {"action": action}
    if context:
        state["context"] = context
    response = call_laya(state, GATE_QUESTIONS, model)
    signals = {qid: (a["noul"] if a["type"] == "noul" else a["score"])
               for qid, a in response["answers"].items()}
    blast = signals["blast_radius"]

    if signals["touches_secrets"] >= GATE_DENY_SECRETS:
        recommendation, why = "block", "touches credentials (%.2f)" % signals["touches_secrets"]
    elif blast >= GATE_DENY_BLAST:
        recommendation, why = "block", "blast radius %.2f of 3 -- shared or production systems" % blast
    elif signals["is_destructive"] >= GATE_DENY_DESTRUCTIVE and blast >= 2.0:
        recommendation, why = "block", "destructive (%.2f) with blast radius %.2f" % (
            signals["is_destructive"], blast)
    else:
        hot = {k: v for k, v in signals.items()
               if k != "blast_radius" and v >= GATE_ASK_SIGNAL}
        if hot or blast >= GATE_ASK_BLAST:
            recommendation = "confirm"
            why = ", ".join("%s %.2f" % kv for kv in sorted(hot.items(), key=lambda kv: -kv[1])) \
                or "blast radius %.2f" % blast
        else:
            recommendation, why = "allow", "no signal above %.2f, blast radius %.2f" % (
                GATE_ASK_SIGNAL, blast)

    payload = {"recommendation": recommendation, "risk": round(blast / 3.0, 4),
               "signals": {k: round(v, 4) for k, v in signals.items()}, "reason": why}
    payload.update(meta(response))
    return result("%s: %s" % (recommendation, why), payload)


@server.tool(
    title="Decide (batched)",
    description=("Ask any number of questions about one state in a single forward pass. This is "
                 "the tool to reach for when you have more than one thing to know: ten questions "
                 "cost about what one costs. `questions` maps an id to "
                 "{type: noul|choice|score, instructions, criteria}."),
    annotations=READ_ONLY, structured_output=False)
def laya_decide(state: Any, questions: Dict[str, Any],
                model: Optional[str] = None) -> CallToolResult:
    require_state(state)
    check_questions(questions)
    response = call_laya(state, questions, model)
    summary = []
    for qid, a in response["answers"].items():
        if a["type"] == "noul":
            summary.append("%s=%.2f" % (qid, a["noul"]))
        elif a["type"] == "choice":
            summary.append("%s=%s" % (qid, a["choice"]))
        else:
            summary.append("%s=%.2f" % (qid, a["score"]))
    payload = {"answers": response["answers"], "usage": response.get("usage", {})}
    payload.update(meta(response))
    return result(", ".join(summary), payload)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
