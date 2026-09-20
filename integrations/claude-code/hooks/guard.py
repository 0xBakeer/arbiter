#!/usr/bin/env python3
"""A PreToolUse hook that asks a local arbiter server whether a Bash command is safe to run.

Claude Code sends the tool call as JSON on stdin and reads a decision as JSON on stdout. What the
hook asks, and how it turns the answers into allow, ask or deny, lives in `guard_policy.py` next
to this file; this file is the wiring: read the event, decide, write the decision, never fail the
tool call. `eval.py` scores the policy against 117 labelled events and prints the matrix.

The first version of this hook sent the raw `cwd` and the tool description to the model and
OR-ed five thresholds. Measured on those events it allowed 52 % of everyday commands and refused
three of them outright -- `git rev-parse` among them -- which is the behaviour of a guard people
turn off. The current policy allows 100 % of them and still never lets a deny-class command
through. See `docs/use-cases.md` for the before and after.

Environment:

    ARBITER_URL                 default http://localhost:8010
    ARBITER_API_KEY             optional bearer token
    ARBITER_GUARD_TIMEOUT       seconds to wait for the server, default 3
    ARBITER_GUARD_MODEL         checkpoint to ask, default laya-english (measured best here)
    ARBITER_GUARD_FAIL_CLOSED   1 = ask the user when the server cannot be reached.
                             The default is fail-open: a decision server that is down must not
                             silently stop the session, so it logs and gets out of the way.
    ARBITER_GUARD_NO_AUTO_ALLOW 1 = never emit "allow"; on a clean verdict the hook says nothing
                             and the normal permission flow applies. More conservative, and
                             what to use if you do not want this hook widening what may run.
    ARBITER_GUARD_NO_FAST_PATH  1 = send read-only commands to the server too, instead of
                             allowing them without a round trip.
    ARBITER_GUARD_LOG           append a line per decision to this file
"""
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request

# A sibling import by path, not by package: the plugin directory gets copied around on its own,
# so the hook must run with nothing on sys.path but itself.
_SPEC = importlib.util.spec_from_file_location(
    "guard_policy", os.path.join(os.path.dirname(os.path.abspath(__file__)), "guard_policy.py"))
policy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(policy)

ARBITER_URL = os.environ.get("ARBITER_URL", "http://localhost:8010").rstrip("/")
ARBITER_API_KEY = os.environ.get("ARBITER_API_KEY")
TIMEOUT = float(os.environ.get("ARBITER_GUARD_TIMEOUT", "3"))
MODEL = os.environ.get("ARBITER_GUARD_MODEL", policy.MODEL)
FAIL_CLOSED = os.environ.get("ARBITER_GUARD_FAIL_CLOSED") == "1"
NO_AUTO_ALLOW = os.environ.get("ARBITER_GUARD_NO_AUTO_ALLOW") == "1"
NO_FAST_PATH = os.environ.get("ARBITER_GUARD_NO_FAST_PATH") == "1"
LOG = os.environ.get("ARBITER_GUARD_LOG")


def log(message, command=None):
    """One line per decision, with the command on it: a guard log nobody can audit is not one."""
    if not LOG:
        return
    try:
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write("%s %s%s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), message,
                                        "  $ %s" % command.replace("\n", " ") if command else ""))
    except OSError:
        pass


def emit(decision, reason, command=None):
    """Write the PreToolUse decision and stop.

    An "allow" with ARBITER_GUARD_NO_AUTO_ALLOW set becomes silence, which is not the same thing:
    silence means "no opinion, run the normal permission flow", while "allow" skips it.
    """
    log("%s: %s" % (decision, reason), command)
    if decision == "allow" and NO_AUTO_ALLOW:
        sys.exit(0)
    json.dump({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                      "permissionDecision": decision,
                                      "permissionDecisionReason": reason}}, sys.stdout)
    sys.stdout.write("\n")
    sys.exit(0)


def ask_arbiter(state):
    body = json.dumps({"state": state, "questions": policy.QUESTIONS, "model": MODEL}).encode()
    headers = {"content-type": "application/json"}
    if ARBITER_API_KEY:
        headers["authorization"] = "Bearer %s" % ARBITER_API_KEY
    request = urllib.request.Request(ARBITER_URL + "/v1/systemone", data=body, headers=headers,
                                     method="POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode())


def main():
    try:
        event = json.load(sys.stdin)
    except ValueError:
        sys.exit(0)                                      # not our business to fail the tool call
    if event.get("tool_name") != "Bash":
        sys.exit(0)
    tool_input = event.get("tool_input") or {}
    command = tool_input.get("command")
    if not command or not command.strip():
        sys.exit(0)

    if not NO_FAST_PATH and policy.is_read_only(command):
        decision, _, reason = policy.decide(None, command)     # no round trip needed
        emit(decision, reason, command)

    state = policy.build_state(command, tool_input.get("description"), event.get("cwd"))
    try:
        response = ask_arbiter(state)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log("unreachable: %s" % exc, command)
        if FAIL_CLOSED:
            emit("ask", "Arbiter guard could not reach %s (%s); asking rather than guessing."
                 % (ARBITER_URL, exc), command)
        print("arbiter guard: %s unreachable (%s); allowing" % (ARBITER_URL, exc), file=sys.stderr)
        sys.exit(0)

    decision, _, reason = policy.decide(response["answers"], command)
    emit(decision, reason, command)


if __name__ == "__main__":
    main()
