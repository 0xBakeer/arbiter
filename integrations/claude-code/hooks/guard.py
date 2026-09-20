#!/usr/bin/env python3
"""A PreToolUse hook that asks a local Laya server whether a Bash command is safe to run.

Claude Code sends the tool call as JSON on stdin and reads a decision as JSON on stdout. This
hook turns that into five questions in one forward pass -- destructive, secrets, outside the
repo, network, blast radius -- and maps the probabilities to allow, ask or deny. It adds tens of
milliseconds to a Bash call, which is why it can run on every one of them.

The thresholds are here, in the hook, not in the model. Edit them: they are the only thing
standing between "the agent works without interruption" and "the agent asks about everything".

Environment:

    LAYA_URL                 default http://localhost:8010
    LAYA_API_KEY             optional bearer token
    LAYA_GUARD_TIMEOUT       seconds to wait for the server, default 3
    LAYA_GUARD_FAIL_CLOSED   1 = ask the user when the server cannot be reached.
                             The default is fail-open: a decision server that is down must not
                             silently stop the session, so it logs and gets out of the way.
    LAYA_GUARD_NO_AUTO_ALLOW 1 = never emit "allow"; on a clean verdict the hook says nothing
                             and the normal permission flow applies. More conservative, and
                             what to use if you do not want this hook widening what may run.
    LAYA_GUARD_LOG           append a line per decision to this file
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

LAYA_URL = os.environ.get("LAYA_URL", "http://localhost:8010").rstrip("/")
LAYA_API_KEY = os.environ.get("LAYA_API_KEY")
TIMEOUT = float(os.environ.get("LAYA_GUARD_TIMEOUT", "3"))
FAIL_CLOSED = os.environ.get("LAYA_GUARD_FAIL_CLOSED") == "1"
NO_AUTO_ALLOW = os.environ.get("LAYA_GUARD_NO_AUTO_ALLOW") == "1"
LOG = os.environ.get("LAYA_GUARD_LOG")

# The same question set as examples/tool_call_guard.py and integrations/mcp/laya_mcp.py. It is
# repeated rather than imported because a plugin directory gets copied around on its own.
QUESTIONS = {
    "is_destructive": {
        "type": "noul",
        "instructions": ("Running this command destroys or irreversibly changes something: "
                         "deleting files, dropping data, force-pushing, overwriting history, "
                         "terminating infrastructure."),
        "criteria": {"true": "data or state is lost and cannot be trivially restored",
                     "false": "it reads, builds, tests, or makes a change that is easy to undo"},
    },
    "touches_secrets": {
        "type": "noul",
        "instructions": ("This command reads, prints, copies or transmits credentials: private "
                         "keys, tokens, passwords, .env files, cloud credentials, keychains."),
    },
    "leaves_repo": {
        "type": "noul",
        "instructions": "This command reads or writes outside the working directory it is run in.",
    },
    "needs_network": {
        "type": "noul",
        "instructions": "This command talks to the network: fetching, uploading, deploying, or calling an API.",
    },
    "blast_radius": {
        "type": "score",
        "instructions": "If this command does the wrong thing, how far does the damage reach?",
        "criteria": ["nothing outside this shell session",
                     "files in this one project",
                     "the whole developer machine",
                     "shared or production systems other people depend on"],
    },
}
DENY_DESTRUCTIVE = 0.75
DENY_SECRETS = 0.70
DENY_BLAST = 2.40
# Well above 0.5: the model is not shy about a mild yes, and `npm test` scores around 0.48 on
# "is this destructive". A guard that stops for that is a guard people turn off.
ASK_SIGNAL = 0.55
ASK_BLAST = 1.50


def log(message):
    if not LOG:
        return
    try:
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write("%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), message))
    except OSError:
        pass


def emit(decision, reason):
    """Write the PreToolUse decision and stop.

    An "allow" with LAYA_GUARD_NO_AUTO_ALLOW set becomes silence, which is not the same thing:
    silence means "no opinion, run the normal permission flow", while "allow" skips it.
    """
    log("%s: %s" % (decision, reason))
    if decision == "allow" and NO_AUTO_ALLOW:
        sys.exit(0)
    json.dump({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                      "permissionDecision": decision,
                                      "permissionDecisionReason": reason}}, sys.stdout)
    sys.stdout.write("\n")
    sys.exit(0)


def ask_laya(state):
    body = json.dumps({"state": state, "questions": QUESTIONS, "model": "auto"}).encode()
    headers = {"content-type": "application/json"}
    if LAYA_API_KEY:
        headers["authorization"] = "Bearer %s" % LAYA_API_KEY
    request = urllib.request.Request(LAYA_URL + "/v1/systemone", data=body, headers=headers,
                                     method="POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode())


def decide(answers):
    """allow | ask | deny, and the sentence that explains it to the user."""
    signal = {qid: (a["noul"] if a["type"] == "noul" else a["score"])
              for qid, a in answers.items()}
    blast = signal["blast_radius"]

    if signal["touches_secrets"] >= DENY_SECRETS:
        return "deny", "Laya: this reads or moves credentials (%.2f)." % signal["touches_secrets"]
    if blast >= DENY_BLAST:
        return "deny", ("Laya: blast radius %.2f of 3 -- shared or production systems." % blast)
    if signal["is_destructive"] >= DENY_DESTRUCTIVE and blast >= 2.0:
        return "deny", ("Laya: destructive (%.2f) with a machine-wide blast radius (%.2f)."
                        % (signal["is_destructive"], blast))

    hot = {k: v for k, v in signal.items() if k != "blast_radius" and v >= ASK_SIGNAL}
    if hot or blast >= ASK_BLAST:
        detail = ", ".join("%s %.2f" % kv for kv in sorted(hot.items(), key=lambda kv: -kv[1]))
        return "ask", "Laya: %s (blast radius %.2f of 3)." % (detail or "elevated risk", blast)
    return "allow", ("Laya: nothing above %.2f, blast radius %.2f of 3."
                     % (ASK_SIGNAL, blast))


def main():
    try:
        event = json.load(sys.stdin)
    except ValueError:
        sys.exit(0)                                      # not our business to fail the tool call
    if event.get("tool_name") != "Bash":
        sys.exit(0)
    command = (event.get("tool_input") or {}).get("command")
    if not command or not command.strip():
        sys.exit(0)

    state = {"command": command, "cwd": event.get("cwd", "")}
    description = (event.get("tool_input") or {}).get("description")
    if description:
        state["intent"] = description

    try:
        response = ask_laya(state)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log("unreachable: %s" % exc)
        if FAIL_CLOSED:
            emit("ask", "Laya guard could not reach %s (%s); asking rather than guessing."
                 % (LAYA_URL, exc))
        print("laya guard: %s unreachable (%s); allowing" % (LAYA_URL, exc), file=sys.stderr)
        sys.exit(0)

    emit(*decide(response["answers"]))


if __name__ == "__main__":
    main()
