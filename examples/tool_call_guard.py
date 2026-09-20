#!/usr/bin/env python3
"""Decide whether a shell command an agent wants to run should be allowed, asked about, or denied.

This is the decision an agent harness makes hundreds of times a session, and the one place where
a 300 ms LLM call is unaffordable and a regex is not good enough. Five questions, one forward
pass, tens of milliseconds: the agent keeps its flow and the dangerous 2% still stop for a human.

`integrations/claude-code/hooks/guard.py` is this same question set wired into a Claude Code
PreToolUse hook.

    python examples/tool_call_guard.py
    python examples/tool_call_guard.py --command 'rm -rf ~/Library'
    echo '{"command": "kubectl delete ns prod", "cwd": "~/infra"}' | python examples/tool_call_guard.py
"""
import sys

import _cli
from arbiter_client import ArbiterClient, Noul, Score

QUESTIONS = {
    "is_destructive": Noul(
        "Running this command destroys or irreversibly changes something: deleting files, "
        "dropping data, force-pushing, overwriting history, terminating infrastructure.",
        {"true": "data or state is lost and cannot be trivially restored",
         "false": "it reads, builds, tests, or makes a change that is easy to undo"},
    ),
    "touches_secrets": Noul(
        "This command reads, prints, copies or transmits credentials: private keys, tokens, "
        "passwords, .env files, cloud credential files, keychains."),
    "leaves_repo": Noul(
        "This command reads or writes outside the working directory it is run in.",
        {"true": "absolute paths elsewhere, the home directory, system paths",
         "false": "everything it touches is inside the project"},
    ),
    "needs_network": Noul(
        "This command talks to the network: fetching, uploading, deploying, or calling an API."),
    "blast_radius": Score(
        "If this command does the wrong thing, how far does the damage reach?",
        ["nothing outside this shell session",
         "files in this one project",
         "the whole developer machine",
         "shared or production systems other people depend on"],
    ),
}

# The guard is a filter on an agent's autonomy, so the two numbers that matter are how often it
# stops something harmless (annoying) and how often it lets something through (expensive).
# Anything touching credentials or production is denied outright; the band in between is where
# the human gets asked rather than guessed at. The ask threshold sits well above 0.5 because the
# model is not shy about a mild yes -- `npm test` scores about 0.48 on "is this destructive", and
# a guard that stops for that is a guard people turn off.
DENY_DESTRUCTIVE = 0.75
DENY_SECRETS = 0.70
DENY_BLAST = 2.40            # between "the whole machine" and "shared or production systems"
ASK_ANY_SIGNAL = 0.55
ASK_BLAST = 1.50

SAMPLE = {
    "command": "kubectl --context prod delete namespace payments",
    "cwd": "~/work/platform/infra",
    "recent_context": ("The user asked me to clean up the leftover namespaces from the staging "
                       "experiment. I listed namespaces a moment ago."),
}


def decide(r):
    """allow | ask | deny."""
    destructive = r.noul("is_destructive")
    secrets = r.noul("touches_secrets")
    outside = r.noul("leaves_repo")
    network = r.noul("needs_network")
    blast = r.score("blast_radius")

    if secrets >= DENY_SECRETS:
        return "deny", "touches_secrets %.2f >= %.2f" % (secrets, DENY_SECRETS)
    if blast >= DENY_BLAST:
        return "deny", "blast_radius %.2f >= %.2f -- shared or production systems" % (blast, DENY_BLAST)
    if destructive >= DENY_DESTRUCTIVE and blast >= 2.0:
        return "deny", "destructive %.2f with blast_radius %.2f" % (destructive, blast)
    signals = {"destructive": destructive, "secrets": secrets,
               "outside the repo": outside, "network": network}
    hot = {k: v for k, v in signals.items() if v >= ASK_ANY_SIGNAL}
    if hot or blast >= ASK_BLAST:
        detail = ", ".join("%s %.2f" % kv for kv in sorted(hot.items(), key=lambda kv: -kv[1]))
        return "ask", (detail or "blast_radius %.2f >= %.2f" % (blast, ASK_BLAST))
    return "allow", "no signal above %.2f, blast_radius %.2f" % (ASK_ANY_SIGNAL, blast)


def main() -> int:
    p = _cli.parser(__doc__)
    p.add_argument("--command", help="the command to judge (shorthand for a one-field state)")
    p.add_argument("--cwd", default=SAMPLE["cwd"], help="working directory to report in the state")
    args = p.parse_args()
    state = ({"command": args.command, "cwd": args.cwd} if args.command
             else _cli.load_state(args, SAMPLE))
    response = ArbiterClient(base_url=args.url).system_one(state, QUESTIONS, model=args.model)
    if args.json:
        return _cli.dump(response)
    action, reason = decide(response)
    command = state.get("command", "") if isinstance(state, dict) else str(state)
    _cli.render("tool-call guard", response, action, reason,
                extra=["%s$ %s%s" % (_cli.DIM, command[:74], _cli.RESET)])
    return {"allow": 0, "ask": 1, "deny": 2}[action]


if __name__ == "__main__":
    sys.exit(_cli.run(main))
