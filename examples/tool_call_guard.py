#!/usr/bin/env python3
"""Decide whether a shell command an agent wants to run should be allowed, asked about, or denied.

This is the decision an agent harness makes hundreds of times a session, and the one place where
a 300 ms LLM call is unaffordable and a regex is not good enough on its own. Six questions, one
forward pass, tens of milliseconds -- and for the read-only half of a session, no call at all.

The policy is shared with the Claude Code PreToolUse hook and the `arbiter_gate` MCP tool, and
it lives in `integrations/claude-code/hooks/guard_policy.py`. That file also carries the
measurements behind it: the question wordings, the weights, the two cut lines and the short list
of text patterns the model was measured to get wrong. `integrations/claude-code/hooks/eval.py`
scores the whole thing against 117 labelled PreToolUse events.

    python examples/tool_call_guard.py
    python examples/tool_call_guard.py --command 'rm -rf ~/Library'
    echo '{"command": "kubectl delete ns prod", "cwd": "~/infra"}' | python examples/tool_call_guard.py
"""
import importlib.util
import os
import sys

import _cli
from arbiter_client import ArbiterClient

POLICY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "integrations", "claude-code", "hooks", "guard_policy.py")
_spec = importlib.util.spec_from_file_location("guard_policy", POLICY_PATH)
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)

SAMPLE = {
    "command": "kubectl --context prod delete namespace payments",
    "cwd": "~/work/platform/infra",
    "description": "Clean up the leftover namespaces from the staging experiment",
}


def main() -> int:
    p = _cli.parser(__doc__)
    p.add_argument("--command", help="the command to judge (shorthand for a one-field state)")
    p.add_argument("--cwd", default=SAMPLE["cwd"], help="working directory the command runs in")
    p.add_argument("--description", help="what the agent says it is doing, as Claude Code sends it")
    p.set_defaults(model=policy.MODEL)
    args = p.parse_args()
    raw = ({"command": args.command, "cwd": args.cwd, "description": args.description}
           if args.command else _cli.load_state(args, SAMPLE))
    if not isinstance(raw, dict):
        raw = {"command": str(raw).strip()}
    command = raw["command"]

    # The read-only fast path: verbs that cannot change or transmit anything are allowed here,
    # with no round trip. It is the same check the hook makes, and it is why the guard costs
    # nothing at all on about half the Bash calls in a session.
    if policy.is_read_only(command):
        action, risk, reason = policy.decide(None, command)
        if args.json:
            return _cli.dump({"decision": action, "risk": risk, "reason": reason,
                              "asked_the_server": False})
        print("  %s$ %s%s" % (_cli.DIM, command[:74], _cli.RESET))
        _cli.verdict(action, reason)
        return 0

    state = policy.build_state(command, raw.get("description"), raw.get("cwd"))
    response = ArbiterClient(base_url=args.url).system_one(state, policy.QUESTIONS,
                                                           model=args.model)
    if args.json:
        return _cli.dump(response)
    action, risk, reason = policy.decide(response["answers"], command)
    _cli.render("tool-call guard", response, action, reason,
                extra=["%s$ %s%s" % (_cli.DIM, command[:74], _cli.RESET)])
    return {"allow": 0, "ask": 1, "deny": 2}[action]


if __name__ == "__main__":
    sys.exit(_cli.run(main))
