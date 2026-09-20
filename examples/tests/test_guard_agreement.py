"""The three copies of the guard, driven side by side, must reach the same verdict.

There are three places a coding agent can meet this decision: the Claude Code PreToolUse hook,
the `arbiter_gate` MCP tool, and `examples/tool_call_guard.py`. They used to hold three copies of
the question set and the thresholds, and three copies drift. They now share
`integrations/claude-code/hooks/guard_policy.py`, and this file is the test that keeps them
honest: the same commands through all three against the same stub, and the same answers out.

The MCP tool speaks its own vocabulary -- allow / confirm / block -- so its verdicts are mapped
back before comparing. That mapping is the only difference the test tolerates.
"""
import importlib.util
import json
import os
import subprocess
import sys

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from stub_server import StubServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOOKS = os.path.join(ROOT, "integrations", "claude-code", "hooks")
GUARD = os.path.join(HOOKS, "guard.py")
MCP_SERVER = os.path.join(ROOT, "integrations", "mcp", "arbiter_mcp.py")
EXAMPLE = os.path.join(ROOT, "examples", "tool_call_guard.py")
FROM_MCP = {"allow": "allow", "confirm": "ask", "block": "deny"}

spec = importlib.util.spec_from_file_location("guard_policy",
                                              os.path.join(HOOKS, "guard_policy.py"))
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

# Six commands, chosen so every layer of the policy decides at least one of them: the fast path,
# the ask floor, the deny floor, and the model in the middle.
COMMANDS = [
    "git status --short",                        # fast path
    "npm run build",                             # the model
    "pip install httpx==0.27.0",                 # the model
    "rm -rf build/",                             # a verb that reaches outside the project
    "cat ~/.ssh/id_rsa",                         # a credential path
    "git push --force origin main",              # never right by accident
]
CWD = "/srv/app"


def hook_verdict(command, url):
    event = {"session_id": "a", "hook_event_name": "PreToolUse", "tool_name": "Bash",
             "tool_input": {"command": command, "description": "do the thing"}, "cwd": CWD}
    done = subprocess.run([sys.executable, GUARD], input=json.dumps(event), capture_output=True,
                          text=True, env=dict(os.environ, ARBITER_URL=url), timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)["hookSpecificOutput"]["permissionDecision"]


def example_verdict(command, url):
    done = subprocess.run([sys.executable, EXAMPLE, "--command", command, "--cwd", CWD,
                           "--description", "do the thing", "--json"],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          env=dict(os.environ, ARBITER_URL=url, NO_COLOR="1"), timeout=60)
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout)
    if "decision" in payload:                                    # the fast path answers locally
        return payload["decision"]
    return policy.decide(payload["answers"], command)[0]


def mcp_verdict(command, url):
    async def main():
        params = StdioServerParameters(command=sys.executable, args=[MCP_SERVER],
                                       env=dict(os.environ, ARBITER_URL=url))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool("arbiter_gate", {
                    "action": command, "context": "do the thing", "cwd": CWD})

    with anyio.from_thread.start_blocking_portal() as portal:
        result = portal.call(main)
    return FROM_MCP[result.structured_content["recommendation"]]


@pytest.mark.parametrize("command", COMMANDS)
def test_the_hook_the_mcp_tool_and_the_example_agree(command):
    with StubServer() as stub:
        verdicts = {"hook": hook_verdict(command, stub.url),
                    "example": example_verdict(command, stub.url),
                    "mcp": mcp_verdict(command, stub.url)}
    assert len(set(verdicts.values())) == 1, "%s: %s" % (command, verdicts)


def test_all_three_send_the_same_state_and_the_same_questions():
    command = "pip install httpx==0.27.0"
    posted = []
    for call in (hook_verdict, example_verdict, mcp_verdict):
        with StubServer() as stub:
            call(command, stub.url)
            assert len(stub.requests) == 1, call.__name__
            posted.append(stub.requests[0])
    assert all(body["questions"] == policy.QUESTIONS for body in posted)
    assert len({json.dumps(body["state"], sort_keys=True) for body in posted}) == 1, \
        [body["state"] for body in posted]


def test_none_of_the_three_keeps_its_own_copy_of_the_questions_or_the_thresholds():
    """A grep, because the cheapest way for the three to drift is for someone to paste again."""
    for path in (GUARD, MCP_SERVER, EXAMPLE):
        source = open(path, encoding="utf-8").read()
        assert "guard_policy" in source, path
        for leaked in ("DENY_SECRETS", "DENY_BLAST", "ASK_BLAST", "ASK_ANY_SIGNAL",
                       "ASK_SIGNAL", "GATE_QUESTIONS", "is_destructive", "leaves_repo"):
            assert leaked not in source, "%s still carries %s" % (path, leaked)
