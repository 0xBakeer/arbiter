"""The Claude Code PreToolUse hook: the decision it returns, and the shape it returns it in.

The hook is run the way Claude Code runs it -- as a subprocess with the event JSON on stdin --
because the contract being tested is that stdout is exactly one JSON object of the documented
shape, and that nothing else ever lands there.
"""
import importlib.util
import json
import os
import subprocess
import sys

import pytest
from stub_server import StubServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLUGIN = os.path.join(ROOT, "integrations", "claude-code")
GUARD = os.path.join(PLUGIN, "hooks", "guard.py")


def load_guard():
    spec = importlib.util.spec_from_file_location("guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load_guard()


def noul(p):
    return {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}


def score(v):
    return {"type": "score", "score": v, "legend": {"0": "a", "1": "b", "2": "c", "3": "d"},
            "probabilities": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}, "confidence": 0.25}


SAFE = {"is_destructive": noul(0.03), "touches_secrets": noul(0.02), "leaves_repo": noul(0.04),
        "needs_network": noul(0.05), "blast_radius": score(0.2)}


def run(event, url, **env):
    environment = dict(os.environ, LAYA_URL=url)
    environment.update(env)
    return subprocess.run([sys.executable, GUARD], input=json.dumps(event), capture_output=True,
                          text=True, env=environment, timeout=30)


def bash(command="ls -la", cwd="/tmp/project"):
    return {"hook_event_name": "PreToolUse", "tool_name": "Bash",
            "tool_input": {"command": command}, "cwd": cwd}


# --------------------------------------------------------------------- the mapping

def test_a_quiet_command_is_allowed():
    assert guard.decide(SAFE)[0] == "allow"


def test_credentials_are_denied():
    assert guard.decide(dict(SAFE, touches_secrets=noul(0.8)))[0] == "deny"


def test_production_blast_radius_is_denied():
    assert guard.decide(dict(SAFE, blast_radius=score(2.7)))[0] == "deny"


def test_the_middle_band_asks():
    decision, reason = guard.decide(dict(SAFE, is_destructive=noul(0.6)))
    assert decision == "ask"
    assert "is_destructive 0.60" in reason


def test_a_mild_yes_is_not_enough_to_interrupt():
    assert guard.decide(dict(SAFE, is_destructive=noul(0.48)))[0] == "allow"


def test_every_reason_names_the_number_behind_it():
    for answers in (SAFE, dict(SAFE, touches_secrets=noul(0.9)), dict(SAFE, blast_radius=score(2.9))):
        assert any(character.isdigit() for character in guard.decide(answers)[1])


# --------------------------------------------------------------------- the wire contract

@pytest.mark.parametrize("answers,expected", [
    (SAFE, "allow"),
    (dict(SAFE, is_destructive=noul(0.6)), "ask"),
    (dict(SAFE, touches_secrets=noul(0.95)), "deny"),
])
def test_the_hook_emits_the_documented_json(answers, expected):
    with StubServer(scripted=answers) as stub:
        done = run(bash("rm -rf build"), stub.url)
    assert done.returncode == 0
    payload = json.loads(done.stdout)["hookSpecificOutput"]
    assert payload["hookEventName"] == "PreToolUse"
    assert payload["permissionDecision"] == expected
    assert payload["permissionDecisionReason"].startswith("Laya:")


def test_the_command_and_the_directory_reach_the_server():
    with StubServer(scripted=SAFE) as stub:
        run(bash("git push --force", cwd="/srv/app"), stub.url)
        state = stub.requests[0]["state"]
    assert state == {"command": "git push --force", "cwd": "/srv/app"}


def test_a_description_is_passed_as_intent():
    event = bash()
    event["tool_input"]["description"] = "list the build output"
    with StubServer(scripted=SAFE) as stub:
        run(event, stub.url)
        assert stub.requests[0]["state"]["intent"] == "list the build output"


@pytest.mark.parametrize("event", [
    {"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}},
    {"tool_name": "Bash", "tool_input": {"command": "   "}},
    {"tool_name": "Bash", "tool_input": {}},
])
def test_the_hook_stays_out_of_calls_that_are_not_its_business(event):
    with StubServer() as stub:
        done = run(event, stub.url)
        assert stub.requests == []
    assert done.returncode == 0
    assert done.stdout == ""


def test_malformed_input_does_not_fail_the_tool_call():
    done = subprocess.run([sys.executable, GUARD], input="not json", capture_output=True,
                          text=True, timeout=30)
    assert done.returncode == 0
    assert done.stdout == ""


# --------------------------------------------------------------------- server down

def test_an_unreachable_server_fails_open():
    done = run(bash(), "http://127.0.0.1:1")
    assert done.returncode == 0
    assert done.stdout == ""
    assert "unreachable" in done.stderr


def test_fail_closed_asks_instead():
    done = run(bash(), "http://127.0.0.1:1", LAYA_GUARD_FAIL_CLOSED="1")
    payload = json.loads(done.stdout)["hookSpecificOutput"]
    assert payload["permissionDecision"] == "ask"
    assert "could not reach" in payload["permissionDecisionReason"]


def test_no_auto_allow_turns_an_allow_into_silence():
    with StubServer(scripted=SAFE) as stub:
        done = run(bash(), stub.url, LAYA_GUARD_NO_AUTO_ALLOW="1")
    assert done.stdout == ""
    with StubServer(scripted=dict(SAFE, touches_secrets=noul(0.95))) as stub:
        done = run(bash(), stub.url, LAYA_GUARD_NO_AUTO_ALLOW="1")
    assert json.loads(done.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_the_log_file_records_the_decision(tmp_path):
    log = tmp_path / "guard.log"
    with StubServer(scripted=SAFE) as stub:
        run(bash(), stub.url, LAYA_GUARD_LOG=str(log))
    assert "allow" in log.read_text()


# --------------------------------------------------------------------- the plugin files

def test_the_plugin_files_are_valid_json_and_point_at_files_that_exist():
    manifest = json.load(open(os.path.join(PLUGIN, ".claude-plugin", "plugin.json")))
    assert manifest["name"] == "laya-decisions"

    hooks = json.load(open(os.path.join(PLUGIN, "hooks", "hooks.json")))
    entry = hooks["hooks"]["PreToolUse"][0]
    assert entry["matcher"] == "Bash"
    assert entry["hooks"][0]["type"] == "command"
    assert "${CLAUDE_PLUGIN_ROOT}/hooks/guard.py" in entry["hooks"][0]["command"]

    mcp = json.load(open(os.path.join(PLUGIN, ".mcp.json")))
    args = mcp["mcpServers"]["laya"]["args"]
    target = args[0].replace("${CLAUDE_PLUGIN_ROOT}", PLUGIN)
    assert os.path.exists(target), target


def test_the_skill_declares_a_name_and_a_description():
    text = open(os.path.join(PLUGIN, "skills", "laya-decisions", "SKILL.md")).read()
    assert text.startswith("---\n")
    front = text.split("---", 2)[1]
    assert "name: laya-decisions" in front
    assert "description:" in front
