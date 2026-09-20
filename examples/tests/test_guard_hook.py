"""The Claude Code PreToolUse hook: the decision it returns, and the shape it returns it in.

The hook is run the way Claude Code runs it -- as a subprocess with the event JSON on stdin --
because the contract being tested is that stdout is exactly one JSON object of the documented
shape, and that nothing else ever lands there.

The policy the hook applies lives in `guard_policy.py` and is tested in `test_guard_policy.py`;
what is tested here is the wiring: which fields of the event reach the server, what the three
environment switches do, and that a server that is not there never fails a tool call.
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
HOOKS = os.path.join(PLUGIN, "hooks")
GUARD = os.path.join(HOOKS, "guard.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load("guard_policy", os.path.join(HOOKS, "guard_policy.py"))


def noul(p):
    return {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}


def score(v):
    return {"type": "score", "score": v, "legend": {"0": "a", "1": "b", "2": "c", "3": "d"},
            "probabilities": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}, "confidence": 0.25}


QUIET = {"destroys_data": noul(0.04), "reads_secrets": noul(0.03),
         "touches_foreign_paths": noul(0.05), "sends_data_to_remote": noul(0.06),
         "serious_harm": noul(0.02), "blast_radius": score(0.2)}
ALARMED = dict(QUIET, serious_harm=noul(0.95), destroys_data=noul(0.92))


def run(event, url, **env):
    environment = dict(os.environ, ARBITER_URL=url)
    environment.update(env)
    return subprocess.run([sys.executable, GUARD], input=json.dumps(event), capture_output=True,
                          text=True, env=environment, timeout=30)


def bash(command="npm test", cwd="/tmp/project", description=None):
    tool_input = {"command": command}
    if description:
        tool_input["description"] = description
    return {"session_id": "3f1a", "hook_event_name": "PreToolUse", "tool_name": "Bash",
            "tool_input": tool_input, "cwd": cwd}


def decision_of(done):
    return json.loads(done.stdout)["hookSpecificOutput"]["permissionDecision"]


# --------------------------------------------------------------------- the wire contract

@pytest.mark.parametrize("answers,expected", [
    (QUIET, "allow"),
    (dict(QUIET, destroys_data=noul(0.92), serious_harm=noul(0.35)), "ask"),
    (ALARMED, "deny"),
])
def test_the_hook_emits_the_documented_json(answers, expected):
    with StubServer(scripted=answers) as stub:
        done = run(bash("npm run build"), stub.url)
    assert done.returncode == 0
    payload = json.loads(done.stdout)["hookSpecificOutput"]
    assert payload["hookEventName"] == "PreToolUse"
    assert payload["permissionDecision"] == expected
    assert payload["permissionDecisionReason"].startswith("Arbiter:")


def test_the_state_is_the_command_the_description_and_a_word_for_the_directory():
    with StubServer(scripted=QUIET) as stub:
        run(bash("npm ci", cwd="/srv/app", description="Install the dependencies"), stub.url)
        state = stub.requests[0]["state"]
    assert state == {"command": "npm ci", "description": "Install the dependencies",
                     "cwd_kind": "system"}


def test_a_missing_description_is_simply_absent():
    with StubServer(scripted=QUIET) as stub:
        run(bash("npm ci", cwd="/srv/app"), stub.url)
        assert stub.requests[0]["state"] == {"command": "npm ci", "cwd_kind": "system"}


def test_the_hook_asks_the_shared_question_set():
    with StubServer(scripted=QUIET) as stub:
        run(bash(), stub.url)
    assert stub.requests[0]["questions"] == policy.QUESTIONS
    assert stub.requests[0]["model"] == policy.MODEL


def test_a_scratch_directory_no_longer_changes_the_verdict():
    """The raw cwd used to turn the same `echo` from allow into ask; a word for it does not."""
    seen = []
    for cwd in ("/tmp/x", "/private/tmp/claude-501/a-long-scratch-path", "/srv/app"):
        with StubServer(scripted=QUIET) as stub:
            done = run(bash("npm run build", cwd=cwd), stub.url)
        seen.append(decision_of(done))
    assert seen == ["allow", "allow", "allow"]


# --------------------------------------------------------------------- the fast path

def test_a_read_only_command_is_allowed_without_asking_anyone():
    with StubServer(scripted=ALARMED) as stub:
        done = run(bash("git rev-parse --show-toplevel"), stub.url)
        assert stub.requests == [], "the fast path made a round trip"
    assert decision_of(done) == "allow"


def test_the_fast_path_can_be_turned_off():
    with StubServer(scripted=ALARMED) as stub:
        done = run(bash("git rev-parse --show-toplevel"), stub.url, ARBITER_GUARD_NO_FAST_PATH="1")
        assert len(stub.requests) == 1
    assert decision_of(done) == "deny"


# --------------------------------------------------------------------- the text floors

def test_a_catastrophic_command_is_denied_even_when_the_model_is_calm():
    with StubServer(scripted=QUIET) as stub:
        done = run(bash("git push --force origin main"), stub.url)
    payload = json.loads(done.stdout)["hookSpecificOutput"]
    assert payload["permissionDecision"] == "deny"
    assert "never right by accident" in payload["permissionDecisionReason"]


def test_a_reaching_command_is_never_silently_allowed():
    with StubServer(scripted=QUIET) as stub:
        done = run(bash("rm -rf build/"), stub.url)
    assert decision_of(done) == "ask"


# --------------------------------------------------------------------- staying out of the way

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


def test_an_unreachable_server_fails_open():
    done = run(bash(), "http://127.0.0.1:1")
    assert done.returncode == 0
    assert done.stdout == ""
    assert "unreachable" in done.stderr


def test_fail_closed_asks_instead():
    done = run(bash(), "http://127.0.0.1:1", ARBITER_GUARD_FAIL_CLOSED="1")
    payload = json.loads(done.stdout)["hookSpecificOutput"]
    assert payload["permissionDecision"] == "ask"
    assert "could not reach" in payload["permissionDecisionReason"]


def test_no_auto_allow_turns_an_allow_into_silence():
    with StubServer(scripted=QUIET) as stub:
        done = run(bash(), stub.url, ARBITER_GUARD_NO_AUTO_ALLOW="1")
    assert done.stdout == ""
    with StubServer(scripted=ALARMED) as stub:
        done = run(bash(), stub.url, ARBITER_GUARD_NO_AUTO_ALLOW="1")
    assert decision_of(done) == "deny"


def test_the_log_file_records_the_decision(tmp_path):
    log = tmp_path / "guard.log"
    with StubServer(scripted=QUIET) as stub:
        run(bash(), stub.url, ARBITER_GUARD_LOG=str(log))
    assert "allow" in log.read_text()


# --------------------------------------------------------------------- the plugin files

def test_the_plugin_files_are_valid_json_and_point_at_files_that_exist():
    manifest = json.load(open(os.path.join(PLUGIN, ".claude-plugin", "plugin.json")))
    assert manifest["name"] == "arbiter"

    hooks = json.load(open(os.path.join(PLUGIN, "hooks", "hooks.json")))
    entry = hooks["hooks"]["PreToolUse"][0]
    assert entry["matcher"] == "Bash"
    assert entry["hooks"][0]["type"] == "command"
    assert "${CLAUDE_PLUGIN_ROOT}/hooks/guard.py" in entry["hooks"][0]["command"]

    mcp = json.load(open(os.path.join(PLUGIN, ".mcp.json")))
    args = mcp["mcpServers"]["arbiter"]["args"]
    target = args[0].replace("${CLAUDE_PLUGIN_ROOT}", PLUGIN)
    assert os.path.exists(target), target


def test_the_hook_runs_with_nothing_on_the_path_but_its_own_directory():
    """The plugin directory gets copied around on its own, so guard.py may not import the repo."""
    done = subprocess.run([sys.executable, GUARD], input="{}", capture_output=True, text=True,
                          cwd=os.path.dirname(ROOT), env=dict(os.environ, PYTHONPATH=""),
                          timeout=30)
    assert done.returncode == 0 and done.stderr == ""


def test_the_skill_declares_a_name_and_a_description():
    text = open(os.path.join(PLUGIN, "skills", "arbiter-decisions", "SKILL.md")).read()
    assert text.startswith("---\n")
    front = text.split("---", 2)[1]
    assert "name: arbiter-decisions" in front
    assert "description:" in front
