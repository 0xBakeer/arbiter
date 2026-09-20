"""The MCP server, driven by a real MCP client over stdio against the stub arbiter server.

Spawning the server as a subprocess and talking to it with the SDK's own client is the only test
that proves the thing an agent will actually do works: the tool schemas, the structured content,
and the errors a badly shaped call gets back.
"""
import importlib.util
import json
import os
import sys

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from stub_server import StubServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVER = os.path.join(ROOT, "integrations", "mcp", "arbiter_mcp.py")
_spec = importlib.util.spec_from_file_location(
    "guard_policy", os.path.join(ROOT, "integrations", "claude-code", "hooks", "guard_policy.py"))
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)
TOOLS = ["arbiter_classify", "arbiter_score", "arbiter_check", "arbiter_gate", "arbiter_decide"]


def talk(url, body):
    """Run `body(session)` against a freshly spawned arbiter-mcp pointed at `url`."""
    async def main():
        params = StdioServerParameters(command=sys.executable, args=[SERVER],
                                       env=dict(os.environ, ARBITER_URL=url))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await body(session)

    with anyio.from_thread.start_blocking_portal() as portal:
        return portal.call(main)


def text_of(result):
    return "".join(c.text for c in result.content if c.type == "text")


# --------------------------------------------------------------------- the tool surface

def test_the_five_tools_are_advertised_with_annotations():
    with StubServer() as stub:
        async def body(session):
            return await session.list_tools()
        tools = talk(stub.url, body).tools
    assert sorted(t.name for t in tools) == sorted(TOOLS)
    for tool in tools:
        assert tool.description and len(tool.description) > 40
        assert tool.annotations is not None, tool.name
        assert tool.annotations.read_only_hint is True, tool.name
        assert tool.annotations.destructive_hint is False, tool.name


def test_check_returns_a_probability_as_structured_content():
    with StubServer() as stub:
        async def body(session):
            return await session.call_tool("arbiter_check", {
                "state": "the build is broken again",
                "instructions": "This message reports a failure."})
        result = talk(stub.url, body)
    assert result.is_error is False
    assert 0.0 <= result.structured_content["probability"] <= 1.0
    assert result.structured_content["checkpoint"] == "english"
    assert text_of(result)


def test_classify_returns_the_full_distribution():
    with StubServer() as stub:
        async def body(session):
            return await session.call_tool("arbiter_classify", {
                "state": "my card was charged twice",
                "instructions": "Which team owns this?",
                "options": {"billing": "payments", "technical": "bugs"}})
        result = talk(stub.url, body)
    payload = result.structured_content
    assert payload["choice"] in ("billing", "technical")
    assert pytest.approx(sum(payload["probabilities"].values()), abs=1e-3) == 1.0


def test_score_reports_the_scale_it_was_given():
    with StubServer() as stub:
        async def body(session):
            return await session.call_tool("arbiter_score", {
                "state": "the site is down for everyone",
                "instructions": "How bad is this?",
                "levels": ["fine", "degraded", "outage"]})
        result = talk(stub.url, body)
    assert result.structured_content["max_score"] == 2
    assert result.structured_content["legend"]["0"] == "fine"


def test_gate_recommends_one_of_three_actions_and_shows_its_signals():
    with StubServer() as stub:
        async def body(session):
            return await session.call_tool("arbiter_gate", {
                "action": "rsync -a ~/Projects backup:/snapshots",
                "context": "taking a backup", "cwd": "/srv/app"})
        result = talk(stub.url, body)
    payload = result.structured_content
    assert payload["recommendation"] in ("allow", "confirm", "block")
    assert sorted(payload["signals"]) == sorted(policy.QUESTIONS)
    assert payload["reason"] in text_of(result)


def test_gate_answers_a_read_only_command_without_calling_the_server_at_all():
    with StubServer() as stub:
        async def body(session):
            return await session.call_tool("arbiter_gate", {"action": "git status --short"})
        result = talk(stub.url, body)
        assert stub.requests == []
    assert result.structured_content["recommendation"] == "allow"


def test_gate_blocks_a_pattern_the_text_settles_however_the_model_answers():
    with StubServer() as stub:
        async def body(session):
            return await session.call_tool("arbiter_gate", {"action": "crontab -r"})
        result = talk(stub.url, body)
    assert result.structured_content["recommendation"] == "block"


def test_decide_answers_every_question_in_one_upstream_request():
    with StubServer() as stub:
        async def body(session):
            return await session.call_tool("arbiter_decide", {
                "state": {"ticket": "charged twice, furious"},
                "questions": {
                    "refund": {"type": "noul", "instructions": "They want money back."},
                    "team": {"type": "choice", "instructions": "Who owns it?",
                             "criteria": {"billing": "payments", "tech": "bugs"}},
                    "urgency": {"type": "score", "instructions": "How urgent?",
                                "criteria": ["low", "high"]}}})
        result = talk(stub.url, body)
        assert len(stub.requests) == 1
    assert sorted(result.structured_content["answers"]) == ["refund", "team", "urgency"]


# --------------------------------------------------------------------- rejecting bad input

def call_expecting_error(url, tool, arguments):
    async def body(session):
        return await session.call_tool(tool, arguments)
    result = talk(url, body)
    assert result.is_error is True
    return text_of(result)


def test_too_many_options_are_refused_with_the_reason():
    with StubServer() as stub:
        message = call_expecting_error(stub.url, "arbiter_classify", {
            "state": "x", "instructions": "pick one",
            "options": {"opt%d" % i: "option %d" % i for i in range(13)}})
    assert "at most 12" in message
    assert "narrower" in message


def test_a_single_option_is_refused():
    with StubServer() as stub:
        message = call_expecting_error(stub.url, "arbiter_classify", {
            "state": "x", "instructions": "pick one", "options": {"only": "one"}})
    assert "at least 2 options" in message


def test_a_score_outside_two_to_ten_levels_is_refused():
    with StubServer() as stub:
        message = call_expecting_error(stub.url, "arbiter_score", {
            "state": "x", "instructions": "how big", "levels": ["only one"]})
    assert "between 2 and 10 levels" in message


def test_an_empty_state_is_refused():
    with StubServer() as stub:
        message = call_expecting_error(stub.url, "arbiter_check", {
            "state": "   ", "instructions": "is it true"})
    assert "state must be" in message


def test_an_empty_instruction_is_refused():
    with StubServer() as stub:
        message = call_expecting_error(stub.url, "arbiter_check", {"state": "x", "instructions": ""})
    assert "instructions must be" in message


def test_an_unknown_question_type_is_refused():
    with StubServer() as stub:
        message = call_expecting_error(stub.url, "arbiter_decide", {
            "state": "x", "questions": {"q": {"type": "guess", "instructions": "hm"}}})
    assert "noul, choice, score" in message


def test_a_choice_question_inside_decide_is_validated_too():
    with StubServer() as stub:
        message = call_expecting_error(stub.url, "arbiter_decide", {
            "state": "x", "questions": {"q": {"type": "choice", "instructions": "pick",
                                              "criteria": {"a": "only one"}}}})
    assert "at least 2 options" in message


def test_a_server_that_is_not_there_is_reported_as_a_tool_error():
    message = call_expecting_error("http://127.0.0.1:1", "arbiter_check",
                                   {"state": "x", "instructions": "is it true"})
    assert "cannot reach the arbiter server" in message
