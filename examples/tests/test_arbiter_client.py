"""The client: what it sends, what it gives back, and what it says when things go wrong."""
import json

import pytest
from stub_server import StubServer

from arbiter_client import (
    AuthError, Choice, ArbiterClient, Noul, OverloadedError, RequestError, Response, Score,
    UnreachableError, encode_questions,
)

QUESTIONS = {
    "flag": Noul("The thing is true."),
    "team": Choice("Who owns it?", {"a": "team a", "b": "team b"}),
    "size": Score("How big?", ["small", "medium", "large"]),
}


# --------------------------------------------------------------------- question encoding

def test_noul_without_criteria_omits_the_key():
    assert encode_questions({"q": Noul("x")}) == {"q": {"type": "noul", "instructions": "x"}}


def test_noul_criteria_are_passed_through():
    encoded = encode_questions({"q": Noul("x", {"true": "t", "false": "f"})})
    assert encoded["q"]["criteria"] == {"true": "t", "false": "f"}


def test_choice_and_score_shapes():
    encoded = encode_questions({"c": Choice("x", {"a": None, "b": "bee"}),
                                "s": Score("y", ["lo", "hi"])})
    assert encoded["c"] == {"type": "choice", "instructions": "x",
                            "criteria": {"a": None, "b": "bee"}}
    assert encoded["s"] == {"type": "score", "instructions": "y", "criteria": ["lo", "hi"]}


def test_a_plain_dict_question_is_accepted_unchanged():
    raw = {"type": "noul", "instructions": "already in wire shape"}
    assert encode_questions({"q": raw}) == {"q": raw}


def test_score_levels_are_copied_not_aliased():
    levels = ["lo", "hi"]
    question = Score("y", levels)
    levels.append("extra")
    assert question.criteria == ["lo", "hi"]


# --------------------------------------------------------------------- the round trip

def test_every_question_goes_in_one_request():
    with StubServer() as stub:
        ArbiterClient(base_url=stub.url).system_one("some state", QUESTIONS)
        assert len(stub.requests) == 1
        assert sorted(stub.requests[0]["questions"]) == ["flag", "size", "team"]
        assert stub.requests[0]["model"] == "auto"


def test_accessors_read_the_three_answer_shapes():
    with StubServer() as stub:
        r = ArbiterClient(base_url=stub.url).system_one("some state", QUESTIONS)
    assert 0.0 <= r.noul("flag") <= 1.0
    assert r.choice("team") in ("a", "b")
    assert 0.0 <= r.score("size") <= 2.0
    assert r.label("size") in ("small", "medium", "large")
    assert pytest.approx(sum(r.probabilities("team").values()), abs=1e-3) == 1.0
    assert pytest.approx(sum(r.probabilities("flag").values()), abs=1e-3) == 1.0
    assert 0.0 <= r.confidence("size") <= 1.0


def test_the_response_is_the_raw_body():
    with StubServer() as stub:
        r = ArbiterClient(base_url=stub.url).system_one("some state", {"flag": Noul("x")})
    assert isinstance(r, Response) and isinstance(r, dict)
    assert json.loads(json.dumps(r))["answers"]["flag"]["type"] == "noul"
    assert r.checkpoint == "english"
    assert r.routing_reason
    assert r.latency_ms > 0 and r.input_tokens > 0


def test_a_non_latin_state_is_routed_to_the_multilingual_checkpoint():
    with StubServer() as stub:
        r = ArbiterClient(base_url=stub.url).system_one("नमस्ते", {"flag": Noul("x")})
    assert r.checkpoint == "multilingual"


def test_label_rejects_a_non_score_answer():
    with StubServer() as stub:
        r = ArbiterClient(base_url=stub.url).system_one("s", {"flag": Noul("x")})
    with pytest.raises(TypeError):
        r.label("flag")


def test_an_unknown_question_id_says_what_was_answered():
    with StubServer() as stub:
        r = ArbiterClient(base_url=stub.url).system_one("s", {"flag": Noul("x")})
    with pytest.raises(KeyError, match="flag"):
        r.noul("nope")


def test_extra_router_hints_are_forwarded():
    with StubServer() as stub:
        ArbiterClient(base_url=stub.url).system_one("s", {"flag": Noul("x")}, task="invoice",
                                                 lang="de")
    assert stub.requests[0]["task"] == "invoice"
    assert stub.requests[0]["lang"] == "de"


# --------------------------------------------------------------------- errors

def test_401_names_the_environment_variable():
    with StubServer(api_key="secret") as stub:
        with pytest.raises(AuthError, match="ARBITER_API_KEY"):
            ArbiterClient(base_url=stub.url).system_one("s", {"flag": Noul("x")})


def test_the_key_is_sent_when_it_is_configured():
    with StubServer(api_key="secret") as stub:
        r = ArbiterClient(base_url=stub.url, api_key="secret").system_one("s", {"flag": Noul("x")})
    assert r.noul("flag") >= 0.0


def test_422_carries_the_server_message():
    with StubServer(fail="budget") as stub:
        with pytest.raises(RequestError, match="head_max_len"):
            ArbiterClient(base_url=stub.url).system_one("s", {"flag": Noul("x")})


def test_an_empty_question_set_is_rejected_by_the_server():
    with StubServer() as stub:
        with pytest.raises(RequestError, match="at least one question"):
            ArbiterClient(base_url=stub.url).system_one("s", {})


def test_529_is_retried_and_then_raised():
    with StubServer(fail="overload") as stub:
        client = ArbiterClient(base_url=stub.url, retries=2)
        with pytest.raises(OverloadedError, match="overloaded"):
            client.system_one("s", {"flag": Noul("x")})


def test_a_server_that_is_not_there_says_so():
    client = ArbiterClient(base_url="http://127.0.0.1:1", retries=0, timeout=2)
    with pytest.raises(UnreachableError, match="cannot reach"):
        client.system_one("s", {"flag": Noul("x")})
    assert client.ready() is False


# --------------------------------------------------------------------- configuration

def test_the_environment_supplies_the_defaults(monkeypatch):
    monkeypatch.setenv("ARBITER_URL", "http://example.invalid:9999/")
    monkeypatch.setenv("ARBITER_API_KEY", "from-env")
    client = ArbiterClient()
    assert client.base_url == "http://example.invalid:9999"
    assert client.api_key == "from-env"


def test_explicit_arguments_win_over_the_environment(monkeypatch):
    monkeypatch.setenv("ARBITER_URL", "http://example.invalid:9999")
    assert ArbiterClient(base_url="http://localhost:1234").base_url == "http://localhost:1234"


def test_ready_and_models():
    with StubServer() as stub:
        client = ArbiterClient(base_url=stub.url)
        assert client.ready() is True
        assert [m["id"] for m in client.models()["data"]] == [
            "laya-english", "laya-multilingual", "laya-typed-decisions"]
