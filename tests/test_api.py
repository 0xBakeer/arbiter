"""The Jev HTTP contract: what is accepted, what is rejected, and in what shape."""
import pytest
from fastapi.testclient import TestClient

from server.app import create_app, resolve_model, validate_questions, ValidationError, Question
from tests.stub import StubEngine

STATE = "I was charged twice and support has not replied."
NOUL = {"type": "noul", "instructions": "The customer is angry."}


def client(engine=None, api_key=None):
    app = create_app(engine or StubEngine())
    app.state.api_key = api_key
    return TestClient(app), app


def post(c, body, **kw):
    return c.post("/v1/systemone", json=body, **kw)


# --------------------------------------------------------------------- validation

def q(**kw):
    return Question(**kw)


def test_noul_criteria_are_optional():
    validate_questions({"a": q(type="noul", instructions="x")})
    validate_questions({"a": q(type="noul", instructions="x", criteria={"true": "yes"})})
    validate_questions({"a": q(type="noul", instructions="x", criteria={"true": "y", "false": "n"})})


def test_noul_rejects_foreign_criteria_keys():
    with pytest.raises(ValidationError, match="maybe"):
        validate_questions({"a": q(type="noul", instructions="x", criteria={"maybe": "?"})})


def test_noul_rejects_list_criteria():
    with pytest.raises(ValidationError, match="must be an object"):
        validate_questions({"a": q(type="noul", instructions="x", criteria=["yes", "no"])})


@pytest.mark.parametrize("n", [2, 5, 10])
def test_score_accepts_two_to_ten_levels(n):
    validate_questions({"a": q(type="score", instructions="x", criteria=["l%d" % i for i in range(n)])})


@pytest.mark.parametrize("n", [0, 1, 11, 20])
def test_score_rejects_level_counts_outside_two_to_ten(n):
    with pytest.raises(ValidationError, match="between 2 and 10 levels"):
        validate_questions({"a": q(type="score", instructions="x", criteria=["l%d" % i for i in range(n)])})


def test_score_rejects_object_criteria():
    with pytest.raises(ValidationError, match="must be an array"):
        validate_questions({"a": q(type="score", instructions="x", criteria={"0": "low"})})


@pytest.mark.parametrize("n", [2, 3, 255])
def test_choice_accepts_up_to_255_options(n):
    validate_questions({"a": q(type="choice", instructions="x",
                               criteria={"o%d" % i: None for i in range(n)})})


def test_choice_rejects_256_options():
    with pytest.raises(ValidationError, match="at most 255"):
        validate_questions({"a": q(type="choice", instructions="x",
                                   criteria={"o%d" % i: None for i in range(256)})})


def test_choice_rejects_a_single_option():
    with pytest.raises(ValidationError, match="at least 2 options"):
        validate_questions({"a": q(type="choice", instructions="x", criteria={"only": None})})


def test_choice_requires_criteria():
    with pytest.raises(ValidationError, match="requires criteria"):
        validate_questions({"a": q(type="choice", instructions="x")})


def test_empty_instructions_rejected():
    with pytest.raises(ValidationError, match="empty instructions"):
        validate_questions({"a": q(type="noul", instructions="   ")})


def test_unknown_question_type_rejected():
    with pytest.raises(ValidationError, match="unknown type"):
        validate_questions({"a": q(type="ranking", instructions="x")})


def test_no_questions_rejected():
    with pytest.raises(ValidationError, match="at least one question"):
        validate_questions({})


@pytest.mark.parametrize("body,fragment", [
    ({"state": STATE, "questions": {"a": {"type": "ranking", "instructions": "x"}}}, "unknown type"),
    ({"state": STATE, "questions": {"a": {"type": "score", "instructions": "x",
                                          "criteria": ["a"] * 11}}}, "between 2 and 10"),
    ({"state": STATE, "questions": {"a": {"type": "choice", "instructions": "x",
                                          "criteria": {"o%d" % i: None for i in range(300)}}}}, "at most 255"),
    ({"state": STATE, "questions": {}}, "at least one question"),
    ({"questions": {"a": NOUL}}, "state"),
    ({"state": STATE}, "questions"),
])
def test_bad_requests_are_422(body, fragment):
    c, _ = client()
    r = post(c, body)
    assert r.status_code == 422
    payload = r.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert fragment in payload["error"]["message"]


def test_body_that_is_not_an_object_is_422():
    c, _ = client()
    r = c.post("/v1/systemone", json=[1, 2, 3])
    assert r.status_code == 422
    assert r.json()["error"]["type"] == "invalid_request_error"


# --------------------------------------------------------------------- model names

@pytest.mark.parametrize("name", [None, "", "auto", "laya", "jev-latest", "JEV-Latest", "default"])
def test_auto_names_leave_the_choice_to_the_router(name):
    assert resolve_model(name) is None


@pytest.mark.parametrize("name,expected", [
    ("english", "english"), ("laya-english", "english"), ("EN", "english"),
    ("multilingual", "multilingual"), ("laya-multilingual", "multilingual"), ("multi", "multilingual"),
    ("typed-decisions", "typed-decisions"), ("laya-typed-decisions", "typed-decisions"),
    ("typed", "typed-decisions"), ("typed_decisions", "typed-decisions"),
])
def test_explicit_names_map_to_checkpoints(name, expected):
    assert resolve_model(name) == expected


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValidationError, match="unknown model"):
        resolve_model("gpt-4")


def test_unknown_model_over_http_is_422():
    c, _ = client()
    r = post(c, {"state": STATE, "model": "gpt-4", "questions": {"a": NOUL}})
    assert r.status_code == 422
    assert "unknown model" in r.json()["error"]["message"]


def test_explicit_model_reaches_the_engine():
    eng = StubEngine()
    c, _ = client(eng)
    r = post(c, {"state": STATE, "model": "laya-typed-decisions", "questions": {"a": NOUL}})
    assert r.status_code == 200
    assert eng.calls == [("typed-decisions", ["a"])]
    assert r.json()["model"] == "laya-typed-decisions"


def test_auto_routing_follows_the_state():
    eng = StubEngine()
    c, _ = client(eng)
    assert post(c, {"state": STATE, "questions": {"a": NOUL}}).json()["routing"]["model"] == "english"
    r = post(c, {"state": "मेरा ऑर्डर नहीं आया", "questions": {"a": NOUL}})
    assert r.json()["routing"]["model"] == "multilingual"


def test_routing_falls_back_to_a_loaded_checkpoint():
    eng = StubEngine(loaded=["english"])
    c, _ = client(eng)
    r = post(c, {"state": "मेरा ऑर्डर नहीं आया", "questions": {"a": NOUL}})
    assert r.status_code == 200
    assert r.json()["routing"]["model"] == "english"
    assert "not loaded here" in r.json()["routing"]["reason"]


# --------------------------------------------------------------------- response shape

def test_response_matches_the_jev_shape():
    c, _ = client()
    body = {"state": STATE, "questions": {
        "angry": NOUL,
        "queue": {"type": "choice", "instructions": "Which queue?",
                  "criteria": {"billing": "money", "technical": None}},
        "urgency": {"type": "score", "instructions": "How urgent?",
                    "criteria": ["low", "medium", "high"]},
    }}
    out = post(c, body).json()

    assert set(out) >= {"model", "answers", "usage"}
    assert isinstance(out["model"], str)
    assert set(out["usage"]) == {"input_tokens", "output_tokens"}
    assert isinstance(out["usage"]["input_tokens"], int)
    assert out["usage"]["output_tokens"] == 0
    assert set(out["answers"]) == {"angry", "queue", "urgency"}

    noul = out["answers"]["angry"]
    assert noul["type"] == "noul" and 0.0 <= noul["noul"] <= 1.0

    choice = out["answers"]["queue"]
    assert choice["type"] == "choice"
    assert choice["choice"] in choice["probabilities"]
    assert set(choice["probabilities"]) == {"billing", "technical"}
    assert 0.0 <= choice["confidence"] <= 1.0

    score = out["answers"]["urgency"]
    assert score["type"] == "score"
    assert score["legend"] == {"0": "low", "1": "medium", "2": "high"}
    assert set(score["probabilities"]) == {"0", "1", "2"}


def test_response_carries_the_extras_jev_clients_ignore():
    c, _ = client()
    out = post(c, {"state": STATE, "questions": {"a": NOUL}}).json()
    assert "routing" in out and "reason" in out["routing"]
    assert isinstance(out["latency_ms"], float)
    assert "act_probability" in out["answers"]["a"]["action"]


def test_predict_is_an_alias_for_systemone():
    c, _ = client()
    a = c.post("/v1/systemone", json={"state": STATE, "questions": {"a": NOUL}}).json()
    b = c.post("/v1/predict", json={"state": STATE, "questions": {"a": NOUL}}).json()
    assert a["answers"] == b["answers"]


@pytest.mark.parametrize("state", ["plain text", {"message": "an object"}, [{"role": "user", "text": "a list"}]])
def test_state_may_be_text_object_or_array(state):
    c, _ = client()
    assert post(c, {"state": state, "questions": {"a": NOUL}}).status_code == 200


# --------------------------------------------------------------------- failure modes

def test_queue_overflow_is_529():
    c, _ = client(StubEngine(fail="overload"))
    r = post(c, {"state": STATE, "questions": {"a": NOUL}})
    assert r.status_code == 529
    assert r.json()["error"]["type"] == "overloaded_error"


def test_option_budget_overflow_is_422():
    c, _ = client(StubEngine(fail="budget"))
    r = post(c, {"state": STATE, "questions": {"a": NOUL}})
    assert r.status_code == 422
    assert "head_max_len" in r.json()["error"]["message"]


# --------------------------------------------------------------------- auth

def test_no_key_configured_means_no_auth():
    c, _ = client(api_key=None)
    assert post(c, {"state": STATE, "questions": {"a": NOUL}}).status_code == 200


def test_key_configured_rejects_anonymous_requests():
    c, _ = client(api_key="secret")
    r = post(c, {"state": STATE, "questions": {"a": NOUL}})
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"


@pytest.mark.parametrize("header", ["secret", "Bearer wrong", "bearer secret", ""])
def test_key_configured_rejects_wrong_headers(header):
    c, _ = client(api_key="secret")
    r = post(c, {"state": STATE, "questions": {"a": NOUL}}, headers={"Authorization": header})
    assert r.status_code == 401


def test_correct_bearer_key_is_accepted():
    c, _ = client(api_key="secret")
    r = post(c, {"state": STATE, "questions": {"a": NOUL}},
             headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200


def test_auth_is_checked_before_validation():
    c, _ = client(api_key="secret")
    assert post(c, {"nonsense": True}).status_code == 401


# --------------------------------------------------------------------- the other endpoints

def test_root_serves_the_playground_page():
    c, _ = client()
    r = c.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "arbiter" in r.text


def test_the_playground_is_not_in_the_openapi_schema():
    c, _ = client()
    assert "/" not in c.get("/openapi.json").json()["paths"]


def test_healthz_and_readyz():
    c, app = client()
    assert c.get("/healthz").json()["status"] == "ok"
    ready = c.get("/readyz").json()
    assert ready["status"] == "ready"
    assert ready["models"] == ["english", "multilingual", "typed-decisions"]
    # the four facts a measurement has to be labelled with
    assert (ready["engine"], ready["mode"], ready["dtype"], ready["device"]) == (
        "laya", "eager", "autocast", "cpu")


def test_readyz_is_503_before_the_models_are_loaded():
    app = create_app(StubEngine())
    app.state.ready = False
    with TestClient(app) as c:
        pass
    app.state.ready = False
    app.state.engine = None
    assert TestClient(app).get("/readyz").status_code == 503


def test_models_lists_the_checkpoints_and_the_auto_alias():
    c, _ = client()
    data = c.get("/v1/models").json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert ids == ["laya-english", "laya-multilingual", "laya-typed-decisions", "jev-latest"]
    english = next(m for m in data["data"] if m["id"] == "laya-english")
    assert "en" in english["aliases"]


def test_metrics_counts_requests_questions_and_batches():
    c, _ = client()
    post(c, {"state": STATE, "questions": {"a": NOUL, "b": NOUL}})
    post(c, {"state": STATE, "questions": {"a": {"type": "ranking", "instructions": "x"}}})
    text = c.get("/metrics").text
    assert 'arbiter_requests_total{model="english",status="200"} 1' in text
    assert 'arbiter_requests_total{model="none",status="422"} 1' in text
    assert 'arbiter_questions_total{model="english"} 2' in text
    assert "arbiter_request_latency_seconds_count 2" in text
    assert "arbiter_batch_rows_count 1" in text
    assert "arbiter_queue_depth 0" in text


# --------------------------------------------------------------------- the showcase

def test_showcase_index_lists_the_games():
    c, _ = client()
    r = c.get("/showcase/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    for game in ("snake", "hopper", "crossing", "paddle", "mines", "dungeon"):
        assert game in r.text


def test_showcase_redirects_the_bare_path():
    c, _ = client()
    r = c.get("/showcase", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/showcase/"


def test_showcase_serves_modules_as_javascript():
    c, _ = client()
    for path in ("_lib/harness.mjs", "_lib/client.mjs", "_lib/recorder.mjs", "snake/logic.mjs"):
        r = c.get("/showcase/" + path)
        assert r.status_code == 200, path
        # a module served as application/octet-stream is a module the browser will not run
        assert r.headers["content-type"].startswith("text/javascript"), path
        assert "export" in r.text


def test_showcase_serves_the_recorded_runs_as_json():
    c, _ = client()
    r = c.get("/showcase/replay/snake.json")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["game"] == "snake"


def test_showcase_serves_a_game_directory_as_its_page():
    c, _ = client()
    r = c.get("/showcase/hopper/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")


def test_showcase_404s_what_is_not_there():
    c, _ = client()
    assert c.get("/showcase/nope/logic.mjs").status_code == 404
    assert c.get("/showcase/snake/nope.mjs").json()["error"]["type"] == "not_found_error"


@pytest.mark.parametrize("path", ["../server/app.py", "..%2Fserver%2Fapp.py", "snake/../../run.sh"])
def test_showcase_refuses_to_leave_its_directory(path):
    c, _ = client()
    assert c.get("/showcase/" + path).status_code == 404


def test_the_showcase_is_not_in_the_openapi_schema():
    c, _ = client()
    paths = c.get("/openapi.json").json()["paths"]
    assert not [p for p in paths if p.startswith("/showcase")]
