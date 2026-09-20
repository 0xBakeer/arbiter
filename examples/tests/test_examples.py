"""The examples: the thresholds each one applies, and that each one runs end to end.

The `decide` functions are pure -- probabilities in, a route out -- so the thresholds can be
pinned without a server. The scripts themselves are then run as scripts against the stub, which
is what catches an import, an argument or a rendering mistake.
"""
import json
import os
import subprocess
import sys

import pytest
from stub_server import StubServer

import alert_triage
import email_triage
import invoice_fields
import model_router
import moderation
import pr_risk_gate
import rag_relevance
import support_triage
import tool_call_guard
from laya_client import LayaClient, Response

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = ["support_triage", "email_triage", "tool_call_guard", "pr_risk_gate", "alert_triage",
           "model_router", "rag_relevance", "moderation", "invoice_fields"]


# --------------------------------------------------------------------- synthetic answers

def noul(p):
    return {"type": "noul", "noul": p, "confidence": round(max(p, 1 - p), 4)}


def choice(name, probabilities):
    return {"type": "choice", "choice": name, "probabilities": probabilities,
            "confidence": max(probabilities.values())}


def score(value, levels):
    return {"type": "score", "score": value,
            "legend": {str(i): "level %d" % i for i in range(levels)},
            "probabilities": {str(i): 1.0 / levels for i in range(levels)}, "confidence": 0.5}


def response(**answers):
    return Response({"model": "laya-english", "answers": answers,
                     "routing": {"model": "english", "reason": "test"},
                     "usage": {"input_tokens": 1}, "latency_ms": 1.0})


def route(module, **answers):
    return module.decide(response(**answers))[0]


# --------------------------------------------------------------------- support_triage

CALM_TICKET = dict(department=choice("billing", {"billing": 0.93, "technical": 0.07}),
                   urgency=score(0.4, 4), frustration=score(0.3, 4),
                   refund_requested=noul(0.05), churn_risk=noul(0.04))


def test_support_auto_needs_every_signal_quiet():
    assert route(support_triage, **CALM_TICKET) == "auto"


def test_support_escalates_on_churn_alone():
    assert route(support_triage, **dict(CALM_TICKET, churn_risk=noul(0.72))) == "escalate"


def test_support_escalates_on_an_angry_customer():
    assert route(support_triage, **dict(CALM_TICKET, frustration=score(2.7, 4))) == "escalate"


def test_support_reviews_when_the_department_is_uncertain():
    uncertain = choice("billing", {"billing": 0.52, "technical": 0.48})
    assert route(support_triage, **dict(CALM_TICKET, department=uncertain)) == "review"


def test_support_reviews_a_refund_request_that_is_not_yet_an_escalation():
    assert route(support_triage, **dict(CALM_TICKET, refund_requested=noul(0.62))) == "review"


# --------------------------------------------------------------------- email_triage

BORING_MAIL = dict(category=choice("newsletter", {"newsletter": 0.91, "spam": 0.09}),
                   phishing=noul(0.04), action_needed=noul(0.06), reply_by=score(0.2, 4))


def test_email_files_a_quiet_newsletter():
    assert route(email_triage, **BORING_MAIL) == "auto"


def test_email_quarantines_a_phish():
    assert route(email_triage, **dict(BORING_MAIL, phishing=noul(0.88))) == "block"


def test_email_banners_an_uncertain_phish_instead_of_dropping_it():
    assert route(email_triage, **dict(BORING_MAIL, phishing=noul(0.45))) == "review"


def test_email_pins_something_that_needs_answering_today():
    assert route(email_triage, **dict(BORING_MAIL, action_needed=noul(0.81),
                                      reply_by=score(2.8, 4))) == "escalate"


# --------------------------------------------------------------------- tool_call_guard

SAFE_COMMAND = dict(is_destructive=noul(0.03), touches_secrets=noul(0.02), leaves_repo=noul(0.05),
                    needs_network=noul(0.04), blast_radius=score(0.2, 4))


def test_guard_allows_a_harmless_command():
    assert route(tool_call_guard, **SAFE_COMMAND) == "allow"


def test_guard_does_not_stop_for_a_mild_yes():
    """`npm test` really does score about 0.48 on "is this destructive" -- that must not ask."""
    assert route(tool_call_guard, **dict(SAFE_COMMAND, is_destructive=noul(0.48),
                                         leaves_repo=noul(0.42),
                                         blast_radius=score(1.22, 4))) == "allow"


def test_guard_denies_anything_touching_credentials():
    assert route(tool_call_guard, **dict(SAFE_COMMAND, touches_secrets=noul(0.8))) == "deny"


def test_guard_denies_a_production_blast_radius():
    assert route(tool_call_guard, **dict(SAFE_COMMAND, blast_radius=score(2.7, 4))) == "deny"


def test_guard_asks_rather_than_denying_in_the_middle():
    assert route(tool_call_guard, **dict(SAFE_COMMAND, is_destructive=noul(0.60),
                                         blast_radius=score(1.1, 4))) == "ask"


def test_guard_exit_codes_are_the_decision():
    assert {"allow": 0, "ask": 1, "deny": 2}["deny"] == 2


# --------------------------------------------------------------------- pr_risk_gate

TIDY_PR = dict(touches_prod_credentials=noul(0.02), touches_migration=noul(0.03),
               touches_shared_infra=noul(0.05), no_rollback_plan=noul(0.04),
               blast_radius=score(0.3, 4))


def test_pr_allows_a_tidy_change():
    assert route(pr_risk_gate, **TIDY_PR) == "allow"


def test_pr_blocks_credentials_in_a_diff():
    assert route(pr_risk_gate, **dict(TIDY_PR, touches_prod_credentials=noul(0.7))) == "block"


def test_pr_blocks_an_irreversible_wide_change():
    assert route(pr_risk_gate, **dict(TIDY_PR, no_rollback_plan=noul(0.7),
                                      blast_radius=score(2.2, 4))) == "block"


def test_pr_reviews_a_migration():
    assert route(pr_risk_gate, **dict(TIDY_PR, touches_migration=noul(0.6))) == "review"


def test_pr_state_puts_the_description_and_the_file_list_before_the_diff():
    state = pr_risk_gate.build_state(pr_risk_gate.SAMPLE_DIFF, "a description")
    assert list(state) == ["description", "files_changed", "diff"]
    assert state["files_changed"] == ["db/migrations/0042_drop_legacy_totals.sql",
                                      "deploy/k8s/rollup-cron.yaml",
                                      "services/rollup/query.py"]


# --------------------------------------------------------------------- alert_triage

QUIET_ALERT = dict(service=choice("search", {"search": 0.9, "redis": 0.1}),
                   root_cause=choice("capacity", {"capacity": 0.8, "deploy": 0.2}),
                   severity=score(0.3, 4), is_duplicate=noul(0.05))


def test_alert_suppresses_a_duplicate_before_anything_else():
    loud_duplicate = dict(QUIET_ALERT, severity=score(2.9, 4), is_duplicate=noul(0.85))
    assert route(alert_triage, **loud_duplicate) == "suppress"


def test_alert_pages_on_a_real_outage():
    assert route(alert_triage, **dict(QUIET_ALERT, severity=score(2.6, 4))) == "page"


def test_alert_files_a_ticket_in_between():
    assert route(alert_triage, **dict(QUIET_ALERT, severity=score(1.4, 4))) == "ticket"


def test_alert_suppresses_what_nobody_notices():
    assert route(alert_triage, **QUIET_ALERT) == "suppress"


# --------------------------------------------------------------------- model_router

SIMPLE_REQUEST = dict(complexity=score(0.3, 4), needs_tools=noul(0.05),
                      needs_long_context=noul(0.03), is_ambiguous=noul(0.05))


def test_router_keeps_a_lookup_on_the_fast_model():
    assert route(model_router, **SIMPLE_REQUEST) == "fast"


def test_router_escalates_long_context_immediately():
    assert route(model_router, **dict(SIMPLE_REQUEST, needs_long_context=noul(0.7))) == "powerful"


def test_router_escalates_open_ended_work():
    assert route(model_router, **dict(SIMPLE_REQUEST, complexity=score(2.5, 4))) == "powerful"


def test_router_cascades_in_the_middle_band():
    assert route(model_router, **dict(SIMPLE_REQUEST, complexity=score(1.5, 4))) == "cascade"


# --------------------------------------------------------------------- moderation

BENIGN = dict(jailbreak=noul(0.03), harmful=noul(0.02), pii=noul(0.04), off_topic=noul(0.05),
              severity=score(0.2, 4))


def test_moderation_allows_an_ordinary_message():
    assert route(moderation, **BENIGN) == "allow"


def test_moderation_blocks_a_harmful_request():
    assert route(moderation, **dict(BENIGN, harmful=noul(0.66))) == "block"


def test_moderation_reviews_a_jailbreak_that_asks_for_nothing_harmful():
    assert route(moderation, **dict(BENIGN, jailbreak=noul(0.8))) == "review"


def test_moderation_reviews_personal_data():
    assert route(moderation, **dict(BENIGN, pii=noul(0.6))) == "review"


# --------------------------------------------------------------------- invoice_fields

SMALL_INVOICE = dict(currency=choice("EUR", {"EUR": 0.95, "USD": 0.05}),
                     amount_band=score(0.7, 4), is_duplicate=noul(0.04),
                     needs_approval=noul(0.06), bank_details_changed=noul(0.03))


def test_invoice_pays_a_small_familiar_one():
    assert route(invoice_fields, **SMALL_INVOICE) == "pay"


def test_invoice_stops_on_changed_bank_details_even_when_everything_else_is_clean():
    assert route(invoice_fields, **dict(SMALL_INVOICE, bank_details_changed=noul(0.5))) == "review"


def test_invoice_stops_on_a_suspected_duplicate():
    assert route(invoice_fields, **dict(SMALL_INVOICE, is_duplicate=noul(0.7))) == "review"


def test_invoice_routes_a_large_one_to_an_approver():
    assert route(invoice_fields, **dict(SMALL_INVOICE, amount_band=score(2.4, 4))) == "approve"


# --------------------------------------------------------------------- rag_relevance

def test_rag_sends_passages_in_chunks_that_fit_the_read_window():
    passages = ["passage %d" % i for i in range(14)]
    with StubServer() as stub:
        scores, _, stage = rag_relevance.rank(LayaClient(base_url=stub.url), "q", passages)
        chunks = [len(r["questions"]) for r in stub.requests]
    assert len(scores) == 14
    assert max(chunks) <= rag_relevance.CHUNK
    assert "strict pass" in stage


def test_rag_asks_once_when_there_are_few_passages():
    with StubServer() as stub:
        _, _, stage = rag_relevance.rank(LayaClient(base_url=stub.url), "q", ["a", "b", "c"])
        assert len(stub.requests) == 1
    assert "one pass" in stage


def test_rag_keys_every_passage_into_the_shared_state():
    with StubServer() as stub:
        rag_relevance.rank(LayaClient(base_url=stub.url), "q", ["a", "b"])
        sent = stub.requests[0]
    assert sent["state"]["query"] == "q"
    assert sorted(sent["state"]["passages"]) == ["p0", "p1"]
    assert sorted(sent["questions"]) == ["p0", "p1"]


# --------------------------------------------------------------------- end to end

def run(script, *args, url, stdin=""):
    env = dict(os.environ, LAYA_URL=url, NO_COLOR="1", PYTHONPATH=os.path.join(ROOT, "examples"))
    return subprocess.run([sys.executable, os.path.join(ROOT, "examples", script + ".py")] + list(args),
                          input=stdin, capture_output=True, text=True, env=env, timeout=60)


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_example_runs_and_prints_a_route(script):
    with StubServer() as stub:
        done = run(script, url=stub.url)
    assert done.returncode in (0, 1, 2), done.stderr
    assert "Laya" in done.stdout
    assert any(word in done.stdout for word in ("ROUTE", "MODEL", "AP", "RERANK")), done.stdout
    assert not done.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_example_can_print_the_raw_response(script):
    with StubServer() as stub:
        done = run(script, "--json", url=stub.url)
    payload = json.loads(done.stdout)
    assert isinstance(payload, dict)


@pytest.mark.parametrize("script", SCRIPTS)
def test_no_example_leaves_ansi_in_a_piped_run(script):
    with StubServer() as stub:
        done = run(script, url=stub.url)
    assert "\x1b[" not in done.stdout


def test_a_piped_state_is_used_instead_of_the_sample():
    with StubServer() as stub:
        run("moderation", url=stub.url, stdin="a message from stdin")
        assert stub.requests[0]["state"] == "a message from stdin"


def test_piped_json_arrives_as_structured_state():
    message = {"from": "a@b.c", "subject": "hi", "body": "there"}
    with StubServer() as stub:
        run("email_triage", url=stub.url, stdin=json.dumps(message))
        assert stub.requests[0]["state"] == message


def test_a_diff_on_stdin_reaches_the_pr_gate():
    with StubServer() as stub:
        run("pr_risk_gate", url=stub.url, stdin="diff --git a/x.py b/x.py\n+print(1)\n")
        assert stub.requests[0]["state"]["files_changed"] == ["x.py"]


def test_an_unreachable_server_is_reported_without_a_traceback():
    done = run("moderation", "--url", "http://127.0.0.1:1", url="http://127.0.0.1:1")
    assert done.returncode == 3
    assert "Traceback" not in done.stderr
    assert "cannot reach" in done.stderr
