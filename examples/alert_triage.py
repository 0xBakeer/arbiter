#!/usr/bin/env python3
"""Triage a monitoring alert against the incidents already open: page, ticket, or suppress.

The state carries the alert *and* the open incidents, so the duplicate question has something to
compare against. This is the cheapest useful thing to put in front of a pager: most alerts at
3 a.m. are the third copy of one that already woke somebody up.

    python examples/alert_triage.py
    python examples/alert_triage.py --state alert.json --json
"""
import sys

import _cli
from arbiter_client import Choice, ArbiterClient, Noul, Score

QUESTIONS = {
    "service": Choice(
        "Which service is most likely at fault, as opposed to merely reporting the symptom?",
        {
            "api-gateway": "the edge: routing, TLS, rate limiting",
            "checkout": "orders, carts, payment initiation",
            "payments": "the payment provider integration and settlement",
            "search": "the search index and query service",
            "postgres": "the primary database",
            "redis": "the cache and queue",
            "kubernetes": "the cluster itself: nodes, scheduling, networking",
        },
    ),
    "root_cause": Choice(
        "What kind of cause does this alert point at?",
        {
            "deploy": "a release or config rollout immediately before the alert",
            "dependency": "an upstream or third-party service failing",
            "capacity": "saturation: CPU, memory, connections, disk, queue depth",
            "config": "a wrong or missing setting, credential or limit",
            "network": "DNS, routing, timeouts between healthy services",
            "data": "bad or unexpected data flowing through a healthy system",
            "unknown": "nothing in the alert points at a cause",
        },
    ),
    "severity": Score(
        "How bad is the user-visible impact right now?",
        ["nothing users notice", "degraded for some", "partial outage", "full outage"],
    ),
    "is_duplicate": Noul(
        "This alert is the same incident as one of the open incidents listed in the state.",
        {"true": "same service and same failure as an open incident",
         "false": "a new and independent problem"},
    ),
}

SUPPRESS_DUPLICATE = 0.70    # attach to the open incident instead of waking anyone
PAGE_SEVERITY = 2.30         # at or past "partial outage"
TICKET_SEVERITY = 0.80

SAMPLE = {
    "alert": {
        "name": "HighErrorRate",
        "service": "checkout",
        "labels": {"severity": "critical", "env": "prod"},
        "message": ("5xx rate on checkout is 34% over the last 5 minutes; upstream payments "
                    "latency p99 is 8.2s, connection pool exhausted"),
        "started_at": "2026-03-11T02:14:07Z",
    },
    "open_incidents": [
        {"id": "INC-2291", "service": "payments", "title": "Payment provider timeouts",
         "opened_at": "2026-03-11T02:05:40Z", "status": "investigating"},
        {"id": "INC-2287", "service": "search", "title": "Index rebuild running long",
         "opened_at": "2026-03-10T19:40:00Z", "status": "monitoring"},
    ],
}


def decide(r):
    """page | ticket | suppress."""
    duplicate = r.noul("is_duplicate")
    severity = r.score("severity")
    if duplicate >= SUPPRESS_DUPLICATE:
        return "suppress", "is_duplicate %.2f >= %.2f -- attach to the open incident" % (
            duplicate, SUPPRESS_DUPLICATE)
    if severity >= PAGE_SEVERITY:
        return "page", "severity %.2f >= %.2f and not a duplicate (%.2f)" % (
            severity, PAGE_SEVERITY, duplicate)
    if severity >= TICKET_SEVERITY or duplicate >= 0.35:
        return "ticket", "severity %.2f, duplicate %.2f -- file it, do not wake anyone" % (
            severity, duplicate)
    return "suppress", "severity %.2f below %.2f -- no user-visible impact" % (
        severity, TICKET_SEVERITY)


def main() -> int:
    args = _cli.parser(__doc__).parse_args()
    state = _cli.load_state(args, SAMPLE)
    response = ArbiterClient(base_url=args.url).system_one(state, QUESTIONS, model=args.model)
    if args.json:
        return _cli.dump(response)
    action, reason = decide(response)
    _cli.render("alert triage", response, action, reason)
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main))
