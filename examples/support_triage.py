#!/usr/bin/env python3
"""Route an inbound support ticket: department, urgency, mood, and what to do about it.

Five questions, one forward pass. The model says how likely each thing is; the thresholds below
decide what happens, and anything that falls between "obviously fine" and "obviously not" goes
to a human rather than being guessed at.

    python examples/support_triage.py
    python examples/support_triage.py --state ticket.txt
    cat ticket.txt | python examples/support_triage.py --json
"""
import sys

import _cli
from laya_client import Choice, LayaClient, Noul, Score

# The questions are data, not code: the same dict is what goes in docs/use-cases.md, what a QA
# harness replays, and what you edit when the routing is wrong.
QUESTIONS = {
    "department": Choice(
        "Which team should own this ticket?",
        {
            "billing": "payments, invoices, refunds, subscriptions, pricing",
            "technical": "bugs, errors, outages, broken features, performance",
            "account": "login, passwords, permissions, profile and settings",
            "shipping": "delivery, tracking, damaged or missing parcels",
            "sales": "pre-purchase questions, quotes, upgrades, plan comparisons",
            "other": "anything that fits none of the above",
        },
    ),
    "urgency": Score(
        "How quickly does this ticket need a reply?",
        ["no rush, informational", "within the week", "today", "immediate, customer is blocked"],
    ),
    "frustration": Score(
        "How frustrated does the customer sound?",
        ["calm and neutral", "mildly annoyed", "clearly angry", "furious, threatening to leave"],
    ),
    "refund_requested": Noul(
        "The customer is asking for a refund, a chargeback or their money back.",
        {"true": "they explicitly want money returned",
         "false": "they want help, information or a fix"},
    ),
    "churn_risk": Noul(
        "The customer is threatening to cancel, downgrade or switch to a competitor.",
    ),
}

# Thresholds live here, in the caller, not in the model. The base checkpoints are calibrated but
# ship slightly over-confident, so these sit further from 0.5 than a calibration plot would
# suggest, and the gap between them is the review band.
AUTO_DEPARTMENT_P = 0.85     # only file a ticket by itself when the department is not in doubt
CALM_FRUSTRATION = 1.2       # below "mildly annoyed"
CHURN_ESCALATE = 0.55
CHURN_REVIEW = 0.25
REFUND_REVIEW = 0.50
URGENT_SCORE = 2.5           # between "today" and "immediate"

SAMPLE = (
    "Hi -- I was charged 79 EUR twice for order #44182 on the 3rd and again on the 4th. I have "
    "written to you twice already and nobody has answered. This is the second billing problem "
    "in three months. Refund the duplicate today or I am cancelling both of our seats and "
    "moving to your competitor."
)


def decide(r):
    """auto | escalate | review, from the five answers. Escalate wins over everything."""
    churn, refund = r.noul("churn_risk"), r.noul("refund_requested")
    urgency, frustration = r.score("urgency"), r.score("frustration")
    dept_p = r.probabilities("department")[r.choice("department")]

    if churn >= CHURN_ESCALATE:
        return "escalate", "churn_risk %.2f >= %.2f -- a human owns retention" % (churn, CHURN_ESCALATE)
    if frustration >= 2.5:
        return "escalate", "frustration %.2f >= 2.50 (angry or worse)" % frustration
    if refund >= 0.80 and urgency >= URGENT_SCORE:
        return "escalate", "refund %.2f and urgency %.2f are both high" % (refund, urgency)
    if (dept_p >= AUTO_DEPARTMENT_P and frustration < CALM_FRUSTRATION
            and churn < CHURN_REVIEW and refund < REFUND_REVIEW):
        return "auto", "department p=%.2f, calm, no refund or churn signal" % dept_p
    return "review", ("department p=%.2f, frustration %.2f, refund %.2f -- inside the review band"
                      % (dept_p, frustration, refund))


def main() -> int:
    args = _cli.parser(__doc__).parse_args()
    state = _cli.load_state(args, SAMPLE)
    response = LayaClient(base_url=args.url).system_one(state, QUESTIONS, model=args.model)
    if args.json:
        return _cli.dump(response)
    action, reason = decide(response)
    _cli.render("support triage", response, action, reason)
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main))
