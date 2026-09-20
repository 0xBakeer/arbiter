#!/usr/bin/env python3
"""Triage an inbox: file it, flag it, or quarantine it -- in one forward pass per message.

The state is the message as an object (`from`, `subject`, `body`) rather than a flattened
string. Structured state is what the model prefers for anything with fields: the sender is
visibly a sender and not a sentence.

Three built-in samples show the router at work. The English one is answered by the English
checkpoint; the German and the Hindi one are answered by the multilingual checkpoint, without
the caller saying anything about language -- watch the badge in the header.

    python examples/email_triage.py
    python examples/email_triage.py --sample de
    python examples/email_triage.py --sample hi --json
    python examples/email_triage.py --state message.json
"""
import sys

import _cli
from arbiter_client import Choice, ArbiterClient, Noul, Score

QUESTIONS = {
    "category": Choice(
        "What kind of email is this?",
        {
            "personal": "from a human who knows the recipient, written to them",
            "work": "colleagues, clients, projects, meetings, internal business",
            "transaction": "receipts, orders, shipping, invoices, bookings, statements",
            "newsletter": "subscriptions, marketing, product news, digests",
            "notification": "automated alerts from a system or service",
            "recruiting": "job offers, recruiters, interview scheduling",
            "spam": "unsolicited bulk mail nobody asked for",
        },
    ),
    "phishing": Noul(
        "This message is a phishing or fraud attempt: it impersonates someone, manufactures "
        "urgency, and pushes the reader to click a link, pay, or hand over credentials.",
        {"true": "impersonation, credential or payment bait, spoofed sender",
         "false": "a legitimate message, even if it is marketing"},
    ),
    "action_needed": Noul(
        "The recipient has to do something for this message; reading it is not enough."),
    "reply_by": Score(
        "How soon does this message need a reply?",
        ["no reply needed", "sometime this month", "this week", "today"],
    ),
}

QUARANTINE_P = 0.70          # above this a message never reaches the inbox
PHISH_REVIEW_P = 0.30        # below QUARANTINE_P but above this, banner it and let a human look
FILE_CATEGORY_P = 0.80       # only auto-file when the folder is not in doubt
ACTION_P = 0.60
SOON_SCORE = 2.0             # "this week" or sooner

SAMPLES = {
    "en": {
        "from": "security-alert@paypa1-support.com",
        "subject": "Unusual sign-in - verify your account within 24 hours",
        "body": ("We detected a login from a new device. Your account will be limited unless "
                 "you confirm your details. Click here to restore access: "
                 "http://paypa1-support.com/verify?id=88213"),
    },
    "de": {
        "from": "buchhaltung@lieferant-nord.de",
        "subject": "Rechnung 2024-0913 überfällig - bitte bis Freitag überweisen",
        "body": ("Sehr geehrte Damen und Herren, unsere Rechnung 2024-0913 über 1.480,00 EUR "
                 "ist seit dem 28. August fällig. Wir bitten Sie, den Betrag bis Freitag zu "
                 "überweisen, andernfalls müssen wir Mahngebühren berechnen."),
    },
    "hi": {
        "from": "priya.sharma@example.in",
        "subject": "कल की मीटिंग का समय बदल गया है",
        "body": ("नमस्ते, कल की प्रोजेक्ट समीक्षा बैठक सुबह 11 बजे के बजाय दोपहर 3 बजे होगी। "
                 "कृपया पुष्टि करें कि आप उपलब्ध हैं, और अपनी स्लाइड आज शाम तक भेज दें।"),
    },
}


def decide(r):
    """block | escalate | auto | review."""
    phish = r.noul("phishing")
    if phish >= QUARANTINE_P:
        return "block", "phishing %.2f >= %.2f -- quarantine, do not deliver" % (phish, QUARANTINE_P)
    if phish >= PHISH_REVIEW_P:
        return "review", "phishing %.2f is in the uncertain band -- deliver with a warning banner" % phish
    action, reply_by = r.noul("action_needed"), r.score("reply_by")
    if action >= ACTION_P and reply_by >= SOON_SCORE:
        return "escalate", "action needed (%.2f) and reply_by %.2f -- pin to the top of the inbox" % (action, reply_by)
    category_p = r.probabilities("category")[r.choice("category")]
    if action < 0.30 and category_p >= FILE_CATEGORY_P:
        return "auto", "no action needed and category p=%.2f -- file into %s" % (category_p, r.choice("category"))
    return "review", "category p=%.2f, action %.2f -- leave it in the inbox" % (category_p, action)


def main() -> int:
    p = _cli.parser(__doc__)
    p.add_argument("--sample", choices=sorted(SAMPLES), default="en",
                   help="which built-in message to triage (ignored with --state or a pipe)")
    args = p.parse_args()
    state = _cli.load_state(args, SAMPLES[args.sample])
    response = ArbiterClient(base_url=args.url).system_one(state, QUESTIONS, model=args.model)
    if args.json:
        return _cli.dump(response)
    action, reason = decide(response)
    _cli.render("email triage", response, action, reason)
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main))
