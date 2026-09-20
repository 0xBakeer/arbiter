#!/usr/bin/env python3
"""Read an incoming invoice: currency, size band, duplicate, and whether a human must approve it.

Invoice processing is one of the four workflows the typed-decisions checkpoint was fine-tuned
on, so this is the one example that asks for it: the request carries `task="typed_decisions"`,
the server's opt-in for that checkpoint. It is never selected silently -- automatic detection is
off by default, and even when it is on it only fires when the question ids exactly match one of
the four fine-tuned schemas (invoice processing, customer service, security incidents, agent
trace observability). The badge in the header says which checkpoint actually answered.

The state carries the invoice *and* the vendor's recent invoices, because "is this a duplicate"
is not a question about one document. Duplicate detection is the single most valuable thing in
an accounts-payable pipeline and the hardest to do with rules: the same invoice arrives with a
different number, the same number arrives with a different total.

    python examples/invoice_fields.py
    python examples/invoice_fields.py --model laya-english      # compare the two checkpoints
    python examples/invoice_fields.py --state invoice.json --json
"""
import sys

import _cli
from laya_client import Choice, LayaClient, Noul, Score

QUESTIONS = {
    "currency": Choice(
        "Which currency is this invoice denominated in?",
        {"EUR": "euro", "USD": "US dollar", "GBP": "pound sterling", "CHF": "Swiss franc",
         "SEK": "Swedish krona", "PLN": "Polish zloty", "JPY": "Japanese yen",
         "other": "a currency not listed here"},
    ),
    "amount_band": Score(
        "How large is the total on this invoice?",
        ["under 100", "100 to 1000", "1000 to 10000", "over 10000"],
    ),
    "is_duplicate": Noul(
        "This invoice has already been received: one of the vendor's recent invoices is the "
        "same charge, even if the invoice number or the date differs.",
        {"true": "same vendor, same work, same amount as one already on file",
         "false": "a new charge, even from a vendor who invoices regularly"},
    ),
    "needs_approval": Noul(
        "This invoice cannot be paid automatically and needs a human approver: no purchase "
        "order, an amount above the vendor's agreed limit, unfamiliar bank details, or line "
        "items that do not match what was ordered."),
    "bank_details_changed": Noul(
        "The payment details on this invoice differ from the ones on file for this vendor."),
}

# The costs here are wildly asymmetric, and the thresholds say so. A false stop costs one email
# to the vendor; a false pass costs the invoice. Invoice fraud almost always arrives as changed
# bank details on a real vendor's letterhead, so that noul stops the payment well below 0.5 --
# it does not have to be probable, only possible.
BANK_CHANGE_STOP = 0.30
DUPLICATE_STOP = 0.55
APPROVAL_P = 0.40
AUTO_PAY_BAND = 1.20         # auto-pay only when the amount is confidently in the low bands

SAMPLE = {
    "invoice": {
        "vendor": "Nordlicht Studio GmbH",
        "invoice_number": "NS-2026-0318",
        "date": "2026-03-04",
        "purchase_order": None,
        "lines": [
            {"description": "Design retainer, February", "amount": 4800.00},
            {"description": "Additional illustration set (6)", "amount": 1450.00},
        ],
        "total": "6.250,00 EUR",
        "payment": {"iban": "DE21 5001 0517 9876 5432 10", "bic": "INGDDEFFXXX"},
    },
    "vendor_history": [
        {"invoice_number": "NS-2026-0244", "date": "2026-02-04", "total": "4.800,00 EUR",
         "iban": "DE89 3704 0044 0532 0130 00", "status": "paid"},
        {"invoice_number": "NS-2026-0190", "date": "2026-01-07", "total": "4.800,00 EUR",
         "iban": "DE89 3704 0044 0532 0130 00", "status": "paid"},
    ],
    "policy": {"auto_pay_limit_eur": 2500, "purchase_order_required_above_eur": 1000},
}


def decide(r):
    """pay | approve | review."""
    duplicate = r.noul("is_duplicate")
    approval = r.noul("needs_approval")
    bank_changed = r.noul("bank_details_changed")
    band = r.score("amount_band")

    if bank_changed >= BANK_CHANGE_STOP:
        return "review", ("bank_details_changed %.2f >= %.2f -- verify with the vendor out of "
                          "band before paying anything" % (bank_changed, BANK_CHANGE_STOP))
    if duplicate >= DUPLICATE_STOP:
        return "review", "is_duplicate %.2f >= %.2f -- match it against the paid invoice" % (
            duplicate, DUPLICATE_STOP)
    if approval >= APPROVAL_P or band >= AUTO_PAY_BAND:
        return "approve", "needs_approval %.2f, amount_band %.2f -- route to an approver" % (
            approval, band)
    return "pay", "small (%.2f), not a duplicate (%.2f), no approval flag (%.2f)" % (
        band, duplicate, approval)


def main() -> int:
    args = _cli.parser(__doc__).parse_args()
    state = _cli.load_state(args, SAMPLE)
    # `task` is the server's router hint. Left on "auto" this example opts in to the
    # typed-decisions checkpoint; naming a model on the command line overrides it.
    task = "typed_decisions" if args.model == "auto" else None
    response = LayaClient(base_url=args.url).system_one(state, QUESTIONS, model=args.model,
                                                        task=task)
    if args.json:
        return _cli.dump(response)
    action, reason = decide(response)
    _cli.render("invoice fields", response, action, reason, label="AP")
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main))
