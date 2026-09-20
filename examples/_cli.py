"""Shared terminal rendering and argument handling for the example scripts.

Plain ANSI, no dependencies. Every example prints the same three things: a header saying which
checkpoint answered and how long it took, one row per question with a probability bar, and the
route the *calling code* chose from those numbers. The last line is the point of the whole
exercise -- the model reports probabilities, the thresholds live here in the script, and they
are visible in the output so a reader can argue with them.
"""
import argparse
import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, Optional, Tuple

from arbiter_client import ArbiterError

WIDTH = 82
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Colours are decided once, at import, so a piped or redirected run is plain text and the
# captured output in docs/use-cases.md is the same text a reader sees in their terminal.
_COLOR = (sys.stdout.isatty() and not os.environ.get("NO_COLOR")
          and os.environ.get("TERM", "") != "dumb")


def _c(code: str) -> str:
    return code if _COLOR else ""


RESET = _c("\x1b[0m")
BOLD = _c("\x1b[1m")
DIM = _c("\x1b[2m")
RED = _c("\x1b[31m")
GREEN = _c("\x1b[32m")
YELLOW = _c("\x1b[33m")
BLUE = _c("\x1b[34m")
MAGENTA = _c("\x1b[35m")
CYAN = _c("\x1b[36m")

# Every example routes to one of a small set of verbs; green means the code acts on its own,
# yellow means a human is asked, red means the code refuses.
VERB_COLOR = {
    "allow": GREEN, "auto": GREEN, "auto_resolve": GREEN, "fast": GREEN, "keep": GREEN,
    "suppress": GREEN, "publish": GREEN, "pay": GREEN,
    "ask": YELLOW, "review": YELLOW, "confirm": YELLOW, "ticket": YELLOW, "approve": YELLOW,
    "cascade": YELLOW,
    "deny": RED, "block": RED, "escalate": RED, "page": RED, "drop": RED,
    "powerful": MAGENTA,
}

# The checkpoint badge in the header, so it is obvious when the router moved a request.
CHECKPOINT_COLOR = {"english": BLUE, "multilingual": MAGENTA, "typed-decisions": CYAN}


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def bar(p: float, width: int = 18) -> str:
    """A horizontal probability bar. Filled cells are the accent colour, the rest is dim."""
    p = max(0.0, min(1.0, float(p)))
    filled = int(round(p * width))
    return "%s%s%s%s%s" % (CYAN, "█" * filled, DIM, "░" * (width - filled), RESET)


def fit(text: str, width: int) -> str:
    """Truncate with an ellipsis, so a clipped label is visibly clipped."""
    return text if len(text) <= width else text[:width - 1] + "…"


def rule(char: str = "─") -> str:
    return DIM + char * WIDTH + RESET


# --------------------------------------------------------------------------- rendering

def _row_fields(answer: Dict[str, Any]) -> Tuple[str, float, str]:
    """(label, what the bar fills to, the number to print) for one answer.

    A noul and a choice print a probability; a score prints the score out of its top level and
    fills the bar to where that score sits between the lowest and the highest level.
    """
    if answer["type"] == "noul":
        p = float(answer["noul"])
        return ("yes" if p >= 0.5 else "no"), p, "%.2f" % p
    if answer["type"] == "choice":
        p = float(answer["probabilities"][answer["choice"]])
        return answer["choice"], p, "%.2f" % p
    top = len(answer["legend"]) - 1
    score = float(answer["score"])
    return (answer["legend"][str(int(round(score)))], (score / top if top else 0.0),
            "%.2f/%d" % (score, top))


def header(title: str, response) -> None:
    ck = response.checkpoint
    badge = "%s%s%s" % (CHECKPOINT_COLOR.get(ck, ""), ck, RESET)
    left = "%sArbiter%s · %s" % (BOLD, RESET, title)
    right = "%s · %s%.1f ms · %d tok%s" % (badge, DIM, response.latency_ms,
                                                     response.input_tokens, RESET)
    pad = max(1, WIDTH - len(strip_ansi(left)) - len(strip_ansi(right)))
    print("%s%s%s" % (left, " " * pad, right))
    print("%s%s%s" % (DIM, fit("  routed: " + response.routing_reason, WIDTH), RESET))
    print(rule())


def answers(response, order=None) -> None:
    """One row per question, in the order they were asked."""
    for qid in (order or response["answers"]):
        a = response.answer(qid)
        label, value, shown = _row_fields(a)
        print("  %-24s %-20s %s %s%7s%s  %sconf %.2f%s" % (
            fit(qid, 24), fit(label, 20), bar(value, 14), BOLD, shown, RESET,
            DIM, a["confidence"], RESET))


def verdict(action: str, reason: str, label: str = "ROUTE") -> None:
    color = VERB_COLOR.get(action, BOLD)
    print(rule())
    head = "  %s%-6s%s %s%s%s%s  " % (DIM, label, RESET, BOLD, color, action.upper(), RESET)
    indent = " " * len(strip_ansi(head))
    lines = textwrap.wrap(reason, WIDTH - len(indent)) or [""]
    print("%s%s%s%s" % (head, DIM, lines[0], RESET))
    for line in lines[1:]:
        print("%s%s%s%s" % (indent, DIM, line, RESET))


def render(title: str, response, action: str, reason: str, order=None,
           label: str = "ROUTE", extra=()) -> None:
    header(title, response)
    answers(response, order)
    for line in extra:
        print("  " + line)
    verdict(action, reason, label)


# --------------------------------------------------------------------------- arguments

def parser(description: str) -> argparse.ArgumentParser:
    """The flags every example shares."""
    p = argparse.ArgumentParser(
        description=description.strip().splitlines()[0] if description else None,
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=description)
    p.add_argument("--json", action="store_true",
                   help="print the raw /v1/systemone response instead of the table")
    p.add_argument("--state", metavar="FILE",
                   help="read the state from FILE ('-' or a pipe reads stdin)")
    p.add_argument("--url", default=None,
                   help="server base URL (default $ARBITER_URL or http://localhost:8010)")
    p.add_argument("--model", default="auto",
                   help="auto | laya-english | laya-multilingual | laya-typed-decisions")
    return p


def load_state(args, sample: Any) -> Any:
    """--state FILE, else stdin if something is piped in, else the script's built-in sample.

    Input that parses as a JSON object or array is passed through as structured state, which is
    what the model prefers for anything with fields; anything else is passed as text.
    """
    if args.state and args.state != "-":
        raw = open(args.state, encoding="utf-8").read()
    elif args.state == "-" or not sys.stdin.isatty():
        raw = sys.stdin.read()
    else:
        return sample
    if not raw.strip():
        return sample
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw
    return parsed if isinstance(parsed, (dict, list)) else raw


def dump(response) -> int:
    print(json.dumps(response, indent=2, ensure_ascii=False))
    return 0


def run(main) -> int:
    """Run an example's `main`, turning a server-side failure into one line instead of a traceback.

    A server that is not running is the commonest way these scripts fail, and it is not a bug in
    them. Exit code 3 keeps that case distinguishable from the 0/1/2 the gates use to report
    their decision.
    """
    try:
        return main()
    except ArbiterError as exc:
        print("arbiter: %s" % exc, file=sys.stderr)
        return 3
    except KeyboardInterrupt:                            # pragma: no cover - interactive only
        return 130
