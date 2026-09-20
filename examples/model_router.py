#!/usr/bin/env python3
"""Decide which model should answer a request before paying for a model to decide.

The classic use: an encoder reads the request in tens of milliseconds and says whether the small
local model can handle it. On this machine the "fast" model can be the LLM served next door on the same box --
Laya is 1.16B parameters across three checkpoints and leaves the GPU almost entirely to it -- so
the whole router costs nothing that the answer was not going to cost anyway.

Three routes, not two: `cascade` is the middle path, where the fast model answers first and its
answer is checked (by another Laya call) before it is shown. That is the pattern that saves the
most money, because most requests in the middle band really are answerable by the small model.

    python examples/model_router.py
    python examples/model_router.py --state request.txt
"""
import sys

import _cli
from arbiter_client import ArbiterClient, Noul, Score

QUESTIONS = {
    "complexity": Score(
        "How much reasoning does answering this request take?",
        ["a lookup or a one-line answer",
         "a short answer with a little reasoning",
         "multi-step reasoning or careful synthesis",
         "open-ended research, design or long-form writing"],
    ),
    "needs_tools": Noul(
        "Answering this properly requires calling tools: running code, searching, reading "
        "files, or hitting an API.",
        {"true": "it cannot be answered from knowledge alone",
         "false": "the answer is knowledge or reasoning about what is already in the request"},
    ),
    "needs_long_context": Noul(
        "Answering this requires holding a large amount of material in mind at once: a whole "
        "codebase, a long document, a long conversation history."),
    "is_ambiguous": Noul(
        "The request is ambiguous or underspecified enough that a small model would guess wrong."),
}

FAST_COMPLEXITY = 0.90       # below this, the small model answers alone
POWERFUL_COMPLEXITY = 2.20
SIGNAL = 0.55                # any single hard requirement sends it straight to the big model

SAMPLE = ("Our checkout service started returning 502s for about 3% of requests after this "
          "morning's deploy, but only for customers in the EU. The change was a connection-pool "
          "refactor. Where should I start looking, and what would you check first?")


def decide(r):
    """fast | cascade | powerful."""
    complexity = r.score("complexity")
    tools, long_context = r.noul("needs_tools"), r.noul("needs_long_context")
    ambiguous = r.noul("is_ambiguous")

    if long_context >= SIGNAL:
        return "powerful", "needs_long_context %.2f >= %.2f" % (long_context, SIGNAL)
    if complexity >= POWERFUL_COMPLEXITY:
        return "powerful", "complexity %.2f >= %.2f" % (complexity, POWERFUL_COMPLEXITY)
    if complexity <= FAST_COMPLEXITY and max(tools, ambiguous) < 0.40:
        return "fast", "complexity %.2f <= %.2f, nothing else flagged" % (complexity, FAST_COMPLEXITY)
    return "cascade", ("complexity %.2f, tools %.2f, ambiguous %.2f -- small model first, "
                       "check the answer, escalate if it fails" % (complexity, tools, ambiguous))


def main() -> int:
    args = _cli.parser(__doc__).parse_args()
    state = _cli.load_state(args, SAMPLE)
    response = ArbiterClient(base_url=args.url).system_one(state, QUESTIONS, model=args.model)
    if args.json:
        return _cli.dump(response)
    action, reason = decide(response)
    _cli.render("model router", response, action, reason, label="MODEL")
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main))
