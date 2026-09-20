#!/usr/bin/env python3
"""Re-rank retrieved passages: one relevance question per passage, all in one forward pass.

A cross-encoder reranker is the standard fix for vector search returning plausible-looking
rubbish, and it normally costs a model call per passage. Here every passage is a row in the same
batch, so twelve passages cost one request and about as long as one.

Two things the model's shape forces, and both are visible in the code:

* The state window is fixed and fairly short, so passages are sent in chunks rather than all at
  once -- past a certain length the tail of the state is simply not read.
* With more than `CHUNK` passages the ranking is hierarchical: a cheap pass over every chunk,
  then a second, stricter pass over the survivors. Two forward passes, not N model calls.

    python examples/rag_relevance.py
    python examples/rag_relevance.py --query "how do I rotate the signing key?"
    python examples/rag_relevance.py --state corpus.json --json
"""
import json
import sys

import _cli
from laya_client import LayaClient, Noul

CHUNK = 6                    # passages per request; keeps each state inside the read window
KEEP = 0.60                  # above this a passage goes to the generator
REVIEW = 0.35                # between REVIEW and KEEP: kept only if nothing better was found
SHORTLIST = 8                # how many survivors the strict second pass re-reads

# Every question is a row over the *same* state, so each one has to say which passage it is
# about. Without the id the rows are byte-identical and the model quite correctly returns the
# same probability for all of them.
COARSE = ("Passage {pid} contains information that helps answer the query. Judge passage {pid} "
          "by its own content, not the other passages in the state.")
STRICT = ("Passage {pid} alone contains the specific facts needed to answer the query -- not "
          "merely the same topic, the same product, or useful background.")

SAMPLE = {
    "query": "How do I rotate the API signing key without downtime?",
    "passages": [
        "Key rotation is a two-phase operation. Publish the new public key to /.well-known/jwks.json first, wait for the cache TTL (600s by default) to expire everywhere, and only then start signing with the new private key.",
        "The API accepts requests signed with any key currently listed in the JWKS document, which is what makes a zero-downtime rotation possible: both keys are valid during the overlap window.",
        "Rate limits are applied per API key and default to 1000 requests per minute. Contact support to raise them.",
        "To create your first API key, open Settings -> Developers -> API keys and press Create. The secret is shown once.",
        "Signing keys are RSA-2048 by default. Ed25519 is available on request and produces shorter signatures.",
        "If a key is compromised, use the emergency revoke endpoint. This is not a rotation: it invalidates the key immediately and every request signed with it starts failing.",
        "Webhook payloads are signed with a separate secret, rotated from the Webhooks tab, and unrelated to API signing keys.",
        "Our status page publishes maintenance windows at least 72 hours in advance.",
        "The JWKS cache TTL can be lowered to 60s for a faster rotation, at the cost of more requests to the discovery endpoint.",
        "Billing is monthly and based on the peak number of active API keys during the period.",
        "During the overlap window both the old and the new key sign valid requests, so no client sees a failure; close the window by removing the old key from the JWKS once traffic has moved.",
        "Sandbox keys are prefixed sk_test_ and never expire. They cannot be used against production.",
        "A rotation that skips the JWKS publication step will fail every request signed with the new key until the cache expires, which is the most common way teams cause an outage during rotation.",
        "SDK versions before 3.2 cached the JWKS document for the process lifetime and ignore the TTL. Upgrade before rotating.",
    ],
}


def ask(client, query, passages, index, instructions, model):
    """One request for up to CHUNK passages; returns {passage index: probability}."""
    state = {"query": query,
             "passages": {"p%d" % i: passages[i] for i in index}}
    questions = {"p%d" % i: Noul(instructions.format(pid="p%d" % i)) for i in index}
    response = client.system_one(state, questions, model=model)
    return {i: response.noul("p%d" % i) for i in index}, response


def rank(client, query, passages, model="auto"):
    """Coarse pass over every chunk, then a strict pass over the shortlist if there were many."""
    scores, last, calls = {}, None, 0
    for start in range(0, len(passages), CHUNK):
        index = list(range(start, min(start + CHUNK, len(passages))))
        part, last = ask(client, query, passages, index, COARSE, model)
        scores.update(part)
        calls += 1

    stage = "one pass over %d passages" % len(passages)
    if len(passages) > 12:
        shortlist = sorted(scores, key=lambda i: -scores[i])[:SHORTLIST]
        strict = {}
        for start in range(0, len(shortlist), CHUNK):
            part, last = ask(client, query, passages, shortlist[start:start + CHUNK], STRICT, model)
            strict.update(part)
            calls += 1
        scores.update(strict)
        stage = "coarse pass over %d, strict pass over the top %d" % (len(passages), len(shortlist))
    return scores, last, "%s (%d requests)" % (stage, calls)


def main() -> int:
    p = _cli.parser(__doc__)
    p.add_argument("--query", help="override the query in the sample corpus")
    args = p.parse_args()
    state = _cli.load_state(args, SAMPLE)
    query = args.query or state["query"]
    passages = state["passages"]

    client = LayaClient(base_url=args.url)
    scores, last, stage = rank(client, query, passages, model=args.model)
    if args.json:
        print(json.dumps({"query": query, "scores": {str(k): v for k, v in scores.items()},
                          "last_response": last}, indent=2, ensure_ascii=False))
        return 0

    _cli.header("rag relevance", last)
    print("  %s%s%s" % (_cli.DIM, _cli.fit("query: " + query, _cli.WIDTH - 2), _cli.RESET))
    print()
    kept = 0
    for rank_, i in enumerate(sorted(scores, key=lambda i: -scores[i]), 1):
        p_ = scores[i]
        mark = "keep" if p_ >= KEEP else ("?" if p_ >= REVIEW else "drop")
        kept += p_ >= KEEP
        color = _cli.GREEN if p_ >= KEEP else (_cli.YELLOW if p_ >= REVIEW else _cli.DIM)
        print("  %s%2d%s %s %s%.2f%s %s%-4s%s %s" % (
            _cli.DIM, rank_, _cli.RESET, _cli.bar(p_, 12), _cli.BOLD, p_, _cli.RESET,
            color, mark, _cli.RESET, _cli.fit(passages[i], _cli.WIDTH - 32)))
    _cli.verdict("keep", "%d of %d passages above %.2f; %s" % (kept, len(passages), KEEP, stage),
                 label="RERANK")
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main))
