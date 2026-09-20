#!/usr/bin/env python3
"""Send the fixed case set to a running server and print the answers and the latency.

This is the check that the whole path works -- routing, batching, decoding -- not a benchmark:
each case runs once, so treat the milliseconds as an order of magnitude.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cases import CASES  # noqa: E402


def one_line(answer):
    t = answer["type"]
    if t == "choice":
        return "%s (p=%.3f, conf=%.3f)" % (answer["choice"], max(answer["probabilities"].values()),
                                           answer["confidence"])
    if t == "score":
        return "%.2f of %d (conf=%.3f)" % (answer["score"], len(answer["legend"]) - 1,
                                           answer["confidence"])
    return "%.3f true (conf=%.3f)" % (answer["noul"], answer["confidence"])


def main():
    import httpx

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8010")
    ap.add_argument("--key", default=os.environ.get("ARBITER_API_KEY") or None)
    args = ap.parse_args()

    headers = {"Authorization": "Bearer %s" % args.key} if args.key else {}
    failures = []

    with httpx.Client(timeout=120.0, headers=headers) as client:
        ready = client.get("%s/readyz" % args.base)
        print("readyz: %s %s" % (ready.status_code, ready.text.strip()))

        page = client.get("%s/" % args.base)
        served = page.status_code == 200 and "arbiter" in page.text
        print("playground: %s %s" % (page.status_code, "ok" if served else "NOT SERVED"))
        if not served:
            failures.append("GET / -> HTTP %d, %d bytes" % (page.status_code, len(page.content)))
        print()

        for case in CASES:
            body = {"state": case["state"], "questions": case["questions"]}
            if case.get("model"):
                body["model"] = case["model"]
            started = time.perf_counter()
            r = client.post("%s/v1/systemone" % args.base, json=body)
            wall = (time.perf_counter() - started) * 1000

            if r.status_code != 200:
                failures.append("%s -> HTTP %d %s" % (case["name"], r.status_code, r.text[:300]))
                print("%-30s FAILED %d %s" % (case["name"], r.status_code, r.text[:200]))
                continue

            out = r.json()
            routed = out["routing"]["model"]
            mark = "ok " if routed == case["expect_route"] else "ROUTE"
            if routed != case["expect_route"]:
                failures.append("%s routed to %s, expected %s (%s)"
                                % (case["name"], routed, case["expect_route"], out["routing"]["reason"]))
            print("%-30s %s  %-16s %6.1f ms wall / %6.1f ms server / %d questions"
                  % (case["name"], mark, routed, wall, out["latency_ms"], len(out["answers"])))
            print("    reason: %s" % out["routing"]["reason"])
            for qid, ans in out["answers"].items():
                print("    %-20s %s" % (qid, one_line(ans)))
            print("    usage: %s" % json.dumps(out["usage"]))
            print()

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  - %s" % f)
        return 1
    print("all %d cases passed" % len(CASES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
