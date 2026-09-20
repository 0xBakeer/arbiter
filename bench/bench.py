#!/usr/bin/env python3
"""Measure a running arbiter server: latency by question count, throughput by concurrency.

Shape (a) matches the table the model card publishes for a T4 -- 1, 5, 10 and 50 questions in
one call -- so the two can be read side by side. Shape (b) is what the T4 table does not answer:
what happens when several callers arrive at once, which is the case the micro-batcher exists for.

The whole run is about ninety seconds.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import date

BASE_QUESTIONS = {
    "urgency": {"type": "score", "instructions": "How urgent is this message?",
                "criteria": ["not urgent", "can wait a day", "same day", "immediate"]},
    "category": {"type": "choice", "instructions": "Which queue should this go to?",
                 "criteria": {"billing": None, "technical": None, "account": None, "sales": None}},
    "needs_human": {"type": "noul", "instructions": "This message needs a human agent."},
    "angry": {"type": "noul", "instructions": "The customer is angry."},
}

STATE = ("I was charged twice for the same subscription this month and the second charge has "
         "not been refunded. I have emailed support twice and nobody has replied. This is the "
         "third billing problem since I signed up in March.")


def make_questions(n: int):
    """n questions built by cycling the four shapes, so every count mixes all three types."""
    base = list(BASE_QUESTIONS.items())
    out = {}
    for i in range(n):
        name, spec = base[i % len(base)]
        out["%s_%d" % (name, i)] = dict(spec)
    return out


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        name = out.stdout.strip().splitlines()[0].strip()
        return name or "unknown GPU"
    except Exception:
        return "unknown GPU"


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def latency_series(client, url, headers, n_questions, calls, warmup=3):
    body = {"state": STATE, "questions": make_questions(n_questions)}
    for _ in range(warmup):
        client.post(url, json=body, headers=headers).raise_for_status()
    samples = []
    for _ in range(calls):
        t0 = time.perf_counter()
        r = client.post(url, json=body, headers=headers)
        r.raise_for_status()
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def throughput(url, headers, concurrency, seconds, n_questions=4):
    import httpx

    body = {"state": STATE, "questions": make_questions(n_questions)}
    stop = time.monotonic() + seconds
    counts = [0] * concurrency
    errors = [0] * concurrency
    latencies = [[] for _ in range(concurrency)]

    def worker(i):
        with httpx.Client(timeout=120.0, headers=headers) as c:
            while time.monotonic() < stop:
                t0 = time.perf_counter()
                try:
                    r = c.post(url, json=body)
                    if r.status_code != 200:
                        errors[i] += 1
                        continue
                except Exception:
                    errors[i] += 1
                    continue
                latencies[i].append((time.perf_counter() - t0) * 1000)
                counts[i] += 1

    started = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - started
    calls = sum(counts)
    flat = [x for sub in latencies for x in sub]
    return {
        "concurrency": concurrency,
        "calls": calls,
        "errors": sum(errors),
        "seconds": round(elapsed, 2),
        "calls_per_s": round(calls / elapsed, 1),
        "questions_per_s": round(calls * n_questions / elapsed, 1),
        "p50_ms": round(percentile(flat, 0.5), 1),
        "p95_ms": round(percentile(flat, 0.95), 1),
    }


def main():
    import httpx

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8010")
    ap.add_argument("--key", default=os.environ.get("ARBITER_API_KEY") or None)
    ap.add_argument("--model", default=None, help="pin a checkpoint instead of auto-routing")
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--label", default=os.environ.get("ARBITER_MODE", "eager"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.md"))
    args = ap.parse_args()

    url = "%s/v1/systemone" % args.base
    headers = {"Authorization": "Bearer %s" % args.key} if args.key else {}

    with httpx.Client(timeout=120.0, headers=headers) as client:
        ready = client.get("%s/readyz" % args.base)
        ready.raise_for_status()
        info = ready.json()

        rows = []
        for n in (1, 5, 10, 50):
            samples = latency_series(client, url, headers, n, args.calls)
            rows.append({
                "questions": n,
                "calls": len(samples),
                "p50_ms": round(percentile(samples, 0.5), 1),
                "p95_ms": round(percentile(samples, 0.95), 1),
                "mean_ms": round(statistics.fmean(samples), 1),
                "ms_per_question": round(percentile(samples, 0.5) / n, 2),
            })
            print("latency  %2d questions  p50 %6.1f ms  p95 %6.1f ms  (%.2f ms/question)"
                  % (n, rows[-1]["p50_ms"], rows[-1]["p95_ms"], rows[-1]["ms_per_question"]))

    conc = []
    for c in (1, 8, 32):
        res = throughput(url, headers, c, args.seconds)
        conc.append(res)
        print("throughput  concurrency %2d  %6.1f questions/s  p50 %6.1f ms  p95 %6.1f ms  errors %d"
              % (c, res["questions_per_s"], res["p50_ms"], res["p95_ms"], res["errors"]))

    gpu = gpu_name()
    payload = {"date": str(date.today()), "gpu": gpu, "mode": info.get("mode", args.label),
               "dtype": info.get("dtype", "autocast"),
               "models": info.get("models"), "version": info.get("version"),
               "latency": rows, "throughput": conc}

    md = ["", "## %s -- %s, `ARBITER_MODE=%s`, `ARBITER_DTYPE=%s`"
          % (payload["date"], gpu, payload["mode"], payload["dtype"]), "",
          "Checkpoints loaded: %s. Server version %s." % (", ".join(payload["models"] or []), payload["version"]),
          "", "### Latency, one caller, auto-routed English state", "",
          "| questions in the call | p50 | p95 | per question |", "|---:|---:|---:|---:|"]
    for r in rows:
        md.append("| %d | %.1f ms | %.1f ms | %.2f ms |" % (r["questions"], r["p50_ms"], r["p95_ms"],
                                                            r["ms_per_question"]))
    md += ["", "### Throughput, 4-question calls, %.0f s per level" % args.seconds, "",
           "| concurrency | questions/s | calls/s | p50 | p95 | errors |", "|---:|---:|---:|---:|---:|---:|"]
    for r in conc:
        md.append("| %d | %.1f | %.1f | %.1f ms | %.1f ms | %d |"
                  % (r["concurrency"], r["questions_per_s"], r["calls_per_s"], r["p50_ms"],
                     r["p95_ms"], r["errors"]))
    md.append("")

    with open(args.out, "a") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print("appended to %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
