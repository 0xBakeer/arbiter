#!/usr/bin/env python3
"""Does the served path answer the same as the SDK does?

Three paths over the same fixed cases:

  reference        laya.Agent.system_one -- fp32 parameters under torch.autocast(bf16), one
                   forward per request, exactly as the SDK ships it
  autocast/eager   the same arithmetic, but all of a checkpoint's rows in one batched forward
  bf16/eager       bf16 parameters and no autocast, batched
  .../graphs       either of those replayed through a captured CUDA graph on a padded bucket

Reported per checkpoint: the largest absolute difference in any reported probability, and
whether the argmax (the chosen option / the rounded score level / true-or-false) ever moves.
Probabilities are compared as the client sees them, rounded to four decimals, which puts a
1e-4 floor under every figure here.

Checkpoints are loaded one at a time and freed in between, so this needs one model in memory,
not nine.
"""
import argparse
import gc
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cases import CASES                                     # noqa: E402
from server.engine import SUBFOLDER, Checkpoint             # noqa: E402


def probs_of(answer: Dict[str, Any]) -> np.ndarray:
    if answer["type"] == "noul":
        return np.array([1.0 - answer["noul"], answer["noul"]])
    return np.array(list(answer["probabilities"].values()), dtype=float)


def pick_of(answer: Dict[str, Any]) -> str:
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        return "true" if answer["noul"] >= 0.5 else "false"
    return str(int(np.argmax(list(answer["probabilities"].values()))))


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_reference(name: str, models_dir: str, device: str, cases: List[Dict]) -> Dict[str, Dict]:
    from laya.agent import Agent

    agent = Agent(models_dir, device=device, subfolder=SUBFOLDER[name])
    out = {}
    for case in cases:
        res = agent.system_one(case["state"], case["questions"])
        for qid, ans in res["answers"].items():
            out["%s/%s" % (case["name"], qid)] = ans
    del agent
    free()
    return out


def run_served(name: str, models_dir: str, device: str, mode: str, cases: List[Dict],
               max_markers: int, dtype_mode: str) -> Tuple[Dict[str, Dict], List[str]]:
    ck = Checkpoint(name, models_dir, device=device, mode=mode, max_markers=max_markers,
                    dtype_mode=dtype_mode)
    out: Dict[str, Dict] = {}
    paths: List[str] = []

    # One forward for everything this checkpoint owns, which is what the micro-batcher does
    # under load: mixed states, mixed question types, padded to the longest row.
    rows, keys = [], []
    for case in cases:
        for row in ck.build_rows(case["state"], case["questions"]):
            rows.append(row)
            keys.append("%s/%s" % (case["name"], row["qid"]))
    logits, act, _, path = ck.forward(rows)
    paths.append(path)
    for i, key in enumerate(keys):
        out[key] = ck.answer(rows[i], logits[i], act[i])

    # And once per case, which is the single-caller shape and hits different buckets.
    per_case: Dict[str, Dict] = {}
    for case in cases:
        crows = ck.build_rows(case["state"], case["questions"])
        clogits, cact, _, cpath = ck.forward(crows)
        paths.append(cpath)
        for i, row in enumerate(crows):
            per_case["%s/%s" % (case["name"], row["qid"])] = ck.answer(row, clogits[i], cact[i])

    if mode == "graphs" and ck.graphs is not None:
        paths.append("captured=%s" % (ck.graphs.captured,))
    del ck
    free()
    out.update({k + " [single]": v for k, v in per_case.items()})
    return out, paths


def compare(a: Dict[str, Dict], b: Dict[str, Dict]) -> Tuple[float, int, int, List[str]]:
    """max |delta p|, agreeing argmax count, total compared, the keys that disagree."""
    keys = [k for k in a if k in b]
    worst, agree, moved = 0.0, 0, []
    for k in keys:
        pa, pb = probs_of(a[k]), probs_of(b[k])
        n = min(len(pa), len(pb))
        worst = max(worst, float(np.abs(pa[:n] - pb[:n]).max()))
        if pick_of(a[k]) == pick_of(b[k]):
            agree += 1
        else:
            moved.append(k)
    return worst, agree, len(keys), moved


def strip_single(d: Dict[str, Dict]) -> Dict[str, Dict]:
    return {k.replace(" [single]", ""): v for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.environ.get("ARBITER_MODELS_DIR", "models/laya"))
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    ap.add_argument("--models", default=os.environ.get("ARBITER_MODELS", ",".join(SUBFOLDER)))
    ap.add_argument("--max-markers", type=int, default=int(os.environ.get("ARBITER_GRAPH_MAX_MARKERS", "32")))
    ap.add_argument("--dtypes", default="autocast,bf16", help="which parameter modes to compare")
    ap.add_argument("--out", default=None, help="append the markdown table to this file")
    args = ap.parse_args()

    names = [n.strip() for n in args.models.split(",") if n.strip()]
    graphs_possible = args.device.startswith("cuda") and torch.cuda.is_available()

    dtype_modes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    modes = ["eager"] + (["graphs"] if graphs_possible else [])

    lines = ["| checkpoint | path | questions | max abs delta p vs reference | argmax agreement |",
             "|---|---|---:|---:|---:|"]
    worst = {}
    agrees = {}

    for name in names:
        cases = [c for c in CASES if c["expect_route"] == name]
        if not cases:
            continue
        print("== %s (%d cases, %d questions)" % (name, len(cases), sum(len(c["questions"]) for c in cases)))

        ref = run_reference(name, args.models_dir, args.device, cases)
        served = {}
        for dtype_mode in dtype_modes:
            for mode in modes:
                label = "%s/%s" % (dtype_mode, mode)
                out, paths = run_served(name, args.models_dir, args.device, mode, cases,
                                        args.max_markers, dtype_mode)
                served[label] = out
                w, agree, total, moved = compare(ref, strip_single(out))
                worst[label] = max(worst.get(label, 0.0), w)
                agrees[label] = agrees.get(label, True) and (agree == total)
                lines.append("| %s | %s | %d | %.2e | %d/%d |" % (name, label, total, w, agree, total))
                print("   %-18s max|dp|=%.2e argmax %d/%d %s   %s"
                      % (label, w, agree, total, moved or "", paths[-1] if mode == "graphs" else ""))

        # and eager against graphs within each dtype, which isolates the graph padding itself
        for dtype_mode in dtype_modes:
            if "graphs" not in modes:
                continue
            w, agree, total, moved = compare(served["%s/eager" % dtype_mode], served["%s/graphs" % dtype_mode])
            lines.append("| %s | %s eager vs graphs | %d | %.2e | %d/%d |"
                         % (name, dtype_mode, total, w, agree, total))
            print("   %-18s max|dp|=%.2e argmax %d/%d %s" % ("%s e-vs-g" % dtype_mode, w, agree, total, moved or ""))

    verdict = []
    for label in sorted(worst):
        ok = worst[label] < 5e-3 and agrees[label]
        verdict.append("%s: max|dp| %.2e, argmax %s -> %s"
                       % (label, worst[label], "unchanged" if agrees[label] else "MOVED",
                          "within the 5e-3 / 100% gate" if ok else "OUTSIDE the gate"))

    block = "\n".join(lines) + "\n\n" + "\n".join("- " + v for v in verdict) + "\n"
    print()
    print(block)
    if args.out:
        with open(args.out, "a") as f:
            f.write("\n" + block)
    return 0 if all("OUTSIDE" not in v for v in verdict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
