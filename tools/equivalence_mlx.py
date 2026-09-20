#!/usr/bin/env python3
"""Does the MLX engine answer the same as the SDK does?

The same 22 questions, the same 5e-3 / 100%-argmax gate and the same reference as
`tools/equivalence.py`: `laya.Agent.system_one` under torch on the **CPU in fp32**, which is the
numerically cleanest thing on a Mac and what the MPS table in `bench/results.md` was measured
against. What changes is the served path -- laya-mlx's own MLX encoder and heads over converted
weights, in each parameter dtype the port offers, batched the way the server batches.

It is a different implementation of the model and not a different precision of the same one, so
this is the measurement that decides whether the engine may ship at all, and in which dtype.

Routing is checked too, and it is cheap: the port carries its own adapted copy of upstream's
router and language detection, neither of which loads a model to decide, so every case is routed
through both and the two decisions have to name the same checkpoint for the same reason.

Checkpoints are loaded one at a time and freed in between, so this needs one model in memory
under torch plus one under MLX, not eight.
"""
import argparse
import gc
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cases import CASES                                                     # noqa: E402
from equivalence import compare, free, run_reference, strip_single          # noqa: E402
from engines.laya_mlx.loader import (                                       # noqa: E402
    HF_REPOS,
    Checkpoint,
    PARAM_DTYPES,
    require_apple_silicon,
)


def free_mlx():
    """Drop the last checkpoint and MLX's buffer cache with it, before loading the next one."""
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()
    free()


def run_served(name: str, models_dir: str, device: str, cases: List[Dict],
               dtype_mode: str) -> Dict[str, Dict]:
    ck = Checkpoint(name, models_dir, device=device, dtype_mode=dtype_mode)
    out: Dict[str, Dict] = {}

    # One forward for everything this checkpoint owns, which is what the micro-batcher does
    # under load: mixed states, mixed question types, padded to the longest row.
    rows, keys = [], []
    for case in cases:
        for row in ck.build_rows(case["state"], case["questions"]):
            rows.append(row)
            keys.append("%s/%s" % (case["name"], row["qid"]))
    logits, act, _, _ = ck.forward(rows)
    for i, key in enumerate(keys):
        out[key] = ck.answer(rows[i], logits[i], act[i])

    # And once per case, which is the single-caller shape.
    for case in cases:
        crows = ck.build_rows(case["state"], case["questions"])
        clogits, cact, _, _ = ck.forward(crows)
        for i, row in enumerate(crows):
            out["%s/%s [single]" % (case["name"], row["qid"])] = ck.answer(row, clogits[i], cact[i])

    del ck
    free_mlx()
    return out


def compare_routing(cases: List[Dict]) -> Tuple[int, int, List[str]]:
    """The two routers over every case: same checkpoint, same reason, or the case is named."""
    from laya.router import Router as SdkRouter
    from laya_mlx.router import Router as MlxRouter

    sdk, mlx_router = SdkRouter(), MlxRouter()
    agree, disagree = 0, []
    for case in cases:
        a = sdk.route(case["state"], case["questions"])
        b = mlx_router.route(case["state"], case["questions"])
        if (a["model"], a["reason"]) == (b["model"], b["reason"]):
            agree += 1
        else:
            disagree.append("%s: %s (%s) vs %s (%s)"
                            % (case["name"], a["model"], a["reason"], b["model"], b["reason"]))
    return agree, len(cases), disagree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.environ.get("ARBITER_MODELS_DIR", "models/laya-mlx"),
                    help="the converted MLX checkpoints, one directory per name")
    ap.add_argument("--upstream-dir", default="models/laya",
                    help="the upstream tree the reference loads from")
    ap.add_argument("--device", default=os.environ.get("ARBITER_DEVICE", "auto"))
    ap.add_argument("--reference-device", default="cpu",
                    help="where laya.Agent runs; the CPU is the cleanest reference on a Mac")
    ap.add_argument("--models", default=os.environ.get("ARBITER_MODELS", ",".join(HF_REPOS)))
    ap.add_argument("--dtypes", default=",".join(PARAM_DTYPES))
    ap.add_argument("--out", default=None, help="append the markdown table to this file")
    args = ap.parse_args()

    require_apple_silicon()
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    dtype_modes = [d.strip() for d in args.dtypes.split(",") if d.strip()]

    print("engine laya_mlx, reference laya.Agent on %s, dtypes %s\n"
          % (args.reference_device, ",".join(dtype_modes)))

    agree, total, disagree = compare_routing(CASES)
    print("routing: %d/%d cases decided identically by laya.Router and laya_mlx.Router %s\n"
          % (agree, total, disagree or ""))

    lines = ["| checkpoint | path | questions | max abs delta p vs reference | argmax agreement |",
             "|---|---|---:|---:|---:|"]
    worst: Dict[str, float] = {}
    agrees: Dict[str, bool] = {}

    for name in names:
        cases = [c for c in CASES if c["expect_route"] == name]
        if not cases:
            continue
        print("== %s (%d cases, %d questions)" % (name, len(cases), sum(len(c["questions"]) for c in cases)))

        ref = run_reference(name, args.upstream_dir, args.reference_device, cases)
        for dtype_mode in dtype_modes:
            label = "%s/eager" % dtype_mode
            out = run_served(name, args.models_dir, args.device, cases, dtype_mode)
            w, ok, n, moved = compare(ref, strip_single(out))
            worst[label] = max(worst.get(label, 0.0), w)
            agrees[label] = agrees.get(label, True) and (ok == n)
            lines.append("| %s | %s | %d | %.2e | %d/%d |" % (name, label, n, w, ok, n))
            print("   %-14s max|dp|=%.2e argmax %d/%d %s" % (label, w, ok, n, moved or ""))

    verdict = ["routing: %d/%d cases identical -> %s"
               % (agree, total, "matches engines/laya" if not disagree else "DIVERGES")]
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
    return 0 if (not disagree and all("OUTSIDE" not in v for v in verdict)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
