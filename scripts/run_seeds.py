"""Runs the pipeline over several seeds (in parallel) and applies the statistical protocol.

    python scripts/run_seeds.py --seeds 15 --jobs 4 --out runs
Aggregation only (no rerun):
    python scripts/run_seeds.py --seeds 15 --out runs --aggregate-only
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qwm.stats import compare  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(seed, args):
    log = os.path.join(args.out, f"log_seed{seed}.txt")
    if os.path.exists(os.path.join(args.out, f"results_seed{seed}.json")):
        return seed, 0
    cmd = [sys.executable, os.path.join(HERE, "run_pipeline.py"), "--seed", str(seed), "--out", args.out]
    cmd += args.extra.split() if args.extra else []
    with open(log, "w") as f:
        return seed, subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--ours", default="wm_full")
    ap.add_argument("--split", choices=["seen", "unseen"], default="unseen",
                    help="which test circuits H1 is evaluated on")
    ap.add_argument("--extra", default="", help="extra arguments passed to run_pipeline.py")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not args.aggregate_only:
        with ThreadPoolExecutor(args.jobs) as ex:
            for seed, code in ex.map(lambda s: run_one(s, args), range(args.seeds)):
                print(f"seed {seed}: {'ok' if code == 0 else f'failed ({code})'}", flush=True)

    res = []
    for s in range(args.seeds):
        p = os.path.join(args.out, f"results_seed{s}.json")
        if os.path.exists(p):
            res.append(json.load(open(p)))
    if len(res) < 2:
        sys.exit("not enough seeds with results")
    key = "h1_mae_unseen" if args.split == "unseen" else "h1_mae"
    methods = sorted(set.intersection(*(set(r[key]) for r in res)))
    per_seed = {m: np.array([r[key][m]["mae"] for r in res]) for m in methods}
    h1 = compare(per_seed, args.ours)

    probe_keys = sorted(set.intersection(*(set(r["h2_linear_probe"]) for r in res)))
    h2_seed = {k: np.array([r["h2_linear_probe"][k]["nrmse_mean"] for r in res]) for k in probe_keys}
    h2 = compare(h2_seed, args.ours) if args.ours in h2_seed else {}

    summary = {"n_seeds": len(res), "h1_split": args.split,
               "h1_mae": {m: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))} for m, v in per_seed.items()},
               "h1_test_vs_" + args.ours: h1,
               "h2_nrmse": {k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))} for k, v in h2_seed.items()},
               "h2_test_vs_" + args.ours: h2}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print(f"\nH1 — test MAE on {args.split} circuits, {len(res)} seeds (one-sided paired Wilcoxon, Holm)")
    print(f"  {'method':18s} {'mean MAE':>10s} {'sd':>8s} {'Δ vs ours':>11s} {'Holm p':>8s}")
    for m in sorted(methods, key=lambda k: per_seed[k].mean()):
        row = h1.get(m)
        extra = f"{-row['mean_difference']:+11.4f} {row['p_holm']:8.4f}" if row else f"{'—':>11s} {'—':>8s}"
        print(f"  {m:18s} {per_seed[m].mean():10.4f} {per_seed[m].std(ddof=1):8.4f} {extra}")
    print("\n(positive Δ = the baseline errs more than the world model)")


if __name__ == "__main__":
    main()
