"""Exploratory sweep over the benchmark regime: drift rate × calibration interval.

One seed per condition (to choose the regime before the 15-seed protocol):
    python scripts/run_sweep.py --drift 1,2,4 --calib 48,144 --jobs 2 --out sweep
Table only, no rerun:
    python scripts/run_sweep.py --drift 1,2,4 --calib 48,144 --out sweep --table-only
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
SHOW = ["raw", "zne_linear", "digital_twin", "mlqem_rf", "mlqem_online",
        "wm_no_predictor", "wm_full", "wm_latent_rf"]


def cond_dir(out, d, c):
    return os.path.join(out, f"drift{d}_calib{c}")


def run(cond, args):
    d, c = cond
    od = cond_dir(args.out, d, c)
    os.makedirs(od, exist_ok=True)
    if os.path.exists(os.path.join(od, f"results_seed{args.seed}.json")):
        return cond, 0
    cmd = [sys.executable, os.path.join(HERE, "run_pipeline.py"), "--seed", str(args.seed),
           "--drift-scale", str(d), "--calib-every", str(c), "--out", od]
    cmd += args.extra.split() if args.extra else []
    with open(os.path.join(od, "log.txt"), "w") as f:
        return cond, subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", default="1,2,4")
    ap.add_argument("--calib", default="48,144")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="sweep")
    ap.add_argument("--extra", default="")
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()
    conds = list(product([float(x) for x in args.drift.split(",")], [int(x) for x in args.calib.split(",")]))
    os.makedirs(args.out, exist_ok=True)

    if not args.table_only:
        with ThreadPoolExecutor(args.jobs) as ex:
            for (d, c), code in ex.map(lambda k: run(k, args), conds):
                print(f"drift {d} / calibration every {c} steps: {'ok' if code == 0 else f'failed ({code})'}", flush=True)

    rows = {}
    for d, c in conds:
        p = os.path.join(cond_dir(args.out, d, c), f"results_seed{args.seed}.json")
        if os.path.exists(p):
            rows[(d, c)] = json.load(open(p))
    table = {}
    for split, key in (("seen", "h1_mae"), ("unseen", "h1_mae_unseen")):
        print(f"\nTest MAE — {split} circuits (seed {args.seed})")
        print(f"  {'drift×':>6s} {'calib':>6s} " + " ".join(f"{m[:13]:>13s}" for m in SHOW))
        for (d, c), r in rows.items():
            vals = [r[key].get(m, {}).get("mae", float("nan")) for m in SHOW]
            table[f"{split}|drift{d}|calib{c}"] = dict(zip(SHOW, vals))
            print(f"  {d:6.1f} {c:6d} " + " ".join(f"{v:13.4f}" for v in vals))
    with open(os.path.join(args.out, "sweep_table.json"), "w") as f:
        json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()
