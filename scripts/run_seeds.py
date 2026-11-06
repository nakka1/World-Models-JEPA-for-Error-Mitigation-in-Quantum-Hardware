"""Roda o pipeline em várias sementes (em paralelo) e aplica o protocolo estatístico.

    python scripts/run_seeds.py --seeds 15 --jobs 4 --out runs
Depois, só a agregação (sem rodar de novo):
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
    ap.add_argument("--ours", default="wm_completo")
    ap.add_argument("--extra", default="", help="argumentos extras para run_pipeline.py")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not args.aggregate_only:
        with ThreadPoolExecutor(args.jobs) as ex:
            for seed, code in ex.map(lambda s: run_one(s, args), range(args.seeds)):
                print(f"semente {seed}: {'ok' if code == 0 else f'falhou ({code})'}", flush=True)

    res = []
    for s in range(args.seeds):
        p = os.path.join(args.out, f"results_seed{s}.json")
        if os.path.exists(p):
            res.append(json.load(open(p)))
    if len(res) < 2:
        sys.exit("poucas sementes com resultado")
    methods = sorted(set.intersection(*(set(r["h1_mae"]) for r in res)))
    per_seed = {m: np.array([r["h1_mae"][m]["mae"] for r in res]) for m in methods}
    h1 = compare(per_seed, args.ours)

    probe_keys = sorted(set.intersection(*(set(r["h2_sonda_linear"]) for r in res)))
    h2_seed = {k: np.array([r["h2_sonda_linear"][k]["nrmse_media"] for r in res]) for k in probe_keys}
    h2 = compare(h2_seed, args.ours) if args.ours in h2_seed else {}

    summary = {"n_sementes": len(res),
               "h1_mae": {m: {"media": float(v.mean()), "dp": float(v.std(ddof=1))} for m, v in per_seed.items()},
               "h1_teste_vs_" + args.ours: h1,
               "h2_nrmse": {k: {"media": float(v.mean()), "dp": float(v.std(ddof=1))} for k, v in h2_seed.items()},
               "h2_teste_vs_" + args.ours: h2}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print(f"\nH1 — MAE no teste, {len(res)} sementes (Wilcoxon pareado unilateral, Holm)")
    print(f"  {'método':18s} {'MAE médio':>10s} {'dp':>8s} {'Δ vs nosso':>11s} {'p Holm':>8s}")
    for m in sorted(methods, key=lambda k: per_seed[k].mean()):
        row = h1.get(m)
        extra = f"{-row['diferenca_media']:+11.4f} {row['p_holm']:8.4f}" if row else f"{'—':>11s} {'—':>8s}"
        print(f"  {m:18s} {per_seed[m].mean():10.4f} {per_seed[m].std(ddof=1):8.4f} {extra}")
    print("\n(Δ positivo = o baseline erra mais que o world model)")


if __name__ == "__main__":
    main()
