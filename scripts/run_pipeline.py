"""Full pipeline for one seed: generate trajectory -> baselines -> world model -> metrics.

Example (Colab A100):
    python scripts/run_pipeline.py --seed 0 --out runs
Quick test:
    python scripts/run_pipeline.py --seed 0 --steps 240 --epochs 5 --out runs_smoke
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwm.baselines import (fit_mlqem, mlqem_features, mlqem_online, mlqem_static,  # noqa: E402
                           physics_predictions)
from qwm.config import ExperimentConfig  # noqa: E402
from qwm.dataset import generate, split_indices, unseen_circuits  # noqa: E402
from qwm.evaluate import SHOT_COST, anomaly_auroc, linear_probe, mae_report  # noqa: E402
from qwm.features import Standardizer, build_table, step_observations, theta_targets  # noqa: E402
from qwm.train import WMData, encode_steps, prediction_error, predict_rows, train_world_model  # noqa: E402

VARIANTS = {
    "wm_full": {},
    "wm_no_predictor": {"use_predictor": False},
    "wm_no_sigreg": {"use_sigreg": False},
    "wm_no_probes": {"use_probes": False},
    "wm_no_stopgrad": {"stop_grad_target": False},
}


def load_or_generate(cfg, path):
    if os.path.exists(path):
        print(f"[data] loading {path}")
        return dict(np.load(path, allow_pickle=False))
    print(f"[data] generating trajectory: {cfg.data.n_steps} steps, seed {cfg.seed}")
    data = generate(cfg)
    np.savez_compressed(path, **data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="number of steps (default: 1440 = 30 days)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--drift-scale", type=float, default=1.0)
    ap.add_argument("--calib-every", type=int, default=None, help="steps between calibrations (default 48 = daily)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--skip-online", action="store_true")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    cfg = ExperimentConfig(seed=args.seed)
    cfg.device.drift_scale = args.drift_scale
    if args.calib_every:
        cfg.device.calib_every_steps = args.calib_every
    if args.steps:
        cfg.data.n_steps = args.steps
        # proportional splits for short trajectories (20/30 and 24/30)
        days = args.steps * cfg.device.dt_hours / 24
        cfg.data.split_days = (days * 20 / 30, days * 24 / 30)
    if args.epochs:
        cfg.model.epochs = args.epochs
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    tag = f"seed{args.seed}_n{cfg.data.n_steps}_d{args.drift_scale}_c{cfg.device.calib_every_steps}"
    data = load_or_generate(cfg, os.path.join(args.out, f"data_{tag}.npz"))
    tab = build_table(data)
    tr, va, te = split_indices(cfg, len(data["age"]))
    unseen = np.isin(tab.circ, unseen_circuits(cfg))
    # train and validation only see labeled circuits; the test split evaluates both groups
    m_tr, m_va = np.isin(tab.t, tr) & ~unseen, np.isin(tab.t, va) & ~unseen
    m_te = np.isin(tab.t, te)
    unseen_te = unseen[m_te]
    spd = int(round(24 / cfg.device.dt_hours))
    y_te = tab.y_ideal[m_te]
    print(f"[split] train/val/test steps = {len(tr)}/{len(va)}/{len(te)}; "
          f"test samples = {m_te.sum()} ({unseen_te.sum()} from unseen circuits)")

    preds = {k: v[m_te] for k, v in physics_predictions(tab).items()}
    print("[baselines] static ML-QEM")
    preds.update(mlqem_static(tab, data["tele"], data["age"], m_tr | m_va, m_te, args.seed))
    if not args.skip_online:
        print("[baselines] online ML-QEM")
        preds["mlqem_online"] = mlqem_online(tab, data["tele"], data["age"], te, args.seed, labeled=~unseen)[m_te]

    theta = theta_targets(data)
    names = [str(s) for s in data["param_names"]]
    results = {"config": cfg.to_dict(), "h2_linear_probe": {}, "h4_tls_auroc": {}, "history": {}}

    # H2/H4 references computed from the raw observations at step t
    x_raw = step_observations(data, use_probes=True)
    results["h2_linear_probe"]["raw_obs_t"] = linear_probe(x_raw[tr], x_raw[te], theta[tr], theta[te], names)
    x_tele = step_observations(data, use_probes=False)
    results["h2_linear_probe"]["telemetry_t"] = linear_probe(x_tele[tr], x_tele[te], theta[tr], theta[te], names)
    sx = Standardizer().fit(x_raw[tr])
    diff = np.linalg.norm(sx(x_raw[te]) - sx(x_raw[te - 1]), axis=1)
    results["h4_tls_auroc"]["raw_obs_difference"] = anomaly_auroc(diff, data["tls_onset"][te])

    X_ml = mlqem_features(tab, data["tele"], data["age"])
    for name in args.variants.split(","):
        mcfg = dataclasses.replace(cfg.model, **VARIANTS[name])
        print(f"[world model] {name}")
        obs = step_observations(data, use_probes=mcfg.use_probes)
        wm = WMData(obs, data["action"], tab, mcfg.window, tr, dev)
        model, hist = train_world_model(wm, mcfg, np.where(m_tr)[0], np.where(m_va)[0], int(tr[-1]), args.seed)
        preds[name] = predict_rows(model, wm, np.where(m_te)[0])
        results["history"][name] = hist
        Z_all = encode_steps(model, wm, np.arange(len(data["age"])))
        results["h2_linear_probe"][name] = linear_probe(Z_all[tr], Z_all[te], theta[tr], theta[te], names)
        if mcfg.use_predictor:
            results["h4_tls_auroc"][name] = anomaly_auroc(prediction_error(model, wm, te), data["tls_onset"][te])
        if name == "wm_full":
            # hybrid: world-model latent as an extra feature for an RF (tree-based head)
            Xz = np.column_stack([X_ml, Z_all[tab.t]])
            rf = fit_mlqem("rf", Xz[m_tr | m_va], (tab.y_ideal - tab.y_ro[:, 0])[m_tr | m_va], args.seed)
            preds["wm_latent_rf"] = tab.y_ro[m_te, 0] + rf.predict(Xz[m_te])

    results["h1_mae"] = mae_report({k: v[~unseen_te] for k, v in preds.items()}, y_te[~unseen_te],
                                   tab.t[m_te][~unseen_te], tab.obs[m_te][~unseen_te], spd)
    results["h1_mae_unseen"] = mae_report({k: v[unseen_te] for k, v in preds.items()}, y_te[unseen_te],
                                          tab.t[m_te][unseen_te], tab.obs[m_te][unseen_te], spd)
    results["shot_cost"] = {k: SHOT_COST.get(k, 1) for k in preds}
    results["n_tls_events_test"] = int(data["tls_onset"][te].sum())
    results["n_calibrations"] = int(data["action"].sum())
    results["time_s"] = time.time() - t0
    path = os.path.join(args.out, f"results_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)

    print(f"\nH1 — test MAE (future), lower is better   [{results['n_calibrations']} calibrations in the trajectory]")
    print(f"  {'method':18s} {'seen':>8s} {'unseen':>9s}  shots")
    for k, v in sorted(results["h1_mae"].items(), key=lambda kv: kv[1]["mae"]):
        print(f"  {k:18s} {v['mae']:8.4f} {results['h1_mae_unseen'][k]['mae']:9.4f}  x{results['shot_cost'][k]}")
    print("H2 — linear probe z -> theta: normalized RMSE (lower is better) and mean R²")
    for k, v in results["h2_linear_probe"].items():
        print(f"  {k:22s} nRMSE {v['nrmse_mean']:.3f}  R² {v['mean']:.3f}   "
              + " ".join(f"{g}={e:.2f}" for g, e in v["nrmse"].items()))
    print(f"H4 — TLS detection AUROC ({results['n_tls_events_test']} events in the test split)")
    for k, v in results["h4_tls_auroc"].items():
        print(f"  {k:22s} {v if v is None else round(v, 3)}")
    print(f"\nresults in {path}  ({results['time_s']:.0f}s)")


if __name__ == "__main__":
    main()
