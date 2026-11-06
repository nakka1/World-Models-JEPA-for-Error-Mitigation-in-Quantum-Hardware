"""Pipeline completo para uma semente: gera trajetória -> baselines -> world model -> métricas.

Exemplo (Colab A100):
    python scripts/run_pipeline.py --seed 0 --out runs
Teste rápido:
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

from qwm.baselines import mlqem_online, mlqem_static, physics_predictions  # noqa: E402
from qwm.config import ExperimentConfig  # noqa: E402
from qwm.dataset import generate, split_indices  # noqa: E402
from qwm.evaluate import SHOT_COST, anomaly_auroc, linear_probe, mae_report  # noqa: E402
from qwm.features import Standardizer, build_table, step_observations, theta_targets  # noqa: E402
from qwm.train import WMData, encode_steps, prediction_error, predict_rows, train_world_model  # noqa: E402

VARIANTS = {
    "wm_completo": {},
    "wm_sem_preditor": {"use_predictor": False},
    "wm_sem_sigreg": {"use_sigreg": False},
    "wm_sem_sondas": {"use_probes": False},
    "wm_sem_stopgrad": {"stop_grad_target": False},
}


def load_or_generate(cfg, path):
    if os.path.exists(path):
        print(f"[dados] carregando {path}")
        return dict(np.load(path, allow_pickle=False))
    print(f"[dados] gerando trajetória: {cfg.data.n_steps} passos, semente {cfg.seed}")
    data = generate(cfg)
    np.savez_compressed(path, **data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="número de passos (padrão: 1440 = 30 dias)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--drift-scale", type=float, default=1.0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--skip-online", action="store_true")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    cfg = ExperimentConfig(seed=args.seed)
    cfg.device.drift_scale = args.drift_scale
    if args.steps:
        cfg.data.n_steps = args.steps
        # splits proporcionais em trajetórias curtas (20/30 e 24/30)
        days = args.steps * cfg.device.dt_hours / 24
        cfg.data.split_days = (days * 20 / 30, days * 24 / 30)
    if args.epochs:
        cfg.model.epochs = args.epochs
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    data = load_or_generate(cfg, os.path.join(args.out, f"data_seed{args.seed}_n{cfg.data.n_steps}_d{args.drift_scale}.npz"))
    tab = build_table(data)
    tr, va, te = split_indices(cfg, len(data["age"]))
    m_tr, m_va, m_te = np.isin(tab.t, tr), np.isin(tab.t, va), np.isin(tab.t, te)
    spd = int(round(24 / cfg.device.dt_hours))
    y_te = tab.y_ideal[m_te]
    print(f"[split] passos treino/val/teste = {len(tr)}/{len(va)}/{len(te)}; amostras de teste = {m_te.sum()}")

    preds = {k: v[m_te] for k, v in physics_predictions(tab).items()}
    print("[baselines] ML-QEM estático")
    preds.update(mlqem_static(tab, data["tele"], data["age"], m_tr | m_va, m_te, args.seed))
    if not args.skip_online:
        print("[baselines] ML-QEM online")
        preds["mlqem_online"] = mlqem_online(tab, data["tele"], data["age"], te, args.seed)[m_te]

    theta = theta_targets(data)
    names = [str(s) for s in data["param_names"]]
    results = {"config": cfg.to_dict(), "h2_sonda_linear": {}, "h4_auroc_tls": {}, "historico": {}}

    # referências para H2/H4 a partir das observações brutas do passo t
    x_raw = step_observations(data, use_probes=True)
    results["h2_sonda_linear"]["obs_brutas_t"] = linear_probe(x_raw[tr], x_raw[te], theta[tr], theta[te], names)
    x_tele = step_observations(data, use_probes=False)
    results["h2_sonda_linear"]["telemetria_t"] = linear_probe(x_tele[tr], x_tele[te], theta[tr], theta[te], names)
    sx = Standardizer().fit(x_raw[tr])
    diff = np.linalg.norm(sx(x_raw[te]) - sx(x_raw[te - 1]), axis=1)
    results["h4_auroc_tls"]["diferenca_obs_brutas"] = anomaly_auroc(diff, data["tls_onset"][te])

    for name in args.variants.split(","):
        mcfg = dataclasses.replace(cfg.model, **VARIANTS[name])
        print(f"[world model] {name}")
        obs = step_observations(data, use_probes=mcfg.use_probes)
        wm = WMData(obs, data["action"], tab, mcfg.window, tr, dev)
        model, hist = train_world_model(wm, mcfg, np.where(m_tr)[0], np.where(m_va)[0], int(tr[-1]), args.seed)
        preds[name] = predict_rows(model, wm, np.where(m_te)[0])
        results["historico"][name] = hist
        Z_tr, Z_te = encode_steps(model, wm, tr), encode_steps(model, wm, te)
        results["h2_sonda_linear"][name] = linear_probe(Z_tr, Z_te, theta[tr], theta[te], names)
        if mcfg.use_predictor:
            results["h4_auroc_tls"][name] = anomaly_auroc(prediction_error(model, wm, te), data["tls_onset"][te])

    results["h1_mae"] = mae_report(preds, y_te, tab.t[m_te], tab.obs[m_te], spd)
    results["custo_shots"] = {k: SHOT_COST.get(k, 1) for k in preds}
    results["n_eventos_tls_teste"] = int(data["tls_onset"][te].sum())
    results["tempo_s"] = time.time() - t0
    path = os.path.join(args.out, f"results_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)

    print("\nH1 — MAE no teste (futuro), menor é melhor")
    for k, v in sorted(results["h1_mae"].items(), key=lambda kv: kv[1]["mae"]):
        print(f"  {k:18s} {v['mae']:.4f}   shots ×{results['custo_shots'][k]}")
    print("H2 — sonda linear z -> theta: RMSE normalizado (menor é melhor) e R² médio")
    for k, v in results["h2_sonda_linear"].items():
        print(f"  {k:22s} nRMSE {v['nrmse_media']:.3f}  R² {v['media']:.3f}   "
              + " ".join(f"{g}={e:.2f}" for g, e in v["nrmse"].items()))
    print(f"H4 — AUROC de detecção de TLS ({results['n_eventos_tls_teste']} eventos no teste)")
    for k, v in results["h4_auroc_tls"].items():
        print(f"  {k:22s} {v if v is None else round(v, 3)}")
    print(f"\nresultados em {path}  ({results['tempo_s']:.0f}s)")


if __name__ == "__main__":
    main()
