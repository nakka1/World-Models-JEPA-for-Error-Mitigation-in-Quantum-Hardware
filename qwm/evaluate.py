"""Metrics for hypotheses H1 (accuracy under drift), H2 (identifiability) and H4 (anticipation).
H3 (planning under a shot budget) comes in a later stage; here only the shot cost is tracked."""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, roc_auc_score

from .device import PER_EDGE, PER_QUBIT
from .features import Standardizer

# executions of the target circuit per mitigated value (multiples of the base shot budget)
SHOT_COST = {"raw": 1, "readout": 1, "zne_linear": 2, "zne_richardson": 3, "digital_twin": 1,
             "mlqem_rf": 1, "mlqem_mlp": 1, "mlqem_online": 1}


def mae_report(preds: dict, y: np.ndarray, t: np.ndarray, obs: np.ndarray, steps_per_day: int):
    rep = {}
    day = (t - t.min()) // steps_per_day
    for name, p in preds.items():
        err = np.abs(p - y)
        rep[name] = {
            "mae": float(err.mean()),
            "mae_Zmean": float(err[obs == 0].mean()),
            "mae_ZZmean": float(err[obs == 1].mean()),
            "mae_per_day": [float(err[day == d].mean()) for d in np.unique(day)],
        }
    return rep


def param_groups(names):
    groups = {}
    for i, nm in enumerate(names):
        groups.setdefault(nm.split("_")[0], []).append(i)
    return groups


def linear_probe(Z_tr, Z_te, th_tr, th_te, names):
    """Mean R² per group of physical parameters for a linear (ridge) probe."""
    sd = Standardizer().fit(th_tr)
    sz = Standardizer().fit(Z_tr)
    m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sz(Z_tr), sd(th_tr))
    pred = m.predict(sz(Z_te))
    y = sd(th_te)
    out, nrmse = {}, {}
    for g, idx in param_groups(names).items():
        out[g] = float(np.mean([r2_score(y[:, i], pred[:, i]) for i in idx]))
        # RMSE in units of the TRAINING standard deviation: stable when the test split has little variance
        nrmse[g] = float(np.mean([np.sqrt(np.mean((y[:, i] - pred[:, i]) ** 2)) for i in idx]))
    out["mean"] = float(np.mean([out[g] for g in PER_QUBIT + PER_EDGE if g in out]))
    out["nrmse"] = nrmse
    out["nrmse_mean"] = float(np.mean(list(nrmse.values())))
    return out


def anomaly_auroc(score: np.ndarray, label: np.ndarray):
    if label.sum() == 0 or label.sum() == len(label):
        return None
    return float(roc_auc_score(label, score))
