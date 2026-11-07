"""Baselines: raw, readout correction, ZNE, calibration-based digital twin, static ML-QEM
(in the style of Liao et al. 2024) and ML-QEM with online retraining."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import Table


def zne_linear(y_scales, scales=(1, 3, 5)):
    """Linear extrapolation over scales (1, 3)."""
    s1, s3 = scales[0], scales[1]
    return (s3 * y_scales[:, 0] - s1 * y_scales[:, 1]) / (s3 - s1)


def zne_richardson(y_scales, scales=(1, 3, 5)):
    """Second-order Richardson over (1, 3, 5): degree-2 polynomial evaluated at 0."""
    s = np.asarray(scales, dtype=float)
    w = np.array([np.prod([s[m] / (s[m] - s[k]) for m in range(len(s)) if m != k]) for k in range(len(s))])
    return y_scales @ w


def physics_predictions(tab: Table) -> dict:
    return {
        "raw": tab.y_raw,
        "readout": tab.y_ro[:, 0],
        "zne_linear": zne_linear(tab.y_ro),
        "zne_richardson": zne_richardson(tab.y_ro),
        "digital_twin": tab.y_raw - (tab.y_twin - tab.y_ideal),
    }


def mlqem_features(tab: Table, tele: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Noisy values + circuit features + observable + noise parameters from the telemetry."""
    return np.column_stack([tab.y_raw, tab.y_ro[:, 0], tab.obs, tab.cfeat, tele[tab.t], age[tab.t]])


def fit_mlqem(kind: str, X, y, seed: int):
    if kind == "rf":
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    else:
        # standardized target: residuals are ~1e-2 and, without this, the default tolerance stops training early
        mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 128), alpha=1e-3, max_iter=600,
                                                             early_stopping=True, random_state=seed))
        m = TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())
    return m.fit(X, y)


def mlqem_static(tab: Table, tele, age, train_mask, test_mask, seed):
    X = mlqem_features(tab, tele, age)
    res = tab.y_ideal - tab.y_ro[:, 0]
    out = {}
    for kind in ("rf", "mlp"):
        m = fit_mlqem(kind, X[train_mask], res[train_mask], seed)
        out[f"mlqem_{kind}"] = tab.y_ro[test_mask, 0] + m.predict(X[test_mask])
    return out


def mlqem_online(tab: Table, tele, age, test_steps, seed, labeled=None, history=240, every=24):
    """RF retrained every `every` steps on the last `history` labeled steps (assumes simulable
    training circuits are run continuously). Uses only data up to t-1 and only samples from
    labeled circuits (`labeled`; unseen circuits never enter training)."""
    X = mlqem_features(tab, tele, age)
    res = tab.y_ideal - tab.y_ro[:, 0]
    pred = np.full(len(tab), np.nan)
    labeled = np.ones(len(tab), dtype=bool) if labeled is None else labeled
    model = None
    for k, t in enumerate(test_steps):
        if model is None or k % every == 0:
            win = (tab.t >= t - history) & (tab.t < t) & labeled
            model = fit_mlqem("rf", X[win], res[win], seed)
        rows = tab.t == t
        pred[rows] = tab.y_ro[rows, 0] + model.predict(X[rows])
    return pred
