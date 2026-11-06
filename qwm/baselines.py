"""Baselines: bruto, correção de leitura, ZNE, gêmeo digital por calibração, ML-QEM estático
(estilo Liao et al. 2024) e ML-QEM com re-treino online."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import Table


def zne_linear(y_scales, scales=(1, 3, 5)):
    """Extrapolação linear em (1, 3)."""
    s1, s3 = scales[0], scales[1]
    return (s3 * y_scales[:, 0] - s1 * y_scales[:, 1]) / (s3 - s1)


def zne_richardson(y_scales, scales=(1, 3, 5)):
    """Richardson de ordem 2 em (1, 3, 5): polinômio de grau 2 avaliado em 0."""
    s = np.asarray(scales, dtype=float)
    w = np.array([np.prod([s[m] / (s[m] - s[k]) for m in range(len(s)) if m != k]) for k in range(len(s))])
    return y_scales @ w


def physics_predictions(tab: Table) -> dict:
    return {
        "bruto": tab.y_raw,
        "leitura": tab.y_ro[:, 0],
        "zne_linear": zne_linear(tab.y_ro),
        "zne_richardson": zne_richardson(tab.y_ro),
        "gemeo_digital": tab.y_raw - (tab.y_twin - tab.y_ideal),
    }


def mlqem_features(tab: Table, tele: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Valores ruidosos + features do circuito + observável + parâmetros de ruído da telemetria."""
    return np.column_stack([tab.y_raw, tab.y_ro[:, 0], tab.obs, tab.cfeat, tele[tab.t], age[tab.t]])


def fit_mlqem(kind: str, X, y, seed: int):
    if kind == "rf":
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    else:
        m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 128), alpha=1e-4, max_iter=400,
                                                           early_stopping=True, random_state=seed))
    return m.fit(X, y)


def mlqem_static(tab: Table, tele, age, train_mask, test_mask, seed):
    X = mlqem_features(tab, tele, age)
    res = tab.y_ideal - tab.y_ro[:, 0]
    out = {}
    for kind in ("rf", "mlp"):
        m = fit_mlqem(kind, X[train_mask], res[train_mask], seed)
        out[f"mlqem_{kind}"] = tab.y_ro[test_mask, 0] + m.predict(X[test_mask])
    return out


def mlqem_online(tab: Table, tele, age, test_steps, seed, history=240, every=24):
    """RF re-treinado a cada `every` passos com os últimos `history` passos rotulados
    (supõe circuitos de treino simuláveis executados continuamente). Usa só dados até t-1."""
    X = mlqem_features(tab, tele, age)
    res = tab.y_ideal - tab.y_ro[:, 0]
    pred = np.full(len(tab), np.nan)
    model = None
    for k, t in enumerate(test_steps):
        if model is None or k % every == 0:
            win = (tab.t >= t - history) & (tab.t < t)
            model = fit_mlqem("rf", X[win], res[win], seed)
        rows = tab.t == t
        pred[rows] = tab.y_ro[rows, 0] + model.predict(X[rows])
    return pred
