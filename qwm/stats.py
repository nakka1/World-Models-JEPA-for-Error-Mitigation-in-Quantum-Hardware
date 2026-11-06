"""Protocolo estatístico: Wilcoxon pareado por semente, correção de Holm, IC bootstrap."""
import numpy as np
from scipy.stats import wilcoxon


def holm(pvals: dict) -> dict:
    """Correção de Holm-Bonferroni (step-down); devolve p ajustados, monotônicos."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - k) * p))
        adj[name] = running
    return adj


def bootstrap_ci(x, n_boot=10_000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def compare(per_seed: dict, ours: str, alternative: str = "less") -> dict:
    """per_seed[método] = array de MAE por semente (mesma ordem de sementes).
    Testa H1 direcional: MAE(ours) < MAE(baseline)."""
    a = np.asarray(per_seed[ours])
    raw_p, rows = {}, {}
    for name, b in per_seed.items():
        if name == ours:
            continue
        d = a - np.asarray(b)
        p = float(wilcoxon(a, b, alternative=alternative).pvalue) if np.any(d != 0) else 1.0
        raw_p[name] = p
        lo, hi = bootstrap_ci(d)
        n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
        rows[name] = {"diferenca_media": float(d.mean()), "ic95": [lo, hi], "p": p,
                      # (sementes em que o nosso venceu − em que perdeu) / total
                      "saldo_de_sinais": (n_neg - n_pos) / max(1, len(d))}
    for name, p in holm(raw_p).items():
        rows[name]["p_holm"] = p
    return rows
