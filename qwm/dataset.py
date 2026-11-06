"""Gera uma trajetória do dispositivo com drift: estado verdadeiro, telemetria, sondas,
ações e execuções dos circuitos-alvo (escalas de ZNE + estimativa do gêmeo digital).

Valores esperados são guardados no formato local (<Z_i> e <Z_iZ_{i+1}>), para permitir correção
de leitura; os observáveis-alvo (<Z>médio, <ZZ>médio) são agregados depois.
"""
import time
import numpy as np
from qiskit_aer import AerSimulator

from .circuits import (OBS_TYPES, build_pool, fold, local_expectations, marginals_p1,
                       probe_circuits, with_measure)
from .config import ExperimentConfig
from .device import DeviceState, DriftProcess, measure_telemetry, param_names
from .noise import build_noise_model


def _run(sim, circuits, noise_model, shots, seed):
    res = sim.run(circuits, noise_model=noise_model, shots=shots, seed_simulator=seed).result()
    return [res.get_counts(i) for i in range(len(circuits))]


def build_noise_model_from_vector(v, cfg: ExperimentConfig):
    """Gêmeo digital por calibração: NoiseModel a partir da última telemetria."""
    n = cfg.device.n_qubits
    st = DeviceState.from_vector(np.asarray(v, dtype=float), n)
    st.T1 = np.clip(st.T1, 1.0, None)
    st.T2 = np.clip(np.minimum(st.T2, 2 * st.T1 * 0.999), 0.5, None)
    st.p01, st.p10 = np.clip(st.p01, 1e-4, 0.4), np.clip(st.p10, 1e-4, 0.4)
    st.p2 = np.clip(st.p2, 1e-5, 0.3)
    return build_noise_model(st, cfg.device)


def generate(cfg: ExperimentConfig, verbose: bool = True) -> dict:
    dc, d = cfg.device, cfg.data
    n = dc.n_qubits
    L = 2 * n - 1
    rng = np.random.default_rng(cfg.seed)
    pool_rng = np.random.default_rng(10_000 + cfg.seed)
    sim = AerSimulator(method="density_matrix")

    pool = build_pool(n, d.pool_size, d.max_trotter, pool_rng)
    scales = list(d.zne_scales)
    folded = {(i, s): with_measure(fold(c["qc"], s)) for i, c in enumerate(pool) for s in scales}
    probes = probe_circuits(n)
    probe_names = list(probes)
    probe_list = [probes[k] for k in probe_names]

    def twin_table(tele_vec, seed):
        """Gêmeo: roda o pool inteiro uma vez por calibração (o modelo só muda quando calibra)."""
        nm_twin = build_noise_model_from_vector(tele_vec, cfg)
        cnt = _run(sim, [folded[(i, 1)] for i in range(d.pool_size)], nm_twin, d.shots_twin, seed)
        return np.stack([local_expectations(c, n) for c in cnt])

    proc = DriftProcess(dc, rng)
    state = proc.state()
    tele = measure_telemetry(state, dc, rng)
    twin_tab = twin_table(tele, int(rng.integers(2**31 - 1)))
    since_cal = 0

    T, S = d.n_steps, d.targets_per_step
    P = len(param_names(n))
    out = {
        "theta": np.zeros((T, P)), "tele": np.zeros((T, P)), "age": np.zeros(T),
        "probe": np.zeros((T, len(probe_names) * n)), "action": np.zeros(T, dtype=np.int64),
        "tls_onset": np.zeros(T, dtype=bool), "tls_depth": np.zeros((T, n)),
        "circ": np.zeros((T, S), dtype=np.int64),
        "loc_noisy": np.zeros((T, S, len(scales), L)),   # dispositivo verdadeiro, por escala de ZNE
        "loc_twin": np.zeros((T, S, L)),                 # gêmeo digital (última calibração)
    }
    t0 = time.time()
    for t in range(T):
        seed_t = int(rng.integers(2**31 - 1))
        nm = build_noise_model(state, dc)
        out["theta"][t] = state.vector()
        out["tele"][t] = tele
        out["age"][t] = since_cal / dc.calib_every_steps
        out["tls_onset"][t] = bool(proc.tls_onset.any())
        out["tls_depth"][t] = proc.tls

        pc = _run(sim, probe_list, nm, d.shots_probe, seed_t)
        out["probe"][t] = np.concatenate([marginals_p1(c, n) for c in pc])

        idx = rng.choice(d.pool_size, size=S, replace=False)
        out["circ"][t] = idx
        batch = [folded[(i, s)] for i in idx for s in scales]
        counts = _run(sim, batch, nm, d.shots_target, seed_t + 1)
        for j, i in enumerate(idx):
            for k in range(len(scales)):
                out["loc_noisy"][t, j, k] = local_expectations(counts[j * len(scales) + k], n)
            out["loc_twin"][t, j] = twin_tab[i]

        since_cal += 1
        recal = since_cal >= dc.calib_every_steps or rng.random() < dc.extra_calib_prob
        out["action"][t] = int(recal)
        state = proc.step(recalibrate=recal)
        if recal:
            since_cal = 0
            tele = measure_telemetry(state, dc, rng)
            twin_tab = twin_table(tele, seed_t + 2)
        if verbose and (t + 1) % max(1, T // 10) == 0:
            print(f"  passo {t + 1}/{T}  ({time.time() - t0:.0f}s)", flush=True)

    out["pool_feats"] = np.stack([c["feats"] for c in pool])
    out["pool_ideal"] = np.array([[c["ideal"][k] for k in OBS_TYPES] for c in pool])
    out["param_names"] = np.array(param_names(n))
    out["probe_names"] = np.array(probe_names)
    out["n_qubits"] = np.array(n)
    return out


def split_indices(cfg: ExperimentConfig, T: int = None):
    steps_per_day = int(round(24 / cfg.device.dt_hours))
    a, b = cfg.data.split_days
    T = T or cfg.data.n_steps
    tr_end, va_end = int(round(min(a * steps_per_day, T))), int(round(min(b * steps_per_day, T)))
    return np.arange(0, tr_end), np.arange(tr_end, va_end), np.arange(va_end, T)
