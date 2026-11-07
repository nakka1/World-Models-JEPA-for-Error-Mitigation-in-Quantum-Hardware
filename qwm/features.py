"""Turns a generated trajectory into tables: mitigation samples (t, circuit, observable)
and per-step observations for the world model."""
from dataclasses import dataclass
import numpy as np

from .circuits import OBS_TYPES, aggregate, readout_correct


@dataclass
class Table:
    t: np.ndarray          # time step of the sample
    circ: np.ndarray       # index in the circuit pool
    obs: np.ndarray        # 0 = mean <Z>, 1 = mean <ZZ>
    y_ideal: np.ndarray
    y_raw: np.ndarray      # scale 1, uncorrected
    y_ro: np.ndarray       # [N, n_scales], readout-corrected with the telemetry
    y_twin: np.ndarray     # noisy prediction of the digital twin
    cfeat: np.ndarray      # circuit features

    def subset(self, mask):
        return Table(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))

    def __len__(self):
        return len(self.t)


def readout_params(tele, n):
    q = tele[..., : 5 * n].reshape(*tele.shape[:-1], n, 5)
    return np.clip(q[..., 3], 0, 0.45), np.clip(q[..., 4], 0, 0.45)


def build_table(data: dict) -> Table:
    n = int(data["n_qubits"])
    T, S = data["circ"].shape
    p01, p10 = readout_params(data["tele"], n)                       # [T, n]
    loc = data["loc_noisy"]                                          # [T, S, K, L]
    ro = readout_correct(loc, p01[:, None, None, :], p10[:, None, None, :], n)
    agg_raw = aggregate(loc[:, :, 0], n)                              # [T, S, 2]
    agg_ro = aggregate(ro, n)                                         # [T, S, K, 2]
    agg_twin = aggregate(data["loc_twin"], n)                         # [T, S, 2]
    ideal = data["pool_ideal"][data["circ"]]                          # [T, S, 2]
    rows = {k: [] for k in Table.__dataclass_fields__}
    for o in range(len(OBS_TYPES)):
        tt, jj = np.meshgrid(np.arange(T), np.arange(S), indexing="ij")
        rows["t"].append(tt.ravel()); rows["circ"].append(data["circ"].ravel())
        rows["obs"].append(np.full(T * S, o))
        rows["y_ideal"].append(ideal[..., o].ravel())
        rows["y_raw"].append(agg_raw[..., o].ravel())
        rows["y_ro"].append(agg_ro[..., o].reshape(T * S, -1))
        rows["y_twin"].append(agg_twin[..., o].ravel())
        rows["cfeat"].append(data["pool_feats"][data["circ"].ravel()])
    return Table(**{k: np.concatenate(v) for k, v in rows.items()})


def step_observations(data: dict, use_probes: bool = True) -> np.ndarray:
    """x_t = [telemetry (T1, T2 in log scale), time since calibration, probes]."""
    n = int(data["n_qubits"])
    tele = data["tele"].copy()
    for q in range(n):
        for k in (0, 1):
            tele[:, 5 * q + k] = np.log(np.clip(tele[:, 5 * q + k], 1e-3, None))
    parts = [tele, data["age"][:, None]]
    if use_probes:
        parts.append(data["probe"])
    return np.concatenate(parts, axis=1)


def theta_targets(data: dict) -> np.ndarray:
    n = int(data["n_qubits"])
    th = data["theta"].copy()
    for q in range(n):
        for k in (0, 1):
            th[:, 5 * q + k] = np.log(th[:, 5 * q + k])
    return th


class Standardizer:
    def fit(self, x):
        self.mu = x.mean(0)
        self.sd = x.std(0) + 1e-8
        return self

    def __call__(self, x):
        return (x - self.mu) / self.sd
