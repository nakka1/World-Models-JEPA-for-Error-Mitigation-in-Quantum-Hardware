"""Drift process for the hidden physical state theta_t of an n-qubit device.

Components:
  - per-qubit T1/T2: OU process in log scale + TLS events (abrupt T1 drop, exponential recovery).
  - coherent control error: over-rotation eps1 per qubit and ZZ error eps2 per edge, as a random
    walk since the last calibration (reset by recalibration).
  - per-qubit readout p(0->1), p(1->0): baseline + drift since calibration.
  - per-edge two-qubit depolarizing p2: OU process.

Physical assumption: running circuits does not change the drift. Only the "recalibrate" action
changes the state, and it corrects coherent error and readout but does not restore T1/T2.
"""
from dataclasses import dataclass
import numpy as np

from .config import DeviceConfig

PER_QUBIT = ("T1", "T2", "eps1", "p01", "p10")
PER_EDGE = ("eps2", "p2")


def param_names(n_qubits: int):
    names = [f"{k}_q{q}" for q in range(n_qubits) for k in PER_QUBIT]
    names += [f"{k}_e{e}" for e in range(n_qubits - 1) for k in PER_EDGE]
    return names


@dataclass
class DeviceState:
    T1: np.ndarray
    T2: np.ndarray
    eps1: np.ndarray
    p01: np.ndarray
    p10: np.ndarray
    eps2: np.ndarray
    p2: np.ndarray

    def vector(self) -> np.ndarray:
        n = len(self.T1)
        per_q = np.stack([self.T1, self.T2, self.eps1, self.p01, self.p10], axis=1).reshape(-1)
        per_e = np.stack([self.eps2, self.p2], axis=1).reshape(-1) if n > 1 else np.zeros(0)
        return np.concatenate([per_q, per_e])

    @staticmethod
    def from_vector(v: np.ndarray, n: int) -> "DeviceState":
        q = v[: 5 * n].reshape(n, 5)
        e = v[5 * n:].reshape(n - 1, 2)
        return DeviceState(q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy(), q[:, 3].copy(),
                           q[:, 4].copy(), e[:, 0].copy(), e[:, 1].copy())


class DriftProcess:
    """Simulates theta_t. Calling step(recalibrate) advances dt and returns the new state."""

    def __init__(self, cfg: DeviceConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        n = cfg.n_qubits
        # static qubit-to-qubit variation (fabrication)
        self.log_t1_mu = np.log(cfg.t1_mean) + rng.normal(0, 0.2, n)
        self.log_t2_mu = np.log(cfg.t2_mean) + rng.normal(0, 0.2, n)
        self.ro_mu01 = np.clip(cfg.ro_base * np.exp(rng.normal(0, 0.3, n)), 0.003, 0.1)
        self.ro_mu10 = np.clip(1.5 * cfg.ro_base * np.exp(rng.normal(0, 0.3, n)), 0.003, 0.12)
        self.p2_mu = np.clip(cfg.p2_mean * np.exp(rng.normal(0, 0.25, n - 1)), 1e-3, 0.05)
        # dynamic state
        self.log_t1 = self.log_t1_mu.copy()
        self.log_t2 = self.log_t2_mu.copy()
        self.tls = np.zeros(n)                 # current depth of the TLS-induced drop, in [0, 1)
        self.eps1 = rng.normal(0, cfg.eps_residual, n)
        self.eps2 = rng.normal(0, cfg.eps_residual, n - 1)
        self.ro_off01 = np.zeros(n)
        self.ro_off10 = np.zeros(n)
        self.p2 = self.p2_mu.copy()
        self.tls_onset = np.zeros(n, dtype=bool)

    def state(self) -> DeviceState:
        t1 = np.exp(self.log_t1) * (1.0 - self.tls)
        t2 = np.minimum(np.exp(self.log_t2), 2.0 * t1 * 0.999)
        p01 = np.clip(self.ro_mu01 + self.ro_off01, 1e-3, 0.25)
        p10 = np.clip(self.ro_mu10 + self.ro_off10, 1e-3, 0.25)
        return DeviceState(t1, t2, self.eps1.copy(), p01, p10, self.eps2.copy(), self.p2.copy())

    def step(self, recalibrate: bool) -> DeviceState:
        c, rng, h = self.cfg, self.rng, self.cfg.dt_hours
        s = c.drift_scale
        n = c.n_qubits
        sq = np.sqrt(h)
        # T1/T2: OU in log scale
        self.log_t1 += c.log_ou_kappa * (self.log_t1_mu - self.log_t1) * h + s * c.log_ou_sigma * sq * rng.normal(size=n)
        self.log_t2 += c.log_ou_kappa * (self.log_t2_mu - self.log_t2) * h + s * c.log_ou_sigma * sq * rng.normal(size=n)
        # TLS: recovery + new events
        self.tls *= np.exp(-h / c.tls_recovery_hours)
        onset = rng.random(n) < (s * c.tls_rate_per_hour * h)
        depth = rng.uniform(*c.tls_depth_range, size=n)
        self.tls = np.where(onset, np.maximum(self.tls, depth), self.tls)
        self.tls_onset = onset
        # coherent error and readout: random walk since calibration
        self.eps1 += s * c.eps1_sigma * sq * rng.normal(size=n)
        self.eps2 += s * c.eps2_sigma * sq * rng.normal(size=n - 1)
        self.ro_off01 += s * c.ro_drift_sigma * sq * rng.normal(size=n)
        self.ro_off10 += s * c.ro_drift_sigma * sq * rng.normal(size=n)
        # two-qubit depolarizing: OU
        self.p2 += 0.05 * (self.p2_mu - self.p2) * h + s * c.p2_sigma * sq * rng.normal(size=n - 1)
        self.p2 = np.clip(self.p2, 5e-4, 0.08)
        if recalibrate:
            self.eps1 = rng.normal(0, c.eps_residual, n)
            self.eps2 = rng.normal(0, c.eps_residual, n - 1)
            self.ro_off01[:] = 0.0
            self.ro_off10[:] = 0.0
        return self.state()


def measure_telemetry(state: DeviceState, cfg: DeviceConfig, rng: np.random.Generator) -> np.ndarray:
    """Calibration measurement: theta with relative noise. Taken right after calibrating,
    so it sees the residual coherent error, not what will accumulate afterwards."""
    v = state.vector().copy()
    noise = 1.0 + cfg.telemetry_noise * rng.normal(size=v.shape)
    n = cfg.n_qubits
    out = v * noise
    # eps are small signed angles: additive noise
    idx_eps1 = [5 * q + 2 for q in range(n)]
    idx_eps2 = [5 * n + 2 * e for e in range(n - 1)]
    for i in idx_eps1 + idx_eps2:
        out[i] = v[i] + cfg.eps_residual * rng.normal()
    return out
