"""Processo de drift do estado físico oculto theta_t de um dispositivo de n qubits.

Componentes:
  - T1/T2 por qubit: OU em log-escala + eventos TLS (queda abrupta de T1, recuperação exponencial).
  - erro coerente de controle: sobre-rotação eps1 por qubit e erro ZZ eps2 por aresta, em passeio
    aleatório desde a última calibração (zerado pela recalibração).
  - leitura p(0->1), p(1->0) por qubit: base + deriva desde a calibração.
  - despolarizante de 2 qubits p2 por aresta: OU.

Ponto físico: executar circuitos não altera o drift. Só a ação "recalibrar" muda o estado,
e ela corrige erro coerente e leitura, mas não restaura T1/T2.
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
    """Simula theta_t. Chamar step(recalibrate) avança dt e devolve o novo estado."""

    def __init__(self, cfg: DeviceConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        n = cfg.n_qubits
        # variações estáticas entre qubits (fabricação)
        self.log_t1_mu = np.log(cfg.t1_mean) + rng.normal(0, 0.2, n)
        self.log_t2_mu = np.log(cfg.t2_mean) + rng.normal(0, 0.2, n)
        self.ro_mu01 = np.clip(cfg.ro_base * np.exp(rng.normal(0, 0.3, n)), 0.003, 0.1)
        self.ro_mu10 = np.clip(1.5 * cfg.ro_base * np.exp(rng.normal(0, 0.3, n)), 0.003, 0.12)
        self.p2_mu = np.clip(cfg.p2_mean * np.exp(rng.normal(0, 0.25, n - 1)), 1e-3, 0.05)
        # estado dinâmico
        self.log_t1 = self.log_t1_mu.copy()
        self.log_t2 = self.log_t2_mu.copy()
        self.tls = np.zeros(n)                 # profundidade atual da queda por TLS, em [0, 1)
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
        # T1/T2: OU em log-escala
        self.log_t1 += c.log_ou_kappa * (self.log_t1_mu - self.log_t1) * h + s * c.log_ou_sigma * sq * rng.normal(size=n)
        self.log_t2 += c.log_ou_kappa * (self.log_t2_mu - self.log_t2) * h + s * c.log_ou_sigma * sq * rng.normal(size=n)
        # TLS: recuperação + novos eventos
        self.tls *= np.exp(-h / c.tls_recovery_hours)
        onset = rng.random(n) < (s * c.tls_rate_per_hour * h)
        depth = rng.uniform(*c.tls_depth_range, size=n)
        self.tls = np.where(onset, np.maximum(self.tls, depth), self.tls)
        self.tls_onset = onset
        # erro coerente e leitura: passeio aleatório desde a calibração
        self.eps1 += s * c.eps1_sigma * sq * rng.normal(size=n)
        self.eps2 += s * c.eps2_sigma * sq * rng.normal(size=n - 1)
        self.ro_off01 += s * c.ro_drift_sigma * sq * rng.normal(size=n)
        self.ro_off10 += s * c.ro_drift_sigma * sq * rng.normal(size=n)
        # despolarizante de 2 qubits: OU
        self.p2 += 0.05 * (self.p2_mu - self.p2) * h + s * c.p2_sigma * sq * rng.normal(size=n - 1)
        self.p2 = np.clip(self.p2, 5e-4, 0.08)
        if recalibrate:
            self.eps1 = rng.normal(0, c.eps_residual, n)
            self.eps2 = rng.normal(0, c.eps_residual, n - 1)
            self.ro_off01[:] = 0.0
            self.ro_off10[:] = 0.0
        return self.state()


def measure_telemetry(state: DeviceState, cfg: DeviceConfig, rng: np.random.Generator) -> np.ndarray:
    """Medida de calibração: theta com ruído relativo. Tomada logo após calibrar,
    então vê o erro coerente residual, não o que vai acumular depois."""
    v = state.vector().copy()
    noise = 1.0 + cfg.telemetry_noise * rng.normal(size=v.shape)
    n = cfg.n_qubits
    out = v * noise
    # eps são ângulos pequenos com sinal: ruído aditivo
    idx_eps1 = [5 * q + 2 for q in range(n)]
    idx_eps2 = [5 * n + 2 * e for e in range(n - 1)]
    for i in idx_eps1 + idx_eps2:
        out[i] = v[i] + cfg.eps_residual * rng.normal()
    return out
