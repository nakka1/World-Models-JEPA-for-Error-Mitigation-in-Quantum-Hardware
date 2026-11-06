"""Constrói o NoiseModel do Qiskit Aer a partir de um DeviceState."""
import numpy as np
from qiskit.circuit.library import RXGate, RZZGate
from qiskit_aer.noise import (NoiseModel, ReadoutError, coherent_unitary_error,
                              depolarizing_error, thermal_relaxation_error)

from .config import DeviceConfig
from .device import DeviceState

BASIS = ["id", "rz", "sx", "x", "cx"]


def build_noise_model(state: DeviceState, cfg: DeviceConfig) -> NoiseModel:
    n = cfg.n_qubits
    nm = NoiseModel(basis_gates=BASIS)
    for q in range(n):
        t1, t2 = float(state.T1[q]), float(min(state.T2[q], 2 * state.T1[q] * 0.999))
        relax_1q = thermal_relaxation_error(t1, t2, cfg.t_1q)
        dep_1q = depolarizing_error(cfg.p1_depol, 1)
        eps = float(state.eps1[q])
        err_sx = coherent_unitary_error(RXGate(eps).to_matrix()).compose(relax_1q).compose(dep_1q)
        err_x = coherent_unitary_error(RXGate(2 * eps).to_matrix()).compose(relax_1q).compose(dep_1q)
        nm.add_quantum_error(err_sx, "sx", [q])
        nm.add_quantum_error(err_x, "x", [q])
        nm.add_quantum_error(thermal_relaxation_error(t1, t2, cfg.t_id), "id", [q])
        p01, p10 = float(state.p01[q]), float(state.p10[q])
        nm.add_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [q])
    for e in range(n - 1):
        a, b = e, e + 1
        tr = {}
        for q in (a, b):
            t1, t2 = float(state.T1[q]), float(min(state.T2[q], 2 * state.T1[q] * 0.999))
            tr[q] = thermal_relaxation_error(t1, t2, cfg.t_cx)
        coh = coherent_unitary_error(RZZGate(float(state.eps2[e])).to_matrix())
        dep = depolarizing_error(float(state.p2[e]), 2)
        # QuantumError em [a, b]: o qubit 0 do erro é 'a'; tensor coloca 'other' no qubit 0
        err_ab = coh.compose(tr[b].tensor(tr[a])).compose(dep)
        err_ba = coh.compose(tr[a].tensor(tr[b])).compose(dep)
        nm.add_quantum_error(err_ab, "cx", [a, b])
        nm.add_quantum_error(err_ba, "cx", [b, a])
    return nm
