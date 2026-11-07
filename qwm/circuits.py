"""Target circuits (fixed pool), cheap probe circuits, ZNE folding and observables."""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap

BASIS_RUN = ["rz", "sx", "x", "cx"]
OBS_TYPES = ("Zmean", "ZZmean")


def _line(n):
    return CouplingMap.from_line(n)


def _to_basis(qc: QuantumCircuit, level: int) -> QuantumCircuit:
    n = qc.num_qubits
    return transpile(qc, basis_gates=BASIS_RUN, coupling_map=_line(n),
                     initial_layout=list(range(n)), layout_method="trivial",
                     routing_method="none" if level == 0 else None,
                     optimization_level=level, seed_transpiler=0)


def tfim_trotter(n, steps, J, h, dt):
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for parity in (0, 1):
            for a in range(parity, n - 1, 2):
                qc.rzz(2 * J * dt, a, a + 1)
        for q in range(n):
            qc.rx(2 * h * dt, q)
    return qc


def random_shallow(n, layers, rng):
    qc = QuantumCircuit(n)
    for layer in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(0, np.pi), q)
            qc.rz(rng.uniform(0, 2 * np.pi), q)
        for a in range(layer % 2, n - 1, 2):
            qc.cx(a, a + 1)
    return qc


def observables(n):
    zmean = SparsePauliOp.from_sparse_list([("Z", [i], 1.0 / n) for i in range(n)], n)
    zz = SparsePauliOp.from_sparse_list([("ZZ", [i, i + 1], 1.0 / (n - 1)) for i in range(n - 1)], n)
    return {"Zmean": zmean, "ZZmean": zz}


def build_pool(n, size, max_trotter, rng):
    """Fixed pool of classically simulable circuits, with ideal values from the statevector."""
    obs = observables(n)
    pool = []
    for i in range(size):
        if i % 2 == 0:
            steps = int(rng.integers(1, max_trotter + 1))
            J, h, dt = rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5), rng.uniform(0.1, 0.3)
            raw, family = tfim_trotter(n, steps, J, h, dt), 0
        else:
            steps = int(rng.integers(1, 5))
            raw, family = random_shallow(n, steps, rng), 1
        qc = _to_basis(raw, level=1)
        sv = Statevector(qc)
        ideal = {k: float(np.real(sv.expectation_value(op))) for k, op in obs.items()}
        ops = qc.count_ops()
        feats = np.array([
            ops.get("cx", 0), ops.get("sx", 0) + ops.get("x", 0), qc.depth(), steps, family,
        ], dtype=float)
        pool.append({"qc": qc, "ideal": ideal, "feats": feats})
    return pool


def fold(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Global folding U (U^dag U)^k, scale = 2k + 1."""
    assert scale % 2 == 1
    out = qc.copy()
    for _ in range((scale - 1) // 2):
        out = out.compose(qc.inverse()).compose(qc)
    return _to_basis(out, level=0)


def with_measure(qc: QuantumCircuit) -> QuantumCircuit:
    m = qc.copy()
    m.measure_all()
    return m


def local_expectations(counts, n):
    """Vector [<Z_0>..<Z_{n-1}>, <Z_0Z_1>..<Z_{n-2}Z_{n-1}>]. Qiskit bitstrings: qubit 0 is rightmost."""
    shots = sum(counts.values())
    zi = np.zeros(n)
    zz = np.zeros(n - 1)
    for bits, c in counts.items():
        b = bits.replace(" ", "")[::-1]
        z = np.array([1.0 if b[q] == "0" else -1.0 for q in range(n)])
        zi += c * z
        zz += c * z[:-1] * z[1:]
    return np.concatenate([zi, zz]) / shots


def aggregate(local, n):
    """From the local vector to (mean <Z>, mean <ZZ>); works on arrays whose last dimension is 2n-1."""
    local = np.asarray(local)
    return np.stack([local[..., :n].mean(-1), local[..., n:].mean(-1)], axis=-1)


def readout_correct(local, p01, p10, n):
    """Inverts independent per-qubit readout error on <Z_i> and <Z_iZ_{i+1}>.
    Z_measured = a Z + b, with a = 1 - p01 - p10, b = p10 - p01 (per qubit)."""
    local = np.asarray(local, dtype=float)
    a = 1.0 - p01 - p10
    b = p10 - p01
    zm, zzm = local[..., :n], local[..., n:]
    z = (zm - b) / a
    ai, aj, bi, bj = a[..., :-1], a[..., 1:], b[..., :-1], b[..., 1:]
    zz = (zzm - ai * bj * z[..., :-1] - bi * aj * z[..., 1:] - bi * bj) / (ai * aj)
    return np.clip(np.concatenate([z, zz], axis=-1), -1.0, 1.0)


def expectations_from_counts(counts, n):
    agg = aggregate(local_expectations(counts, n), n)
    return {"Zmean": float(agg[0]), "ZZmean": float(agg[1])}


def marginals_p1(counts, n):
    shots = sum(counts.values())
    p = np.zeros(n)
    for bits, c in counts.items():
        b = bits.replace(" ", "")[::-1]
        p += c * np.array([b[q] == "1" for q in range(n)], dtype=float)
    return p / shots


def _h(qc, q):
    qc.rz(np.pi / 2, q)
    qc.sx(q)
    qc.rz(np.pi / 2, q)


def probe_circuits(n, n_idle=20, n_sx=16):
    """Cheap probe circuits, built directly in the basis (no transpilation, so the 'id' gates are kept)."""
    probes = {}
    qc = QuantumCircuit(n); qc.measure_all(); probes["ro0"] = qc
    qc = QuantumCircuit(n); qc.x(range(n)); qc.measure_all(); probes["ro1"] = qc
    qc = QuantumCircuit(n)
    for _ in range(n_sx):
        qc.sx(range(n))
    qc.measure_all(); probes["overrot"] = qc
    qc = QuantumCircuit(n); qc.x(range(n))
    for _ in range(n_idle):
        qc.id(range(n))
    qc.measure_all(); probes["t1"] = qc
    qc = QuantumCircuit(n)
    for q in range(n):
        _h(qc, q)
    for _ in range(n_idle):
        qc.id(range(n))
    for q in range(n):
        _h(qc, q)
    qc.measure_all(); probes["t2"] = qc
    qc = QuantumCircuit(n); _h(qc, 0)
    for _ in range(2):
        for a in range(n - 1):
            qc.cx(a, a + 1)
        for a in reversed(range(n - 1)):
            qc.cx(a, a + 1)
    _h(qc, 0); qc.measure_all(); probes["cx_echo"] = qc
    return probes
