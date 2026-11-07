"""Unit tests for the numerical building blocks (no full simulation)."""
import numpy as np
import pytest
import torch

from qwm.baselines import zne_linear, zne_richardson
from qwm.circuits import aggregate, local_expectations, readout_correct
from qwm.config import DeviceConfig, ExperimentConfig
from qwm.dataset import split_indices, unseen_circuits
from qwm.device import DeviceState, DriftProcess, param_names
from qwm.model import NoiseWorldModel, sigreg
from qwm.stats import holm


def test_local_expectations_bit_order():
    # Qiskit bitstrings put qubit 0 on the right: "001" means qubit 0 is |1>
    loc = local_expectations({"001": 100}, 3)
    np.testing.assert_allclose(loc, [-1, 1, 1, -1, 1])
    np.testing.assert_allclose(aggregate(loc, 3), [1 / 3, 0.0])


def test_readout_correct_inverts_independent_readout_error():
    n = 3
    rng = np.random.default_rng(0)
    z = rng.uniform(-1, 1, n)
    zz = z[:-1] * z[1:]                       # product state: <ZiZj> = <Zi><Zj>
    p01, p10 = np.array([0.02, 0.05, 0.03]), np.array([0.04, 0.01, 0.06])
    a, b = 1 - p01 - p10, p10 - p01
    zm = a * z + b
    zzm = zm[:-1] * zm[1:]
    out = readout_correct(np.concatenate([zm, zzm]), p01, p10, n)
    np.testing.assert_allclose(out, np.concatenate([z, zz]), atol=1e-12)


def test_zne_extrapolations_are_exact_for_polynomials():
    s = np.array([1.0, 3.0, 5.0])
    lin = (0.8 - 0.05 * s)[None, :]
    quad = (0.8 - 0.05 * s + 0.004 * s ** 2)[None, :]
    assert zne_linear(lin)[0] == pytest.approx(0.8)
    assert zne_richardson(quad)[0] == pytest.approx(0.8)


def test_holm_is_monotone_and_bounded():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)      # monotone: never below the previous step
    assert all(0 <= p <= 1 for p in adj.values())


def test_sigreg_prefers_gaussian_over_collapsed():
    torch.manual_seed(0)
    gauss = torch.randn(2048, 16)
    collapsed = torch.zeros(2048, 16) + 0.01 * torch.randn(2048, 16)
    assert sigreg(gauss) < 0.01 < sigreg(collapsed)


def test_world_model_shapes():
    m = NoiseWorldModel(obs_dim=10, window=4, latent_dim=8, cfeat_dim=5, hidden=32)
    z = m.encode(torch.randn(6, 4, 10), torch.zeros(6, 4))
    assert z.shape == (6, 8)
    assert m.predict(z, torch.ones(6)).shape == (6, 8)
    r = m.mitigate(z, torch.randn(6, 5), torch.eye(2)[[0, 1, 0, 1, 0, 1]], torch.randn(6), torch.randn(6))
    assert r.shape == (6,)


def test_device_state_roundtrip_and_recalibration():
    cfg = DeviceConfig()
    proc = DriftProcess(cfg, np.random.default_rng(0))
    st = proc.state()
    v = st.vector()
    assert len(v) == len(param_names(cfg.n_qubits))
    np.testing.assert_allclose(DeviceState.from_vector(v, cfg.n_qubits).vector(), v)
    for _ in range(200):
        proc.step(recalibrate=False)
    drifted = np.abs(proc.eps1).max()
    st = proc.step(recalibrate=True)
    assert np.abs(st.eps1).max() < drifted
    assert np.all(st.T2 <= 2 * st.T1)


def test_splits_are_temporal_and_unseen_covers_both_families():
    cfg = ExperimentConfig()
    tr, va, te = split_indices(cfg)
    assert tr[-1] < va[0] and va[-1] < te[0] and te[-1] == cfg.data.n_steps - 1
    u = unseen_circuits(cfg)
    assert len(u) == 10 and set(u % 2) == {0, 1}
