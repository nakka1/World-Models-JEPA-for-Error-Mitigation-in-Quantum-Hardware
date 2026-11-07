"""Central experiment configuration (Phase 1, simulator with drift)."""
from dataclasses import dataclass, field, asdict


@dataclass
class DeviceConfig:
    n_qubits: int = 5                 # linear topology 0-1-2-3-4
    dt_hours: float = 0.5             # one step = 30 simulated minutes
    # gate durations (us)
    t_1q: float = 0.035
    t_cx: float = 0.300
    t_id: float = 1.0
    # T1/T2 (us): Ornstein-Uhlenbeck process in log scale
    t1_mean: float = 100.0
    t2_mean: float = 80.0
    log_ou_kappa: float = 0.02        # mean reversion per hour
    log_ou_sigma: float = 0.03        # diffusion per sqrt(hour)
    # TLS events: abrupt T1 drop that recovers over time
    tls_rate_per_hour: float = 0.004  # per qubit
    tls_depth_range: tuple = (0.4, 0.75)
    tls_recovery_hours: float = 8.0
    # coherent control error (rad): random walk since the last calibration
    eps1_sigma: float = 0.006         # per sqrt(hour), single-qubit gates
    eps2_sigma: float = 0.010         # per sqrt(hour), CX (ZZ error)
    eps_residual: float = 0.003       # residual right after calibration
    # readout
    ro_base: float = 0.02
    ro_drift_sigma: float = 0.0012    # per sqrt(hour)
    # depolarizing
    p1_depol: float = 2e-4
    p2_mean: float = 0.008
    p2_sigma: float = 0.0004
    # calibration
    calib_every_steps: int = 48       # daily
    # extra random recalibrations, to vary the actions; None = 1 / (4 * calib_every_steps),
    # so that the effective calibration interval is not shortened
    extra_calib_prob: float = None
    telemetry_noise: float = 0.05     # relative noise of calibration measurements
    drift_scale: float = 1.0          # global multiplier (stress tests)


@dataclass
class DataConfig:
    n_steps: int = 1440               # 30 days
    targets_per_step: int = 4
    pool_size: int = 40
    max_trotter: int = 6
    shots_target: int = 2000
    shots_probe: int = 256
    shots_twin: int = 8192
    zne_scales: tuple = (1, 3, 5)
    split_days: tuple = (20, 24)      # train up to day 20, validation up to day 24, test afterwards
    # unseen circuits: never labeled in train/validation, evaluated only at test time
    # (indices i with i % 8 in {0, 1}: 10 out of 40, half from each family)
    unseen_mod: int = 8


@dataclass
class ModelConfig:
    window: int = 8
    latent_dim: int = 32
    hidden: int = 256
    lam_sigreg: float = 0.05
    beta_mit: float = 1.0
    sigreg_slices: int = 256
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_predictor: bool = True
    use_sigreg: bool = True
    use_probes: bool = True
    stop_grad_target: bool = True


@dataclass
class ExperimentConfig:
    seed: int = 0
    device: DeviceConfig = field(default_factory=DeviceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self):
        return asdict(self)
