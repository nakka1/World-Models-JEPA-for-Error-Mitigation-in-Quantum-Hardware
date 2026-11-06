"""Configuração central do experimento (Fase 1, simulador com drift)."""
from dataclasses import dataclass, field, asdict


@dataclass
class DeviceConfig:
    n_qubits: int = 5                 # topologia linear 0-1-2-3-4
    dt_hours: float = 0.5             # um passo = 30 min simulados
    # tempos de porta (us)
    t_1q: float = 0.035
    t_cx: float = 0.300
    t_id: float = 1.0
    # T1/T2 (us): OU em log-escala
    t1_mean: float = 100.0
    t2_mean: float = 80.0
    log_ou_kappa: float = 0.02        # reversão à média por hora
    log_ou_sigma: float = 0.03        # difusão por sqrt(hora)
    # eventos TLS: queda abrupta de T1 que se recupera
    tls_rate_per_hour: float = 0.004  # por qubit
    tls_depth_range: tuple = (0.4, 0.75)
    tls_recovery_hours: float = 8.0
    # erro coerente de controle (rad): passeio aleatório desde a calibração
    eps1_sigma: float = 0.006         # por sqrt(hora), portas de 1 qubit
    eps2_sigma: float = 0.010         # por sqrt(hora), CX (erro ZZ)
    eps_residual: float = 0.003       # resíduo logo após calibrar
    # leitura
    ro_base: float = 0.02
    ro_drift_sigma: float = 0.0012    # por sqrt(hora)
    # despolarizante
    p1_depol: float = 2e-4
    p2_mean: float = 0.008
    p2_sigma: float = 0.0004
    # calibração
    calib_every_steps: int = 48       # diária
    extra_calib_prob: float = 0.02    # recalibrações extras, para variar as ações
    telemetry_noise: float = 0.05     # ruído relativo das medidas de calibração
    drift_scale: float = 1.0          # multiplicador global (testes de estresse)


@dataclass
class DataConfig:
    n_steps: int = 1440               # 30 dias
    targets_per_step: int = 4
    pool_size: int = 40
    max_trotter: int = 6
    shots_target: int = 2000
    shots_probe: int = 256
    shots_twin: int = 8192
    zne_scales: tuple = (1, 3, 5)
    split_days: tuple = (20, 24)      # treino até dia 20, validação até 24, teste no resto


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
