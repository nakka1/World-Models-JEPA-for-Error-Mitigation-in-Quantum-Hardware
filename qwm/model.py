"""JEPA world model of the device noise state: window encoder, action-conditioned predictor,
SIGReg and mitigation head."""
import torch
import torch.nn as nn


def sigreg(z: torch.Tensor, num_slices: int = 256, t_max: float = 3.0, n_t: int = 17) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization (LeJEPA, Balestriero & LeCun 2025).

    Projects z onto random directions and compares the empirical characteristic function of
    each projection with that of N(0, 1) (Epps-Pulley statistic with a Gaussian window).
    Own implementation from the paper's description; the official package is `lejepa`.
    """
    N, D = z.shape
    A = torch.randn(D, num_slices, device=z.device, dtype=z.dtype)
    A = A / A.norm(dim=0, keepdim=True)
    proj = z @ A                                                    # [N, S]
    t = torch.linspace(-t_max, t_max, n_t, device=z.device, dtype=z.dtype)
    arg = proj.unsqueeze(-1) * t                                     # [N, S, T]
    ecf_re, ecf_im = torch.cos(arg).mean(0), torch.sin(arg).mean(0)  # [S, T]
    phi = torch.exp(-0.5 * t ** 2)
    err = (ecf_re - phi) ** 2 + ecf_im ** 2
    stat = torch.trapezoid(err * phi, t, dim=-1)                     # [S]
    return stat.mean()


def mlp(i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class NoiseWorldModel(nn.Module):
    def __init__(self, obs_dim: int, window: int, latent_dim: int, cfeat_dim: int, hidden: int = 256):
        super().__init__()
        self.window = window
        self.encoder = mlp(window * (obs_dim + 1), hidden, latent_dim)
        self.predictor = mlp(latent_dim + 1, hidden, latent_dim)
        # mitigation head: z, circuit features, observable (one-hot), noisy values
        self.mitigation = mlp(latent_dim + cfeat_dim + 2 + 2, hidden, 1)

    def encode(self, obs_win: torch.Tensor, act_win: torch.Tensor) -> torch.Tensor:
        """obs_win [B, W, F], act_win [B, W] -> z [B, D]."""
        x = torch.cat([obs_win, act_win.unsqueeze(-1)], dim=-1).flatten(1)
        return self.encoder(x)

    def predict(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([z, a.unsqueeze(-1)], dim=-1))

    def mitigate(self, z, cfeat, obs_onehot, y_raw, y_ro) -> torch.Tensor:
        """Returns the residual r such that y_hat = y_ro + r."""
        x = torch.cat([z, cfeat, obs_onehot, y_raw.unsqueeze(-1), y_ro.unsqueeze(-1)], dim=-1)
        return self.mitigation(x).squeeze(-1)


def effective_rank(z: torch.Tensor) -> float:
    """Effective rank (entropy of the singular values): monitors representation collapse."""
    s = torch.linalg.svdvals(z - z.mean(0))
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
