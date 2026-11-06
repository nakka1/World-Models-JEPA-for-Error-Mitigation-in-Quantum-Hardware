"""Treino do world model (JEPA + SIGReg + cabeça de mitigação)."""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig
from .features import Standardizer, Table
from .model import NoiseWorldModel, effective_rank, sigreg


class WMData:
    """Tensores prontos: observações por passo, ações e amostras de mitigação."""

    def __init__(self, obs: np.ndarray, action: np.ndarray, table: Table, window: int,
                 train_steps: np.ndarray, device: str):
        self.W = window
        self.dev = device
        self.obs_std = Standardizer().fit(obs[train_steps])
        self.cf_std = Standardizer().fit(table.cfeat[np.isin(table.t, train_steps)])
        pad = np.repeat(obs[:1], window - 1, axis=0)                 # preenche o início da trajetória
        self.obs = torch.tensor(self.obs_std(np.concatenate([pad, obs])), dtype=torch.float32, device=device)
        self.act = torch.tensor(np.concatenate([np.zeros(window - 1), action]), dtype=torch.float32, device=device)
        self.T = len(obs)
        self.tab = table
        f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
        self.t = torch.tensor(table.t, dtype=torch.long, device=device)
        self.cfeat = f(self.cf_std(table.cfeat))
        self.obs_oh = F.one_hot(torch.tensor(table.obs, device=device), 2).float()
        self.y_raw, self.y_ro, self.y_ideal = f(table.y_raw), f(table.y_ro[:, 0]), f(table.y_ideal)

    def windows(self, t: torch.Tensor):
        """Janela de observações que termina no passo t (já deslocada pelo padding)."""
        idx = t.unsqueeze(1) + torch.arange(self.W, device=self.dev).unsqueeze(0)
        return self.obs[idx], self.act[idx]

    def action_at(self, t: torch.Tensor):
        return self.act[t + self.W - 1]


def make_model(data: WMData, cfg: ModelConfig) -> NoiseWorldModel:
    return NoiseWorldModel(data.obs.shape[1], cfg.window, cfg.latent_dim, data.cfeat.shape[1], cfg.hidden).to(data.dev)


def losses(model, data: WMData, rows: torch.Tensor, cfg: ModelConfig, last_step: int):
    t = data.t[rows]
    z = model.encode(*data.windows(t))
    out = {}
    r_hat = model.mitigate(z, data.cfeat[rows], data.obs_oh[rows], data.y_raw[rows], data.y_ro[rows])
    out["mit"] = F.mse_loss(r_hat, data.y_ideal[rows] - data.y_ro[rows])
    if cfg.use_predictor:
        ok = t < last_step                                   # o alvo t+1 precisa estar no mesmo split
        if ok.any():
            t1 = t[ok] + 1
            z1 = model.encode(*data.windows(t1))
            if cfg.stop_grad_target:
                z1 = z1.detach()
            z_hat = model.predict(z[ok], data.action_at(t[ok]))
            out["pred"] = F.mse_loss(z_hat, z1)
    if cfg.use_sigreg:
        out["sigreg"] = sigreg(z, cfg.sigreg_slices)
    total = cfg.beta_mit * out["mit"] + out.get("pred", 0.0) + cfg.lam_sigreg * out.get("sigreg", 0.0)
    return total, out


@torch.no_grad()
def predict_rows(model, data: WMData, rows: np.ndarray, bs: int = 4096) -> np.ndarray:
    model.eval()
    res = []
    for k in range(0, len(rows), bs):
        r = torch.tensor(rows[k:k + bs], device=data.dev)
        z = model.encode(*data.windows(data.t[r]))
        res.append((data.y_ro[r] + model.mitigate(z, data.cfeat[r], data.obs_oh[r], data.y_raw[r], data.y_ro[r])).cpu().numpy())
    return np.concatenate(res) if res else np.zeros(0)


@torch.no_grad()
def encode_steps(model, data: WMData, steps: np.ndarray) -> np.ndarray:
    model.eval()
    t = torch.tensor(steps, device=data.dev)
    return model.encode(*data.windows(t)).cpu().numpy()


@torch.no_grad()
def prediction_error(model, data: WMData, steps: np.ndarray) -> np.ndarray:
    """||P(z_{t-1}, a_{t-1}) - z_t||^2 para cada t em steps (t >= 1)."""
    model.eval()
    t = torch.tensor(steps, device=data.dev)
    z_prev = model.encode(*data.windows(t - 1))
    z_now = model.encode(*data.windows(t))
    return ((model.predict(z_prev, data.action_at(t - 1)) - z_now) ** 2).sum(-1).cpu().numpy()


def train_world_model(data: WMData, cfg: ModelConfig, train_rows, val_rows, train_last_step, seed, log=print):
    torch.manual_seed(seed)
    model = make_model(data, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    gen = np.random.default_rng(seed)
    best, best_state, hist = np.inf, None, []
    for ep in range(cfg.epochs):
        model.train()
        perm = gen.permutation(train_rows)
        agg = {}
        for k in range(0, len(perm), cfg.batch_size):
            rows = torch.tensor(perm[k:k + cfg.batch_size], device=data.dev)
            loss, parts = losses(model, data, rows, cfg, train_last_step)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for name, v in parts.items():
                agg[name] = agg.get(name, 0.0) + float(v.detach()) * len(rows)
        sched.step()
        val_mae = float(np.abs(predict_rows(model, data, val_rows) - data.tab.y_ideal[val_rows]).mean())
        rec = {"epoch": ep, "val_mae": val_mae, **{k: v / len(perm) for k, v in agg.items()}}
        hist.append(rec)
        if val_mae < best:
            best, best_state = val_mae, copy.deepcopy(model.state_dict())
        if ep % 10 == 0 or ep == cfg.epochs - 1:
            z = model.encode(*data.windows(data.t[torch.tensor(train_rows[:2048], device=data.dev)]))
            log(f"    ep {ep:3d}  val MAE {val_mae:.4f}  " + "  ".join(f"{k} {v / len(perm):.4f}" for k, v in agg.items())
                + f"  posto efetivo {effective_rank(z):.1f}")
    model.load_state_dict(best_state)
    return model, hist
