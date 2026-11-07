# qwm-qem

**Latent world models (JEPA) for quantum error mitigation under hardware drift.**

Machine-learning error mitigation (ML-QEM) learns a static map from noisy to corrected
expectation values, and degrades when the hardware noise drifts. Calibration-based digital twins
are tied to a fixed parametric noise model. This project treats the device noise state as a
latent variable with its own dynamics: a JEPA-style world model encodes calibration telemetry and
cheap probe circuits into a latent `z_t`, predicts `z_{t+1}` conditioned on the action (run or
recalibrate), and uses `z_t` to correct expectation values.

This repository holds **Phase 1**: everything runs on a simulator with controlled drift, where the
true physical state of the device is known. See [`docs/proposal.md`](docs/proposal.md) for the
research question, hypotheses H1–H4 and the experimental protocol.

> Status: research prototype. Results below come from **one seed per condition** and are
> exploratory; the 15-seed protocol has not been run yet.

## Method

```
observations o_{t-W+1..t}             action a_t ∈ {run, recalibrate}
(telemetry, time since calibration,            │
 probe circuits)                               ▼
        │                       ┌──────────────────────────┐
        └──► Encoder E ──► z_t ─┤ Predictor P(z_t, a_t)    ├──► ẑ_{t+1}  ≈  sg[E(o_{..t+1})]
                           │    └──────────────────────────┘
                           │    SIGReg(z): isotropic Gaussian regularizer (LeJEPA)
                           ▼
            Mitigation head M(z_t, circuit features, ỹ)  ──►  ŷ = y_readout + M(·)
```

Loss: latent prediction + λ · SIGReg + β · mitigation MSE (against ideal values of classically
simulable circuits, as in ML-QEM).

## Repository layout

```
qwm/
  config.py      experiment configuration (device, data, model)
  device.py      drift process of the hidden state θ_t (T1/T2 + TLS, coherent error, readout, p2)
  noise.py       θ_t -> Qiskit Aer NoiseModel
  circuits.py    target-circuit pool (TFIM Trotter, shallow random layers), probes, folding,
                 readout correction
  dataset.py     trajectory generation (state, telemetry, probes, actions, 3 ZNE scales, twin)
  features.py    mitigation tables and per-step observations
  model.py       encoder, predictor, mitigation head, SIGReg
  train.py       world-model training
  baselines.py   raw, readout, linear/Richardson ZNE, digital twin, static and online ML-QEM
  evaluate.py    H1 (MAE), H2 (linear probe), H4 (TLS AUROC) metrics
  stats.py       paired Wilcoxon, Holm correction, bootstrap CI
scripts/
  run_pipeline.py  one seed, end to end
  run_seeds.py     many seeds in parallel + statistical test
  run_sweep.py     exploratory sweep: drift rate × calibration interval
tests/             unit tests for the numerical building blocks
results/sweep_seed0/  sweep results (one seed per condition)
docs/proposal.md      research proposal (English summary)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"                             # optional: package + pytest
```

## Usage

```bash
# quick smoke test (~2 min of CPU data generation)
python scripts/run_pipeline.py --seed 0 --steps 240 --epochs 8 --out runs_smoke

# one full seed: 30 simulated days, 1440 steps (~15 min of CPU data generation)
python scripts/run_pipeline.py --seed 0 --out runs

# regime sweep (one seed per condition, ~1 h on 2 cores)
python scripts/run_sweep.py --drift 1,2,4 --calib 48,144 --jobs 2 --out sweep

# full protocol: 15 seeds, one-sided paired Wilcoxon + Holm, on unseen circuits
python scripts/run_seeds.py --seeds 15 --jobs 4 --out runs \
    --extra "--drift-scale 4 --calib-every 144"

# unit tests
pytest -q
```

Data generation is CPU-bound (Qiskit Aer, density matrix, 5 qubits) and is cached as
`data_*.npz`; training uses a GPU when available. On Colab, generation is the bottleneck, so
set `--jobs` to the number of cores.

## Modeling decisions

- **Exogenous dynamics.** Running circuits does not change the drift; only `recalibrate` changes
  the state, and it resets coherent error and readout drift but does not restore T1/T2.
- **Telemetry is blind to post-calibration coherent error.** Calibration measures the residual
  right after calibrating; what accumulates later only shows up in the probes (`overrot`,
  `cx_echo`). This is where a world model can beat telemetry-based ML-QEM.
- **Mitigation as a residual** on top of the readout-corrected value: `ŷ = y_ro + M(z, x, ỹ)`.
- **Ideal values** are available because the pool is classically simulable (as in Liao et al. 2024).
- **Digital twin** = NoiseModel built from the latest telemetry; additive mitigation
  `ŷ = ỹ − (y_twin − y_ideal)`.
- **ZNE** uses readout-corrected values (folding does not amplify readout error).
- **Unseen circuits.** 10 of the 40 pool circuits (`i % 8 in {0, 1}`, 5 per family) are never
  labeled in training or validation; the test split reports MAE separately for seen and unseen.
- **Extra calibrations** occur with probability `1 / (4 · interval)`, so the effective interval
  is not shortened.
- **Static ML-QEM** trains on train + validation; the world model uses validation for epoch selection.
- **`wm_latent_rf`**: the world-model latent `z` as an extra feature for a random forest.
- **SIGReg** is our own implementation from the LeJEPA paper; the official package is `lejepa`.

## Preliminary results (exploratory sweep, seed 0)

Test MAE, shown as **unseen / seen** circuits. Drift = multiplier of the drift rate;
calibration every 48 steps (daily) or 144 steps (every 3 days). ZNE uses 2× the shots of the
other methods.

| Drift | Calib. | ZNE linear | Digital twin | ML-QEM RF | ML-QEM online | WM full | WM latent + RF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1× | 48 | 0.0188 / 0.0187 | **0.0149** / 0.0124 | 0.0375 / **0.0039** | 0.0301 / 0.0080 | 0.0273 / 0.0227 | 0.0387 / 0.0045 |
| 1× | 144 | 0.0205 / 0.0232 | **0.0144** / 0.0143 | 0.0767 / **0.0075** | 0.0400 / 0.0080 | 0.0224 / 0.0219 | 0.0743 / 0.0093 |
| 2× | 48 | 0.0280 / 0.0250 | **0.0230** / 0.0187 | 0.0462 / **0.0116** | 0.0427 / 0.0137 | 0.0397 / 0.0317 | 0.0479 / 0.0135 |
| 2× | 144 | 0.0309 / 0.0352 | **0.0228** / 0.0239 | 0.0862 / **0.0157** | 0.0459 / 0.0159 | 0.0408 / 0.0355 | 0.0793 / 0.0180 |
| 4× | 48 | 0.0512 / 0.0450 | **0.0382** / 0.0321 | 0.0831 / **0.0196** | 0.0749 / 0.0256 | 0.0631 / 0.0516 | 0.0803 / 0.0218 |
| 4× | 144 | 0.0549 / 0.0603 | **0.0417** / 0.0447 | 0.1014 / 0.0360 | 0.0813 / 0.0307 | 0.0685 / 0.0486 | 0.1026 / **0.0290** |

Full per-condition outputs (including ablations, H2 and H4) are in `results/sweep_seed0/`.

What the sweep shows:

1. **Unseen circuits flip the ranking.** The ML-QEM random forest is best on seen circuits but
   degrades 5–10× on unseen ones. The full world model is the best *learned* method on unseen
   circuits in 5 of 6 conditions, 1.3–3.4× better than the random forest.
2. **Strong drift + rare calibration favors latent dynamics.** At 4× drift with calibration every
   3 days, adding `z` to the random forest (0.0290) beat the static (0.0360) and online (0.0307)
   ML-QEM on seen circuits — the only regime where `z` added anything.
3. **The latent predictor helps** at 1–2× drift (10–30% lower world-model MAE) but not at 4×.
4. **The digital twin wins on unseen circuits everywhere, but the comparison favors it.** The
   simulator and the twin share the exact same noise-model family, and telemetry measures the true
   parameters with only 5% error. On real hardware, crosstalk, non-Markovian noise and leakage
   fall outside the twin's model. **This is the main weakness of the current design.**
5. **H2 and H4 are inconclusive.** The latent beats raw observations on the linear probe at 1×
   drift but not at 4×; with 0–11 TLS events per trajectory, AUROC ranges from 0.37 to 0.82.

## Roadmap

- [ ] **Model mismatch**: noise the twin does not model (idle ZZ crosstalk, time-correlated 1/f
      noise, systematic telemetry bias). Without it the twin is an unrealistic upper bound.
- [ ] **World model as a correction on top of the twin**: feed `y_twin` to the mitigation head so
      the world model only learns what the twin cannot explain.
- [ ] Fix the main regime at 4× drift / calibration every 144 steps and run 15 seeds.
- [ ] H3: planning in latent space to choose the mitigation configuration under a shot budget.
- [ ] Out-of-distribution drift shapes (steps, sinusoids).
- [ ] Validation against real IBM calibration history.
- [ ] Phase 2 (conditional on H1): hybrid classical-quantum encoder.

## References

- Liao et al., [Machine Learning for Practical Quantum Error Mitigation](https://arxiv.org/abs/2309.17368),
  *Nature Machine Intelligence* 6, 1478–1486 (2024).
- Balestriero & LeCun, [LeJEPA: Provable and Scalable Self-Supervised Learning Without the
  Heuristics](https://arxiv.org/abs/2511.08544) (2025).
- Müller et al., [Towards a Digital Twin of Noisy Quantum Computers](https://arxiv.org/abs/2504.08313) (2025).
- LeCun, A Path Towards Autonomous Machine Intelligence (2022).

## Author

Alexandre O. S. Santos — Latin America Quantum Computing Center (LAQCC).
