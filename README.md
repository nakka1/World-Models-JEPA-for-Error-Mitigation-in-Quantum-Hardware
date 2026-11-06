# qwm-qem — Latent world model (JEPA) for quantum error mitigation under drift

Phase 1 of the preliminary project: everything runs in a simulator with controlled drift, where the device’s true physical state is known. The world model learns a latent state z_t from calibration telemetry and low-cost probes, predicts z_{t+1} conditioned on the action (run or recalibrate), and uses z_t to correct expected values.

## Estrutura

```
qwm/
  config.py      settings (device, data, model)
  device.py      hidden-state drift process θ_t (T1/T2 + TLS, coherent error, readout, p2)
  noise.py       θ_t -> Qiskit Aer NoiseModel
  circuits.py    target-circuit pool (Trotter TFIM, shallow layers), probes, folding, readout correction
  dataset.py     generates the trajectory (state, telemetry, probes, actions, runs at 3 ZNE scales, twin)
  features.py    mitigation tables and per-step observations
  model.py       encoder, predictor, mitigation head, SIGReg
  train.py       world model training (latent prediction + SIGReg + mitigation)
  baselines.py   raw, readout, linear/Richardson ZNE, digital twin, static and online ML-QEM
  evaluate.py    H1 (MAE), H2 (linear probe), H4 (TLS AUROC) metrics
  stats.py       paired Wilcoxon, Holm, bootstrap CI
scripts/
  run_pipeline.py  one seed, end to end
  run_seeds.py     multiple seeds in parallel + statistical test
```

## How to run

```bash
pip install -r requirements.txt
# quick test (~2 min of CPU data generation)
python scripts/run_pipeline.py --seed 0 --steps 240 --epochs 8 --out runs_smoke
# full run for one seed: 30 days, 1440 steps (~10 min of CPU data generation)
python scripts/run_pipeline.py --seed 0 --out runs
# full protocol: 15 seeds, Wilcoxon + Holm
python scripts/run_seeds.py --seeds 15 --jobs 4 --out runs
# stress test: 2x faster drift
python scripts/run_seeds.py --seeds 15 --jobs 4 --out runs_drift2 --extra "--drift-scale 2"
```

Data generation runs on the CPU (Qiskit Aer, density matrix, 5 qubits) and is cached in
data_seed*.npz; training uses the GPU if available. In Colab, data generation is the bottleneck: set --jobs to the number of cores.

## Modeling decisions (for review)

- Exogenous dynamics. Running circuits does not change the drift; only recalibrate changes the state. It resets coherent error and readout drift, but does not restore T1/T2.
- Telemetry is blind to post-calibration coherent error. Calibration measures the residual immediately after recalibration; the error that accumulates afterward is only visible in the probes (overrot, cx_echo). This gives the world model room to outperform ML-QEM with telemetry.
- Mitigation as a residual over the readout-corrected value: ŷ = y_ro + M(z, x, ỹ).
- Ideal value is available because the pool can be simulated classically (as in Liao et al. 2024).
- Digital twin = NoiseModel built from the latest telemetry; additive mitigation ŷ = ỹ − (y_twin − y_ideal).
- ZNE uses readout-corrected values (folding does not amplify readout error).
- Static ML-QEM trains on the training and validation sets; the world model uses the validation set to select the epoch.
- SIGReg is a custom implementation based on the LeJEPA paper; the official package is lejepa.

## Preliminary result (seed 0, 30 days, default configuration)

| Method | Test MAE | Shots |
| --- | --- | --- |
| ML-QEM RF (telemetry) | 0.0049 | ×1 |
| Online ML-QEM | 0.0093 | ×1 |
| Digital twin | 0.0110 | ×1 |
| Full world model | 0.0165 | ×1 |
| Linear ZNE (with readout correction) | 0.0174 | ×2 |
| World model without predictor | 0.0245 | ×1 |
| Raw | 0.0519 | ×1 |

Interpretation: the latent predictor improves the world model (0.0245 → 0.0165), and the latent state outperforms the raw observations on the linear probe (nRMSE 1.11 vs. 1.64), but the RF with telemetry performs best. Adding probes or z itself to the RF did not help (0.0054 and 0.0056). Diagnosis: the benchmark does not yet stress the drift. The same pool of 40 circuits is used for training and testing, so a per-circuit correction solves almost everything. Next experimental design adjustments: unseen circuits in the test set, sweep --drift-scale, and calibrate less frequently.

## Known outstanding items

- [ ] Split with unseen circuits in the test set (circuit generalization, not just temporal generalization)
- [ ] Sweep drift rate (--drift-scale 1, 2, 4) and calibration interval
- [ ] Investigate the sklearn ML-QEM MLP (MAE 0.13, likely a fitting issue)
- [ ] More strongly regularized or tree-based mitigation head over z

- [ ] H3: latent-space planning to choose the mitigation configuration under a shot budget
- [ ] Crosstalk (spurious ZZ coupling between idle neighbors) in the simulator
- [ ] Out-of-distribution tests with other drift patterns (steps, sinusoids)
- [ ] Few TLS events per test trajectory (~3): H4 only makes sense when aggregating across seeds
- [ ] World model still overfits on short trajectories: regularize the mitigation head
- [ ] Validation against real IBM calibration history
