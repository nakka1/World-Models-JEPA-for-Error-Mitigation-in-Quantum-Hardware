# Research proposal — Latent world models (JEPA) for quantum error mitigation under hardware drift

English summary of the Portuguese pre-project.

## Abstract

We propose a latent world model, trained JEPA-style, that learns the noise state of a quantum
processor and how it drifts over time, and uses it to guide error mitigation. Current ML-based
mitigation (ML-QEM) learns a static map from noisy to corrected values and degrades when hardware
noise drifts. Calibration-based digital twins are tied to a fixed parametric form and miss
crosstalk and unmodeled drift. The proposed model encodes calibration telemetry and cheap probe
circuits into a latent state `z`, predicts its evolution conditioned on actions (run or
recalibrate), and uses `z` to correct expectation values and to choose the mitigation
configuration under a shot budget.

## Gap

No current approach models the device noise state as a latent with action-conditioned temporal
dynamics.

| Approach | What it learns | Handles drift? | Main limitation |
| --- | --- | --- | --- |
| ZNE / PEC / CDR | Nothing (fixed protocol) or a per-layer noise model | Only if re-characterized | Shot and circuit overhead |
| ML-QEM (Liao et al., 2024) | Static map: noisy values + features → ideal value | No | Degrades outside the training window |
| Physics-informed GNN QEM (2026) | Local noise propagation, transfers across sizes | No | Assumes stationary noise |
| ML-QEM with online learning | Continuously retrained correction | Reactively | No state model; reacts after the error |
| Calibration-based digital twin | Physical parametric model fitted at each calibration | Not between calibrations | Misses crosstalk and unmodeled drift |
| JEPA / LeJEPA | Predictive representations in latent space | — | Never applied to quantum noise |

Why JEPA: reconstructing the full noise channel (process tomography) is the quantum analogue of
predicting pixels — exponential and full of detail irrelevant to the decision. A JEPA predicts
only the abstract part of the state that matters, and an auxiliary regression head formally
determines what the latent must preserve.

## Research question

Does a latent world model of the device noise state, trained by prediction in representation
space, improve error mitigation under hardware drift compared with static methods, at lower shot
cost?

## Hypotheses

- **H1 — accuracy under drift.** Conditioning the correction on `z` reduces the absolute error of
  the mitigated expectation value in future windows, compared with static ML-QEM using the same
  telemetry.
- **H2 — identifiability.** A linear probe on `z` recovers the hidden physical parameters (T1, T2,
  coherent error, readout) with lower error than on the raw lagged telemetry.
- **H3 — cost.** Choosing the mitigation configuration by planning in latent space matches the
  error of fixed ZNE with fewer shots.
- **H4 — anticipation.** The latent prediction error flags degradation (TLS events, accumulated
  coherent error) before the mitigation error rises, enabling on-demand recalibration.

## Formulation

The device has a hidden physical state `θ_t` that drifts over time. The model never sees `θ_t`:
it sees partial, lagged observations and learns a latent `z_t`.

- **Observation** `o_t` over a window of `W` steps: calibration telemetry (T1, T2, gate and
  readout errors per qubit), refreshed only every `K` steps and with measurement noise, plus time
  since the last calibration; results of cheap probe circuits with few shots.
- **Action** `a_t ∈ {run, recalibrate}`. Running circuits does not change the drift; recalibration
  resets coherent control error and readout but does not restore T1/T2.
- **Model**: encoder `E`, action-conditioned predictor `P`, mitigation head `M`.

```
z_t = E(o_{t-W+1:t}),   ẑ_{t+1} = P(z_t, a_t)

L = || ẑ_{t+1} − sg[E(o_{t-W+2:t+1})] ||²  +  λ · SIGReg(z)  +  β · (M(z_t, x, ỹ) − y*)²
```

`x` = target-circuit features, `ỹ` = its noisy expectation value, `y*` = ideal value (available
for classically simulable circuits), `sg` = stop-gradient. SIGReg pushes embeddings toward an
isotropic Gaussian through random projections and characteristic-function matching, preventing
collapse without EMA teachers or heuristics.

## Experimental design (Phase 1)

- **Simulator** (Qiskit Aer, density matrix): 5 qubits, linear topology; 30-minute steps;
  30-day trajectories (1440 steps). T1/T2 as log-scale OU processes plus abrupt TLS events;
  coherent control error as a random walk since calibration; readout drift; recalibration daily
  by default.
- **Target circuits**: TFIM Trotter evolution (1–6 steps) and shallow random layers, with
  observables mean ⟨Z⟩ and mean ⟨ZZ⟩ and ideal values from the statevector. 10 of 40 circuits are
  held out as unseen.
- **Baselines**: raw, readout correction, ZNE (linear and Richardson), calibration-based digital
  twin, static ML-QEM (RF, MLP), online ML-QEM; world-model ablations without predictor, SIGReg,
  probes or stop-gradient.
- **Metrics**: test MAE on seen and unseen circuits (H1); linear-probe error from `z` to `θ` (H2);
  shots per mitigated value (H3); AUROC and lead time of TLS detection (H4).
- **Temporal splits**: train on days 1–20, validate on 21–24, test on 25–30, always in the future.
- **Statistics**: 15 seeds per condition, one-sided paired Wilcoxon per hypothesis, Holm
  correction, effect sizes and 95% bootstrap CIs.

## Phase 2 (optional): hybrid classical-quantum encoder

Starts only if Phase 1 confirms H1. Replaces the last encoder layer by a 4–8 qubit variational
circuit with SIGReg applied to its measurements, run on a noiseless simulator to avoid
circularity. Question: how does SIGReg interact with concentration of measure? A negative result
is acceptable.

## Risks

| Risk | Mitigation |
| --- | --- |
| The simulator favors the model (generated drift has the shape the model assumes) | OOD tests with drift shapes absent from training; validation on real calibration history |
| Ideal value y* unavailable for large circuits | Train on classically simulable circuits; mimic ZNE on large ones, as in Liao et al. |
| Latent collapse | SIGReg + auxiliary head; monitor effective rank of z |
| Density-matrix simulation cost | 5 qubits in Phase 1; cached noise models; GPU training |
| Small gain over telemetry-based ML-QEM | Still informative: quantifies when latent dynamics pays off (drift rate as a variable) |
