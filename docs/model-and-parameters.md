# Model assumptions and parameters

FDR asks how a structured state space, perturbation, and external cue can produce dispersion and return. The implementation is a synthetic stochastic particle demonstration, with a fixed three-well potential and a W-shaped trench centerline. It is not fitted to neural recordings or behavioral data. Existing potential-landscape and stochastic-dynamics ideas motivate the construction; no novelty or priority claim is made.

## Assumptions

1. State coordinates and time use arbitrary model units. Sample particles are not defined as biological memory elements.
2. Landscape geometry is fixed during each run. Only the coefficient `alpha` varies between the LOW, MEDIUM, and HIGH columns.
3. The same initial states and standard normal innovations are reused across comparisons. State-dependent diffusion means noise displacements need not be identical.
4. An imposed cue pulls toward the initially occupied region A after dissipation. It does not simulate learned retrieval.
5. The diffuse-stress regime deliberately alters three mechanisms together. It does not isolate their causal effects or establish spontaneous forgetting.
6. Euler–Maruyama is used at one fixed timestep. Finiteness and gradient checks do not establish numerical convergence.

## Experiment

Three barrier levels × Cue OFF/ON × two regimes = 12 simulated combinations, presented as six panels per regime. Each panel has 180 samples and 301 stored time points, from 0 to 10.5. Timecourse CSV rows are sampled every five steps (61 per combination).

| Parameter | Value / interpretation |
|---|---|
| Seed | 20261107; a single-seed demonstration |
| Timestep / end time | 0.035 / 10.5 |
| Phase boundaries | Flow until 2.0; dissipation until 6.0; outcome thereafter |
| L / H | 2.20 / 0.75 |
| Trench beta / gamma | 1.10 / 0.050 |
| Barrier alpha LOW / MEDIUM / HIGH | 0.012 / 0.030 / 0.072 |
| Along-trench barrier | `4 * alpha * L**6 / 27`; about 0.20 / 0.50 / 1.21 |
| Cue strength / ramp duration | 0.46 / 0.60 |
| Baseline / peak diffusion | 0.014 / 0.62 |
| Diffusion ramp up / down | 0.85 / 0.90 |
| Stress outcome diffusion | 0.34 |
| Stress restoring fraction | 0.22 |
| Stress escape drift multiplier | 1.0 |
| Local readout radius | 0.52 |
| Initial distribution | Region A center plus independent Gaussian coordinates, standard deviation 0.12 |
| State diffusion factor | `0.80 + 0.70 / (1 + exp(-clip((U - 0.10) / 0.45, -60, 60)))` |
| Video sampling | Every five simulation steps; final point included; 12 frames per second |
| Surface display maximum | 2.40; display clipping only |

The parameter JSON contains the principal numerical settings; the source and this table also specify ramp durations and initialization details. Source is authoritative for exact formulas, including the explicit escape vector field. Animation playback time is independent of model time.

## What the readouts mean

Each sample is assigned its nearest minimum. It contributes to A, B, or C only if its distance is at most 0.52; otherwise it contributes to unbound. The four fractions partition the sample population. Displayed integer percentages can sum to 99% or 101% because of rounding. These local disks are operational measurement regions, not a formal stochastic basin decomposition.

The [README](../README.md) gives the potential and SDE. The [verification record](verification.md) states what has actually been checked.
