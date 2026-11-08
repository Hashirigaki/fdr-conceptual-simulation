# Verification record — FDR v0.5

Date: 2026-11-08. Scope: reproducibility and release preparation of a conceptual simulation.

## Current independent rerun

| Check | Result |
|---|---|
| Bundled smoke tests | 4/4 passed: stationary minima / finite-difference gradient, barrier order, operational partition, fixed-seed determinism |
| Python execution | Completed; two MP4s, four PNGs, two CSVs, and one parameter JSON generated |
| Notebook execution | All eight code cells executed in a local Jupyter kernel, including the optional explorer's initial render |
| Python / Notebook source correspondence | All 27 function definitions and 35 shared top-level assignments match by Python AST; notebook adds the execution calls |
| Notebook / Python outputs in this environment | All nine output files match byte-for-byte |
| Archived summary CSV | 12 rows; numeric values agree within absolute/relative tolerance 1e-12; largest absolute difference 2.22e-16 |
| Archived timecourse CSV | 732 rows; same tolerance; largest absolute difference 4.44e-16 |
| Archived parameter JSON | Same parsed content; byte formatting differs |
| Videos | Both newly generated MP4s decode without errors; original archived MP4s are retained for publication |
| Static figures | Original and regenerated Standard final figures inspected; same displayed readouts and qualitative configuration |
| Colab cloud | Not executed; local Jupyter success is not a Colab cloud verification |

The current environment was Windows 11, Python 3.12.14, NumPy 2.5.4, Matplotlib 3.11.2, and Pillow 12.3.0. FFmpeg was supplied by imageio-ffmpeg 0.8.1 for this check. The recorded Python package snapshot is [verification-environment.txt](verification-environment.txt); it is an environment record rather than a cross-platform lockfile. FFmpeg must also be installed separately and available on PATH when reproducing the normal CLI.

The incoming release README reported an earlier local run with byte-identical archived assets and successful execution of all eight notebook code cells. No separate raw log for that earlier run was included in the incoming ZIP. The current independent rerun is the evidence summarized above. Its PNG/MP4 bytes differ from the archive, so cross-environment byte identity is not claimed. Rendering, fonts, encoding, line endings, and small floating-point differences can affect artifact bytes.

## Commands

Run from the repository root, using Python 3.12 or another compatible Python environment:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/fdr_v0_5_showcase.py
```

For local notebook execution:

```bash
python -m pip install jupyterlab ipykernel
python -m jupyterlab
```

Open `notebooks/FDR_v0.5_Showcase_Colab.ipynb` and run all cells. Install FFmpeg with H.264 encoding support and verify `ffmpeg -version` before generating videos. Without FFmpeg, the script warns and omits MP4s while still generating other outputs. Run in a fresh copy if the bundled outputs should be preserved: the default output directory is overwritten.

## Interpretation boundaries

- Checks concern implementation consistency, fixed-seed determinism, and export integrity.
- No timestep convergence study, multi-seed uncertainty analysis, chaos diagnostic, Lyapunov exponent, empirical memory experiment, or biological validation was performed.
- A/B/C are local radius-based readouts; unbound is a finite-time category outside all three disks. It does not mean permanent loss of memory.
- Standard demonstrates one chosen parameter regime; cue-assisted return is not universal.
- Diffuse stress explicitly maintains diffusion, weakens restoring drift, and adds escape drift. It is an imposed stress protocol.
- The figures use fixed viewing windows; samples outside the plotted window remain part of the readout statistics.
- The notebook's optional widget initialization executed locally; all possible interactive selections were not tested.

## Release contents

The original ZIP contained 15 intended release files and two Python bytecode cache files. The caches were excluded. Python source, notebook, tests, dependencies, and the nine archived output assets were preserved without changes. README formatting and verification documentation were updated; no research feature or parameter was changed. See [parameters and assumptions](model-and-parameters.md).

No project license has been selected. No LICENSE file or citation/author identity metadata has been added.
