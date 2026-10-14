# A Wasserstein Trust-Region Method for Smooth Potential–Interaction Energies

Code for the numerical experiments in Section 4 of

> You Wan, Ting Gao, and Jinqiao Duan, *A Wasserstein Trust-Region Method for
> Smooth Potential–Interaction Energies*.

The method (WTR) minimizes energies
F(ρ) = ∫V dρ + ½∬W(x−y) dρ(x) dρ(y) over probability measures. Each step
solves a quadratic model in the L²(ρ) tangent norm with truncated conjugate
gradients and moves the measure by a pushforward. For an empirical measure
ρ_N = N⁻¹ Σ δ_{X_i}, it is a trust-region method for the particles X_i in the
inner product ⟨S, T⟩ = N⁻¹ Σ S_i·T_i.

## Contents

| Path | Content |
|---|---|
| `wtr/trust_region.py` | WTR for particle energies (NumPy): truncated CG and the outer iteration, with the Cauchy-only and no-radius-growth variants |
| `wtr/mmd.py` | Empirical MMD energy with a Gaussian-mixture kernel, its cached version for large target samples, and the banana, two-moons and ring targets |
| `wtr/gaussian_core.py` | Gaussian core model in a harmonic trap and WTR on the GPU (PyTorch) |
| `wtr/lbfgs.py` | L-BFGS baselines for the Gaussian core model (PyTorch and SciPy) |
| `experiments/gaussian_core.py` | Section 4.2: Table 1, Figures 1 and 2 |
| `experiments/mmd_ablation.py` | Section 4.3: Figure 3 |
| `experiments/integration.py` | Section 4.4: Table 2, Figures 4 and 5 |

## Installation

Python 3.10 or later.

```sh
python -m pip install -r requirements.txt
```

The Gaussian-core experiment requires a CUDA GPU. The other two experiments run
on the CPU.

## Reproducing the results

Run all commands from the repository root. Results go to `results/` by default;
use `--out` to change this.

**Gaussian core model (Table 1, Figures 1–2).**

```sh
python -m experiments.gaussian_core check   --device cuda:0   # oracle checks
python -m experiments.gaussian_core tune    --device cuda:0   # optional, see below
python -m experiments.gaussian_core run     --device cuda:0
python -m experiments.gaussian_core figures
```

WTR uses the same parameters for every (κ, N):
(Δ₀, Δ_max) = (2, 8), J_max = 3, ε_cg = 0.1, (η₁, η₂) = (0.1, 0.75),
(γ₁, γ₂) = (0.25, 2), θ = 0.8. For each (κ, N), the L-BFGS baseline is the best
of twelve variants on the initial configurations of seeds 101 and 103.
`tune` repeats this selection and writes `lbfgs-selection.json`. If that file
is absent, `run` uses the selection reported in the paper. `run` solves each of
the 24 test problems three times with each solver, one run at a time. It saves
every run with its force history and its initial and final configurations. The
paper's timings were measured in float64 on one NVIDIA RTX 4090 with
PyTorch 2.12.0 and CUDA 12.6.

**MMD fitting (Figure 3).**

```sh
python -m experiments.mmd_ablation run --workers 8
python -m experiments.mmd_ablation figure
```

The three variants run on three targets from eight initial configurations each
(1000 particles). `--workers` sets the number of parallel processes. This study
reports iteration counts and losses, not timings, so parallel runs are fine.

**Equal-weight integration (Table 2, Figures 4–5).**

```sh
python -m experiments.integration prepare   # target points and exact test integrals
python -m experiments.integration run
python -m experiments.integration figures
```

`run` times the 108 solves one at a time with single-threaded numerical
libraries. Absolute times depend on the machine. The paper's times were measured
on an Apple M3 with NumPy 2.2.5 and SciPy 1.15.3.

## Reproducibility notes

- The random initial configurations and target samples are generated from
  fixed seeds in the scripts.
- On the same platform, WTR and both L-BFGS variants reproduce their iterates
  exactly from run to run. SciPy's `trust-krylov` does not: repeated runs of the
  same problem can differ in the last digits and in the number of iterations.
- Banana samples are platform independent. Two-moons and ring samples use sine
  and cosine, so they can differ in the last bits across math libraries. On the
  MMD problems this changes no iteration count, and the losses agree to
  about 10⁻¹¹ relative.
