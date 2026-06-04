# Reproducibility repository for CUDA FDFD RF-cavity eigenmode calculations

This repository contains the CUDA/C++ numerical implementation and reproducibility material used for the RF-cavity eigenmode calculations reported in the associated manuscript. It documents the implementation details, benchmark configuration, reference output, and exported field data used in the study.

The numerical method is a 3D finite-difference frequency-domain (FDFD) eigenmode formulation for coaxial half-wave resonator (HWR) cavities. Curved internal boundaries are resolved with Dey–Mittra conformal subcell weighting on a structured Yee grid, and the lossy impedance-boundary-condition (IBC) eigenmode is obtained by a complex Rayleigh-quotient iteration with a GMRES inner solve. The implementation is CUDA-accelerated.

## Scope of this repository

This repository is provided as reproducibility material for the associated Computational Physics Paper. It contains the research implementation used to generate the benchmark eigenmode calculations, reference console output, and exported field data discussed in the manuscript. It is not presented as a standalone software package, general-purpose cavity-design tool, or CPC Program Library contribution; it is made available to support transparency of the numerical implementation and reproducibility of the reported results.

The version corresponding to the revised manuscript is tagged **`v1.0-cpc-revision`**, which identifies the implementation and reference output used for the CPC submission.

## Repository contents

- CUDA/C++ source files for the numerical implementation (`src/`, `include/`);
- benchmark drivers used in the manuscript (`apps/`);
- the grid-convergence diagnostic (`tests/`);
- reference console output for the reported benchmark case (`results/baseline/`);
- a description of the exported field data;
- build and reproduction instructions;
- implementation notes on the discretization, boundary-condition treatment, and eigensolver workflow.

## Numerical method

The wave equation ∇×∇×**E** = *k²***E** is discretized on the Yee staggered grid in cylindrical coordinates (*r*, *φ*, *z*). Curved boundaries are resolved with Dey–Mittra subcell weighting: edge vacuum-fractions and face vacuum-areas (analytic where possible, 8-point Gauss–Legendre where R varies along a face) are baked into the curl kernels, replacing the staircased *O*(d*r*) boundary error with an *O*(α) ≈ 10⁻⁴ term. The lossy eigenmode is found by complex Rayleigh-quotient iteration with GMRES inner solves, seeded by the converged real PEC eigenvector. The lossy eigenvalue uses a finite surface impedance *Z_s = R_s(1+j)*, and the unloaded quality factor is read directly from *Q₀ = −Re(k²)/Im(k²)*. The geometric *G*-factor and *R/Q* are obtained by post-processing the eigenvector with surface integrals on the same conformal weights used by the operator.

For the coaxial HWR cavity (optionally perturbed by radial beam ports and axial coupling ports), each calculation reports the resonant frequency *f*, the unloaded quality factor *Q₀*, the geometric *G*-factor, *R/Q* (per crossing, per pass, total), the power-dissipation breakdown by surface, and the eigenmode E- and H-field maps on the staggered Yee grid.

## Benchmark result

Canonical 10-port HWR benchmark case (a = 0.333 m, b = 1.0 m, L = 1.395 m, 20 radial beam ports, 2 axial coupling ports):

| Quantity | Analytical (unperturbed TEM HWR) | This implementation |
| --- | --- | --- |
| *f* (MHz) | 107.4525 | 107.4178 |
| *Q₀* | 48202 | 48126 |
| *R/Q* per crossing (Ω) | 83.88 | 81.15 |

## Performance information for the manuscript benchmark

| Item | Value |
| --- | --- |
| GPU (production) | NVIDIA RTX A5000 (24 GB) |
| GPU (verification) | NVIDIA RTX 4060 Mobile |
| CUDA toolkit | 12.x |
| Grid (Nr × Nφ × Nz) | 88 × 256 × 117 |
| Degrees of freedom | ≈ 8.0 M |
| GPU memory used | ≈ 6 GB |
| Eigensolver | complex Rayleigh-quotient iteration, GMRES(30) inner solve |
| Wall-clock | ≈ 10 min (RTX 4060), ≈ 4 min (RTX A5000) |
| Wall conductivity | σ = 5.8 × 10⁷ S/m (copper) |
| Result | *f* = 107.4178 MHz, *Q₀* = 48126, *R/Q* = 81.15 Ω per crossing |

For the tested CUDA environments, the reported scalar benchmark values are reproducible to the printed precision; wall-clock times vary across hardware; 
An empty diff indicates that the printed benchmark values match the reference output. If using a different CUDA version, GPU architecture, 
or compiler configuration, small last-digit differences may occur.

## Reproducing the benchmark calculation

To reproduce the benchmark calculation reported in the manuscript, build the CUDA/C++ implementation and run the supplied benchmark case.

Requirements:

- CUDA Toolkit ≥ 11.x
- NVIDIA GPU with compute capability ≥ 8.0
- GCC ≥ 7, GNU Make
- ≈ 6 GB of GPU memory for the benchmark grid

Build:

```
make
```

Override the compute capability if your GPU is not Ampere:

```
make CUDA_ARCH=sm_89
```

This builds three benchmark drivers in the repository root:

| Driver | Purpose |
| --- | --- |
| `solve_full_model` | Full benchmark solve (PEC + conformal IBC); exports field maps and computes *R/Q* |
| `manuscript_3stage` | Three-stage calculation (clean cavity → radial ports → full model) producing the manuscript table |
| `convergence_study` | Grid-convergence diagnostic with Richardson extrapolation |

Run the benchmark and compare against the reference output:

```
./solve_full_model 2>&1 | tee my_run.log
diff <(grep -E "f       =|Q       =|R/Q" results/baseline/console.log) \
     <(grep -E "f       =|Q       =|R/Q" my_run.log)
```

An empty diff indicates the reported benchmark numerics were reproduced exactly.

## Field data

The repository saves the computed field data and reference numerical outputs used for the manuscript. Plotting scripts are not included; the exported field arrays and tabulated reference results are provided so that the reported numerical values and field distributions can be independently inspected or post-processed. The reference console output for the benchmark case is in [`results/baseline/console.log`](results/baseline/console.log).

## Source layout

```
include/             Header files for the numerical implementation
src/                 Source files for the numerical implementation
apps/                Benchmark drivers
  solve_full_model.cpp
  manuscript_3stage.cpp
tests/               Grid-convergence diagnostic
  convergence_study.cpp
scripts/             Auxiliary checks or lightweight post-processing helpers, where applicable
results/baseline/    Reference output for the benchmark case
Makefile             Build entry point
```

The source files and benchmark drivers are kept separate so that the same numerical implementation is used consistently for all reported calculations.

## Citation

## Citation

If you use this repository, numerical implementation, reference outputs, or exported field data, please cite the associated manuscript. 
A citation entry for the tagged reproducibility version is provided through `CITATION.cff`.
When the manuscript is published, this section will be updated to reference the published article.

```
@misc{GK_cuda_fdfd_hwr_2026,
  author  = {Krishna, Gopi},
  title   = {CUDA FDFD numerical implementation and reproducibility material for RF-cavity eigenmode calculations},
  year    = {2026},
  url     = {https://github.com/gopikrishna911/CUDA_FDFD_HWR},
  version = {v1.0-cpc-revision},
  note    = {Tagged reproducibility repository for the associated manuscript}
}
```

When the manuscript is published, the citation will be updated to reference the published article.

## License

MIT. See [LICENSE](LICENSE).

## Author

**Gopi Krishna** — Indian Institute of Technology Roorkee. ORCID: [0009-0001-8511-8272](https://orcid.org/0009-0001-8511-8272).