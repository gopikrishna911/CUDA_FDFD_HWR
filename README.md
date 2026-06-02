# CUDA_FDFD_HWR

A 3D finite-difference frequency-domain (FDFD) eigensolver for coaxial half-wave resonator (HWR) cavities. Uses Dey–Mittra conformal subcell weighting to resolve curved boundaries on a structured Yee grid, and a complex Rayleigh-quotient iteration with GMRES inner solve for the lossy impedance-boundary-condition (IBC) eigenmode. Entirely CUDA-accelerated.

## What it computes

For a coaxial HWR cavity, optionally perturbed by radial beam ports and axial coupling ports, the solver returns:

- Resonant frequency *f* and unloaded quality factor *Q₀*
- Geometric *G*-factor and *R/Q* (per crossing, per pass, total)
- Power-dissipation breakdown by surface (inner conductor, outer conductor, endplates)
- Eigenmode E- and H-field maps on the staggered Yee grid

The lossy eigenvalue is solved directly with finite surface impedance *Z_s = R_s(1+j)*; *Q₀* is read from *−Re(k²)/Im(k²)*.

## Headline result

Canonical 10-port HWR test case (a=0.333 m, b=1.0 m, L=1.395 m, 20 radial beam ports, 2 axial coupling ports):

| Quantity | Analytical (unperturbed TEM HWR) | This code |
|---|---|---|
| *f* (MHz) | 107.4525 | 107.4178 |
| *Q₀* | 48202 | 48126 |
| *R/Q* per crossing (Ω) | 83.88 | 81.15 |

Wall-clock: ~10 minutes on an RTX 4060, ~4 minutes on an RTX A5000, for ~8.0 M DOFs.

Full reference output: [`results/baseline/console.log`](results/baseline/console.log).

## Build

Requirements:
- CUDA Toolkit ≥ 11.x
- NVIDIA GPU with compute capability ≥ 8.0
- GCC ≥ 7, GNU Make
- ~6 GB of GPU memory for the production grid

```sh
make
```

Override compute capability if your GPU is not Ampere:

```sh
make CUDA_ARCH=sm_89
```

Three binaries are built in the repo root:

| Binary | What it does |
|---|---|
| `solve_full_model` | Production solve: PEC + conformal IBC, exports field maps and computes *R/Q* |
| `manuscript_3stage` | Three-stage solve (clean cavity → radial ports → full model) producing the manuscript table |
| `convergence_study` | Grid-convergence study with Richardson extrapolation |

## Reproduce the baseline

```sh
./solve_full_model 2>&1 | tee my_run.log
diff <(grep -E "f       =|Q       =|R/Q" results/baseline/console.log) \
     <(grep -E "f       =|Q       =|R/Q" my_run.log)
```

Empty diff means you reproduced the baseline numerics exactly. Numerics are deterministic; only wall-clock times will vary across hardware.

## Repository layout

```
include/             Library headers
src/                 Library implementation (host .cpp + device .cu)
apps/                Production drivers
  solve_full_model.cpp
  manuscript_3stage.cpp
tests/               Diagnostic test
  convergence_study.cpp
scripts/             Post-processing helpers (Python)
results/baseline/    Canonical reference run for verification
Makefile             Single build entry point
```

Library and apps are kept separate so the same library backs all three binaries.

## Algorithm in one paragraph

The wave equation ∇×∇×**E** = *k²***E** is discretized on the Yee staggered grid in cylindrical coordinates (*r*, *φ*, *z*). Curved boundaries are resolved with Dey–Mittra subcell weighting: edge vacuum-fractions and face vacuum-areas (analytic where possible, 8-point Gauss–Legendre where R varies along a face) are baked into the curl kernels, replacing the staircased *O*(d*r*) boundary error with an *O*(α) ≈ 10⁻⁴ term. The lossy eigenmode is then found by complex Rayleigh-quotient iteration, with GMRES(30) inner solves, seeded by the converged real PEC eigenvector. *G* and *R/Q* are computed by post-processing the eigenvector with surface integrals on the same conformal weights used by the operator.

## Citation

```bibtex
@software{krishna_cuda_fdfd_hwr_2026,
  author       = {Krishna, Gopi},
  title        = {CUDA\_FDFD\_HWR: CUDA-accelerated FDFD eigensolver for coaxial half-wave resonator cavities},
  year         = 2026,
  url          = {https://github.com/<your-username>/CUDA_FDFD_HWR},
  orcid        = {0009-0001-8511-8272}
}
```

When the journal paper appears, replace this block with the published citation.

## License

MIT. See [LICENSE](LICENSE).

## Author

**Gopi Krishna** — Indian Institute of Technology Roorkee. ORCID: [0009-0001-8511-8272](https://orcid.org/0009-0001-8511-8272).