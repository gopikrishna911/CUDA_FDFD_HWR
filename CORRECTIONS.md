# Corrections applied (GPU bug fix + endplate physics)

This bundle fixes three linked issues found by the conformal cross-check.

## 1. GPU out-of-bounds bug — `cuda_conformal_pipe.cu`

`rezero_endplate_surface` computed the conductor-side surface layer as
`ks = k_zL + 1` and wrote to plane `ks` via `pec_Er_at_k_kernel` /
`pec_Ephi_at_k_kernel`. Those kernels only bounds-check the in-plane
index (`gid < Nr*Nphi`), never the `k` plane. So when `k_zL == Nz`,
the GPU wrote to `k = Nz+1` — one full plane past the end of the
Er/Ephi arrays, silently corrupting adjacent device memory.

**Fix (minimal, 5-line diff):**
- added `int Nz` to the `rezero_endplate_surface` signature,
- guarded the zL branch with `&& k_zL + 1 <= Nz`,
- passed `Nz` at the two call sites.

In production (`test_conformal_ibc_full`) the endplates are interior
(`k_zL + 1 <= Nz` already holds), so the bug was dormant there — but it
is one refactor away from biting, and it *was* corrupting the
radial-only cross-check. Drop the file in `gpu_fix/` over your
`cuda_conformal_pipe.cu` and rebuild.

A defensive belt-and-suspenders alternative (not applied here, but
recommended) is to also add `if (k < 0 || k > Nz) return;` to
`pec_Er_at_k_kernel` / `pec_Ephi_at_k_kernel` in `cuda_pipe_model.cu`,
so any future bad endplate index fails safe instead of corrupting memory.

## 2. Endplate physics — Q restored to the right ballpark (was ~86k)

The earlier radial-only test set `k_z0 = -1, k_zL = -1` to dodge the GPU
OOB. That **disabled the endplate grid-plane IBC entirely**. The two
endplates carry ~44% of the cavity wall loss, so disabling them raised Q
by ~1.79x — turning a physical ~48-59k into the bogus ~86,000 we saw.

**Fix:** both `test_conformal_cpu.cpp` and `bench_gpu_conformal_xcheck.cu`
now use a **z-extended grid with interior endplates**:
- grid built with `grid_init_with_all_pipes` (small z-extension each end),
- endplates at interior planes `k_z0 = Nz_pipe_z0`,
  `k_zL = Nz_pipe_z0 + Nz_cavity` (so `k_z0-1 >= 0` and `k_zL+1 <= Nz`),
- an empty `EndcapPipeConfig` with nonzero `z0_extension`/`zL_extension`
  makes the regions beyond the endplates fully PEC (solid endplates, no
  apertures), via `material_mask_build_full`.

With endplates restored, Q drops from the bogus ~86k into the physical
range (48-59k across the coarse benchmark grids), with CPU and GPU
agreeing on each grid.

### IMPORTANT caveat on Q convergence

Q does **not** cleanly converge to the analytical 48,200 on these coarse
benchmark grids (Nz <= 40). Across a grid sweep it wanders ~48-59k,
tracking the frequency scatter. This is NOT a bug — it is because the
12.5 mm beam pipes are badly under-resolved on ~50 mm cells, so the
conformal cut cells (and the endplate surface discretization) change
non-smoothly with grid. Clean Q convergence needs production-scale grids
(Nr_cavity ~ 81, Nphi ~ 256), not these benchmark sizes. The frequency
is robust to this; Q (which depends on the small imaginary eigenvalue
k2_im ~ 1e-4) is far more sensitive.

### Q is also inner-GMRES-tolerance sensitive

Because Q = -k2_re / k2_im and k2_im is tiny (~1e-4), the loose inner
GMRES tolerance (1e-1, the production default in gpu_rqi_complex_ws)
leaves k2_im imprecise. At 24^3 this produced a ~12% CPU/GPU Q split
(54,340 vs 48,362) even though k2_re agreed to 2e-6 — the two loose
solves took slightly different paths and landed on different k2_im.
Tightening the inner tolerance to 1e-4 closes that split (both -> 48,360),
confirming it as solver-path divergence, not an operator difference.

We have now APPLIED the tightened tolerance (1e-4) to BOTH files in this
bundle:
  - cpu_conformal.cpp ~line 858:   `5e-1 : 1e-1` -> `1e-1 : 1e-4`
  - cuda_eigensolver.cu ~line 1111: `5e-1 : 1e-1` -> `1e-1 : 1e-4`
This makes CPU and GPU agree on Q at a given grid (verified: at Nz_cav=24
both give Q ~= 48,360, k2_im ~= -1.047e-4). It does NOT fix the
cross-GRID Q variation — that is a separate resolution+geometry effect
that needs Phase B (real endcap pipes + inner ports) at production
resolution. Apply cuda_eigensolver.cu from gpu_fix/ along with
cuda_conformal_pipe.cu.

## 3. Geometry — production pipe parameters

The earlier test used oversized toy pipes (100 mm radius). Both harnesses
now use the production geometry: 12.5 mm pipe radius, 17.5 mm aperture,
50 mm extension, 10 beam passes — matching `test_conformal_ibc_full`.

## Rebuild & run

```bash
# apply the GPU fix
cp gpu_fix/cuda_conformal_pipe.cu .

# rebuild both sides
make -f Makefile.conformal        clean && make -f Makefile.conformal        test_conformal_cpu
make -f Makefile.conformal_xcheck clean && make -f Makefile.conformal_xcheck bench_gpu_conformal_xcheck

# cross-check (operator + eigensolve)
./compare_conformal_xcheck.sh 16 20 24
```

Expect: the single-matvec checksums agree to ~1e-12 or exactly (operator
identical), the eigensolve frequency near 107 MHz, and Q ~ 48k on both
sides, trending to 48,200 as the grid refines.

## Note on the argument-order inconsistency in your tree

While building the harness I noticed `test_conformal_ibc_full.cpp`
(line ~601) calls `gpu_conformal_pipe_operator_init` with what looks like
a different argument pattern than the 9-parameter signature declared in
`cuda_conformal_pipe.h`. My harness follows the **header** signature
(`..., int k_z0, int k_zL, const EndcapPipeConfig*, const GridParams*,
double z0_extension`). If your production test currently compiles against
a different signature, double-check which one is live before applying the
fix — the `Nz` parameter I added is positional and assumes the header
signature.
