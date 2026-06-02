# CPU reference benchmark for Rhodotron FDFD eigensolver

This directory contains a CPU-only port of the GPU FDFD eigensolver,
written so that wall-clock numbers can be compared apples-to-apples
against the existing CUDA implementation. It builds two binaries from
the same source:

| Binary          | Compiled with | Use |
|-----------------|---------------|-----|
| `bench_serial`  | no `-fopenmp` | single-thread reference |
| `bench_omp`     | with `-fopenmp` | multi-thread (set via `OMP_NUM_THREADS` or `--threads`) |

The point is to measure the **same algorithm** running on CPU vs GPU,
not to produce a numerically identical run — parallel reductions on
either side change rounding, so equivalence is at the level of
*converged eigenvalue agreement to ~10 digits*, not bit-exactness.

## Files (Stage 1 — what's in this drop)

| File | Purpose |
|------|---------|
| `cpu_reference.h`   | Public API: real PEC + complex IBC declarations |
| `cpu_reference.cpp` | Real curl-E / curl-H / curl-curl / MINRES / real RQI (mirrors `cuda_curls.cu` + `cuda_eigensolver.cu`) |
| `cpu_complex.cpp`   | Complex IBC matvec + complex GMRES(m) + complex RQI (mirrors `gpu_curlcurl_matvec_complex` + `gpu_gmres_solve_shifted_complex_ws` + `gpu_rqi_complex_ws`) |
| `bench_solver.cpp`  | CLI harness: grid sweep, warm-up, timing, pretty-print + CSV |
| `Makefile.bench`    | Self-contained build, reuses `curl_E.cpp`, `curl_H.cpp`, `curlcurl_operator.cpp` from main project |

**Scope:** PEC eigenproblem on Yee grid (cylindrical) + perturbative
IBC via flat-wall surface correction. **Not yet ported:** conformal
Dey-Mittra subcell curls (`cuda_conformal_curls.cu`) and the conformal
pipe matvec (`cuda_conformal_pipe.cu`). Those are Stage 2 — let me know
if you want them next.

## Build

Place the four new files alongside the existing project sources (same
directory as `curl_E.cpp` etc.), then:

```bash
# Both binaries
make -f Makefile.bench all

# Just one
make -f Makefile.bench bench_serial
make -f Makefile.bench bench_omp

# Clean
make -f Makefile.bench clean
```

The Makefile uses `-O3 -march=native -funroll-loops` by default.
`-ffast-math` is **off** by default because it changes reduction order
enough to perturb converged k² at the 1e-10 level, which defeats the
GPU vs CPU comparison. Opt in with `FFAST=1`:

```bash
make -f Makefile.bench FFAST=1 all
```

## Usage

```
./bench_serial [options]
./bench_omp    [options]
```

| Flag                       | Default      | Meaning |
|----------------------------|--------------|---------|
| `--sizes=N1,N2,...`        | `16,24,32,48`| Grid sizes (cube, Nr=Nphi=Nz=N) |
| `--mode=pec|ibc|both`      | `both`       | Which solver path to run |
| `--matvec-iters=K`         | `200`        | Pure matvec timing iterations |
| `--max-rqi=K`              | `8`          | Outer RQI iteration cap |
| `--rqi-tol=X`              | `1e-6`       | RQI convergence tolerance |
| `--gmres-restart=M`        | `30`         | Complex GMRES(m) restart length (IBC only) |
| `--conductivity=σ`         | `5.8e7`      | Wall conductivity for IBC (default: Cu) |
| `--threads=N`              | (env)        | Override `OMP_NUM_THREADS` (OMP build only) |
| `--csv=path`               | (none)       | Append results to CSV (with header) |

### Examples

```bash
# Quick smoke test (small grid, both paths)
./bench_serial --sizes=16 --mode=both --matvec-iters=50 --max-rqi=4

# Single-thread reference, full sweep, save CSV
./bench_serial --sizes=16,24,32,48 --mode=both --csv=cpu_serial.csv

# 8-thread sweep
OMP_NUM_THREADS=8 ./bench_omp --sizes=16,24,32,48 --mode=both --csv=cpu_omp8.csv

# OMP scaling study (one grid, vary threads)
for T in 1 2 4 8 16; do
  OMP_NUM_THREADS=$T ./bench_omp --sizes=32 --mode=pec --csv=scaling.csv
done
```

## Interpreting output

### PEC (real) section

```
Grid       DOFs    Thr   Matvec [µs]  Eigensolve[s]  k²            Iter  f [MHz]    Conv
16x16x16   13328   1     316.107      3.480          5272.368838   6     3464.5252  no
```

- **DOFs** = `n_total` = `Nr·Nphi·(Nz+1) + (Nr+1)·Nphi·(Nz+1) + (Nr+1)·Nphi·Nz`
- **Matvec [µs]** = average over `--matvec-iters` warm-iterations, in
  microseconds. This is the single most important number for raw GPU
  vs CPU comparison: it isolates the curl-curl operator from the
  Krylov machinery.
- **Eigensolve[s]** = wall time for the full RQI loop.
- **k²** = converged eigenvalue (units 1/m²). `f = c·sqrt(k²)/(2π)`.
- **Conv** = `yes` if `r.converged == 1`, otherwise iteration cap was
  hit.

### IBC (complex) section

```
Grid       DOFs   Thr  Matvec [µs] Eigensolve[s] Re(k²)    Iter  f [MHz]  Q        Conv
16x16x16   13328  1    810.37      1.79          0.057269  4     11.418   40404.8  yes
```

- **Matvec [µs]** for IBC is roughly **2.5×** the PEC cost because the
  complex matvec is two real curl-curls (real + imag) plus the surface
  correction.
- **Q** = `−Re(k²) / Im(k²)`. Numbers near `σ_Cu`-derived
  surface-loss estimates are sane (typically 10⁴–10⁵ for copper).
- **Note:** the perturbative IBC path on a *naked* coax (no port
  masks, no Dey-Mittra subcell) will relax to a near-DC mode under
  pure RQI — exactly as on the GPU. This is fine for benchmarking
  matvec throughput and Krylov iteration cost; if you want the
  physical TM mode, port masks and conformal subcell are required
  (Stage 2).

## What's checked vs the GPU code

| Component                          | GPU file/function | CPU mirror |
|------------------------------------|-------------------|------------|
| Real curl-E                        | `cuda_curls.cu::compute_curl_E_kernel` | `cpu_compute_curl_E_omp` |
| Real curl-H                        | `cuda_curls.cu::compute_curl_H_kernel` | `cpu_compute_curl_H_omp` |
| Real curl-curl matvec (PEC)        | `cuda_operator.cu::gpu_curlcurl_matvec` | `cpu_curlcurl_matvec_omp` |
| Weighted dot ⟨·,·⟩_r               | `cuda_vector_ops.cu::weighted_dot_kernel` | `cpu_vec_dot_weighted_omp` |
| MINRES with Paige–Saunders rotation | `cuda_eigensolver.cu::gpu_minres_solve_shifted_ws` | `cpu_minres_shifted` |
| Real RQI                           | `cuda_eigensolver.cu::gpu_rqi_ws` | `cpu_rqi_minres_omp` |
| Complex matvec (IBC surface correction) | `cuda_operator.cu::gpu_curlcurl_matvec_complex` | `cpu_curlcurl_matvec_complex_omp` |
| Restarted complex GMRES(m)         | `cuda_eigensolver.cu::gpu_gmres_solve_shifted_complex_ws` | `cpu_gmres_solve_complex_omp` |
| Complex RQI                        | `cuda_eigensolver.cu::gpu_rqi_complex_ws` | `cpu_rqi_complex_omp` |
| IBC α and β coefficients           | `cuda_operator.cu::ibc_compute_alpha` | `cpu_ibc_compute_alpha` (file-static, identical formula) |

### Memory layout

Complex vectors are packed as `[real_part | imag_part]`, each
`n_total` long; within each half, fields are `[Er | Eφ | Ez]` in the
same order/index convention as the GPU `gpu_pack_field_complex` /
`gpu_unpack_field_complex`. This makes byte-for-byte cross-checking
possible if you ever want to copy a GPU vector to CPU memory and run
the CPU operator on it (or vice versa).

### Inner product convention

Both sides use a r-weighted, dV-scaled inner product:

```
⟨x,y⟩_r = Σ_(i,j,k)  r_{loc} · x · y · dV
```

with `r_{Er} = a + (i+½)·dr`, `r_{Eφ} = r_{Ez} = a + i·dr`, and
`dV = dr·dφ·dz`. The complex inner product is `Σ conj(x)·y`, so
the real and imaginary blocks combine as in `cpu_vec_dot_weighted_complex_omp`.

## Expected timings (rough, single-socket workstation)

These are order-of-magnitude on a recent x86 server with AVX2; your
mileage will vary with cache and memory bandwidth.

| Grid    | DOFs  | Matvec serial | Matvec OMP-8 | Matvec GPU (RTX-class) |
|---------|-------|---------------|--------------|------------------------|
| 16³     | 13K   | ~0.3 ms       | ~0.1 ms      | ~0.02 ms               |
| 32³     | 100K  | ~3 ms         | ~0.6 ms      | ~0.05 ms               |
| 48³     | 350K  | ~12 ms        | ~2 ms        | ~0.12 ms               |
| 64³     | 830K  | ~30 ms        | ~5 ms        | ~0.25 ms               |

OMP scaling is bandwidth-bound past ~4–8 threads on a single socket;
the matvec touches roughly 6× the field memory per call, so once you
saturate DDR you stop scaling. The GPU win grows with grid size.

**Note on tiny grids:** at 12³–16³ the per-call work is small enough
that OpenMP overhead can make the threaded build *slower* than serial.
Don't draw scaling conclusions from sub-millisecond matvec times.

## CSV columns

```
mode,Nr,Nphi,Nz,n_total,threads,
matvec_total_s,matvec_per_us,
eigensolve_s,rqi_iters,
k2_re,k2_im_or_freqMHz,freqMHz,Q,converged
```

For PEC rows, the `k2_im_or_freqMHz` column repeats `freqMHz` and `Q`
is zero; for IBC rows it carries Im(k²) and `Q` is the loaded Q.

## Known gaps / Stage 2 wishlist

1. **Conformal subcell curls.** GPU has `cuda_conformal_curls.cu`
   reading precomputed `edge_frac` and `face_area` arrays from
   `conformal_geometry.h`. Straight port to `cpu_conformal_curls.cpp`
   is mechanical but ~600 LOC and best done after Stage 1 is verified.
2. **Conformal + pipe matvec.** Mirror of `cuda_conformal_pipe.cu`
   including endplate grid-plane IBC and staircase-IBC fallback for
   ports.
3. **Port masks for the perturbative IBC path.** Currently the CPU IBC
   applies surface correction on all four flat walls; the GPU
   `build_port_masks` carves out beam-pipe apertures. Without those,
   converged Q includes spurious surface losses through the port
   holes. Easy ~50-LOC add when Stage 2 lands.

Let me know whether you want Stage 2 next, or whether you'd like the
existing Stage 1 verified against your GPU numbers first.
