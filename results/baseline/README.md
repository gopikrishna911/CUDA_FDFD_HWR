# Baseline reference run

Canonical output of `./solve_full_model` for the 10-port HWR test case.

**Hardware:** NVIDIA RTX 4060 Mobile (8 GB, sm_89)
**CUDA:** 12.x
**Date:** June 2026
**Grid:** Nr=88, Nφ=256, Nz=117 (8.0 M DOFs)
**Wall clock:** ~10 minutes

## Headline numbers

| Quantity | Value |
|---|---|
| Frequency, *f* | 107.4178 MHz |
| Quality factor, *Q₀* | 48126 |
| *R/Q* per crossing (linac) | 81.15 Ω |
| *G* factor | 130.42 Ω |

Analytical (unperturbed TEM HWR) reference: *f* = 107.4525 MHz, *Q₀* = 48202.

## How to reproduce

```sh
./solve_full_model 2>&1 | tee my_run.log
diff <(grep -E "f       =|Q       =|R/Q" console.log) \
     <(grep -E "f       =|Q       =|R/Q" my_run.log)
```

Empty diff = reproduced. Numerics are deterministic; only wall-clock times vary.