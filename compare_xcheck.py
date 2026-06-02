#!/usr/bin/env python3
"""
compare_xcheck.py — cross-check CPU reference vs GPU.

Reads the two CSVs produced by bench_serial (or bench_omp) and
bench_gpu_xcheck, joins rows by (mode, grid), and reports:
  - relative difference in converged k2_re   (the correctness check)
  - frequency / Q comparison
  - matvec and eigensolve speedup (CPU_time / GPU_time)

Usage:
  python3 compare_xcheck.py cpu_serial.csv gpu_xcheck.csv
  python3 compare_xcheck.py cpu_omp8.csv  gpu_xcheck.csv --tol 1e-9
"""
import csv, sys, argparse

def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["mode"], int(r["Nr"]), int(r["Nphi"]), int(r["Nz"]))
            rows[key] = r
    return rows

def getf(r, *names):
    """First present, non-empty column among names -> float, else None."""
    for n in names:
        if n in r and r[n] not in (None, ""):
            try: return float(r[n])
            except ValueError: pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpu_csv")
    ap.add_argument("gpu_csv")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="rel-diff threshold for PASS on k2_re (default 1e-9)")
    args = ap.parse_args()

    cpu = load(args.cpu_csv)
    gpu = load(args.gpu_csv)
    keys = sorted(set(cpu) & set(gpu), key=lambda k: (k[0], k[1]))
    if not keys:
        print("No matching (mode,grid) rows between the two CSVs.")
        print("  CPU keys:", sorted(cpu))
        print("  GPU keys:", sorted(gpu))
        sys.exit(1)

    print("="*92)
    print(f"  Cross-check: {args.cpu_csv}  vs  {args.gpu_csv}")
    print(f"  PASS if |k2_cpu - k2_gpu| / |k2_cpu| < {args.tol:g}")
    print("="*92)
    hdr = (f"{'mode':4} {'grid':10} {'k2_cpu':>16} {'k2_gpu':>16} "
           f"{'rel.diff':>11} {'mv x':>7} {'eig x':>7} {'':>5}")
    print(hdr); print("-"*len(hdr))

    worst = 0.0
    n_fail = 0
    for k in keys:
        rc, rg = cpu[k], gpu[k]
        k2c = getf(rc, "k2_re")
        k2g = getf(rg, "k2_re")
        if k2c is None or k2g is None:
            continue
        rel = abs(k2c - k2g) / abs(k2c) if k2c != 0 else abs(k2g)
        worst = max(worst, rel)
        verdict = "PASS" if rel < args.tol else "FAIL"
        if verdict == "FAIL": n_fail += 1

        mv_c  = getf(rc, "matvec_per_us")
        mv_g  = getf(rg, "matvec_per_us")
        eig_c = getf(rc, "eigensolve_s")
        eig_g = getf(rg, "eigensolve_s")
        mv_sp  = (mv_c / mv_g)   if (mv_c and mv_g)   else float('nan')
        eig_sp = (eig_c / eig_g) if (eig_c and eig_g and eig_g > 0) else float('nan')

        grid = f"{k[1]}x{k[2]}x{k[3]}"
        print(f"{k[0]:4} {grid:10} {k2c:16.10f} {k2g:16.10f} "
              f"{rel:11.2e} {mv_sp:7.1f} {eig_sp:7.1f}  {verdict}")

    print("-"*len(hdr))
    print(f"  worst rel.diff in k2_re : {worst:.2e}")
    print(f"  result                  : "
          f"{'ALL PASS' if n_fail==0 else str(n_fail)+' FAILED'}")
    print()
    print("  mv x  = CPU matvec time / GPU matvec time   (raw operator throughput)")
    print("  eig x = CPU eigensolve  / GPU eigensolve     (full solve, Krylov-bound)")
    print()
    print("  Note: rel.diff up to ~1e-8 is normal — GPU and CPU/OMP sum their")
    print("  reductions in different orders, so the converged k2 differs in the")
    print("  last few digits even though the algorithm is identical. A rel.diff")
    print("  of 1e-3 or worse indicates a genuine operator/setup mismatch.")

if __name__ == "__main__":
    main()
