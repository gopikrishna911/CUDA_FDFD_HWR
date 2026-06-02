#!/usr/bin/env bash
# run_production_compare.sh
#
# Runs BOTH the CPU reference and the GPU cross-check at production-scale
# grids, captures their TIMING + RESULT + MATVEC_CHECK, and prints a side-by-
# side comparison (matvec speedup, eigensolve speedup, f and Q agreement).
#
# This is the run that gives you EVERYTHING: CPU vs GPU performance AND the
# converged physical f/Q with the shift-updating RQI.
#
# Usage:
#   ./run_production_compare.sh 48 64 72        # grid sizes (Nz_cavity)
#   OMP_NUM_THREADS=32 ./run_production_compare.sh 72
#
# Each grid runs the CPU (use OMP_NUM_THREADS) then the GPU. Large grids on
# the CPU are slow — start small and watch the per-run ETA.

set -u
SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then SIZES=(48 64 72); fi

CPU=./test_conformal_cpu
GPU=./bench_gpu_conformal_xcheck
: "${OMP_NUM_THREADS:=1}"
export OMP_NUM_THREADS

[ -x "$CPU" ] || { echo "missing $CPU (make -f Makefile.conformal OMP=1 test_conformal_cpu)"; exit 1; }
[ -x "$GPU" ] || { echo "missing $GPU (make -f Makefile.conformal_xcheck bench_gpu_conformal_xcheck)"; exit 1; }

g() { grep -Eo "$1 = [-0-9.eE+]+" | head -1 | awk '{print $3}'; }

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
printf "%-6s %-9s | %-11s %-11s %-8s | %-11s %-11s %-8s | %-9s %-9s %-9s %-9s\n" \
  "grid" "DOFs" "mv_cpu_us" "mv_gpu_us" "mv_spd" "eig_cpu_s" "eig_gpu_s" "eig_spd" \
  "f_cpu" "f_gpu" "Q_cpu" "Q_gpu"
printf -- "---------------------------------------------------------------------------------------------------------------------------------\n"

for N in "${SIZES[@]}"; do
    co=$("$CPU" "$N" 2>/dev/null)
    go=$("$GPU" "$N" 2>/dev/null)

    dof=$(echo "$co" | grep -Eo 'DOFs=[0-9]+' | head -1 | cut -d= -f2)
    mvc=$(echo "$co" | grep 'TIMING: matvec' | grep -Eo '= [0-9.]+' | head -1 | awk '{print $2}')
    mvg=$(echo "$go" | grep 'TIMING: matvec' | grep -Eo '= [0-9.]+' | head -1 | awk '{print $2}')
    eic=$(echo "$co" | grep 'TIMING: eigensolve' | grep -Eo '= [0-9.]+' | head -1 | awk '{print $2}')
    eig=$(echo "$go" | grep 'TIMING: eigensolve' | grep -Eo '= [0-9.]+' | head -1 | awk '{print $2}')
    fc=$(echo "$co" | grep RESULT | g 'f'); fg=$(echo "$go" | grep RESULT | g 'f')
    qc=$(echo "$co" | grep RESULT | g 'Q'); qg=$(echo "$go" | grep RESULT | g 'Q')
    cc=$(echo "$co" | grep -Eo 'conv=[01]' | head -1); cg=$(echo "$go" | grep -Eo 'conv=[01]' | head -1)

    mvspd=$(awk -v c="${mvc:-0}" -v g="${mvg:-0}" 'BEGIN{print (g>0)?sprintf("%.1fx",c/g):"-"}')
    eispd=$(awk -v c="${eic:-0}" -v g="${eig:-0}" 'BEGIN{print (g>0)?sprintf("%.1fx",c/g):"-"}')

    printf "%-6s %-9s | %-11s %-11s %-8s | %-11s %-11s %-8s | %-9s %-9s %-9s %-9s\n" \
      "${N}" "${dof:-?}" "${mvc:-?}" "${mvg:-?}" "$mvspd" \
      "${eic:-?}" "${eig:-?}" "$eispd" "${fc:-?}" "${fg:-?}" "${qc:-?}" "${qg:-?}"
    # convergence flags on a second line if either didn't converge
    if [ "${cc:-conv=1}" != "conv=1" ] || [ "${cg:-conv=1}" != "conv=1" ]; then
        printf "       ^ convergence: CPU %s, GPU %s\n" "${cc:-?}" "${cg:-?}"
    fi
done
printf -- "---------------------------------------------------------------------------------------------------------------------------------\n"
echo "mv_spd/eig_spd = CPU/GPU time ratio (higher = GPU faster). f in MHz, Q dimensionless."
echo "If a 'convergence' line appears, that solve hit the iteration cap — treat its Q as not-yet-converged."
