#!/usr/bin/env bash
# compare_conformal_xcheck.sh
#
# Runs the CPU (test_conformal_cpu) and GPU (bench_gpu_conformal_xcheck)
# conformal solvers at one or more grid sizes and reports the relative
# difference in converged k2_re. A rel.diff around 1e-10..1e-8 is the
# expected PASS (parallel-reduction order); 1e-3 or worse is a real
# operator mismatch.
#
# Usage:
#   ./compare_conformal_xcheck.sh 24 32 40
#   ./compare_conformal_xcheck.sh           # defaults to 24 32

set -u
SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then SIZES=(24 32); fi

CPU=./test_conformal_cpu
GPU=./bench_gpu_conformal_xcheck

if [ ! -x "$CPU" ]; then echo "missing $CPU (make -f Makefile.conformal)"; exit 1; fi
if [ ! -x "$GPU" ]; then echo "missing $GPU (make -f Makefile.conformal_xcheck)"; exit 1; fi

# extract "k2_re = <number>" from a RESULT line
extract_k2() { grep -Eo 'k2_re = [-0-9.eE+]+' | head -1 | awk '{print $3}'; }
extract_f()  { grep -Eo 'f = [-0-9.eE+]+'     | head -1 | awk '{print $3}'; }
extract_Q()  { grep -Eo 'Q = [-0-9.eE+]+'     | head -1 | awk '{print $3}'; }
# extract the matvec checksum ||Ax|| and <x,Ax> from the MATVEC_CHECK line
extract_mvnorm() { grep MATVEC_CHECK | grep -Eo '\|\|Ax\|\| = [-0-9.eE+]+' | head -1 | awk '{print $3}'; }
extract_mvdot()  { grep MATVEC_CHECK | grep -Eo '<x,Ax> = [-0-9.eE+]+'     | head -1 | awk '{print $3}'; }

printf "SINGLE-MATVEC operator check (decisive: removes the solver path):\n"
printf "%-8s %-22s %-22s %-12s %s\n" "grid" "<x,Ax>_cpu" "<x,Ax>_gpu" "rel.diff" "verdict"
printf -- "----------------------------------------------------------------------------------------------\n"
mv_fail=0
for N in "${SIZES[@]}"; do
    co=$("$CPU" "$N" 2>/dev/null)
    go=$("$GPU" "$N" 2>/dev/null)
    dc=$(echo "$co" | extract_mvdot); dg=$(echo "$go" | extract_mvdot)
    if [ -z "$dc" ] || [ -z "$dg" ]; then
        printf "%-8s  (no MATVEC_CHECK line)\n" "${N}^3"; mv_fail=1; continue
    fi
    rel=$(awk -v a="$dc" -v b="$dg" 'BEGIN{ d=(a-b); if(d<0)d=-d; r=(a!=0)?d/(a<0?-a:a):d; printf "%.3e", r }')
    verdict=$(awk -v r="$rel" 'BEGIN{ print (r<1e-9)?"PASS":"CHECK" }')
    [ "$verdict" = "CHECK" ] && mv_fail=1
    printf "%-8s %-22s %-22s %-12s %s\n" "${N}^3" "$dc" "$dg" "$rel" "$verdict"
done
printf -- "----------------------------------------------------------------------------------------------\n"
if [ $mv_fail -eq 0 ]; then
    echo "  -> single matvec agrees to <1e-9 on all grids: the OPERATOR is identical."
    echo "     Any eigenvalue spread below is solver-path divergence, not an operator bug."
else
    echo "  -> single matvec differs: the discrepancy is in the OPERATOR itself."
fi
echo

printf "FULL EIGENSOLVE (includes iterative solver path):\n"
printf "%-7s %-15s %-15s %-10s %-9s %-9s %-9s %-9s %s\n" \
       "grid" "k2_cpu" "k2_gpu" "rel.diff" "f_cpu" "f_gpu" "Q_cpu" "Q_gpu" "verdict"
printf -- "--------------------------------------------------------------------------------------------------------\n"

worst=0
fail=0
for N in "${SIZES[@]}"; do
    co=$("$CPU" "$N" 2>/dev/null | grep RESULT)
    go=$("$GPU" "$N" 2>/dev/null | grep RESULT)
    k2c=$(echo "$co" | extract_k2); k2g=$(echo "$go" | extract_k2)
    fc=$(echo  "$co" | extract_f);  fg=$(echo  "$go" | extract_f)
    qc=$(echo  "$co" | extract_Q);  qg=$(echo  "$go" | extract_Q)
    if [ -z "$k2c" ] || [ -z "$k2g" ]; then
        printf "%-8s  (no RESULT line — CPU='%s' GPU='%s')\n" "${N}^3" "$co" "$go"
        fail=1; continue
    fi
    rel=$(awk -v a="$k2c" -v b="$k2g" 'BEGIN{ d=(a-b); if(d<0)d=-d; r=(a!=0)?d/(a<0?-a:a):d; printf "%.3e", r }')
    # Eigensolve threshold reflects the INNER GMRES tolerance (loose, ~1e-1),
    # not machine epsilon: the converged eigenvector is approximate, so CPU/GPU
    # land within ~1e-5 of each other from solver-path divergence. That is a
    # PASS. Only >1e-3 indicates a genuine operator/setup mismatch.
    verdict=$(awk -v r="$rel" 'BEGIN{ print (r<1e-5)?"PASS":((r<1e-3)?"CLOSE":"CHECK") }')
    [ "$verdict" = "CHECK" ] && fail=1
    worst=$(awk -v w="$worst" -v r="$rel" 'BEGIN{ print (r>w)?r:w }')
    printf "%-7s %-15s %-15s %-10s %-9s %-9s %-9s %-9s %s\n" \
           "${N}^3" "$k2c" "$k2g" "$rel" "$fc" "$fg" "$qc" "$qg" "$verdict"
done

printf -- "----------------------------------------------------------------------------------------------\n"
printf "worst rel.diff in k2_re: %s   ->  %s\n" "$worst" \
       "$([ $fail -eq 0 ] && echo ALL PASS || echo 'CHECK (see rows above)')"
echo
echo "Verdicts: single-matvec PASS (<1e-9) certifies the OPERATOR is identical."
echo "Eigensolve PASS (<1e-5) = solver-path divergence at the inner-GMRES tol level"
echo "(benign). CLOSE (1e-5..1e-3) = looser but still consistent. CHECK (>1e-3) ="
echo "a genuine operator/setup mismatch worth investigating."
