/*=============================================================================
 * test_conformal_cuda.cpp
 *
 * CUDA tests for the Dey-Mittra conformal curl kernels (Phase 2).
 *
 * TEST A: Reference Cavity (no pipes)
 *   Conformal curl-curl on a grid with all fracs=1, areas=standard
 *   must produce IDENTICAL output to the standard curl-curl.
 *
 * TEST B: Pipe Model — Matvec Comparison
 *   Conformal vs staircase on the 20-pipe model. Both should give
 *   reasonable Rayleigh quotients and produce no NaN/Inf.
 *
 * TEST C: Conformal Mask vs Material Mask
 *   Both should agree at non-boundary cells (deep vacuum and deep PEC).
 *
 * Compile (adjust to your Makefile structure):
 *   nvcc -O2 -o test_conformal_cuda test_conformal_cuda.cpp \
 *        cuda_conformal_curls_cu.cpp conformal_geometry.c \
 *        cuda_curls_cu.cpp cuda_fields_cu.cpp cuda_operator_cu.cpp \
 *        cuda_vector_ops_cu.cpp cuda_pipe_model_cu.cpp \
 *        cuda_eigensolver_cu.cpp \
 *        curlcurl_operator.cpp curl_E.cpp curl_H.cpp pipe_model.cpp \
 *        -lm
 *============================================================================*/

#include "cuda_conformal.h"
#include "cuda_pipe_model.h"
#include "cuda_eigensolver.h"
#include "conformal_geometry.h"
#include "pipe_model.h"
#include "curlcurl_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define C0  299792458.0
#define MU0 (4.0e-7 * M_PI)

#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int g_pass = 0, g_fail = 0;

static void check(const char* name, int condition) {
    printf("    %-55s %s\n", name, condition ? PASS : FAIL);
    if (condition) g_pass++; else g_fail++;
}

static void checkf(const char* name, double got, double expected, double reltol) {
    double err = (fabs(expected) > 1e-30) ?
        fabs(got - expected) / fabs(expected) : fabs(got);
    char buf[256];
    snprintf(buf, sizeof(buf), "%s: %.6e vs %.6e (err %.2e)",
             name, got, expected, err);
    check(buf, err < reltol);
}

/*=============================================================================
 * Helper: max |a-b| / max(|a|,|b|) between two GPU arrays
 *============================================================================*/
static double gpu_max_reldiff(const double* d_a, const double* d_b, int n) {
    double* h_a = (double*)malloc(n * sizeof(double));
    double* h_b = (double*)malloc(n * sizeof(double));
    cudaMemcpy(h_a, d_a, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, n * sizeof(double), cudaMemcpyDeviceToHost);

    double max_diff = 0.0, max_val = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = fabs(h_a[i] - h_b[i]);
        double val = fmax(fabs(h_a[i]), fabs(h_b[i]));
        if (diff > max_diff) max_diff = diff;
        if (val > max_val) max_val = val;
    }
    free(h_a);
    free(h_b);
    return (max_val > 1e-30) ? max_diff / max_val : max_diff;
}

/*=============================================================================
 * Helper: fill TEM initial guess Er = sin(πz/L)/r on host
 *============================================================================*/
static void fill_tem(double* h_x, const CurlCurlOperator* op,
                     const GridParams* grid, double b_cav, double L_cav)
{
    int n = op->n_total;
    memset(h_x, 0, n * sizeof(double));
    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                double r = grid->a + (i + 0.5) * grid->dr;
                if (r <= b_cav && z >= 0.0 && z <= L_cav) {
                    int idx = op->offset_Er + idx_Er(grid, i, j, k);
                    h_x[idx] = sin(M_PI * z / L_cav) / r;
                }
            }
        }
    }
}

/*=============================================================================
 * TEST A: Reference Cavity — conformal MUST equal standard
 *============================================================================*/
static void test_A_reference_cavity(void) {
    printf("\n  TEST A: Reference Cavity — Conformal vs Standard Curl-Curl\n");
    printf("  ===========================================================\n");

    double a = 0.3333, b = 1.0, L = 1.395;
    int Nr = 60, Nphi = 64, Nz = 60;

    /* Setup grid and CPU operator */
    GridParams grid;
    grid_init(&grid, a, b, L, Nr, Nphi, Nz);
    printf("    Grid: Nr=%d, Nphi=%d, Nz=%d, dr=%.2f mm\n",
           Nr, Nphi, Nz, grid.dr * 1e3);

    CurlCurlOperator cpu_op;
    curlcurl_op_init(&cpu_op, &grid);
    int n = cpu_op.n_total;
    printf("    DOFs: %d\n", n);

    cuda_grid_init(&grid);

    /* Build conformal data with 0 pipes → all fracs=1, areas=standard */
    PipeConfig no_pipes;
    pipe_config_init(&no_pipes, a, b, 0.0125, 0.0175, 0.050, 0.0);
    /* num_pipes stays 0 */

    ConformalData cd;
    conformal_data_build(&cd, &no_pipes, &grid, b, 0.0);
    int total_cut = cd.num_cut_Er + cd.num_cut_Ephi + cd.num_cut_Ez;
    printf("    Conformal: %d cut edges (expect 0)\n", total_cut);
    check("No cut edges on reference cavity", total_cut == 0);

    /* Upload to GPU */
    GPU_ConformalData gpu_cd;
    gpu_conformal_data_init(&gpu_cd, &cd);

    /* Allocate GPU fields */
    GPU_EField d_E, d_result_std, d_result_cfm;
    GPU_HField d_H_std, d_H_cfm;
    gpu_efield_alloc(&d_E, &grid);
    gpu_efield_alloc(&d_result_std, &grid);
    gpu_efield_alloc(&d_result_cfm, &grid);
    gpu_hfield_alloc(&d_H_std, &grid);
    gpu_hfield_alloc(&d_H_cfm, &grid);

    /* Fill TEM guess on host, upload to GPU packed vector, unpack to E-field */
    double* h_x = (double*)calloc(n, sizeof(double));
    fill_tem(h_x, &cpu_op, &grid, b, L);

    double* d_x;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_to_device(d_x, h_x, n);
    gpu_unpack_field(d_x, &d_E,
        cpu_op.offset_Er, cpu_op.offset_Ephi, cpu_op.offset_Ez);

    /* Apply PEC boundary */
    gpu_apply_PEC_boundary(&d_E, &grid);

    /* Standard curl-curl */
    gpu_compute_curl_curl_E(&d_E, &d_result_std, &d_H_std, &grid);

    /* Conformal curl-curl */
    gpu_compute_curl_curl_E_conformal(&d_E, &d_result_cfm, &d_H_cfm,
                                      &grid, &gpu_cd);

    /* Compare all 3 components */
    /* Er has real signal; Ephi and Ez are ~0 for TEM mode so use absolute diff */
    double diff_Er = gpu_max_reldiff(d_result_std.Er, d_result_cfm.Er,
        d_E.size_Er);

    /* For Ephi/Ez: compute max absolute diff, compare to max |Er| */
    double max_abs_Er = 0.0;
    {
        double* h_tmp = (double*)malloc(d_E.size_Er * sizeof(double));
        cudaMemcpy(h_tmp, d_result_std.Er, d_E.size_Er * sizeof(double),
            cudaMemcpyDeviceToHost);
        for (int ii = 0; ii < d_E.size_Er; ii++)
            if (fabs(h_tmp[ii]) > max_abs_Er) max_abs_Er = fabs(h_tmp[ii]);
        free(h_tmp);
    }

    double diff_Ep_abs = 0.0, diff_Ez_abs = 0.0;
    {
        double* ha = (double*)malloc(d_E.size_Ephi * sizeof(double));
        double* hb = (double*)malloc(d_E.size_Ephi * sizeof(double));
        cudaMemcpy(ha, d_result_std.Ephi, d_E.size_Ephi * sizeof(double),
            cudaMemcpyDeviceToHost);
        cudaMemcpy(hb, d_result_cfm.Ephi, d_E.size_Ephi * sizeof(double),
            cudaMemcpyDeviceToHost);
        for (int ii = 0; ii < d_E.size_Ephi; ii++) {
            double d = fabs(ha[ii] - hb[ii]);
            if (d > diff_Ep_abs) diff_Ep_abs = d;
        }
        free(ha); free(hb);
    }
    {
        double* ha = (double*)malloc(d_E.size_Ez * sizeof(double));
        double* hb = (double*)malloc(d_E.size_Ez * sizeof(double));
        cudaMemcpy(ha, d_result_std.Ez, d_E.size_Ez * sizeof(double),
            cudaMemcpyDeviceToHost);
        cudaMemcpy(hb, d_result_cfm.Ez, d_E.size_Ez * sizeof(double),
            cudaMemcpyDeviceToHost);
        for (int ii = 0; ii < d_E.size_Ez; ii++) {
            double d = fabs(ha[ii] - hb[ii]);
            if (d > diff_Ez_abs) diff_Ez_abs = d;
        }
        free(ha); free(hb);
    }

    double diff_Ep_rel = (max_abs_Er > 1e-30) ? diff_Ep_abs / max_abs_Er : diff_Ep_abs;
    double diff_Ez_rel = (max_abs_Er > 1e-30) ? diff_Ez_abs / max_abs_Er : diff_Ez_abs;

    printf("    Max relative diff:  Er=%.2e\n", diff_Er);
    printf("    Max abs diff / |Er|: Ephi=%.2e, Ez=%.2e\n", diff_Ep_rel, diff_Ez_rel);

    check("Er:   conformal == standard (reldiff < 1e-10)", diff_Er < 1e-10);
    check("Ephi: conformal == standard (absdiff/|Er| < 1e-10)", diff_Ep_rel < 1e-10);
    check("Ez:   conformal == standard (absdiff/|Er| < 1e-10)", diff_Ez_rel < 1e-10);

    /* Compare Rayleigh quotients */
    double* d_y_std, *d_y_cfm;
    gpu_vector_alloc(&d_y_std, n);
    gpu_vector_alloc(&d_y_cfm, n);

    gpu_pack_field(&d_result_std, d_y_std,
        cpu_op.offset_Er, cpu_op.offset_Ephi, cpu_op.offset_Ez);
    gpu_pack_field(&d_result_cfm, d_y_cfm,
        cpu_op.offset_Er, cpu_op.offset_Ephi, cpu_op.offset_Ez);

    ReductionWorkspace rws;
    reduction_workspace_init(&rws, n);

    double xAx_std, xAx_cfm, xx;
    gpu_vec_dot_weighted_ws(d_x, d_y_std, &xAx_std, &cpu_op, &rws);
    gpu_vec_dot_weighted_ws(d_x, d_y_cfm, &xAx_cfm, &cpu_op, &rws);
    gpu_vec_dot_weighted_ws(d_x, d_x,     &xx,      &cpu_op, &rws);

    double rq_std = xAx_std / xx;
    double rq_cfm = xAx_cfm / xx;
    double k2_analytical = (M_PI / L) * (M_PI / L);

    printf("    Rayleigh quotient: std=%.8f, cfm=%.8f, analytical=%.8f\n",
           rq_std, rq_cfm, k2_analytical);
    checkf("RQ: conformal == standard", rq_cfm, rq_std, 1e-10);

    /* Cleanup */
    reduction_workspace_free(&rws);
    gpu_vector_free(d_x);
    gpu_vector_free(d_y_std);
    gpu_vector_free(d_y_cfm);
    gpu_efield_free(&d_E);
    gpu_efield_free(&d_result_std);
    gpu_efield_free(&d_result_cfm);
    gpu_hfield_free(&d_H_std);
    gpu_hfield_free(&d_H_cfm);
    gpu_conformal_data_free(&gpu_cd);
    conformal_data_free(&cd);
    pipe_config_free(&no_pipes);
    curlcurl_op_free(&cpu_op);
    free(h_x);
}

/*=============================================================================
 * TEST B: Pipe Model — Conformal vs Staircase Matvec
 *============================================================================*/
static void test_B_pipe_model(void) {
    printf("\n  TEST B: Pipe Model — Conformal vs Staircase Matvec\n");
    printf("  ====================================================\n");

    double a = 0.3333, b = 1.0, L = 1.395;
    double R_pipe = 0.0125, R_aper = 0.0175, pipe_len = 0.050;
    int num_passes = 10;

    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, R_pipe, R_aper, pipe_len, 0.0);
    pipe_config_add_multi_pass(&pipes, L / 2.0, num_passes);

    int Nr_cav = 81, Nr_p = 7, Nphi = 256, Nz = 84;
    GridParams grid;
    grid_init_with_pipes(&grid, a, b, L, pipe_len, Nr_cav, Nr_p, Nphi, Nz);
    printf("    Grid: Nr=%d, Nphi=%d, Nz=%d\n", grid.Nr, Nphi, Nz);

    MaterialMask mask;
    material_mask_build(&mask, &pipes, &grid);

    ConformalData cd;
    conformal_data_build(&cd, &pipes, &grid, b, 0.0);
    conformal_data_print_stats(&cd, &grid);

    CurlCurlOperator cpu_op;
    curlcurl_op_init(&cpu_op, &grid);
    int n = cpu_op.n_total;
    printf("    DOFs: %d (%.1f M)\n", n, n / 1e6);

    cuda_grid_init(&grid);

    GPU_ConformalData gpu_cd;
    gpu_conformal_data_init(&gpu_cd, &cd);

    GPU_MaterialMask gpu_mask;
    gpu_material_mask_init(&gpu_mask, &mask);

    /* GPU fields — standard path */
    GPU_EField d_E_std, d_result_std;
    GPU_HField d_H_std;
    gpu_efield_alloc(&d_E_std, &grid);
    gpu_efield_alloc(&d_result_std, &grid);
    gpu_hfield_alloc(&d_H_std, &grid);

    /* GPU fields — conformal path */
    GPU_EField d_E_cfm, d_result_cfm;
    GPU_HField d_H_cfm;
    gpu_efield_alloc(&d_E_cfm, &grid);
    gpu_efield_alloc(&d_result_cfm, &grid);
    gpu_hfield_alloc(&d_H_cfm, &grid);

    /* TEM guess */
    double* h_x = (double*)calloc(n, sizeof(double));
    fill_tem(h_x, &cpu_op, &grid, b, L);

    double* d_x;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_to_device(d_x, h_x, n);

    /* --- Staircase path --- */
    gpu_unpack_field(d_x, &d_E_std,
        cpu_op.offset_Er, cpu_op.offset_Ephi, cpu_op.offset_Ez);
    gpu_apply_PEC_boundary(&d_E_std, &grid);
    gpu_apply_material_mask(&d_E_std, &gpu_mask);
    gpu_compute_curl_curl_E(&d_E_std, &d_result_std, &d_H_std, &grid);
    gpu_apply_PEC_boundary(&d_result_std, &grid);
    gpu_apply_material_mask(&d_result_std, &gpu_mask);

    double* d_y_std;
    gpu_vector_alloc(&d_y_std, n);
    gpu_pack_field(&d_result_std, d_y_std,
        cpu_op.offset_Er, cpu_op.offset_Ephi, cpu_op.offset_Ez);

    /* --- Conformal path --- */
    gpu_unpack_field(d_x, &d_E_cfm,
        cpu_op.offset_Er, cpu_op.offset_Ephi, cpu_op.offset_Ez);
    gpu_apply_PEC_boundary(&d_E_cfm, &grid);
    gpu_apply_conformal_mask(&d_E_cfm, &gpu_cd);
    gpu_compute_curl_curl_E_conformal(&d_E_cfm, &d_result_cfm, &d_H_cfm,
                                      &grid, &gpu_cd);
    gpu_apply_PEC_boundary(&d_result_cfm, &grid);
    gpu_apply_conformal_mask(&d_result_cfm, &gpu_cd);

    double* d_y_cfm;
    gpu_vector_alloc(&d_y_cfm, n);
    gpu_pack_field(&d_result_cfm, d_y_cfm,
        cpu_op.offset_Er, cpu_op.offset_Ephi, cpu_op.offset_Ez);

    /* Rayleigh quotients */
    ReductionWorkspace rws;
    reduction_workspace_init(&rws, n);

    double xAx_std, xAx_cfm, xx;
    gpu_vec_dot_weighted_ws(d_x, d_y_std, &xAx_std, &cpu_op, &rws);
    gpu_vec_dot_weighted_ws(d_x, d_y_cfm, &xAx_cfm, &cpu_op, &rws);
    gpu_vec_dot_weighted_ws(d_x, d_x,     &xx,      &cpu_op, &rws);

    double rq_std = xAx_std / xx;
    double rq_cfm = xAx_cfm / xx;
    double k2_ref = (M_PI / L) * (M_PI / L);

    printf("    Rayleigh quotient (1 matvec, TEM guess):\n");
    printf("      Standard  (staircase): %.8f\n", rq_std);
    printf("      Conformal (Dey-Mittra): %.8f\n", rq_cfm);
    printf("      Analytical (no pipes):  %.8f\n", k2_ref);
    printf("      Std vs analytical: %+.4f%%\n", (rq_std - k2_ref) / k2_ref * 100);
    printf("      Cfm vs analytical: %+.4f%%\n", (rq_cfm - k2_ref) / k2_ref * 100);

    check("Staircase RQ is reasonable (within 20% of analytical)",
          fabs(rq_std - k2_ref) / k2_ref < 0.20);
    check("Conformal RQ is reasonable (within 20% of analytical)",
          fabs(rq_cfm - k2_ref) / k2_ref < 0.20);

    /* Check for NaN/Inf in conformal result */
    {
        double* h_y = (double*)malloc(n * sizeof(double));
        cudaMemcpy(h_y, d_y_cfm, n * sizeof(double), cudaMemcpyDeviceToHost);
        int has_nan = 0, has_inf = 0;
        for (int i = 0; i < n; i++) {
            if (isnan(h_y[i])) has_nan++;
            if (isinf(h_y[i])) has_inf++;
        }
        check("Conformal result: no NaN", has_nan == 0);
        check("Conformal result: no Inf", has_inf == 0);
        if (has_nan || has_inf)
            printf("      NaN=%d, Inf=%d\n", has_nan, has_inf);
        free(h_y);
    }

    /* Cleanup */
    reduction_workspace_free(&rws);
    gpu_vector_free(d_x);
    gpu_vector_free(d_y_std);
    gpu_vector_free(d_y_cfm);
    gpu_efield_free(&d_E_std);
    gpu_efield_free(&d_E_cfm);
    gpu_efield_free(&d_result_std);
    gpu_efield_free(&d_result_cfm);
    gpu_hfield_free(&d_H_std);
    gpu_hfield_free(&d_H_cfm);
    gpu_conformal_data_free(&gpu_cd);
    gpu_material_mask_free(&gpu_mask);
    conformal_data_free(&cd);
    material_mask_free(&mask);
    pipe_config_free(&pipes);
    curlcurl_op_free(&cpu_op);
    free(h_x);
}

/*=============================================================================
 * TEST C: Conformal Mask vs Material Mask
 *============================================================================*/
static void test_C_mask_consistency(void) {
    printf("\n  TEST C: Conformal Mask vs Material Mask\n");
    printf("  ========================================\n");

    double a = 0.3333, b = 1.0, L = 1.395;
    double R_pipe = 0.0125, R_aper = 0.0175, pipe_len = 0.050;

    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, R_pipe, R_aper, pipe_len, 0.0);
    pipe_config_add_multi_pass(&pipes, L / 2.0, 10);

    int Nr_cav = 81, Nr_p = 7, Nphi = 256, Nz = 84;
    GridParams grid;
    grid_init_with_pipes(&grid, a, b, L, pipe_len, Nr_cav, Nr_p, Nphi, Nz);

    MaterialMask mask;
    material_mask_build(&mask, &pipes, &grid);

    ConformalData cd;
    conformal_data_build(&cd, &pipes, &grid, b, 0.0);

    CurlCurlOperator cpu_op;
    curlcurl_op_init(&cpu_op, &grid);
    int n = cpu_op.n_total;

    cuda_grid_init(&grid);

    GPU_ConformalData gpu_cd;
    gpu_conformal_data_init(&gpu_cd, &cd);
    GPU_MaterialMask gpu_mask;
    gpu_material_mask_init(&gpu_mask, &mask);

    /* Create all-ones test vector, upload to GPU */
    double* h_ones = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) h_ones[i] = 1.0;

    double* d_ones;
    gpu_vector_alloc(&d_ones, n);
    gpu_vector_to_device(d_ones, h_ones, n);

    /* Path 1: material mask */
    GPU_EField d_E_mat;
    gpu_efield_alloc(&d_E_mat, &grid);
    gpu_unpack_field(d_ones, &d_E_mat,
        cpu_op.offset_Er, cpu_op.offset_Ephi, cpu_op.offset_Ez);
    gpu_apply_material_mask(&d_E_mat, &gpu_mask);

    /* Path 2: conformal mask */
    GPU_EField d_E_cfm;
    gpu_efield_alloc(&d_E_cfm, &grid);
    gpu_unpack_field(d_ones, &d_E_cfm,
        cpu_op.offset_Er, cpu_op.offset_Ephi, cpu_op.offset_Ez);
    gpu_apply_conformal_mask(&d_E_cfm, &gpu_cd);

    /* Download Er and compare */
    double* h_mat = (double*)malloc(mask.size_Er * sizeof(double));
    double* h_cfm = (double*)malloc(cd.size_Er * sizeof(double));
    cudaMemcpy(h_mat, d_E_mat.Er, mask.size_Er * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cfm, d_E_cfm.Er, cd.size_Er * sizeof(double),
               cudaMemcpyDeviceToHost);

    int agree_vac = 0, agree_pec = 0, disagree = 0;
    for (int i = 0; i < mask.size_Er; i++) {
        int mat_zero = (h_mat[i] == 0.0);
        int cfm_zero = (h_cfm[i] == 0.0);
        if (!mat_zero && !cfm_zero) agree_vac++;
        else if (mat_zero && cfm_zero) agree_pec++;
        else disagree++;
    }

    printf("    Er: vac_agree=%d, pec_agree=%d, disagree=%d\n",
           agree_vac, agree_pec, disagree);

    check("Mask disagreements < 5% of total PEC cells",
          disagree < 0.05 * (agree_pec + disagree));

    /* Conformal should never kill a cell that staircase says is vacuum */
    {
        int cfm_killed = 0;
        for (int i = 0; i < mask.size_Er; i++) {
            if (h_mat[i] != 0.0 && h_cfm[i] == 0.0)
                cfm_killed++;
        }
        check("Conformal kills < 0.1% of staircase-vacuum cells",
            cfm_killed < 0.001 * agree_vac);
        if (cfm_killed > 0)
            printf("      (%d vacuum cells killed!)\n", cfm_killed);
    }

    /* Cleanup */
    free(h_ones);
    free(h_mat);
    free(h_cfm);
    gpu_vector_free(d_ones);
    gpu_efield_free(&d_E_mat);
    gpu_efield_free(&d_E_cfm);
    gpu_conformal_data_free(&gpu_cd);
    gpu_material_mask_free(&gpu_mask);
    conformal_data_free(&cd);
    material_mask_free(&mask);
    pipe_config_free(&pipes);
    curlcurl_op_free(&cpu_op);
}

/*=============================================================================
 * MAIN
 *============================================================================*/
int main(void) {
    printf("\n");
    printf("================================================================\n");
    printf("  CONFORMAL CUDA KERNELS — PHASE 2 VALIDATION\n");
    printf("================================================================\n");

    cuda_print_device_info();
    cuda_print_memory_info("startup");

    test_A_reference_cavity();
    test_B_pipe_model();
    test_C_mask_consistency();

    printf("\n================================================================\n");
    printf("  RESULTS: %d passed, %d failed\n", g_pass, g_fail);
    printf("================================================================\n\n");

    return g_fail > 0 ? 1 : 0;
}
