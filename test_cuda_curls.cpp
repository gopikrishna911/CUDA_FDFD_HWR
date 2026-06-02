/*=============================================================================
 * test_cuda_curls.cpp
 *
 * Phase 1C verification: GPU curl operations
 * Tests: curl_E, curl_H, curl_curl_E vs CPU
 *============================================================================*/

#include "cuda_curls.h"
#include "cuda_fields.h"
#include "curlcurl_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

 /*=============================================================================
  * Helper: fill E-field with TEM-like pattern
  *============================================================================*/
static void fill_TEM_field(EField* E, const GridParams* grid) {
    efield_zero(E);

    /* Er = sin(pi*z/L) / r */
    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                double r = r_at_i_half(grid, i);
                int idx = idx_Er(grid, i, j, k);
                E->Er[idx] = sin(M_PI * z / grid->L) / r;
            }
        }
    }
}

/*=============================================================================
 * Helper: fill E-field with all-component pattern (tests all curls)
 *============================================================================*/
static void fill_mixed_field(EField* E, const GridParams* grid) {
    efield_zero(E);

    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            for (int i = 0; i < grid->Nr; i++) {
                double r = r_at_i_half(grid, i);
                E->Er[idx_Er(grid, i, j, k)] =
                    sin(M_PI * z / grid->L) * cos(phi) / r;
            }
        }
    }

    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            for (int i = 0; i <= grid->Nr; i++) {
                double r = r_at_i(grid, i);
                if (r < 1e-14) r = 1e-14;
                E->Ephi[idx_Ephi(grid, i, j, k)] =
                    sin(M_PI * z / grid->L) * sin(phi) / r;
            }
        }
    }

    for (int k = 0; k < grid->Nz; k++) {
        double z = (k + 0.5) * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            double phi = j * grid->dphi;
            for (int i = 0; i <= grid->Nr; i++) {
                double r = r_at_i(grid, i);
                if (r < 1e-14) r = 1e-14;
                E->Ez[idx_Ez(grid, i, j, k)] =
                    cos(M_PI * z / grid->L) * cos(phi) * r;
            }
        }
    }
}

/*=============================================================================
 * Helper: compare arrays, return max absolute and relative error
 *============================================================================*/
static void compare_arrays(const double* cpu, const double* gpu, int n,
    double* max_abs, double* max_rel) {
    *max_abs = 0.0;
    *max_rel = 0.0;

    double max_val = 0.0;
    for (int i = 0; i < n; i++) {
        if (fabs(cpu[i]) > max_val) max_val = fabs(cpu[i]);
    }

    for (int i = 0; i < n; i++) {
        double err = fabs(cpu[i] - gpu[i]);
        if (err > *max_abs) *max_abs = err;
        if (max_val > 1e-14) {
            double rel = err / max_val;
            if (rel > *max_rel) *max_rel = rel;
        }
    }
}

/*=============================================================================
 * Test 1: Curl of E (GPU vs CPU) with TEM field
 *============================================================================*/
static int test_curl_E_TEM(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 1: Curl of E (TEM field) - GPU vs CPU\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    /* CPU computation */
    EField E_cpu;
    HField G_cpu;
    efield_alloc(&E_cpu, &grid);
    hfield_alloc(&G_cpu, &grid);
    fill_TEM_field(&E_cpu, &grid);

    /* CPU curl */
    for (int k = 0; k < grid.Nz; k++)
        for (int j = 0; j < grid.Nphi; j++)
            for (int i = 0; i <= grid.Nr; i++)
                G_cpu.Hr[idx_Hr(&grid, i, j, k)] = curl_E_r_at(&E_cpu, &grid, i, j, k);

    for (int k = 0; k < grid.Nz; k++)
        for (int j = 0; j < grid.Nphi; j++)
            for (int i = 0; i < grid.Nr; i++)
                G_cpu.Hphi[idx_Hphi(&grid, i, j, k)] = curl_E_phi_at(&E_cpu, &grid, i, j, k);

    for (int k = 0; k <= grid.Nz; k++)
        for (int j = 0; j < grid.Nphi; j++)
            for (int i = 0; i < grid.Nr; i++)
                G_cpu.Hz[idx_Hz(&grid, i, j, k)] = curl_E_z_at(&E_cpu, &grid, i, j, k);

    /* GPU computation */
    GPU_EField d_E;
    GPU_HField d_G;
    gpu_efield_alloc(&d_E, &grid);
    gpu_hfield_alloc(&d_G, &grid);
    gpu_efield_to_device(&d_E, &E_cpu);

    gpu_compute_curl_E(&d_E, &d_G, &grid);
    cudaDeviceSynchronize();

    /* Copy back */
    HField G_gpu;
    hfield_alloc(&G_gpu, &grid);
    gpu_hfield_to_host(&G_gpu, &d_G);

    /* Compare */
    double abs_err, rel_err;

    compare_arrays(G_cpu.Hr, G_gpu.Hr, G_cpu.size_Hr, &abs_err, &rel_err);
    printf("  (curl E)_r:   max_abs=%.3e  max_rel=%.3e\n", abs_err, rel_err);

    compare_arrays(G_cpu.Hphi, G_gpu.Hphi, G_cpu.size_Hphi, &abs_err, &rel_err);
    printf("  (curl E)_phi: max_abs=%.3e  max_rel=%.3e\n", abs_err, rel_err);

    compare_arrays(G_cpu.Hz, G_gpu.Hz, G_cpu.size_Hz, &abs_err, &rel_err);
    printf("  (curl E)_z:   max_abs=%.3e  max_rel=%.3e\n", abs_err, rel_err);

    /* Overall check */
    double total_rel = 0.0;
    compare_arrays(G_cpu.Hr, G_gpu.Hr, G_cpu.size_Hr, &abs_err, &rel_err);
    if (rel_err > total_rel) total_rel = rel_err;
    compare_arrays(G_cpu.Hphi, G_gpu.Hphi, G_cpu.size_Hphi, &abs_err, &rel_err);
    if (rel_err > total_rel) total_rel = rel_err;
    compare_arrays(G_cpu.Hz, G_gpu.Hz, G_cpu.size_Hz, &abs_err, &rel_err);
    if (rel_err > total_rel) total_rel = rel_err;

    int passed = (total_rel < 1e-12);
    printf("  Overall max relative error: %.3e\n", total_rel);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    gpu_efield_free(&d_E);
    gpu_hfield_free(&d_G);
    hfield_free(&G_gpu);
    hfield_free(&G_cpu);
    efield_free(&E_cpu);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 2: Curl of E with mixed field (all components active)
 *============================================================================*/
static int test_curl_E_mixed(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 2: Curl of E (mixed field) - GPU vs CPU\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    EField E_cpu;
    efield_alloc(&E_cpu, &grid);
    fill_mixed_field(&E_cpu, &grid);

    /* Full CPU curl_curl to get intermediate G = curl E */
    HField G_cpu;
    hfield_alloc(&G_cpu, &grid);

    for (int k = 0; k < grid.Nz; k++)
        for (int j = 0; j < grid.Nphi; j++)
            for (int i = 0; i <= grid.Nr; i++)
                G_cpu.Hr[idx_Hr(&grid, i, j, k)] = curl_E_r_at(&E_cpu, &grid, i, j, k);

    for (int k = 0; k < grid.Nz; k++)
        for (int j = 0; j < grid.Nphi; j++)
            for (int i = 0; i < grid.Nr; i++)
                G_cpu.Hphi[idx_Hphi(&grid, i, j, k)] = curl_E_phi_at(&E_cpu, &grid, i, j, k);

    for (int k = 0; k <= grid.Nz; k++)
        for (int j = 0; j < grid.Nphi; j++)
            for (int i = 0; i < grid.Nr; i++)
                G_cpu.Hz[idx_Hz(&grid, i, j, k)] = curl_E_z_at(&E_cpu, &grid, i, j, k);

    /* GPU */
    GPU_EField d_E;
    GPU_HField d_G;
    gpu_efield_alloc(&d_E, &grid);
    gpu_hfield_alloc(&d_G, &grid);
    gpu_efield_to_device(&d_E, &E_cpu);

    gpu_compute_curl_E(&d_E, &d_G, &grid);
    cudaDeviceSynchronize();

    HField G_gpu;
    hfield_alloc(&G_gpu, &grid);
    gpu_hfield_to_host(&G_gpu, &d_G);

    double abs_err, rel_err, total_rel = 0.0;

    compare_arrays(G_cpu.Hr, G_gpu.Hr, G_cpu.size_Hr, &abs_err, &rel_err);
    printf("  (curl E)_r:   max_abs=%.3e  max_rel=%.3e\n", abs_err, rel_err);
    if (rel_err > total_rel) total_rel = rel_err;

    compare_arrays(G_cpu.Hphi, G_gpu.Hphi, G_cpu.size_Hphi, &abs_err, &rel_err);
    printf("  (curl E)_phi: max_abs=%.3e  max_rel=%.3e\n", abs_err, rel_err);
    if (rel_err > total_rel) total_rel = rel_err;

    compare_arrays(G_cpu.Hz, G_gpu.Hz, G_cpu.size_Hz, &abs_err, &rel_err);
    printf("  (curl E)_z:   max_abs=%.3e  max_rel=%.3e\n", abs_err, rel_err);
    if (rel_err > total_rel) total_rel = rel_err;

    int passed = (total_rel < 1e-12);
    printf("  Overall max relative error: %.3e\n", total_rel);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    gpu_efield_free(&d_E);
    gpu_hfield_free(&d_G);
    hfield_free(&G_gpu);
    hfield_free(&G_cpu);
    efield_free(&E_cpu);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 3: Full curl-curl (GPU vs CPU) — the critical test
 *============================================================================*/
static int test_curl_curl(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 3: Curl-Curl E (GPU vs CPU) — full operator\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    /* CPU: fill field and compute curl-curl */
    EField E_cpu, ccE_cpu;
    HField G_temp;
    efield_alloc(&E_cpu, &grid);
    efield_alloc(&ccE_cpu, &grid);
    hfield_alloc(&G_temp, &grid);

    fill_mixed_field(&E_cpu, &grid);
    compute_curl_curl_E_with_temp(&E_cpu, &ccE_cpu, &G_temp, &grid);

    /* GPU: same computation */
    GPU_EField d_E, d_ccE;
    GPU_HField d_G;
    gpu_efield_alloc(&d_E, &grid);
    gpu_efield_alloc(&d_ccE, &grid);
    gpu_hfield_alloc(&d_G, &grid);

    gpu_efield_to_device(&d_E, &E_cpu);
    gpu_compute_curl_curl_E(&d_E, &d_ccE, &d_G, &grid);
    cudaDeviceSynchronize();

    /* Copy back */
    EField ccE_gpu;
    efield_alloc(&ccE_gpu, &grid);
    gpu_efield_to_host(&ccE_gpu, &d_ccE);

    /* Compare */
    double abs_err, rel_err, total_rel = 0.0;

    compare_arrays(ccE_cpu.Er, ccE_gpu.Er, ccE_cpu.size_Er, &abs_err, &rel_err);
    printf("  (∇×∇×E)_r:   max_abs=%.3e  max_rel=%.3e\n", abs_err, rel_err);
    if (rel_err > total_rel) total_rel = rel_err;

    compare_arrays(ccE_cpu.Ephi, ccE_gpu.Ephi, ccE_cpu.size_Ephi, &abs_err, &rel_err);
    printf("  (∇×∇×E)_phi: max_abs=%.3e  max_rel=%.3e\n", abs_err, rel_err);
    if (rel_err > total_rel) total_rel = rel_err;

    compare_arrays(ccE_cpu.Ez, ccE_gpu.Ez, ccE_cpu.size_Ez, &abs_err, &rel_err);
    printf("  (∇×∇×E)_z:   max_abs=%.3e  max_rel=%.3e\n", abs_err, rel_err);
    if (rel_err > total_rel) total_rel = rel_err;

    int passed = (total_rel < 1e-12);
    printf("  Overall max relative error: %.3e\n", total_rel);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    gpu_efield_free(&d_E);
    gpu_efield_free(&d_ccE);
    gpu_hfield_free(&d_G);
    efield_free(&ccE_gpu);
    efield_free(&ccE_cpu);
    efield_free(&E_cpu);
    hfield_free(&G_temp);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 4: Curl-curl timing
 *============================================================================*/
static int test_curl_curl_timing(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 4: Curl-curl timing (GPU)\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    GPU_EField d_E, d_ccE;
    GPU_HField d_G;
    gpu_efield_alloc(&d_E, &grid);
    gpu_efield_alloc(&d_ccE, &grid);
    gpu_hfield_alloc(&d_G, &grid);

    /* Fill with something */
    EField E_cpu;
    efield_alloc(&E_cpu, &grid);
    fill_mixed_field(&E_cpu, &grid);
    gpu_efield_to_device(&d_E, &E_cpu);

    /* Warm up */
    gpu_compute_curl_curl_E(&d_E, &d_ccE, &d_G, &grid);
    cudaDeviceSynchronize();

    /* Time it */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iter = 100;
    cudaEventRecord(start);
    for (int i = 0; i < num_iter; i++) {
        gpu_compute_curl_curl_E(&d_E, &d_ccE, &d_G, &grid);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    int n_total = E_cpu.size_Er + E_cpu.size_Ephi + E_cpu.size_Ez;
    printf("  Grid: Nr=%d, Nphi=%d, Nz=%d  (%.1fM DOFs)\n",
        grid.Nr, grid.Nphi, grid.Nz, n_total / 1.0e6);
    printf("  %d curl-curl ops in %.1f ms\n", num_iter, ms);
    printf("  Average: %.3f ms per curl-curl\n", ms / num_iter);
    printf("  Throughput: %.1f Gdof/s\n",
        (double)n_total * num_iter / (ms * 1e-3) / 1e9);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("  PASSED ✓\n");

    gpu_efield_free(&d_E);
    gpu_efield_free(&d_ccE);
    gpu_hfield_free(&d_G);
    efield_free(&E_cpu);

    return 0;
}

/*=============================================================================
 * Test 5: TEM eigenvalue test (curl-curl of TEM mode)
 *============================================================================*/
static int test_TEM_eigenvalue(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 5: TEM mode eigenvalue via GPU curl-curl\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);

    /* Create TEM field */
    EField E_cpu;
    efield_alloc(&E_cpu, &grid);
    fill_TEM_field(&E_cpu, &grid);

    /* GPU curl-curl */
    GPU_EField d_E, d_ccE;
    GPU_HField d_G;
    gpu_efield_alloc(&d_E, &grid);
    gpu_efield_alloc(&d_ccE, &grid);
    gpu_hfield_alloc(&d_G, &grid);

    gpu_efield_to_device(&d_E, &E_cpu);
    gpu_compute_curl_curl_E(&d_E, &d_ccE, &d_G, &grid);
    cudaDeviceSynchronize();

    /* Pack both into vectors for Rayleigh quotient */
    double* h_x = (double*)malloc(op.n_total * sizeof(double));
    double* h_Ax = (double*)malloc(op.n_total * sizeof(double));
    pack_field(&E_cpu, h_x, &op);

    EField ccE_gpu;
    efield_alloc(&ccE_gpu, &grid);
    gpu_efield_to_host(&ccE_gpu, &d_ccE);
    pack_field(&ccE_gpu, h_Ax, &op);

    /* Rayleigh quotient on CPU using GPU-computed Ax */
    double xAx = vec_dot_product_weighted(h_x, h_Ax, &op);
    double xx = vec_dot_product_weighted(h_x, h_x, &op);
    double k2_gpu = xAx / xx;

    /* Compare with exact */
    double k2_exact = (M_PI / grid.L) * (M_PI / grid.L);
    double rel_err = fabs(k2_gpu - k2_exact) / k2_exact;

    double c = 299792458.0;
    double f_gpu = sqrt(k2_gpu) * c / (2.0 * M_PI);

    printf("  k² exact:  %.10f\n", k2_exact);
    printf("  k² GPU:    %.10f\n", k2_gpu);
    printf("  Relative error: %.6e\n", rel_err);
    printf("  Frequency: %.6f MHz\n", f_gpu / 1e6);

    int passed = (rel_err < 0.001);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    free(h_x);
    free(h_Ax);
    gpu_efield_free(&d_E);
    gpu_efield_free(&d_ccE);
    gpu_hfield_free(&d_G);
    efield_free(&ccE_gpu);
    efield_free(&E_cpu);
    curlcurl_op_free(&op);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Main
 *============================================================================*/
int main(void) {
    printf("\n");
    printf("****************************************************************\n");
    printf("*          CUDA PHASE 1C: CURL KERNELS TESTS                  *\n");
    printf("****************************************************************\n");

    int failures = 0;

    failures += (test_curl_E_TEM() != 0);
    failures += (test_curl_E_mixed() != 0);
    failures += (test_curl_curl() != 0);
    failures += (test_curl_curl_timing() != 0);
    failures += (test_TEM_eigenvalue() != 0);

    printf("\n");
    printf("****************************************************************\n");
    if (failures == 0)
        printf("*          ALL PHASE 1C TESTS PASSED ✓                        *\n");
    else
        printf("*          %d TEST(S) FAILED ✗                                *\n", failures);
    printf("****************************************************************\n\n");

    return failures;
}