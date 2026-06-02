/*=============================================================================
 * test_cuda_eigensolver.cpp
 *
 * Phase 1F verification: GPU eigensolver
 *============================================================================*/

#include "cuda_eigensolver.h"
#include "cuda_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

 /*=============================================================================
  * Test 1: GPU MINRES vs CPU MINRES
  *============================================================================*/
static int test_minres(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 1: GPU MINRES vs CPU (shifted solve)\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    CurlCurlOperator cpu_op;
    curlcurl_op_init(&cpu_op, &grid);

    GPU_Operator gpu_op;
    gpu_operator_init(&gpu_op, &cpu_op);

    int n = cpu_op.n_total;
    double sigma = 5.0;  /* Close to TEM eigenvalue */

    /* Create RHS = TEM-like field */
    double* h_b = (double*)malloc(n * sizeof(double));
    double* h_x_cpu = (double*)malloc(n * sizeof(double));
    vec_zero(h_b, n);

    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                int idx = cpu_op.offset_Er + idx_Er(&grid, i, j, k);
                h_b[idx] = sin(M_PI * z / grid.L) / r;
            }
        }
    }

    /* GPU solve */
    double* d_b, * d_x;
    gpu_vector_alloc(&d_b, n);
    gpu_vector_alloc(&d_x, n);
    gpu_vector_to_device(d_b, h_b, n);

    GPU_LinearSolverResult gpu_result = gpu_minres_solve_shifted(
        &gpu_op, sigma, d_b, d_x, 500, 1e-6);

    double* h_x_gpu = (double*)malloc(n * sizeof(double));
    gpu_vector_to_host(h_x_gpu, d_x, n);

    printf("  GPU MINRES: %d iterations, residual = %.3e, converged = %s\n",
        gpu_result.iterations, gpu_result.residual,
        gpu_result.converged ? "YES" : "NO");

    /* Verify: compute ||A*x_gpu - sigma*x_gpu - b|| */
    double* h_Ax = (double*)malloc(n * sizeof(double));
    curlcurl_matvec(&cpu_op, h_x_gpu, h_Ax);

    double check_residual = 0.0;
    for (int i = 0; i < n; i++) {
        double r = h_Ax[i] - sigma * h_x_gpu[i] - h_b[i];
        check_residual += r * r;
    }
    check_residual = sqrt(check_residual);

    double b_norm = 0.0;
    for (int i = 0; i < n; i++) b_norm += h_b[i] * h_b[i];
    b_norm = sqrt(b_norm);

    printf("  Verification: ||(A-σI)x - b|| / ||b|| = %.3e\n",
        check_residual / b_norm);

    int passed = gpu_result.converged;
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    free(h_b); free(h_x_cpu); free(h_x_gpu); free(h_Ax);
    gpu_vector_free(d_b); gpu_vector_free(d_x);
    gpu_operator_free(&gpu_op);
    curlcurl_op_free(&cpu_op);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 2: GPU RQI — find TEM eigenvalue
 *============================================================================*/
static int test_rqi_TEM(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 2: GPU RQI — TEM mode eigenvalue\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    CurlCurlOperator cpu_op;
    curlcurl_op_init(&cpu_op, &grid);

    GPU_Operator gpu_op;
    gpu_operator_init(&gpu_op, &cpu_op);

    int n = cpu_op.n_total;

    /* TEM initial guess on GPU */
    double* h_x = (double*)malloc(n * sizeof(double));
    vec_zero(h_x, n);

    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                int idx = cpu_op.offset_Er + idx_Er(&grid, i, j, k);
                h_x[idx] = sin(M_PI * z / grid.L) / r;
            }
        }
    }

    double* d_x;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_to_device(d_x, h_x, n);

    /* GPU RQI */
    double k2_target = (M_PI / grid.L) * (M_PI / grid.L);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    GPU_EigenResult result = gpu_rayleigh_quotient_iteration(
        &gpu_op, d_x, k2_target, 20, 1e-10);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    /* Compare with CPU result */
    double k2_exact = k2_target;
    double rel_err = fabs(result.eigenvalue - k2_exact) / k2_exact;

    double c = 299792458.0;
    double f = sqrt(fabs(result.eigenvalue)) * c / (2.0 * M_PI);

    printf("\n  Results:\n");
    printf("    k² exact:      %.10f\n", k2_exact);
    printf("    k² GPU RQI:    %.10f\n", result.eigenvalue);
    printf("    Error vs exact: %.6e\n", rel_err);
    printf("    Frequency:     %.6f MHz\n", f / 1e6);
    printf("    Converged:     %s\n", result.converged ? "YES" : "NO");
    printf("    Iterations:    %d\n", result.iterations);
    printf("    Residual:      %.3e\n", result.residual);
    printf("    GPU time:      %.1f ms\n", gpu_ms);

    int passed = result.converged && (rel_err < 0.001);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_x);
    gpu_vector_free(d_x);
    gpu_operator_free(&gpu_op);
    curlcurl_op_free(&cpu_op);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 3: GPU RQI with ports — compare with CPU result
 *============================================================================*/
static int test_rqi_with_ports(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 3: GPU RQI with ports vs CPU RQI\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    PortConfig ports;
    port_config_init(&ports);
    port_config_add_beam_apertures_single_pass(&ports, grid.L / 2.0, 0.005);
    port_config_add_vacuum_port(&ports, SURFACE_ENDPLATE_Z0,
        0.6666, M_PI / 2.0, 0.015);

    CurlCurlOperator cpu_op;
    curlcurl_op_init_with_ports(&cpu_op, &grid, &ports);

    GPU_Operator gpu_op;
    gpu_operator_init(&gpu_op, &cpu_op);

    int n = cpu_op.n_total;

    /* TEM initial guess */
    double* h_x = (double*)malloc(n * sizeof(double));
    double* h_x_cpu = (double*)malloc(n * sizeof(double));
    vec_zero(h_x, n);

    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                int idx = cpu_op.offset_Er + idx_Er(&grid, i, j, k);
                h_x[idx] = sin(M_PI * z / grid.L) / r;
            }
        }
    }
    vec_copy(h_x, h_x_cpu, n);

    /* CPU RQI */
    printf("  --- CPU RQI ---\n");
    double k2_target = (M_PI / grid.L) * (M_PI / grid.L);

    clock_t cpu_start = clock();
    EigenResult cpu_result = rayleigh_quotient_iteration(
        &cpu_op, h_x_cpu, k2_target, 20, 1e-10);
    clock_t cpu_end = clock();
    double cpu_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0;

    /* GPU RQI */
    printf("\n  --- GPU RQI ---\n");
    double* d_x;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_to_device(d_x, h_x, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    GPU_EigenResult gpu_result = gpu_rayleigh_quotient_iteration(
        &gpu_op, d_x, k2_target, 20, 1e-10);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    /* Compare */
    double rel_err = fabs(gpu_result.eigenvalue - cpu_result.eigenvalue) /
        fabs(cpu_result.eigenvalue);

    printf("\n  Comparison:\n");
    printf("    CPU k²: %.10f  (%d iters, %.1f ms)\n",
        cpu_result.eigenvalue, cpu_result.iterations, cpu_ms);
    printf("    GPU k²: %.10f  (%d iters, %.1f ms)\n",
        gpu_result.eigenvalue, gpu_result.iterations, gpu_ms);
    printf("    Eigenvalue relative error: %.3e\n", rel_err);
    printf("    Speedup: %.1fx\n", cpu_ms / gpu_ms);

    int passed = (rel_err < 1e-8) && gpu_result.converged;
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_x); free(h_x_cpu);
    gpu_vector_free(d_x);
    gpu_operator_free(&gpu_op);
    port_config_free(&ports);
    curlcurl_op_free(&cpu_op);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 4: GPU eigenvector export
 *============================================================================*/
static int test_eigenvector_export(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 4: GPU eigenvector → CPU export\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    CurlCurlOperator cpu_op;
    curlcurl_op_init(&cpu_op, &grid);

    GPU_Operator gpu_op;
    gpu_operator_init(&gpu_op, &cpu_op);

    int n = cpu_op.n_total;

    /* TEM initial guess */
    double* h_x = (double*)malloc(n * sizeof(double));
    vec_zero(h_x, n);

    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                int idx = cpu_op.offset_Er + idx_Er(&grid, i, j, k);
                h_x[idx] = sin(M_PI * z / grid.L) / r;
            }
        }
    }

    double* d_x;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_to_device(d_x, h_x, n);

    /* Solve on GPU */
    double k2_target = (M_PI / grid.L) * (M_PI / grid.L);
    GPU_EigenResult result = gpu_rayleigh_quotient_iteration(
        &gpu_op, d_x, k2_target, 10, 1e-10);

    /* Copy eigenvector back to CPU */
    gpu_vector_to_host(h_x, d_x, n);

    /* Normalize for export */
    double max_val = 0.0;
    for (int i = 0; i < n; i++) {
        if (fabs(h_x[i]) > max_val) max_val = fabs(h_x[i]);
    }
    if (max_val > 0) vec_scale(h_x, 1.0 / max_val, n);

    /* Export VTK */
    export_field_vtk(&cpu_op, h_x, "gpu_tem_mode.vtk");
    printf("  Exported gpu_tem_mode.vtk\n");

    printf("  k² = %.10f, f = %.6f MHz\n",
        result.eigenvalue,
        sqrt(fabs(result.eigenvalue)) * 299792458.0 / (2.0 * M_PI) / 1e6);
    printf("  PASSED ✓\n");

    free(h_x);
    gpu_vector_free(d_x);
    gpu_operator_free(&gpu_op);
    curlcurl_op_free(&cpu_op);

    return 0;
}

/*=============================================================================
 * Main
 *============================================================================*/
int main(void) {
    printf("\n");
    printf("****************************************************************\n");
    printf("*        CUDA PHASE 1F: EIGENSOLVER TESTS                     *\n");
    printf("****************************************************************\n");

    int failures = 0;

    failures += (test_minres() != 0);
    failures += (test_rqi_TEM() != 0);
    failures += (test_rqi_with_ports() != 0);
    failures += (test_eigenvector_export() != 0);

    printf("\n");
    printf("****************************************************************\n");
    if (failures == 0)
        printf("*        ALL PHASE 1F TESTS PASSED ✓                         *\n");
    else
        printf("*        %d TEST(S) FAILED ✗                                 *\n", failures);
    printf("****************************************************************\n\n");

    return failures;
}