/*=============================================================================
 * test_cuda_operator.cpp
 *
 * Phase 1D+1E verification: GPU boundary conditions + full operator
 *============================================================================*/

#include "cuda_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <ctime>

 /*=============================================================================
  * Test 1: PEC boundary (no ports) — GPU vs CPU
  *============================================================================*/
static int test_pec_boundary(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 1: PEC boundary conditions (no ports) GPU vs CPU\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    /* Fill with random data */
    EField E_cpu, E_gpu_result;
    efield_alloc(&E_cpu, &grid);
    efield_alloc(&E_gpu_result, &grid);

    srand(12345);
    for (int i = 0; i < E_cpu.size_Er; i++) E_cpu.Er[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < E_cpu.size_Ephi; i++) E_cpu.Ephi[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < E_cpu.size_Ez; i++) E_cpu.Ez[i] = (double)rand() / RAND_MAX;

    /* GPU: copy, apply BC, copy back */
    GPU_EField d_E;
    gpu_efield_alloc(&d_E, &grid);
    gpu_efield_to_device(&d_E, &E_cpu);
    gpu_apply_PEC_boundary(&d_E, &grid);
    cudaDeviceSynchronize();
    gpu_efield_to_host(&E_gpu_result, &d_E);

    /* CPU: apply BC */
    apply_PEC_boundary(&E_cpu, &grid);

    /* Compare */
    double max_err = 0.0;
    for (int i = 0; i < E_cpu.size_Er; i++) {
        double err = fabs(E_cpu.Er[i] - E_gpu_result.Er[i]);
        if (err > max_err) max_err = err;
    }
    for (int i = 0; i < E_cpu.size_Ephi; i++) {
        double err = fabs(E_cpu.Ephi[i] - E_gpu_result.Ephi[i]);
        if (err > max_err) max_err = err;
    }
    for (int i = 0; i < E_cpu.size_Ez; i++) {
        double err = fabs(E_cpu.Ez[i] - E_gpu_result.Ez[i]);
        if (err > max_err) max_err = err;
    }

    printf("  Max BC error: %.3e\n", max_err);
    int passed = (max_err == 0.0);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    gpu_efield_free(&d_E);
    efield_free(&E_cpu);
    efield_free(&E_gpu_result);
    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 2: Full matvec (no ports) — GPU vs CPU
 *============================================================================*/
static int test_matvec_no_ports(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 2: Full matvec (no ports) GPU vs CPU\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    CurlCurlOperator cpu_op;
    curlcurl_op_init(&cpu_op, &grid);

    int n = cpu_op.n_total;
    printf("  DOFs: %d\n", n);

    /* Create TEM-like input */
    double* h_x = (double*)malloc(n * sizeof(double));
    double* h_y_cpu = (double*)malloc(n * sizeof(double));
    double* h_y_gpu = (double*)malloc(n * sizeof(double));
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

    /* CPU matvec */
    curlcurl_matvec(&cpu_op, h_x, h_y_cpu);

    /* GPU matvec */
    GPU_Operator gpu_op;
    gpu_operator_init(&gpu_op, &cpu_op);

    double* d_x, * d_y;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_alloc(&d_y, n);
    gpu_vector_to_device(d_x, h_x, n);

    gpu_curlcurl_matvec(&gpu_op, d_x, d_y);
    cudaDeviceSynchronize();

    gpu_vector_to_host(h_y_gpu, d_y, n);

    /* Compare */
    double max_err = 0.0;
    double max_val = 0.0;
    for (int i = 0; i < n; i++) {
        if (fabs(h_y_cpu[i]) > max_val) max_val = fabs(h_y_cpu[i]);
    }
    for (int i = 0; i < n; i++) {
        double err = fabs(h_y_cpu[i] - h_y_gpu[i]);
        if (err > max_err) max_err = err;
    }
    double rel_err = max_err / (max_val + 1e-30);

    printf("  Max absolute error: %.3e\n", max_err);
    printf("  Max value: %.3e\n", max_val);
    printf("  Relative error: %.3e\n", rel_err);

    int passed = (rel_err < 1e-12);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    gpu_vector_free(d_x);
    gpu_vector_free(d_y);
    gpu_operator_free(&gpu_op);
    free(h_x);
    free(h_y_cpu);
    free(h_y_gpu);
    curlcurl_op_free(&cpu_op);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 3: Full matvec WITH ports — GPU vs CPU
 *============================================================================*/
static int test_matvec_with_ports(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 3: Full matvec (with ports) GPU vs CPU\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    /* Setup ports */
    PortConfig ports;
    port_config_init(&ports);
    port_config_add_beam_apertures_single_pass(&ports, grid.L / 2.0, 0.005);
    port_config_add_vacuum_port(&ports, SURFACE_ENDPLATE_Z0,
        0.6666, M_PI / 2.0, 0.015);

    CurlCurlOperator cpu_op;
    curlcurl_op_init_with_ports(&cpu_op, &grid, &ports);

    int n = cpu_op.n_total;
    printf("  DOFs: %d, Ports: %d\n", n, ports.num_ports);

    /* Random input */
    double* h_x = (double*)malloc(n * sizeof(double));
    double* h_y_cpu = (double*)malloc(n * sizeof(double));
    double* h_y_gpu = (double*)malloc(n * sizeof(double));
    vec_random(h_x, n, 54321);

    /* CPU matvec */
    curlcurl_matvec(&cpu_op, h_x, h_y_cpu);

    /* GPU matvec */
    GPU_Operator gpu_op;
    gpu_operator_init(&gpu_op, &cpu_op);

    double* d_x, * d_y;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_alloc(&d_y, n);
    gpu_vector_to_device(d_x, h_x, n);

    gpu_curlcurl_matvec(&gpu_op, d_x, d_y);
    cudaDeviceSynchronize();

    gpu_vector_to_host(h_y_gpu, d_y, n);

    /* Compare */
    double max_err = 0.0;
    double max_val = 0.0;
    for (int i = 0; i < n; i++) {
        if (fabs(h_y_cpu[i]) > max_val) max_val = fabs(h_y_cpu[i]);
    }
    for (int i = 0; i < n; i++) {
        double err = fabs(h_y_cpu[i] - h_y_gpu[i]);
        if (err > max_err) max_err = err;
    }
    double rel_err = max_err / (max_val + 1e-30);

    printf("  Max absolute error: %.3e\n", max_err);
    printf("  Relative error: %.3e\n", rel_err);

    int passed = (rel_err < 1e-12);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    gpu_vector_free(d_x);
    gpu_vector_free(d_y);
    gpu_operator_free(&gpu_op);
    free(h_x);
    free(h_y_cpu);
    free(h_y_gpu);
    port_config_free(&ports);
    curlcurl_op_free(&cpu_op);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 4: GPU Rayleigh quotient matches CPU
 *============================================================================*/
static int test_rayleigh_quotient(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 4: GPU Rayleigh quotient vs CPU\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    CurlCurlOperator cpu_op;
    curlcurl_op_init(&cpu_op, &grid);

    int n = cpu_op.n_total;

    /* TEM mode */
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

    /* CPU Rayleigh quotient */
    double rq_cpu = rayleigh_quotient_weighted(&cpu_op, h_x);

    /* GPU Rayleigh quotient */
    GPU_Operator gpu_op;
    gpu_operator_init(&gpu_op, &cpu_op);

    double* d_x;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_to_device(d_x, h_x, n);

    double rq_gpu;
    gpu_rayleigh_quotient(&gpu_op, d_x, &rq_gpu);

    double rel_err = fabs(rq_gpu - rq_cpu) / fabs(rq_cpu);
    double k2_exact = (M_PI / grid.L) * (M_PI / grid.L);

    printf("  k² exact:     %.10f\n", k2_exact);
    printf("  k² CPU (RQ):  %.10f\n", rq_cpu);
    printf("  k² GPU (RQ):  %.10f\n", rq_gpu);
    printf("  GPU vs CPU relative error: %.3e\n", rel_err);

    int passed = (rel_err < 1e-10);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    gpu_vector_free(d_x);
    gpu_operator_free(&gpu_op);
    free(h_x);
    curlcurl_op_free(&cpu_op);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 5: Matvec timing
 *============================================================================*/
static int test_matvec_timing(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 5: GPU matvec timing\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    CurlCurlOperator cpu_op;
    curlcurl_op_init(&cpu_op, &grid);

    GPU_Operator gpu_op;
    gpu_operator_init(&gpu_op, &cpu_op);

    int n = cpu_op.n_total;

    double* d_x, * d_y;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_alloc(&d_y, n);

    double* h_x = (double*)malloc(n * sizeof(double));
    vec_random(h_x, n, 99999);
    gpu_vector_to_device(d_x, h_x, n);

    /* Warm up */
    gpu_curlcurl_matvec(&gpu_op, d_x, d_y);
    cudaDeviceSynchronize();

    /* Time GPU */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iter = 200;
    cudaEventRecord(start);
    for (int i = 0; i < num_iter; i++) {
        gpu_curlcurl_matvec(&gpu_op, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    /* Time CPU */
    double* h_y = (double*)malloc(n * sizeof(double));
    clock_t cpu_start = clock();
    for (int i = 0; i < 10; i++) {
        curlcurl_matvec(&cpu_op, h_x, h_y);
    }
    clock_t cpu_end = clock();
    double cpu_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0;

    printf("  Grid: Nr=%d, Nphi=%d, Nz=%d  (%.1fM DOFs)\n",
        grid.Nr, grid.Nphi, grid.Nz, n / 1.0e6);
    printf("\n  GPU: %d matvecs in %.1f ms → %.3f ms each\n",
        num_iter, gpu_ms, gpu_ms / num_iter);
    printf("  CPU: 10 matvecs in %.1f ms → %.3f ms each\n",
        cpu_ms, cpu_ms / 10.0);
    printf("  Speedup: %.1fx\n", (cpu_ms / 10.0) / (gpu_ms / num_iter));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("  PASSED ✓\n");

    free(h_x);
    free(h_y);
    gpu_vector_free(d_x);
    gpu_vector_free(d_y);
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
    printf("*      CUDA PHASE 1D+1E: BOUNDARY + FULL OPERATOR TESTS      *\n");
    printf("****************************************************************\n");

    int failures = 0;

    failures += (test_pec_boundary() != 0);
    failures += (test_matvec_no_ports() != 0);
    failures += (test_matvec_with_ports() != 0);
    failures += (test_rayleigh_quotient() != 0);
    failures += (test_matvec_timing() != 0);

    printf("\n");
    printf("****************************************************************\n");
    if (failures == 0)
        printf("*      ALL PHASE 1D+1E TESTS PASSED ✓                        *\n");
    else
        printf("*      %d TEST(S) FAILED ✗                                   *\n", failures);
    printf("****************************************************************\n\n");

    return failures;
}