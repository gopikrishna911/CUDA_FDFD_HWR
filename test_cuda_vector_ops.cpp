/*=============================================================================
 * test_cuda_vector_ops.cpp
 *
 * Phase 1B verification: GPU vector operations
 * Tests: scale, axpy, weighted dot product, norm
 *============================================================================*/

#include "cuda_vector_ops.h"
#include "curlcurl_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

 /*=============================================================================
  * Test 1: Basic vector operations (scale, axpy, copy)
  *============================================================================*/
static int test_basic_ops(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 1: Basic vector operations (scale, axpy, copy)\n");
    printf("============================================================\n");

    int n = 100000;
    double* h_x = (double*)malloc(n * sizeof(double));
    double* h_y = (double*)malloc(n * sizeof(double));
    double* h_result = (double*)malloc(n * sizeof(double));

    /* Initialize */
    for (int i = 0; i < n; i++) {
        h_x[i] = sin(0.01 * i);
        h_y[i] = cos(0.01 * i);
    }

    /* Allocate GPU vectors */
    double* d_x, * d_y;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_alloc(&d_y, n);
    gpu_vector_to_device(d_x, h_x, n);
    gpu_vector_to_device(d_y, h_y, n);

    /* Test scale: x = 2.5 * x */
    gpu_vec_scale(d_x, 2.5, n);
    gpu_vector_to_host(h_result, d_x, n);

    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double expected = 2.5 * sin(0.01 * i);
        double err = fabs(h_result[i] - expected);
        if (err > max_err) max_err = err;
    }
    printf("  Scale error: %.3e\n", max_err);

    /* Reset d_x */
    gpu_vector_to_device(d_x, h_x, n);

    /* Test axpy: y = 3.0 * x + y */
    gpu_vec_axpy(3.0, d_x, d_y, n);
    gpu_vector_to_host(h_result, d_y, n);

    max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double expected = 3.0 * sin(0.01 * i) + cos(0.01 * i);
        double err = fabs(h_result[i] - expected);
        if (err > max_err) max_err = err;
    }
    printf("  AXPY error: %.3e\n", max_err);

    /* Test copy */
    gpu_vector_to_device(d_x, h_x, n);
    gpu_vec_copy(d_x, d_y, n);
    gpu_vector_to_host(h_result, d_y, n);

    max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs(h_result[i] - h_x[i]);
        if (err > max_err) max_err = err;
    }
    printf("  Copy error: %.3e\n", max_err);

    int passed = (max_err < 1e-14);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    gpu_vector_free(d_x);
    gpu_vector_free(d_y);
    free(h_x);
    free(h_y);
    free(h_result);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 2: Weighted dot product vs CPU
 *============================================================================*/
static int test_weighted_dot(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 2: Weighted dot product (GPU vs CPU)\n");
    printf("============================================================\n");

    /* Setup grid and operator */
    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);

    int n = op.n_total;
    printf("  Grid: Nr=%d, Nphi=%d, Nz=%d\n", grid.Nr, grid.Nphi, grid.Nz);
    printf("  DOFs: %d\n", n);

    /* Create two test vectors on CPU */
    double* h_x = (double*)malloc(n * sizeof(double));
    double* h_y = (double*)malloc(n * sizeof(double));

    /* Fill with TEM-like pattern */
    vec_zero(h_x, n);
    vec_zero(h_y, n);

    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                int idx = op.offset_Er + idx_Er(&grid, i, j, k);
                h_x[idx] = sin(M_PI * z / grid.L) / r;
                h_y[idx] = sin(M_PI * z / grid.L) / (r * r);
            }
        }
    }

    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        for (int j = 0; j < grid.Nphi; j++) {
            double phi = (j + 0.5) * grid.dphi;
            for (int i = 0; i <= grid.Nr; i++) {
                double r = r_at_i(&grid, i);
                int idx_ep = op.offset_Ephi + idx_Ephi(&grid, i, j, k);
                h_x[idx_ep] = cos(phi) * sin(M_PI * z / grid.L) / (r + 0.01);
                h_y[idx_ep] = sin(phi) * cos(M_PI * z / grid.L) / (r + 0.01);
            }
        }
    }

    /* CPU weighted dot product */
    double cpu_dot = vec_dot_product_weighted(h_x, h_y, &op);

    /* GPU weighted dot product */
    double* d_x, * d_y;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_alloc(&d_y, n);
    gpu_vector_to_device(d_x, h_x, n);
    gpu_vector_to_device(d_y, h_y, n);

    double gpu_dot;
    gpu_vec_dot_weighted(d_x, d_y, &gpu_dot, &op);

    double rel_err = fabs(gpu_dot - cpu_dot) / (fabs(cpu_dot) + 1e-30);
    printf("  CPU dot: %.15e\n", cpu_dot);
    printf("  GPU dot: %.15e\n", gpu_dot);
    printf("  Relative error: %.3e\n", rel_err);

    /* Also test self-dot (for norm) */
    double cpu_norm_sq = vec_dot_product_weighted(h_x, h_x, &op);
    double gpu_norm_sq;
    gpu_vec_dot_weighted(d_x, d_x, &gpu_norm_sq, &op);

    double norm_rel_err = fabs(gpu_norm_sq - cpu_norm_sq) / (fabs(cpu_norm_sq) + 1e-30);
    printf("  CPU norm²: %.15e\n", cpu_norm_sq);
    printf("  GPU norm²: %.15e\n", gpu_norm_sq);
    printf("  Norm² relative error: %.3e\n", norm_rel_err);

    /* Acceptable tolerance: different summation order gives ~1e-10 for doubles */
    int passed = (rel_err < 1e-10) && (norm_rel_err < 1e-10);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    gpu_vector_free(d_x);
    gpu_vector_free(d_y);
    free(h_x);
    free(h_y);
    curlcurl_op_free(&op);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 3: Weighted norm and normalize
 *============================================================================*/
static int test_weighted_norm(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 3: Weighted norm and normalize\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);

    int n = op.n_total;

    /* Create TEM-like vector */
    double* h_x = (double*)malloc(n * sizeof(double));
    vec_zero(h_x, n);

    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                int idx = op.offset_Er + idx_Er(&grid, i, j, k);
                h_x[idx] = sin(M_PI * z / grid.L) / r;
            }
        }
    }

    double cpu_norm = vec_norm2_weighted(h_x, &op);

    /* GPU norm */
    double* d_x;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_to_device(d_x, h_x, n);

    double gpu_norm;
    gpu_vec_norm_weighted(d_x, &gpu_norm, &op);

    double rel_err = fabs(gpu_norm - cpu_norm) / cpu_norm;
    printf("  CPU norm: %.15e\n", cpu_norm);
    printf("  GPU norm: %.15e\n", gpu_norm);
    printf("  Relative error: %.3e\n", rel_err);

    /* Test normalize: after normalization, norm should be 1.0 */
    gpu_vec_normalize_weighted(d_x, &op);

    double norm_after;
    gpu_vec_norm_weighted(d_x, &norm_after, &op);
    printf("  Norm after normalize: %.15e (should be 1.0)\n", norm_after);
    printf("  Error from 1.0: %.3e\n", fabs(norm_after - 1.0));

    int passed = (rel_err < 1e-10) && (fabs(norm_after - 1.0) < 1e-12);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    gpu_vector_free(d_x);
    free(h_x);
    curlcurl_op_free(&op);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Test 4: Workspace-based dot product (performance test)
 *============================================================================*/
static int test_workspace_dot(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 4: Workspace-based dot product + timing\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);

    int n = op.n_total;

    /* Create random vectors */
    double* h_x = (double*)malloc(n * sizeof(double));
    vec_random(h_x, n, 12345);

    double* d_x;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_to_device(d_x, h_x, n);

    /* Initialize workspace */
    ReductionWorkspace ws;
    reduction_workspace_init(&ws, n);

    /* Warm up */
    double result;
    gpu_vec_dot_weighted_ws(d_x, d_x, &result, &op, &ws);

    /* Time many iterations (simulating MINRES inner loop) */
    int num_iter = 1000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iter; i++) {
        gpu_vec_dot_weighted_ws(d_x, d_x, &result, &op, &ws);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("  %d weighted dot products in %.1f ms\n", num_iter, ms);
    printf("  Average: %.3f ms per dot product\n", ms / num_iter);
    printf("  Throughput: %.1f Gdof/s\n",
        (double)n * num_iter / (ms * 1e-3) / 1e9);

    /* Also time without workspace (with malloc each call) */
    cudaEventRecord(start);
    for (int i = 0; i < num_iter; i++) {
        gpu_vec_dot_weighted(d_x, d_x, &result, &op);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_no_ws = 0;
    cudaEventElapsedTime(&ms_no_ws, start, stop);

    printf("\n  Without workspace: %.3f ms per dot\n", ms_no_ws / num_iter);
    printf("  Workspace speedup: %.1fx\n", ms_no_ws / ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("  PASSED ✓\n");

    reduction_workspace_free(&ws);
    gpu_vector_free(d_x);
    free(h_x);
    curlcurl_op_free(&op);

    return 0;
}

/*=============================================================================
 * Test 5: Orthogonality test (dot product of orthogonal fields = 0)
 *============================================================================*/
static int test_orthogonality(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 5: Orthogonality test\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);

    int n = op.n_total;

    /* Create two orthogonal vectors:
     * x: Er = sin(pi*z/L) / r  (TEM mode n=1)
     * y: Er = sin(2*pi*z/L) / r  (TEM mode n=2)
     * These should be orthogonal under r-weighted inner product */
    double* h_x = (double*)malloc(n * sizeof(double));
    double* h_y = (double*)malloc(n * sizeof(double));
    vec_zero(h_x, n);
    vec_zero(h_y, n);

    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                int idx = op.offset_Er + idx_Er(&grid, i, j, k);
                h_x[idx] = sin(M_PI * z / grid.L) / r;
                h_y[idx] = sin(2.0 * M_PI * z / grid.L) / r;
            }
        }
    }

    /* CPU dot products */
    double cpu_xx = vec_dot_product_weighted(h_x, h_x, &op);
    double cpu_yy = vec_dot_product_weighted(h_y, h_y, &op);
    double cpu_xy = vec_dot_product_weighted(h_x, h_y, &op);

    /* GPU dot products */
    double* d_x, * d_y;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_alloc(&d_y, n);
    gpu_vector_to_device(d_x, h_x, n);
    gpu_vector_to_device(d_y, h_y, n);

    double gpu_xx, gpu_yy, gpu_xy;
    gpu_vec_dot_weighted(d_x, d_x, &gpu_xx, &op);
    gpu_vec_dot_weighted(d_y, d_y, &gpu_yy, &op);
    gpu_vec_dot_weighted(d_x, d_y, &gpu_xy, &op);

    printf("  <x,x> CPU: %.15e  GPU: %.15e\n", cpu_xx, gpu_xx);
    printf("  <y,y> CPU: %.15e  GPU: %.15e\n", cpu_yy, gpu_yy);
    printf("  <x,y> CPU: %.15e  GPU: %.15e\n", cpu_xy, gpu_xy);
    printf("  <x,y> / sqrt(<x,x>*<y,y>): %.3e (should be ~0)\n",
        fabs(gpu_xy) / sqrt(gpu_xx * gpu_yy));

    int passed = (fabs(gpu_xy) / sqrt(gpu_xx * gpu_yy) < 1e-10) &&
        (fabs(gpu_xx - cpu_xx) / cpu_xx < 1e-10) &&
        (fabs(gpu_yy - cpu_yy) / cpu_yy < 1e-10);
    printf("  %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    gpu_vector_free(d_x);
    gpu_vector_free(d_y);
    free(h_x);
    free(h_y);
    curlcurl_op_free(&op);

    return passed ? 0 : -1;
}

/*=============================================================================
 * Main
 *============================================================================*/
int main(void) {
    printf("\n");
    printf("****************************************************************\n");
    printf("*        CUDA PHASE 1B: VECTOR OPERATIONS TESTS               *\n");
    printf("****************************************************************\n");

    int failures = 0;

    failures += (test_basic_ops() != 0);
    failures += (test_weighted_dot() != 0);
    failures += (test_weighted_norm() != 0);
    failures += (test_workspace_dot() != 0);
    failures += (test_orthogonality() != 0);

    printf("\n");
    printf("****************************************************************\n");
    if (failures == 0)
        printf("*        ALL PHASE 1B TESTS PASSED ✓                         *\n");
    else
        printf("*        %d TEST(S) FAILED ✗                                 *\n", failures);
    printf("****************************************************************\n\n");

    return failures;
}