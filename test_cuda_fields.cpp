/*=============================================================================
 * test_cuda_fields.c
 *
 * Phase 1A verification: GPU memory management
 * Tests: alloc, copy to device, copy back, verify match
 *============================================================================*/

#include "cuda_fields.h"
#include "curlcurl_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

 /*=============================================================================
  * Helper: fill E-field with known pattern
  *============================================================================*/
static void fill_efield_pattern(EField* E, const GridParams* grid) {
    /* Er = sin(pi*r) * cos(phi) * sin(pi*z/L) */
    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            for (int i = 0; i < grid->Nr; i++) {
                double r = r_at_i_half(grid, i);
                int idx = idx_Er(grid, i, j, k);
                E->Er[idx] = sin(M_PI * r) * cos(phi) * sin(M_PI * z / grid->L);
            }
        }
    }

    /* Ephi = cos(pi*r) * sin(phi) * cos(pi*z/L) */
    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            for (int i = 0; i <= grid->Nr; i++) {
                double r = r_at_i(grid, i);
                int idx = idx_Ephi(grid, i, j, k);
                E->Ephi[idx] = cos(M_PI * r) * sin(phi) * cos(M_PI * z / grid->L);
            }
        }
    }

    /* Ez = r * sin(2*phi) * cos(pi*z/L) */
    for (int k = 0; k < grid->Nz; k++) {
        double z = (k + 0.5) * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            double phi = j * grid->dphi;
            for (int i = 0; i <= grid->Nr; i++) {
                double r = r_at_i(grid, i);
                int idx = idx_Ez(grid, i, j, k);
                E->Ez[idx] = r * sin(2.0 * phi) * cos(M_PI * z / grid->L);
            }
        }
    }
}

/*=============================================================================
 * Helper: compare two E-fields, return max absolute error
 *============================================================================*/
static double compare_efields(const EField* A, const EField* B) {
    double max_err = 0.0;

    for (int i = 0; i < A->size_Er; i++) {
        double err = fabs(A->Er[i] - B->Er[i]);
        if (err > max_err) max_err = err;
    }
    for (int i = 0; i < A->size_Ephi; i++) {
        double err = fabs(A->Ephi[i] - B->Ephi[i]);
        if (err > max_err) max_err = err;
    }
    for (int i = 0; i < A->size_Ez; i++) {
        double err = fabs(A->Ez[i] - B->Ez[i]);
        if (err > max_err) max_err = err;
    }

    return max_err;
}

/*=============================================================================
 * Test 1: GPU Device Info
 *============================================================================*/
static int test_device_info(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 1: GPU Device Information\n");
    printf("============================================================\n");

    cuda_print_device_info();
    cuda_print_memory_info("initial");

    return 0;
}

/*=============================================================================
 * Test 2: E-field round-trip (CPU → GPU → CPU)
 *============================================================================*/
static int test_efield_roundtrip(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 2: E-field round-trip (CPU → GPU → CPU)\n");
    printf("============================================================\n");

    /* Setup grid */
    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    printf("  Grid: Nr=%d, Nphi=%d, Nz=%d\n", grid.Nr, grid.Nphi, grid.Nz);

    /* Allocate CPU fields */
    EField E_original, E_returned;
    efield_alloc(&E_original, &grid);
    efield_alloc(&E_returned, &grid);

    /* Fill with known pattern */
    fill_efield_pattern(&E_original, &grid);

    printf("  E-field sizes: Er=%d, Ephi=%d, Ez=%d\n",
        E_original.size_Er, E_original.size_Ephi, E_original.size_Ez);

    /* Allocate GPU field */
    GPU_EField d_E;
    cuda_print_memory_info("before alloc");

    if (gpu_efield_alloc(&d_E, &grid) != 0) {
        printf("  FAILED: GPU allocation\n");
        return -1;
    }
    cuda_print_memory_info("after alloc");

    /* CPU → GPU */
    if (gpu_efield_to_device(&d_E, &E_original) != 0) {
        printf("  FAILED: Copy to device\n");
        return -1;
    }

    /* GPU → CPU */
    if (gpu_efield_to_host(&E_returned, &d_E) != 0) {
        printf("  FAILED: Copy to host\n");
        return -1;
    }

    /* Compare */
    double max_err = compare_efields(&E_original, &E_returned);
    printf("  Max round-trip error: %.3e\n", max_err);

    if (max_err == 0.0) {
        printf("  PASSED ✓ (exact bit-for-bit match)\n");
    }
    else {
        printf("  FAILED ✗\n");
        return -1;
    }

    /* Cleanup */
    gpu_efield_free(&d_E);
    efield_free(&E_original);
    efield_free(&E_returned);

    cuda_print_memory_info("after free");
    return 0;
}

/*=============================================================================
 * Test 3: H-field round-trip
 *============================================================================*/
static int test_hfield_roundtrip(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 3: H-field round-trip (CPU → GPU → CPU)\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);

    /* Allocate and fill H-field with pattern */
    HField H_original, H_returned;
    hfield_alloc(&H_original, &grid);
    hfield_alloc(&H_returned, &grid);

    for (int i = 0; i < H_original.size_Hr; i++)
        H_original.Hr[i] = sin(0.1 * i);
    for (int i = 0; i < H_original.size_Hphi; i++)
        H_original.Hphi[i] = cos(0.2 * i);
    for (int i = 0; i < H_original.size_Hz; i++)
        H_original.Hz[i] = sin(0.3 * i) * cos(0.1 * i);

    printf("  H-field sizes: Hr=%d, Hphi=%d, Hz=%d\n",
        H_original.size_Hr, H_original.size_Hphi, H_original.size_Hz);

    /* Round-trip */
    GPU_HField d_H;
    if (gpu_hfield_alloc(&d_H, &grid) != 0) return -1;
    if (gpu_hfield_to_device(&d_H, &H_original) != 0) return -1;
    if (gpu_hfield_to_host(&H_returned, &d_H) != 0) return -1;

    /* Compare */
    double max_err = 0.0;
    for (int i = 0; i < H_original.size_Hr; i++) {
        double err = fabs(H_original.Hr[i] - H_returned.Hr[i]);
        if (err > max_err) max_err = err;
    }
    for (int i = 0; i < H_original.size_Hphi; i++) {
        double err = fabs(H_original.Hphi[i] - H_returned.Hphi[i]);
        if (err > max_err) max_err = err;
    }
    for (int i = 0; i < H_original.size_Hz; i++) {
        double err = fabs(H_original.Hz[i] - H_returned.Hz[i]);
        if (err > max_err) max_err = err;
    }

    printf("  Max round-trip error: %.3e\n", max_err);
    if (max_err == 0.0)
        printf("  PASSED ✓\n");
    else {
        printf("  FAILED ✗\n");
        return -1;
    }

    gpu_hfield_free(&d_H);
    hfield_free(&H_original);
    hfield_free(&H_returned);
    return 0;
}

/*=============================================================================
 * Test 4: Packed vector round-trip with pack/unpack kernels
 *============================================================================*/
static int test_packed_vector_roundtrip(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 4: Packed vector round-trip with GPU pack/unpack\n");
    printf("============================================================\n");

    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 32, 128, 128);
    cuda_grid_init(&grid);

    /* Setup operator for offsets and sizes */
    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);

    printf("  Total DOFs: %d\n", op.n_total);
    printf("  Offsets: Er=%d, Ephi=%d, Ez=%d\n",
        op.offset_Er, op.offset_Ephi, op.offset_Ez);

    /* Create and fill CPU E-field */
    EField E_original, E_returned;
    efield_alloc(&E_original, &grid);
    efield_alloc(&E_returned, &grid);
    fill_efield_pattern(&E_original, &grid);

    /* Allocate GPU E-field and packed vector */
    GPU_EField d_E;
    double* d_x;
    gpu_efield_alloc(&d_E, &grid);
    gpu_vector_alloc(&d_x, op.n_total);

    /* CPU E-field → GPU E-field → GPU packed vector → GPU E-field → CPU E-field */

    /* Step 1: CPU → GPU E-field */
    gpu_efield_to_device(&d_E, &E_original);

    /* Step 2: GPU E-field → GPU packed vector */
    gpu_pack_field(&d_E, d_x, op.offset_Er, op.offset_Ephi, op.offset_Ez);

    /* Step 3: Zero out d_E to prove unpack works */
    gpu_efield_zero(&d_E);

    /* Step 4: GPU packed vector → GPU E-field */
    gpu_unpack_field(d_x, &d_E, op.offset_Er, op.offset_Ephi, op.offset_Ez);

    /* Step 5: GPU E-field → CPU */
    gpu_efield_to_host(&E_returned, &d_E);

    /* Compare */
    double max_err = compare_efields(&E_original, &E_returned);
    printf("  Max pack/unpack round-trip error: %.3e\n", max_err);

    if (max_err == 0.0)
        printf("  PASSED ✓ (exact bit-for-bit match)\n");
    else {
        printf("  FAILED ✗\n");
        return -1;
    }

    /* Also test: CPU pack → GPU vector → GPU unpack → GPU → CPU */
    double* h_x = (double*)malloc(op.n_total * sizeof(double));
    pack_field(&E_original, h_x, &op);                         /* CPU pack */
    gpu_vector_to_device(d_x, h_x, op.n_total);                /* CPU → GPU */
    gpu_efield_zero(&d_E);
    gpu_unpack_field(d_x, &d_E, op.offset_Er, op.offset_Ephi, op.offset_Ez);
    gpu_efield_to_host(&E_returned, &d_E);

    max_err = compare_efields(&E_original, &E_returned);
    printf("  Max CPU-pack → GPU-unpack error: %.3e\n", max_err);

    if (max_err == 0.0)
        printf("  PASSED ✓\n");
    else {
        printf("  FAILED ✗\n");
        return -1;
    }

    /* Cleanup */
    free(h_x);
    gpu_vector_free(d_x);
    gpu_efield_free(&d_E);
    efield_free(&E_original);
    efield_free(&E_returned);
    curlcurl_op_free(&op);
    return 0;
}

/*=============================================================================
 * Test 5: Memory stress test (large grid)
 *============================================================================*/
static int test_memory_large_grid(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 5: Large grid memory test (production size)\n");
    printf("============================================================\n");

    /* Production-size grid for 25mm apertures */
    int Nr = 64, Nphi = 512, Nz = 256;
    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, Nr, Nphi, Nz);

    int size_Er = Nr * Nphi * (Nz + 1);
    int size_Ephi = (Nr + 1) * Nphi * (Nz + 1);
    int size_Ez = (Nr + 1) * Nphi * Nz;
    int n_total = size_Er + size_Ephi + size_Ez;

    double mem_E = (size_Er + size_Ephi + size_Ez) * sizeof(double) / 1.0e6;
    //double mem_H_est = mem_E;  /* roughly same */

    printf("  Grid: Nr=%d, Nphi=%d, Nz=%d\n", Nr, Nphi, Nz);
    printf("  Total DOFs: %d (%.1f M)\n", n_total, n_total / 1.0e6);
    printf("  E-field memory: %.1f MB\n", mem_E);
    printf("  Estimated total (E + H + 6 work vectors): %.1f MB\n",
        mem_E * 8);

    cuda_print_memory_info("before large alloc");

    /* Try to allocate everything we'd need */
    GPU_EField d_E, d_result;
    GPU_HField d_H;
    double* d_x, * d_y;

    int ok = 1;
    if (gpu_efield_alloc(&d_E, &grid) != 0) { ok = 0; printf("  E alloc FAILED\n"); }
    if (gpu_efield_alloc(&d_result, &grid) != 0) { ok = 0; printf("  result alloc FAILED\n"); }
    if (gpu_hfield_alloc(&d_H, &grid) != 0) { ok = 0; printf("  H alloc FAILED\n"); }
    if (gpu_vector_alloc(&d_x, n_total) != 0) { ok = 0; printf("  x alloc FAILED\n"); }
    if (gpu_vector_alloc(&d_y, n_total) != 0) { ok = 0; printf("  y alloc FAILED\n"); }

    if (ok) {
        cuda_print_memory_info("after large alloc");
        printf("  PASSED ✓ (all allocations succeeded)\n");
    }

    /* Free everything */
    if (ok) {
        gpu_efield_free(&d_E);
        gpu_efield_free(&d_result);
        gpu_hfield_free(&d_H);
        gpu_vector_free(d_x);
        gpu_vector_free(d_y);
    }

    cuda_print_memory_info("after free");
    return ok ? 0 : -1;
}

/*=============================================================================
 * Main
 *============================================================================*/
int main(void) {
    printf("\n");
    printf("****************************************************************\n");
    printf("*          CUDA PHASE 1A: MEMORY MANAGEMENT TESTS             *\n");
    printf("****************************************************************\n");

    int failures = 0;

    failures += (test_device_info() != 0);
    failures += (test_efield_roundtrip() != 0);
    failures += (test_hfield_roundtrip() != 0);
    failures += (test_packed_vector_roundtrip() != 0);
    failures += (test_memory_large_grid() != 0);

    printf("\n");
    printf("****************************************************************\n");
    if (failures == 0)
        printf("*          ALL PHASE 1A TESTS PASSED ✓                        *\n");
    else
        printf("*          %d TEST(S) FAILED ✗                                *\n", failures);
    printf("****************************************************************\n\n");

    return failures;
}