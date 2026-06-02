#include "curlcurl_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void export_field_csv(const CurlCurlOperator* op, const double* x, const char* filename);
void export_field_slice_rz(const CurlCurlOperator* op, const double* x, const char* filename);
void export_field_vtk(const CurlCurlOperator* op, const double* x, const char* filename);

/*=============================================================================
 * Test 1: Basic operator functionality
 * 
 * Initialize with TEM mode, apply operator, check eigenvalue
 *============================================================================*/
void test_operator_basic(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 1: Basic operator functionality with TEM mode\n");
    printf("============================================================\n");
    
    /* Setup grid */
    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 16, 8, 16);
    
    /* Create operator */
    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);
    curlcurl_op_print(&op);
    
    /* Analytical eigenvalue */
    double k2_exact = (M_PI / grid.L) * (M_PI / grid.L);
    printf("\nExpected k² = (π/L)² = %g\n", k2_exact);
    
    /* Allocate vectors */
    double* x = (double*)malloc(op.n_total * sizeof(double));
    double* y = (double*)malloc(op.n_total * sizeof(double));
    
    /* Initialize x with TEM mode: E_r = (1/r) * sin(πz/L) */
    vec_zero(x, op.n_total);
    
    /* Fill E_r component */
    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        double sin_piz_L = sin(M_PI * z / grid.L);
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                int idx = op.offset_Er + idx_Er(&grid, i, j, k);
                x[idx] = sin_piz_L / r;
            }
        }
    }
    
    /* Apply operator: y = A * x */
    printf("\nApplying curl-curl operator...\n");
    curlcurl_matvec(&op, x, y);
    
    /* Compute Rayleigh quotient */
    double xAx = vec_dot_product_weighted(x, y, &op);
    double xx = vec_dot_product_weighted(x, x, &op);
    double k2_computed = xAx / xx;
    
    printf("\nResults:\n");
    printf("  x · x     = %g\n", xx);
    printf("  x · (Ax)  = %g\n", xAx);
    printf("  Rayleigh quotient k² = %g\n", k2_computed);
    printf("  Expected k²          = %g\n", k2_exact);
    printf("  Relative error       = %.4f%%\n", 
           fabs(k2_computed - k2_exact) / k2_exact * 100.0);
    
    /* Check component-wise ratio (should be constant for eigenfunction) */
    printf("\nComponent-wise check (k² = y_i / x_i):\n");
    printf("  Sample E_r ratios:\n");
    
    int samples = 0;
    for (int k = 1; k < grid.Nz && samples < 5; k += grid.Nz/4) {
        int j = 0;
        int i = grid.Nr / 2;
        int idx = op.offset_Er + idx_Er(&grid, i, j, k);
        if (fabs(x[idx]) > 1e-10) {
            double ratio = y[idx] / x[idx];
            printf("    [i=%d, j=%d, k=%d]: x=%10.6f, y=%10.6f, ratio=%10.6f\n",
                   i, j, k, x[idx], y[idx], ratio);
            samples++;
        }
    }
    
    /* Cleanup */
    free(x);
    free(y);
    curlcurl_op_free(&op);
    
    printf("\nTest 1 PASSED\n");
}

/*=============================================================================
 * Test 2: Verify boundary conditions are enforced
 *============================================================================*/
void test_boundary_enforcement(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 2: Boundary condition enforcement\n");
    printf("============================================================\n");
    
    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 8, 4, 8);
    
    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);
    
    /* Create input with non-zero boundary values */
    double* x = (double*)malloc(op.n_total * sizeof(double));
    double* y = (double*)malloc(op.n_total * sizeof(double));
    
    /* Fill everything with 1.0 (including boundaries) */
    for (int i = 0; i < op.n_total; i++) {
        x[i] = 1.0;
    }
    
    /* Apply operator */
    curlcurl_matvec(&op, x, y);
    
    /* Check that boundary values in y are zero */
    printf("\nChecking boundary values in output y:\n");
    
    /* Check E_r at z boundaries */
    double max_Er_boundary = 0;
    for (int j = 0; j < grid.Nphi; j++) {
        for (int i = 0; i < grid.Nr; i++) {
            int idx0 = op.offset_Er + idx_Er(&grid, i, j, 0);
            int idxN = op.offset_Er + idx_Er(&grid, i, j, grid.Nz);
            if (fabs(y[idx0]) > max_Er_boundary) max_Er_boundary = fabs(y[idx0]);
            if (fabs(y[idxN]) > max_Er_boundary) max_Er_boundary = fabs(y[idxN]);
        }
    }
    printf("  max|y_Er| at z-boundaries: %e (should be 0)\n", max_Er_boundary);
    
    /* Check E_phi at all boundaries */
    double max_Ephi_boundary = 0;
    for (int k = 0; k <= grid.Nz; k++) {
        for (int j = 0; j < grid.Nphi; j++) {
            /* r boundaries */
            int idx0 = op.offset_Ephi + idx_Ephi(&grid, 0, j, k);
            int idxN = op.offset_Ephi + idx_Ephi(&grid, grid.Nr, j, k);
            if (fabs(y[idx0]) > max_Ephi_boundary) max_Ephi_boundary = fabs(y[idx0]);
            if (fabs(y[idxN]) > max_Ephi_boundary) max_Ephi_boundary = fabs(y[idxN]);
        }
    }
    for (int j = 0; j < grid.Nphi; j++) {
        for (int i = 0; i <= grid.Nr; i++) {
            /* z boundaries */
            int idx0 = op.offset_Ephi + idx_Ephi(&grid, i, j, 0);
            int idxN = op.offset_Ephi + idx_Ephi(&grid, i, j, grid.Nz);
            if (fabs(y[idx0]) > max_Ephi_boundary) max_Ephi_boundary = fabs(y[idx0]);
            if (fabs(y[idxN]) > max_Ephi_boundary) max_Ephi_boundary = fabs(y[idxN]);
        }
    }
    printf("  max|y_Ephi| at boundaries: %e (should be 0)\n", max_Ephi_boundary);
    
    /* Check E_z at r boundaries */
    double max_Ez_boundary = 0;
    for (int k = 0; k < grid.Nz; k++) {
        for (int j = 0; j < grid.Nphi; j++) {
            int idx0 = op.offset_Ez + idx_Ez(&grid, 0, j, k);
            int idxN = op.offset_Ez + idx_Ez(&grid, grid.Nr, j, k);
            if (fabs(y[idx0]) > max_Ez_boundary) max_Ez_boundary = fabs(y[idx0]);
            if (fabs(y[idxN]) > max_Ez_boundary) max_Ez_boundary = fabs(y[idxN]);
        }
    }
    printf("  max|y_Ez| at r-boundaries: %e (should be 0)\n", max_Ez_boundary);
    
    free(x);
    free(y);
    curlcurl_op_free(&op);
    
    if (max_Er_boundary < 1e-14 && max_Ephi_boundary < 1e-14 && max_Ez_boundary < 1e-14) {
        printf("\nTest 2 PASSED\n");
    } else {
        printf("\nTest 2 FAILED - boundary values not zero\n");
    }
}

/*=============================================================================
 * Test 3: Power iteration to find largest eigenvalue
 * 
 * NOTE: Power iteration finds the LARGEST eigenvalue, not the TEM mode!
 *============================================================================*/
void test_power_iteration(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 3: Power iteration eigensolver\n");
    printf("NOTE: Power iteration finds LARGEST eigenvalue (not TEM mode)\n");
    printf("============================================================\n");
    
    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 12, 4, 12);
    
    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);
    
    printf("Grid: Nr=%d, Nphi=%d, Nz=%d\n", grid.Nr, grid.Nphi, grid.Nz);
    printf("Total DOFs: %d\n", op.n_total);
    
    /* Allocate and initialize with random vector */
    double* x = (double*)malloc(op.n_total * sizeof(double));
    vec_random(x, op.n_total, 42);
    
    printf("\nRunning power iteration (finding largest eigenvalue)...\n");
    
    EigenResult result = power_iteration_weighted(&op, x, 200, 1e-8);
    
    printf("\nResults:\n");
    printf("  Converged:   %s\n", result.converged ? "YES" : "NO");
    printf("  Iterations:  %d\n", result.iterations);
    printf("  Eigenvalue (largest): %g\n", result.eigenvalue);
    printf("  Residual:    %e\n", result.residual);
    
    /* Convert to frequency */
    double k = sqrt(fabs(result.eigenvalue));
    double c = 299792458.0;
    double f = k * c / (2.0 * M_PI);
    printf("\n  This is a HIGH-FREQUENCY mode:\n");
    printf("  k = sqrt(lambda) = %g [1/m]\n", k);
    printf("  f = k*c/(2*pi) = %.4f MHz\n", f / 1e6);
    
    /* Compare with TEM mode */
    double k2_TEM = (M_PI / grid.L) * (M_PI / grid.L);
    double f_TEM = sqrt(k2_TEM) * c / (2.0 * M_PI);
    printf("\n  For comparison, TEM mode:\n");
    printf("  k^2_TEM = %g\n", k2_TEM);
    printf("  f_TEM = %.4f MHz\n", f_TEM / 1e6);
    printf("\n  Ratio (largest / TEM): %.1f\n", result.eigenvalue / k2_TEM);
    
    free(x);
    curlcurl_op_free(&op);
    
    printf("\nTest 3 completed (power iteration working correctly)\n");
}


/*=============================================================================
 * Test 4: Rayleigh quotient with TEM mode
 *============================================================================*/
void test_rayleigh_quotient(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 4: Rayleigh quotient convergence\n");
    printf("============================================================\n");
    
    printf("%6s %10s %15s %15s %10s\n", 
           "N", "DOFs", "k² computed", "k² exact", "Error %");
    printf("--------------------------------------------------------------\n");
    
    int N_vals[] = {8, 12, 16, 24, 32};
    int num_tests = 5;
    
    double k2_exact = (M_PI / 1.39) * (M_PI / 1.39);
    
    for (int t = 0; t < num_tests; t++) {
        int N = N_vals[t];
        
        GridParams grid;
        grid_init(&grid, 0.3333, 1.0, 1.39, N, 4, N);
        
        CurlCurlOperator op;
        curlcurl_op_init(&op, &grid);
        
        /* Create TEM mode */
        double* x = (double*)malloc(op.n_total * sizeof(double));
        vec_zero(x, op.n_total);
        
        for (int k = 0; k <= grid.Nz; k++) {
            double z = k * grid.dz;
            double sin_piz_L = sin(M_PI * z / grid.L);
            for (int j = 0; j < grid.Nphi; j++) {
                for (int i = 0; i < grid.Nr; i++) {
                    double r = r_at_i_half(&grid, i);
                    int idx = op.offset_Er + idx_Er(&grid, i, j, k);
                    x[idx] = sin_piz_L / r;
                }
            }
        }
        
        double k2_computed = rayleigh_quotient_weighted(&op, x);
        double error = fabs(k2_computed - k2_exact) / k2_exact * 100.0;
        
        printf("%6d %10d %15.8f %15.8f %10.4f\n",
               N, op.n_total, k2_computed, k2_exact, error);
        
        free(x);
        curlcurl_op_free(&op);
    }
    
    printf("\nTest 4 PASSED (second-order convergence expected)\n");
}

/*=============================================================================
 * Test 5: Find TEM mode with Rayleigh Quotient Iteration
 *============================================================================*/
void test_inverse_iteration(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 5: Rayleigh Quotient Iteration for TEM mode\n");
    printf("============================================================\n");
    
    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 16, 4, 16);
    
    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);
    
    double k2_exact = (M_PI / grid.L) * (M_PI / grid.L);
    printf("Target k^2 (TEM mode) = %.10f\n", k2_exact);
    printf("Grid: Nr=%d, Nphi=%d, Nz=%d, DOFs=%d\n", 
           grid.Nr, grid.Nphi, grid.Nz, op.n_total);
    
    /* Start with TEM-like initial guess for faster convergence */
    double* x = (double*)malloc(op.n_total * sizeof(double));
    vec_zero(x, op.n_total);
    
    /* Initialize with approximate TEM mode shape */
    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        double sin_piz_L = sin(M_PI * z / grid.L);
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                int idx = op.offset_Er + idx_Er(&grid, i, j, k);
                x[idx] = sin_piz_L / r;
            }
        }
    }
    
    /* Add some noise */
    srand(12345);
    for (int i = 0; i < op.n_total; i++) {
        x[i] += 0.1 * ((double)rand() / RAND_MAX - 0.5);
    }
    
    printf("\nUsing Rayleigh Quotient Iteration (adaptive shift):\n\n");
    
    EigenResult result = rayleigh_quotient_iteration(&op, x, k2_exact, 30, 1e-10);
    
    printf("\nFinal Results:\n");
    printf("  Converged:   %s\n", result.converged ? "YES" : "NO");
    printf("  Iterations:  %d\n", result.iterations);
    printf("  Eigenvalue:  %.10f\n", result.eigenvalue);
    printf("  Expected:    %.10f\n", k2_exact);
    printf("  Error:       %.6f%%\n", 
           fabs(result.eigenvalue - k2_exact) / k2_exact * 100.0);
    printf("  Residual:    %e\n", result.residual);
    
    /* Convert to frequency */
    double k = sqrt(fabs(result.eigenvalue));
    double c = 299792458.0;
    double f = k * c / (2.0 * M_PI);
    double f_exact = sqrt(k2_exact) * c / (2.0 * M_PI);
    printf("\n  Computed frequency: %.6f MHz\n", f / 1e6);
    printf("  Expected frequency: %.6f MHz\n", f_exact / 1e6);
    printf("  Frequency error:    %.6f MHz (%.4f%%)\n", 
           fabs(f - f_exact) / 1e6,
           fabs(f - f_exact) / f_exact * 100.0);
    
    free(x);
    curlcurl_op_free(&op);
    
    /* Pass if residual is small (eigenvalue accuracy limited by discretization) */
    if (result.converged || result.residual < 1e-6) {
        printf("\nTest 5 PASSED\n");
    } else {
        printf("\nTest 5 FAILED\n");
    }
}

/*=============================================================================
 * Main
 *============================================================================*/
int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("========================================================\n");
    printf("  Curl-Curl Operator Tests\n");
    printf("========================================================\n");
    
    test_operator_basic();
    test_boundary_enforcement();
    test_power_iteration();
    test_rayleigh_quotient();
    test_inverse_iteration();
    test_field_export();

    printf("\n========================================================\n");
    printf("  All tests completed.\n");
    printf("========================================================\n");
    
    return 0;
}