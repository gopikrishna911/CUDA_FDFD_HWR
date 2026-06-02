#include "curl_E.h"
#include "curl_H.h"
#include <stdio.h>
#include <math.h>

/*=============================================================================
 * Test Case 1: TEM mode eigenfunction
 * 
 * E = (1/r) * sin(pi*z/L) * r_hat
 * 
 * curl E = (1/r) * (pi/L) * cos(pi*z/L) * phi_hat
 * 
 * curl(curl E) = (pi/L)^2 * (1/r) * sin(pi*z/L) * r_hat = (pi/L)^2 * E
 * 
 * So: curl(curl E) = k^2 * E  where k = pi/L
 *============================================================================*/
void test_TEM_curlcurl(GridParams* grid) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 1: TEM mode curl(curl E) eigenvalue\n");
    printf("E = (1/r) * sin(pi*z/L) * r_hat\n");
    printf("Expected: curl(curl E) = (pi/L)^2 * E\n");
    printf("============================================================\n");
    
    EField E, curlcurlE;
    efield_alloc(&E, grid);
    efield_alloc(&curlcurlE, grid);
    
    double L = grid->L;
    double k = M_PI / L;
    double k2 = k * k;
    
    printf("Expected k^2 = (pi/L)^2 = %g\n\n", k2);
    
    /* Set Er = (1/r) * sin(pi*z/L) at (i+1/2, j, k) */
    for (int kk = 0; kk <= grid->Nz; kk++) {
        double z = kk * grid->dz;
        double sin_piz_L = sin(M_PI * z / L);
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                double r = r_at_i_half(grid, i);
                E.Er[idx_Er(grid, i, j, kk)] = sin_piz_L / r;
            }
        }
    }
    
    /* Compute curl(curl E) */
    compute_curl_curl_E(&E, &curlcurlE, grid);
    
    /* Check that curl(curl E)_r = k^2 * E_r */
    printf("Checking (curl curl E)_r = k^2 * E_r:\n");
    printf("%5s %5s %5s %12s %12s %12s %12s\n", 
           "i", "j", "k", "E_r", "curlcurl_r", "k^2*E_r", "ratio");
    printf("----------------------------------------------------------------------\n");
    
    double sum_ratio = 0.0;
    int count = 0;
    
    for (int kk = 1; kk < grid->Nz; kk += grid->Nz/4) {  /* Skip boundaries */
        for (int j = 0; j < 1; j++) {  /* Just check j=0 */
            for (int i = 0; i < grid->Nr; i += grid->Nr/4) {
                int idx = idx_Er(grid, i, j, kk);
                double Er_val = E.Er[idx];
                double cc_r = curlcurlE.Er[idx];
                double expected = k2 * Er_val;
                double ratio = (fabs(Er_val) > 1e-10) ? cc_r / Er_val : 0.0;
                
                printf("%5d %5d %5d %12.6f %12.6f %12.6f %12.6f\n",
                       i, j, kk, Er_val, cc_r, expected, ratio);
                
                if (fabs(Er_val) > 1e-10) {
                    sum_ratio += ratio;
                    count++;
                }
            }
        }
    }
    
    double avg_ratio = sum_ratio / count;
    double rel_error = fabs(avg_ratio - k2) / k2 * 100.0;
    
    printf("----------------------------------------------------------------------\n");
    printf("Average ratio (curl curl E)_r / E_r = %g\n", avg_ratio);
    printf("Expected k^2 = %g\n", k2);
    printf("Relative error = %.4f%%\n", rel_error);
    
    /* Check that other components are small */
    double max_Ephi = 0, max_Ez = 0;
    double max_cc_phi = 0, max_cc_z = 0;
    
    for (int i = 0; i < E.size_Ephi; i++) {
        if (fabs(E.Ephi[i]) > max_Ephi) max_Ephi = fabs(E.Ephi[i]);
        if (fabs(curlcurlE.Ephi[i]) > max_cc_phi) max_cc_phi = fabs(curlcurlE.Ephi[i]);
    }
    for (int i = 0; i < E.size_Ez; i++) {
        if (fabs(E.Ez[i]) > max_Ez) max_Ez = fabs(E.Ez[i]);
        if (fabs(curlcurlE.Ez[i]) > max_cc_z) max_cc_z = fabs(curlcurlE.Ez[i]);
    }
    
    printf("\nOther components (should be ~0):\n");
    printf("  max|E_phi| = %e, max|(curl curl E)_phi| = %e\n", max_Ephi, max_cc_phi);
    printf("  max|E_z|   = %e, max|(curl curl E)_z|   = %e\n", max_Ez, max_cc_z);
    
    /* Add this to test_TEM_curlcurl */
    printf("\nChecking (curl curl E)_z at INTERIOR points only:\n");
    double max_cc_z_interior = 0;
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 1; i < grid->Nr; i++) {  /* Skip i=0 and i=Nr */
                int idx = idx_Ez(grid, i, j, k);
                if (fabs(curlcurlE.Ez[idx]) > max_cc_z_interior) {
                    max_cc_z_interior = fabs(curlcurlE.Ez[idx]);
                }
            }
        }
    }
    printf("max|(curl curl E)_z| at interior = %e\n", max_cc_z_interior);

    efield_free(&E);
    efield_free(&curlcurlE);
}

/*=============================================================================
 * Test Case 2: Uniform field (should give zero curl-curl)
 * 
 * E = E0 * z_hat (uniform)
 * curl E = 0
 * curl(curl E) = 0
 *============================================================================*/
void test_uniform_curlcurl(GridParams* grid) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 2: Uniform field curl(curl E)\n");
    printf("E = E0 * z_hat (uniform)\n");
    printf("Expected: curl(curl E) = 0\n");
    printf("============================================================\n");
    
    EField E, curlcurlE;
    efield_alloc(&E, grid);
    efield_alloc(&curlcurlE, grid);
    
    double E0 = 1.0;
    
    /* Set Ez = E0 everywhere */
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++) {
                E.Ez[idx_Ez(grid, i, j, k)] = E0;
            }
        }
    }
    
    compute_curl_curl_E(&E, &curlcurlE, grid);
    
    /* Find max values */
    double max_cc_r = 0, max_cc_phi = 0, max_cc_z = 0;
    
    for (int i = 0; i < curlcurlE.size_Er; i++) {
        if (fabs(curlcurlE.Er[i]) > max_cc_r) max_cc_r = fabs(curlcurlE.Er[i]);
    }
    for (int i = 0; i < curlcurlE.size_Ephi; i++) {
        if (fabs(curlcurlE.Ephi[i]) > max_cc_phi) max_cc_phi = fabs(curlcurlE.Ephi[i]);
    }
    for (int i = 0; i < curlcurlE.size_Ez; i++) {
        if (fabs(curlcurlE.Ez[i]) > max_cc_z) max_cc_z = fabs(curlcurlE.Ez[i]);
    }
    
    printf("max|(curl curl E)_r|   = %e (should be ~0)\n", max_cc_r);
    printf("max|(curl curl E)_phi| = %e (should be ~0)\n", max_cc_phi);
    printf("max|(curl curl E)_z|   = %e (should be ~0)\n", max_cc_z);
    
    efield_free(&E);
    efield_free(&curlcurlE);
}

/*=============================================================================
 * Test Case 3: Vector identity verification
 * 
 * For E_z = sin(pi*r/(b-a)) * sin(pi*z/L), we can compute curl curl E
 * analytically and compare.
 * 
 * This tests all components working together.
 *============================================================================*/
void test_Ez_mode_curlcurl(GridParams* grid) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 3: TM-like mode with E_z\n");
    printf("E_z = sin(pi*(r-a)/(b-a)) * sin(pi*z/L)\n");
    printf("============================================================\n");
    
    EField E, curlcurlE;
    efield_alloc(&E, grid);
    efield_alloc(&curlcurlE, grid);
    
    double L = grid->L;
    double a = grid->a;
    double b = grid->b;
    double kr = M_PI / (b - a);
    double kz = M_PI / L;
    
    /* Set Ez = sin(kr*(r-a)) * sin(kz*z) at (i, j, k+1/2) */
    for (int k = 0; k < grid->Nz; k++) {
        double z = (k + 0.5) * grid->dz;
        double sin_kz_z = sin(kz * z);
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++) {
                double r = r_at_i(grid, i);
                double sin_kr_r = sin(kr * (r - a));
                E.Ez[idx_Ez(grid, i, j, k)] = sin_kr_r * sin_kz_z;
            }
        }
    }
    
    compute_curl_curl_E(&E, &curlcurlE, grid);
    
    /* For this mode, curl curl E should have only E_z component 
     * (approximately, since it's not an exact eigenmode due to 
     * the 1/r terms in cylindrical coordinates) */
    
    printf("\nSample curl(curl E)_z values:\n");
    printf("%5s %5s %5s %12s %12s\n", "i", "j", "k", "E_z", "curlcurl_z");
    printf("--------------------------------------------------\n");
    
    int j = 0;
    for (int k = grid->Nz/4; k < grid->Nz; k += grid->Nz/4) {
        for (int i = 1; i < grid->Nr; i += grid->Nr/4) {
            int idx = idx_Ez(grid, i, j, k);
            printf("%5d %5d %5d %12.6f %12.6f\n",
                   i, j, k, E.Ez[idx], curlcurlE.Ez[idx]);
        }
    }
    
    /* Check magnitude of other components */
    double max_cc_r = 0, max_cc_phi = 0, max_cc_z = 0;
    
    for (int i = 0; i < curlcurlE.size_Er; i++) {
        if (fabs(curlcurlE.Er[i]) > max_cc_r) max_cc_r = fabs(curlcurlE.Er[i]);
    }
    for (int i = 0; i < curlcurlE.size_Ephi; i++) {
        if (fabs(curlcurlE.Ephi[i]) > max_cc_phi) max_cc_phi = fabs(curlcurlE.Ephi[i]);
    }
    for (int i = 0; i < curlcurlE.size_Ez; i++) {
        if (fabs(curlcurlE.Ez[i]) > max_cc_z) max_cc_z = fabs(curlcurlE.Ez[i]);
    }
    
    printf("\nComponent magnitudes:\n");
    printf("  max|(curl curl E)_r|   = %e\n", max_cc_r);
    printf("  max|(curl curl E)_phi| = %e\n", max_cc_phi);
    printf("  max|(curl curl E)_z|   = %e\n", max_cc_z);
    
    efield_free(&E);
    efield_free(&curlcurlE);
}

/*=============================================================================
 * Test Case 4: Convergence test for TEM mode
 *============================================================================*/
void test_TEM_convergence(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 4: TEM mode convergence study\n");
    printf("============================================================\n");
    printf("%6s %10s %15s %10s\n", "N", "h", "Rel Error", "Order");
    printf("----------------------------------------------------\n");
    
    double prev_error = 0;
    double prev_h = 0;
    
    int N_values[] = {8, 16, 32, 64};
    int num_tests = 4;
    
    for (int t = 0; t < num_tests; t++) {
        int N = N_values[t];
        GridParams grid;
        grid_init(&grid, 0.3333, 1.0, 1.39, N, 4, N);
        
        EField E, curlcurlE;
        efield_alloc(&E, &grid);
        efield_alloc(&curlcurlE, &grid);
        
        double L = grid.L;
        double k2 = (M_PI / L) * (M_PI / L);
        
        /* Set TEM mode */
        for (int k = 0; k <= grid.Nz; k++) {
            double z = k * grid.dz;
            double sin_piz_L = sin(M_PI * z / L);
            for (int j = 0; j < grid.Nphi; j++) {
                for (int i = 0; i < grid.Nr; i++) {
                    double r = r_at_i_half(&grid, i);
                    E.Er[idx_Er(&grid, i, j, k)] = sin_piz_L / r;
                }
            }
        }
        
        compute_curl_curl_E(&E, &curlcurlE, &grid);
        
        /* Compute average ratio at interior points */
        double sum_ratio = 0.0;
        int count = 0;
        
        for (int k = 1; k < grid.Nz; k++) {
            for (int j = 0; j < grid.Nphi; j++) {
                for (int i = 0; i < grid.Nr; i++) {
                    int idx = idx_Er(&grid, i, j, k);
                    double Er_val = E.Er[idx];
                    if (fabs(Er_val) > 1e-10) {
                        double ratio = curlcurlE.Er[idx] / Er_val;
                        sum_ratio += ratio;
                        count++;
                    }
                }
            }
        }
        
        double avg_ratio = sum_ratio / count;
        double rel_error = fabs(avg_ratio - k2) / k2;
        double h = grid.dz;  /* Characteristic mesh size */
        
        double order = 0;
        if (prev_error > 0 && prev_h > 0) {
            order = log(prev_error / rel_error) / log(prev_h / h);
        }
        
        printf("%6d %10.6f %15.6e %10.2f\n", N, h, rel_error, order);
        
        prev_error = rel_error;
        prev_h = h;
        
        efield_free(&E);
        efield_free(&curlcurlE);
    }
}

/*=============================================================================
 * Main
 *============================================================================*/
int main(void) {
    printf("========================================================\n");
    printf("  Curl of Curl Verification Tests\n");
    printf("  Cylindrical Coordinates (r, phi, z)\n");
    printf("========================================================\n");
    
    /* Setup grid */
    GridParams grid;
    grid_init(&grid, 
              0.3333,   /* a: inner radius */
              1.0,      /* b: outer radius */
              1.39,     /* L: length */
              16,       /* Nr */
              8,        /* Nphi */
              16);      /* Nz */
    
    grid_print(&grid);
    
    /* Run tests */
    test_TEM_curlcurl(&grid);
    test_uniform_curlcurl(&grid);
    test_Ez_mode_curlcurl(&grid);
    test_TEM_convergence();
    
    printf("\n========================================================\n");
    printf("  All tests completed.\n");
    printf("========================================================\n");
    
    return 0;
}