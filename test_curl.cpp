#include "curl_E.h"
#include <stdio.h>
#include <math.h>

/*=============================================================================
 * Test Case 1: Uniform E_z field
 * 
 * E = E0 * z_hat
 * curl E = 0
 *============================================================================*/
void test_uniform_Ez(GridParams* grid) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 1: Uniform E_z field (E = E0 * z_hat)\n");
    printf("Expected: curl E = 0 everywhere\n");
    printf("============================================================\n");
    
    EField E;
    CurlE curl;
    efield_alloc(&E, grid);
    curl_alloc(&curl, grid);
    
    double E0 = 1.0;
    
    /* Set Ez = E0 everywhere */
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++) {
                E.Ez[idx_Ez(grid, i, j, k)] = E0;
            }
        }
    }
    
    compute_curl_E(&E, &curl, grid);
    
    /* Check max values */
    double max_curl_r = 0, max_curl_phi = 0, max_curl_z = 0;
    for (int i = 0; i < curl.size_curl_r; i++) {
        if (fabs(curl.curl_r[i]) > max_curl_r) max_curl_r = fabs(curl.curl_r[i]);
    }
    for (int i = 0; i < curl.size_curl_phi; i++) {
        if (fabs(curl.curl_phi[i]) > max_curl_phi) max_curl_phi = fabs(curl.curl_phi[i]);
    }
    for (int i = 0; i < curl.size_curl_z; i++) {
        if (fabs(curl.curl_z[i]) > max_curl_z) max_curl_z = fabs(curl.curl_z[i]);
    }
    
    printf("Max |curl_r|   = %e (should be ~0)\n", max_curl_r);
    printf("Max |curl_phi| = %e (should be ~0)\n", max_curl_phi);
    printf("Max |curl_z|   = %e (should be ~0)\n", max_curl_z);
    
    efield_free(&E);
    curl_free(&curl);
}

/*=============================================================================
 * Test Case 2: E_phi linear in z
 * 
 * E = (E0 * z) * phi_hat
 * curl E = E0 * r_hat
 *============================================================================*/
void test_Ephi_linear_z(GridParams* grid) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 2: E_phi linear in z (E = E0*z * phi_hat)\n");
    printf("Expected: curl E = E0 * r_hat\n");
    printf("============================================================\n");
    
    EField E;
    CurlE curl;
    efield_alloc(&E, grid);
    curl_alloc(&curl, grid);
    
    double E0 = 2.0;
    
    /* Set Ephi = E0 * z at (i, j+1/2, k) */
    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++) {
                E.Ephi[idx_Ephi(grid, i, j, k)] = E0 * z;
            }
        }
    }
    
    compute_curl_E(&E, &curl, grid);
    
    /* curl_r should be -dEphi/dz = -E0 */
    /* (Note: curl_r = (1/r)*dEz/dphi - dEphi/dz, with dEz/dphi = 0) */
    printf("\nSample curl_r values (should be %g):\n", -E0);
    for (int i = 0; i <= grid->Nr; i += grid->Nr/4) {
        int j = 0, k = grid->Nz / 2;
        if (k < grid->Nz) {
            printf("  curl_r[i=%d, j=%d, k=%d] = %g\n", 
                   i, j, k, curl.curl_r[idx_curl_r(grid, i, j, k)]);
        }
    }
    
    efield_free(&E);
    curl_free(&curl);
}

/*=============================================================================
 * Test Case 3: E_r linear in z
 * 
 * E = (E0 * z) * r_hat
 * curl E = -E0 * phi_hat
 *============================================================================*/
void test_Er_linear_z(GridParams* grid) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 3: E_r linear in z (E = E0*z * r_hat)\n");
    printf("Expected: curl E = -E0 * phi_hat\n");
    printf("============================================================\n");
    
    EField E;
    CurlE curl;
    efield_alloc(&E, grid);
    curl_alloc(&curl, grid);
    
    double E0 = 3.0;
    
    /* Set Er = E0 * z at (i+1/2, j, k) */
    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                E.Er[idx_Er(grid, i, j, k)] = E0 * z;
            }
        }
    }
    
    compute_curl_E(&E, &curl, grid);
    
    /* curl_phi should be dEr/dz - dEz/dr = E0 - 0 = E0 */
    printf("\nSample curl_phi values (should be %g):\n", E0);
    for (int i = 0; i < grid->Nr; i += grid->Nr/4) {
        int j = 0, k = grid->Nz / 2;
        if (k < grid->Nz) {
            printf("  curl_phi[i=%d, j=%d, k=%d] = %g\n", 
                   i, j, k, curl.curl_phi[idx_curl_phi(grid, i, j, k)]);
        }
    }
    
    efield_free(&E);
    curl_free(&curl);
}

/*=============================================================================
 * Test Case 4: E_r linear in phi
 * 
 * E = (E0 * phi) * r_hat  (Note: phi is the coordinate, not unit vector)
 * curl E = -(E0/r) * z_hat
 *============================================================================*/
void test_Er_linear_phi(GridParams* grid) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 4: E_r linear in phi (E = E0*phi * r_hat)\n");
    printf("Expected: curl E = -(E0/r) * z_hat\n");
    printf("============================================================\n");
    
    EField E;
    CurlE curl;
    efield_alloc(&E, grid);
    curl_alloc(&curl, grid);
    
    double E0 = 1.0;
    
    /* Set Er = E0 * phi at (i+1/2, j, k) */
    for (int k = 0; k <= grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            double phi = j * grid->dphi;
            for (int i = 0; i < grid->Nr; i++) {
                E.Er[idx_Er(grid, i, j, k)] = E0 * phi;
            }
        }
    }
    
    compute_curl_E(&E, &curl, grid);
    
    /* curl_z should be -(1/r) * dEr/dphi = -E0/r */
    printf("\nSample curl_z values:\n");
    int k = grid->Nz / 2;
    int j = 0;
    for (int i = 0; i < grid->Nr; i += grid->Nr/4) {
        double r = r_at_i_half(grid, i);
        double expected = -E0 / r;
        double computed = curl.curl_z[idx_curl_z(grid, i, j, k)];
        printf("  i=%d (r=%g): computed=%g, expected=%g, error=%e\n", 
               i, r, computed, expected, fabs(computed - expected));
    }
    
    efield_free(&E);
    curl_free(&curl);
}

/*=============================================================================
 * Test Case 5: E_phi proportional to r
 * 
 * E = (E0 * r) * phi_hat
 * curl E = (1/r) * d(r * E0*r)/dr * z_hat = (1/r) * d(E0*r^2)/dr * z_hat
 *        = (1/r) * 2*E0*r * z_hat = 2*E0 * z_hat
 *============================================================================*/
void test_Ephi_linear_r(GridParams* grid) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 5: E_phi linear in r (E = E0*r * phi_hat)\n");
    printf("Expected: curl E = 2*E0 * z_hat\n");
    printf("============================================================\n");
    
    EField E;
    CurlE curl;
    efield_alloc(&E, grid);
    curl_alloc(&curl, grid);
    
    double E0 = 1.5;
    
    /* Set Ephi = E0 * r at (i, j+1/2, k) */
    for (int k = 0; k <= grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++) {
                double r = r_at_i(grid, i);
                E.Ephi[idx_Ephi(grid, i, j, k)] = E0 * r;
            }
        }
    }
    
    compute_curl_E(&E, &curl, grid);
    
    /* curl_z should be 2*E0 */
    printf("\nSample curl_z values (should be %g):\n", 2.0 * E0);
    int k = grid->Nz / 2;
    int j = 0;
    for (int i = 0; i < grid->Nr; i += grid->Nr/4) {
        double computed = curl.curl_z[idx_curl_z(grid, i, j, k)];
        printf("  curl_z[i=%d, j=%d, k=%d] = %g\n", i, j, k, computed);
    }
    
    efield_free(&E);
    curl_free(&curl);
}

/*=============================================================================
 * Test Case 6: TEM mode field
 * 
 * E = (V0/r) * sin(pi*z/L) * r_hat
 * 
 * curl E = (1/r) * d(r*Er)/dr * ... - no, let's compute properly
 * 
 * For E = Er(r,z) * r_hat:
 * curl E = -dEr/dz * phi_hat = -(V0/r) * (pi/L) * cos(pi*z/L) * phi_hat
 *============================================================================*/
void test_TEM_mode(GridParams* grid) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 6: TEM mode (E = (V0/r)*sin(pi*z/L) * r_hat)\n");
    printf("Expected: curl E = -(V0*pi)/(r*L)*cos(pi*z/L) * phi_hat\n");
    printf("============================================================\n");
    
    EField E;
    CurlE curl;
    efield_alloc(&E, grid);
    curl_alloc(&curl, grid);
    
    double V0 = 1.0;
    double L = grid->L;
    
    /* Set Er = (V0/r) * sin(pi*z/L) at (i+1/2, j, k) */
    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;
        double sin_piz_L = sin(M_PI * z / L);
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                double r = r_at_i_half(grid, i);
                E.Er[idx_Er(grid, i, j, k)] = (V0 / r) * sin_piz_L;
            }
        }
    }
    
    compute_curl_E(&E, &curl, grid);
    
    /* curl_phi should be dEr/dz = (V0/r) * (pi/L) * cos(pi*z/L) */
    printf("\nSample curl_phi values at z = L/4:\n");
    int k = grid->Nz / 4;
    double z = (k + 0.5) * grid->dz;  /* curl_phi is at k+1/2 */
    int j = 0;
    
    for (int i = 0; i < grid->Nr; i += grid->Nr/4) {
        double r = r_at_i_half(grid, i);
        double expected = (V0 / r) * (M_PI / L) * cos(M_PI * z / L);
        double computed = curl.curl_phi[idx_curl_phi(grid, i, j, k)];
        double rel_error = fabs(computed - expected) / fabs(expected);
        printf("  i=%d (r=%6.4f): computed=%10.6f, expected=%10.6f, rel_err=%e\n", 
               i, r, computed, expected, rel_error);
    }
    
    efield_free(&E);
    curl_free(&curl);
}

/*=============================================================================
 * Test Case 7: E_z linear in phi (tests 1/r term in curl_r)
 * 
 * E = (E0 * phi) * z_hat
 * curl E = (1/r) * dEz/dphi * r_hat = (E0/r) * r_hat
 *============================================================================*/
void test_Ez_linear_phi(GridParams* grid) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 7: E_z linear in phi (E = E0*phi * z_hat)\n");
    printf("Expected: curl E = (E0/r) * r_hat\n");
    printf("============================================================\n");
    
    EField E;
    CurlE curl;
    efield_alloc(&E, grid);
    curl_alloc(&curl, grid);
    
    double E0 = 2.5;
    
    /* Set Ez = E0 * phi at (i, j, k+1/2) */
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            double phi = j * grid->dphi;
            for (int i = 0; i <= grid->Nr; i++) {
                E.Ez[idx_Ez(grid, i, j, k)] = E0 * phi;
            }
        }
    }
    
    compute_curl_E(&E, &curl, grid);
    
    /* curl_r should be (1/r) * dEz/dphi = E0/r */
    printf("\nSample curl_r values:\n");
    int k = grid->Nz / 2;
    int j = 0;
    for (int i = 0; i <= grid->Nr; i += grid->Nr/4) {
        double r = r_at_i(grid, i);
        double expected = E0 / r;
        double computed = curl.curl_r[idx_curl_r(grid, i, j, k)];
        double rel_error = (fabs(expected) > 1e-10) ? 
                           fabs(computed - expected) / fabs(expected) : fabs(computed);
        printf("  i=%d (r=%6.4f): computed=%10.6f, expected=%10.6f, rel_err=%e\n", 
               i, r, computed, expected, rel_error);
    }
    
    efield_free(&E);
    curl_free(&curl);
}

/*=============================================================================
 * Test Case 8: E_z linear in r (tests dEz/dr in curl_phi)
 * 
 * E = (E0 * r) * z_hat
 * curl E = -dEz/dr * phi_hat = -E0 * phi_hat
 *============================================================================*/
void test_Ez_linear_r(GridParams* grid) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 8: E_z linear in r (E = E0*r * z_hat)\n");
    printf("Expected: curl E = -E0 * phi_hat\n");
    printf("============================================================\n");
    
    EField E;
    CurlE curl;
    efield_alloc(&E, grid);
    curl_alloc(&curl, grid);
    
    double E0 = 1.0;
    
    /* Set Ez = E0 * r at (i, j, k+1/2) */
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++) {
                double r = r_at_i(grid, i);
                E.Ez[idx_Ez(grid, i, j, k)] = E0 * r;
            }
        }
    }
    
    compute_curl_E(&E, &curl, grid);
    
    /* curl_phi should be -dEz/dr = -E0 */
    printf("\nSample curl_phi values (should be %g):\n", -E0);
    int k = grid->Nz / 2;
    int j = 0;
    for (int i = 0; i < grid->Nr; i += grid->Nr/4) {
        double computed = curl.curl_phi[idx_curl_phi(grid, i, j, k)];
        printf("  curl_phi[i=%d, j=%d, k=%d] = %g\n", i, j, k, computed);
    }
    
    efield_free(&E);
    curl_free(&curl);
}

/*=============================================================================
 * Test Case 9: E_phi = r * z  (critical cylindrical metric test)
 *
 * E = (r*z) * phi_hat
 *
 * curl E = -(1/r) * d(r*Ephi)/dz * r_hat = -r * r_hat
 *============================================================================*/
void test_Ephi_rz(GridParams* grid) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 9: E_phi = r * z (metric consistency test)\n");
    printf("Expected: curl_r = -r\n");
    printf("============================================================\n");

    EField E;
    CurlE curl;
    efield_alloc(&E, grid);
    curl_alloc(&curl, grid);

    /* Set Ephi = r * z at (i, j+1/2, k) */
    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++) {
                double r = r_at_i(grid, i);
                E.Ephi[idx_Ephi(grid, i, j, k)] = r * z;
            }
        }
    }

    compute_curl_E(&E, &curl, grid);

    /* Inspect curl_r */
    printf("\nSample curl_r values at mid-z plane:\n");
    int k = grid->Nz / 2;
    int j = 0;

    for (int i = 1; i <= grid->Nr; i += grid->Nr / 4) {
        double r = r_at_i(grid, i);
        double expected = -r;
        double computed = curl.curl_r[idx_curl_r(grid, i, j, k)];
        double rel_err = fabs(computed - expected) / fabs(expected);

        printf("  i=%d (r=%6.4f): computed=%10.6f, expected=%10.6f, rel_err=%e\n",
               i, r, computed, expected, rel_err);
    }

    efield_free(&E);
    curl_free(&curl);
}


void test_TEM_convergence() {
    printf("\n");
    printf("============================================================\n");
    printf("Convergence Test: TEM mode with varying resolution\n");
    printf("============================================================\n");
    printf("%6s %12s %12s %12s\n", "Nz", "dz", "Rel Error", "Order");
    printf("----------------------------------------------------\n");
    
    double prev_error = 0;
    double prev_dz = 0;
    
    int Nz_values[] = {8, 16, 32, 64, 128};
    int num_tests = 5;
    
    for (int t = 0; t < num_tests; t++) {
        GridParams grid;
        grid_init(&grid, 0.3333, 1.0, 1.39, 8, 8, Nz_values[t]);
        
        EField E;
        CurlE curl;
        efield_alloc(&E, &grid);
        curl_alloc(&curl, &grid);
        
        double V0 = 1.0;
        double L = grid.L;
        
        /* Set Er = (V0/r) * sin(pi*z/L) */
        for (int k = 0; k <= grid.Nz; k++) {
            double z = k * grid.dz;
            double sin_piz_L = sin(M_PI * z / L);
            for (int j = 0; j < grid.Nphi; j++) {
                for (int i = 0; i < grid.Nr; i++) {
                    double r = r_at_i_half(&grid, i);
                    E.Er[idx_Er(&grid, i, j, k)] = (V0 / r) * sin_piz_L;
                }
            }
        }
        
        compute_curl_E(&E, &curl, &grid);
        
        /* Compute max relative error */
        double max_rel_error = 0;
        for (int k = 0; k < grid.Nz; k++) {
            double z = (k + 0.5) * grid.dz;
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                double expected = (V0 / r) * (M_PI / L) * cos(M_PI * z / L);
                double computed = curl.curl_phi[idx_curl_phi(&grid, i, 0, k)];
                double rel_err = fabs(computed - expected) / fabs(expected);
                if (rel_err > max_rel_error) max_rel_error = rel_err;
            }
        }
        
        double order = 0;
        if (prev_error > 0 && prev_dz > 0) {
            order = log(prev_error / max_rel_error) / log(prev_dz / grid.dz);
        }
        
        printf("%6d %12.6f %12.6e %12.2f\n", 
               Nz_values[t], grid.dz, max_rel_error, order);
        
        prev_error = max_rel_error;
        prev_dz = grid.dz;
        
        efield_free(&E);
        curl_free(&curl);
    }
}


/*=============================================================================
 * Main
 *============================================================================*/
int main() {
    printf("========================================================\n");
    printf("  Curl of E - Yee Grid Verification Tests\n");
    printf("  Cylindrical Coordinates (r, phi, z)\n");
    printf("========================================================\n");
    
    /* Setup grid */
    GridParams grid;
    grid_init(&grid, 
              0.3333,   /* a: inner radius */
              1.0,      /* b: outer radius */
              1.39,     /* L: length */
              8,        /* Nr */
              8,        /* Nphi */
              8);       /* Nz */
    
    grid_print(&grid);
    
    /* Run tests */
    test_uniform_Ez(&grid);
    test_Ephi_linear_z(&grid);
    test_Er_linear_z(&grid);
    test_Er_linear_phi(&grid);
    test_Ephi_linear_r(&grid);
    test_TEM_mode(&grid);
    test_Ez_linear_phi(&grid);
    test_Ez_linear_r(&grid);
    test_Ephi_rz(&grid);
    
    printf("\n========================================================\n");
    printf("  All tests completed.\n");
    printf("========================================================\n");
    
    /* Run convergence test */
    test_TEM_convergence();

    return 0;
}

