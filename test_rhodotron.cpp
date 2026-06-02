/*=============================================================================
 * test_rhodotron.c
 * 
 * Rhodotron cavity simulation with beam apertures and vacuum port
 *============================================================================*/

#include "curlcurl_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    printf("\n");
    printf("****************************************************************\n");
    printf("*              RHODOTRON CAVITY SIMULATION                     *\n");
    printf("*              Single Pass with Vacuum Port                    *\n");
    printf("****************************************************************\n");

    /*=========================================================================
     * Geometry Parameters
     *========================================================================*/
    double a = 0.3333;                      /* Inner radius [m] */
    double b = 1.0;                         /* Outer radius [m] */
    double L = 1.39;                        /* Cavity length [m] */
    
    double beam_aperture_radius = 0.005;    /* 12.5 mm */
    double vacuum_port_radius = 0.015;      /* 75 mm (DN200) for UHV */
    double z_beam = L / 2.0;                /* Beam at cavity center */
    double r_vacuum = (a + b) / 2.0;        /* Vacuum port radial position */
    double phi_vacuum = M_PI / 2.0;         /* Vacuum port at 90 degrees */

    printf("\n=== GEOMETRY ===\n");
    printf("  Inner radius a = %.4f m\n", a);
    printf("  Outer radius b = %.4f m\n", b);
    printf("  Length L = %.4f m\n", L);
    printf("  Beam aperture radius = %.1f mm\n", beam_aperture_radius * 1000);
    printf("  Vacuum port radius = %.1f mm\n", vacuum_port_radius * 1000);

    /*=========================================================================
     * Grid Setup
     *========================================================================*/
    int Nr = 32;
    int Nphi = 128;
    int Nz = 128;

    GridParams grid;
    grid_init(&grid, a, b, L, Nr, Nphi, Nz);

    printf("\n=== GRID ===\n");
    printf("  Nr=%d, Nphi=%d, Nz=%d\n", Nr, Nphi, Nz);
    printf("  dr = %.2f mm, dz = %.2f mm\n", grid.dr * 1000, grid.dz * 1000);

    /*=========================================================================
     * Configure Ports
     *========================================================================*/
    PortConfig ports;
    port_config_init(&ports);

    port_config_add_beam_apertures_single_pass(&ports, z_beam, beam_aperture_radius);
    port_config_add_vacuum_port(&ports, SURFACE_ENDPLATE_Z0,
                                r_vacuum, phi_vacuum, vacuum_port_radius);

    port_config_print(&ports, &grid);

    /*=========================================================================
     * Create Operators
     *========================================================================*/
    
    /* Reference: No ports (ideal cavity) */
    CurlCurlOperator op_ref;
    curlcurl_op_init(&op_ref, &grid);

    /* With ports */
    CurlCurlOperator op;
    curlcurl_op_init_with_ports(&op, &grid, &ports);

    printf("\n=== OPERATOR ===\n");
    printf("  Total DOFs: %d\n", op.n_total);
    printf("  Has ports: %s\n", op.has_ports ? "YES" : "NO");

    /*=========================================================================
     * Initialize TEM Mode
     *========================================================================*/
    double* x_ref = (double*)malloc(op_ref.n_total * sizeof(double));
    double* x = (double*)malloc(op.n_total * sizeof(double));
    vec_zero(x_ref, op_ref.n_total);

    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        double sin_piz_L = sin(M_PI * z / L);
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                int idx = op_ref.offset_Er + idx_Er(&grid, i, j, k);
                x_ref[idx] = sin_piz_L / r;
            }
        }
    }

    /*=========================================================================
     * Reference: Ideal Cavity (No Ports)
     *========================================================================*/
    printf("\n=== REFERENCE: IDEAL CAVITY ===\n");

    double k2_exact = (M_PI / L) * (M_PI / L);
    double k2_ref = rayleigh_quotient_weighted(&op_ref, x_ref);

    double c = 299792458.0;
    double f_ref = sqrt(k2_ref) * c / (2.0 * M_PI);

    printf("  Analytical k^2 = %.10f\n", k2_exact);
    printf("  Numerical k^2  = %.10f (error: %.4f%%)\n", 
           k2_ref, fabs(k2_ref - k2_exact) / k2_exact * 100.0);
    printf("  Frequency = %.6f MHz\n", f_ref / 1e6);

    /*=========================================================================
     * With Ports
     *========================================================================*/
    printf("\n=== CAVITY WITH PORTS ===\n");

    vec_copy(x_ref, x, op.n_total);

    /* Add small perturbation */
    srand(12345);
    for (int i = 0; i < op.n_total; i++) {
        x[i] += 0.01 * x[i] * ((double)rand() / RAND_MAX - 0.5);
    }

    printf("\nRunning Rayleigh Quotient Iteration...\n\n");
    EigenResult result = rayleigh_quotient_iteration(&op, x, k2_ref, 20, 1e-10);

    double k2_ports = result.eigenvalue;
    double f_ports = sqrt(fabs(k2_ports)) * c / (2.0 * M_PI);

    /*=========================================================================
     * Results
     *========================================================================*/
    printf("\n");
    printf("****************************************************************\n");
    printf("*                     RESULTS SUMMARY                          *\n");
    printf("****************************************************************\n");
    printf("\n  Eigenvalue k^2:\n");
    printf("    Without ports: %.10f\n", k2_ref);
    printf("    With ports:    %.10f\n", k2_ports);
    printf("    Difference:    %.6e (%.4f%%)\n",
           k2_ports - k2_ref, (k2_ports - k2_ref) / k2_ref * 100.0);

    printf("\n  Frequency:\n");
    printf("    Without ports: %.6f MHz\n", f_ref / 1e6);
    printf("    With ports:    %.6f MHz\n", f_ports / 1e6);
    printf("    Shift:         %.3f kHz\n", (f_ports - f_ref) / 1e3);

    printf("\n  Convergence:\n");
    printf("    Converged: %s\n", result.converged ? "YES" : "NO");
    printf("    Iterations: %d\n", result.iterations);
    printf("    Residual: %.3e\n", result.residual);

    /*=========================================================================
     * Export VTK
     *========================================================================*/
    printf("\n=== EXPORTING VTK FILES ===\n");
    
    /* Normalize for export */
    double max_val = 0.0;
    for (int i = 0; i < op.n_total; i++) {
        if (fabs(x[i]) > max_val) max_val = fabs(x[i]);
    }
    if (max_val > 0) {
        vec_scale(x, 1.0 / max_val, op.n_total);
        vec_scale(x_ref, 1.0 / max_val, op_ref.n_total);
    }

    export_field_vtk(&op_ref, x_ref, "rhodotron_ideal.vtk");
    export_field_vtk(&op, x, "rhodotron_with_ports.vtk");

    /*=========================================================================
     * Cleanup
     *========================================================================*/
    free(x);
    free(x_ref);
    port_config_free(&ports);
    curlcurl_op_free(&op);
    curlcurl_op_free(&op_ref);

    printf("\n****************************************************************\n");
    printf("*                    SIMULATION COMPLETE                       *\n");
    printf("****************************************************************\n\n");

    return 0;
}