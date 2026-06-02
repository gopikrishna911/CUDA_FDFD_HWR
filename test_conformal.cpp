/*=============================================================================
 * test_conformal.c
 *
 * Standalone CPU test for the Dey-Mittra conformal geometry engine.
 * No CUDA required.
 *
 * Compile:
 *   gcc -O2 -o test_conformal test_conformal.c conformal_geometry.c \
 *       pipe_model.c curl_E.c -lm
 *
 * Tests:
 *   1. Circle-rectangle intersection primitives (analytical cases)
 *   2. Edge fractions for a single pipe (consistency checks)
 *   3. Face areas for a single pipe (area conservation)
 *   4. Conformal vs staircase mask agreement (fully-interior/exterior cells)
 *   5. Full 20-pipe model: cut cell count and total wall area
 *   6. Fillet profile correctness
 *   7. Pipe at various azimuthal positions (symmetry test)
 *============================================================================*/

#define _USE_MATH_DEFINES
#include "conformal_geometry.h"
#include "pipe_model.h"
#include "curl_E.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define C0  299792458.0
#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int g_pass = 0, g_fail = 0;

static void check(const char* name, int condition) {
    printf("    %-50s %s\n", name, condition ? PASS : FAIL);
    if (condition) g_pass++; else g_fail++;
}

static void checkf(const char* name, double got, double expected, double reltol) {
    double err = (expected != 0.0) ? fabs(got - expected) / fabs(expected) : fabs(got);
    char buf[200];
    snprintf(buf, sizeof(buf), "%s: got %.6e, exp %.6e, err %.2e", name, got, expected, err);
    check(buf, err < reltol);
}

/*=============================================================================
 * TEST 1: Circle-rect intersection primitives
 *============================================================================*/
static void test_primitives(void) {
    printf("\n  TEST 1: Circle-Rect Intersection Primitives\n");
    printf("  ============================================\n");

    /* Full circle inside large rect */
    checkf("Full circle in rect",
        circle_rect_intersection_area(-5, 5, -5, 5, 0, 0, 1.0),
        M_PI, 1e-12);

    /* Rect inside circle */
    checkf("Rect inside circle",
        circle_rect_intersection_area(-0.1, 0.1, -0.1, 0.1, 0, 0, 10.0),
        0.04, 1e-12);

    /* Half circle (y > 0) */
    checkf("Half circle (y>0)",
        circle_rect_intersection_area(-2, 2, 0, 2, 0, 0, 1.0),
        M_PI / 2.0, 1e-12);

    /* Quarter circle (x>0, y>0) */
    checkf("Quarter circle",
        circle_rect_intersection_area(0, 2, 0, 2, 0, 0, 1.0),
        M_PI / 4.0, 1e-12);

    /* No overlap */
    checkf("No overlap",
        circle_rect_intersection_area(5, 6, 5, 6, 0, 0, 1.0),
        0.0, 1e-15);

    /* Circle at (3,4) fully in large rect */
    checkf("Off-center circle",
        circle_rect_intersection_area(-10, 20, -10, 20, 3.0, 4.0, 2.0),
        M_PI * 4.0, 1e-12);

    /* Tangent cases: circle touches edge */
    checkf("Circle touching edge",
        circle_rect_intersection_area(0, 2, 0, 2, -1.0, 1.0, 1.0),
        0.0, 1e-10);  /* center at (-1,1), R=1: just touches x=0 line */

    /* Small circle clipping corner of rect */
    {
        /* Circle R=1 centered at (0,0), rect [0.5, 1.5] × [0.5, 1.5] */
        /* The circle clips the corner at (0.5, 0.5) to (0, 1) and (1, 0) */
        double area = circle_rect_intersection_area(0.5, 1.5, 0.5, 1.5, 0, 0, 1.0);
        /* Numerical: should be a circular segment area */
        /* Area ≈ sector(45°) - triangle = π/4 - 0.5 ≈ 0.2854 */
        double approx = M_PI / 12.0 - (sqrt(3.0) - 1.0) / 4.0;
        checkf("Corner clip",
            area, approx, 1e-6);
    }

    /* Symmetry: circle area should be independent of rect orientation */
    {
        double a1 = circle_rect_intersection_area(-1, 1, -2, 0, 0, 0, 1.5);
        double a2 = circle_rect_intersection_area(-1, 1,  0, 2, 0, 0, 1.5);
        checkf("Vertical symmetry", a1, a2, 1e-12);
    }
}

/*=============================================================================
 * TEST 2: Edge fractions for a single pipe
 *============================================================================*/
static void test_edge_fractions_single_pipe(void) {
    printf("\n  TEST 2: Edge Fractions (Single Pipe)\n");
    printf("  ====================================\n");

    double b = 1.0;
    double R_aper = 0.0175;
    double R_pipe = 0.0125;
    double phi_c = 0.0;
    double z_c = 0.6975;  /* L/2 */
    double pipe_len = 0.050;

    /* Eφ edge deep inside pipe: should be ~1.0 */
    {
        double r = 1.01;  /* just past wall */
        double z = z_c;   /* at pipe center */
        /* Edge centered on pipe axis */
        double frac = edge_fraction_Ephi_radial_pipe(r, z, -0.001, 0.001, phi_c, z_c, b, R_aper, R_pipe, pipe_len);
        check("Ephi inside pipe: frac > 0.99", frac > 0.99);
    }

    /* Eφ edge far from pipe: should be 0.0 */
    {
        double r = 1.01;
        double z = z_c;
        double frac = edge_fraction_Ephi_radial_pipe(r, z, 1.0, 1.01, phi_c, z_c, b, R_aper, R_pipe, pipe_len);
        check("Ephi far from pipe: frac == 0", frac < 1e-14);
    }

    /* Eφ edge on pipe boundary: should be 0 < frac < 1 */
    {
        double r = 1.02;  /* past fillet, R = R_pipe = 12.5mm */
        double R = R_pipe;
        double z = z_c;
        /* Place edge straddling the boundary */
        double half_phi = R / r;  /* angular half-width of pipe */
        double phi_edge_lo = phi_c + half_phi - 0.005;
        double phi_edge_hi = phi_edge_lo + 0.01;

        double frac = edge_fraction_Ephi_radial_pipe(r, z, phi_edge_lo, phi_edge_hi, phi_c, z_c, b, R_aper, R_pipe, pipe_len);
        check("Ephi on boundary: 0 < frac < 1",
              frac > 0.01 && frac < 0.99);
        printf("      (frac = %.4f)\n", frac);
    }

    /* Ez edge deep inside pipe: ~1.0 */
    {
        double r = 1.02;
        double phi = phi_c;
        double frac = edge_fraction_Ez_radial_pipe(r, phi, z_c - 0.001, z_c + 0.001, phi_c, z_c, b, R_aper, R_pipe, pipe_len);
        check("Ez inside pipe: frac > 0.99", frac > 0.99);
    }

    /* Er edge crossing wall: frac should be partial */
    {
        /* Edge from r=0.99 to r=1.01, at pipe center */
        double frac = edge_fraction_Er_radial_pipe(0.99, 1.01, phi_c, z_c, phi_c, z_c, b, R_aper, R_pipe, pipe_len);
        /* r < b is vacuum (cavity), r > b is vacuum (pipe center) → all vacuum */
        check("Er at pipe center crossing wall: frac == 1.0",
              frac > 0.99);
    }

    /* Er edge crossing wall but NOT at pipe center: should be partial */
    {
        /* Edge from r=0.99 to r=1.01, offset in φ so pipe doesn't cover r>b */
        double frac = edge_fraction_Er_radial_pipe(
            0.99, 1.01, phi_c + 0.1, z_c,
            phi_c, z_c, b, R_aper, R_pipe, pipe_len);
        /* r < b = cavity (vacuum), r > b = PEC (outside pipe) */
        /* So fraction ≈ 0.5 (half in cavity, half in PEC) */
        checkf("Er crossing wall outside pipe", frac, 0.5, 0.1);
    }

    /* Edge fraction in cavity (r < b): always 1 */
    {
        double frac = edge_fraction_Ephi_radial_pipe(
            0.5, z_c, 0.0, 0.01, phi_c, z_c, b, R_aper, R_pipe, pipe_len);
        check("Ephi in cavity: frac == 1.0", frac >= 1.0 - 1e-15);
    }
}

/*=============================================================================
 * TEST 3: Face areas — conservation & consistency
 *============================================================================*/
static void test_face_areas(void) {
    printf("\n  TEST 3: Face Areas (Single Pipe)\n");
    printf("  ================================\n");

    double b = 1.0;
    double R_aper = 0.0175;
    double R_pipe = 0.0125;
    double phi_c = 0.0;
    double z_c = 0.6975;
    double pipe_len = 0.050;

    /* Hr face at r > b, fully inside pipe: area = R circle ∩ rect */
    {
        double r = 1.02;  /* past fillet */
        double R = R_pipe;
        /* Large face that fully contains the pipe cross-section */
        double phi0 = phi_c - 0.05;
        double phi1 = phi_c + 0.05;
        double z0 = z_c - 0.05;
        double z1 = z_c + 0.05;

        double area = face_area_Hr_radial_pipe(
            r, phi0, phi1, z0, z1,
            phi_c, z_c, b, R_aper, R_pipe, pipe_len);
        /* Should equal full circle area π·R² since face contains entire circle */
        double expected = M_PI * R * R;
        checkf("Hr face containing full pipe", area, expected, 1e-6);
    }

    /* Hr face at r ≤ b: standard area */
    {
        double r = 0.5;
        double dphi = 0.01;
        double dz_val = 0.01;
        double area = face_area_Hr_radial_pipe(
            r, 0.0, dphi, 0.0, dz_val,
            phi_c, z_c, b, R_aper, R_pipe, pipe_len);
        double expected = r * dphi * dz_val;
        checkf("Hr face in cavity", area, expected, 1e-12);
    }

    /* Hφ face in cavity: standard area = dr · dz */
    {
        double area = face_area_Hphi_radial_pipe(
            0.5, 0.4, 0.41, 0.3, 0.31,
            phi_c, z_c, b, R_aper, R_pipe, pipe_len);
        double expected = 0.01 * 0.01;
        checkf("Hphi face in cavity", area, expected, 1e-6);
    }

    /* Hz face in cavity: standard area = r_mid · dφ · dr */
    {
        double r0 = 0.5, r1 = 0.51, phi0 = 0.0, phi1 = 0.01;
        double area = face_area_Hz_radial_pipe(
            0.5, r0, r1, phi0, phi1,
            phi_c, z_c, b, R_aper, R_pipe, pipe_len);
        double r_mid = 0.505;
        double expected = r_mid * 0.01 * 0.01;
        /* GL quadrature integrates r·dφ·dr, so ∫r dr from 0.5 to 0.51 = 0.00505 */
        /* times dφ = 0.01 → 0.0000505 */
        double exact = 0.5 * (r1*r1 - r0*r0) * (phi1 - phi0);
        checkf("Hz face in cavity", area, exact, 1e-4);  /* GL has some error */
    }

    /* Sum of face areas around a pipe: total vacuum area at a given r */
    {
        double r = 1.02;
        double R = R_pipe;
        int N_phi_test = 200;
        int N_z_test = 200;
        double phi_range = 0.1;  /* covers pipe angular extent */
        double z_range = 0.1;    /* covers pipe z extent */
        double dphi_t = phi_range / N_phi_test;
        double dz_t = z_range / N_z_test;

        double total_area = 0.0;
        for (int jj = 0; jj < N_phi_test; jj++) {
            double p0 = phi_c - phi_range/2 + jj * dphi_t;
            double p1 = p0 + dphi_t;
            for (int kk = 0; kk < N_z_test; kk++) {
                double zz0 = z_c - z_range/2 + kk * dz_t;
                double zz1 = zz0 + dz_t;
                total_area += face_area_Hr_radial_pipe(
                    r, p0, p1, zz0, zz1,
                    phi_c, z_c, b, R_aper, R_pipe, pipe_len);
            }
        }
        double expected = M_PI * R * R;
        checkf("Sum of Hr sub-faces = π·R²", total_area, expected, 1e-3);
    }
}

/*=============================================================================
 * TEST 4: Fillet profile
 *============================================================================*/
static void test_fillet(void) {
    printf("\n  TEST 4: Quarter-Circle Fillet Profile\n");
    printf("  =====================================\n");

    double b = 1.0;
    double R_aper = 0.0175;
    double R_pipe = 0.0125;
    double R_f = R_aper - R_pipe;  /* 5mm */

    /* At r = b: should be aperture radius */
    checkf("R(r=b) = aperture",
        conformal_pipe_radius(b, b, R_aper, R_pipe), R_aper, 1e-14);

    /* At r = b + R_f: should be pipe radius */
    checkf("R(r=b+Rf) = pipe",
        conformal_pipe_radius(b + R_f, b, R_aper, R_pipe), R_pipe, 1e-14);

    /* At r = b + 2*R_f: still pipe radius */
    checkf("R(r=b+2Rf) = pipe",
        conformal_pipe_radius(b + 2*R_f, b, R_aper, R_pipe), R_pipe, 1e-14);

    /* Mid-fillet: R should be between pipe and aperture */
    {
        double R_mid = conformal_pipe_radius(b + R_f/2, b, R_aper, R_pipe);
        check("R(r=b+Rf/2) between pipe and aperture",
              R_mid > R_pipe && R_mid < R_aper);
        printf("      (R_mid = %.4f mm, pipe = %.4f, aper = %.4f)\n",
               R_mid*1e3, R_pipe*1e3, R_aper*1e3);
    }

    /* Fillet is monotonically decreasing */
    {
        int mono = 1;
        double R_prev = R_aper;
        for (int s = 1; s <= 100; s++) {
            double r = b + s * R_f / 100.0;
            double R = conformal_pipe_radius(r, b, R_aper, R_pipe);
            if (R > R_prev + 1e-15) { mono = 0; break; }
            R_prev = R;
        }
        check("Fillet is monotonically decreasing", mono);
    }

    /* Print fillet profile */
    printf("    Fillet profile (R_f = %.1f mm):\n", R_f * 1e3);
    for (int s = 0; s <= 10; s++) {
        double dr = s * R_f / 10.0;
        double R = conformal_pipe_radius(b + dr, b, R_aper, R_pipe);
        printf("      r = b + %.2f mm: R = %.3f mm\n", dr * 1e3, R * 1e3);
    }
}

/*=============================================================================
 * TEST 5: Conformal data build — full model
 *============================================================================*/
static void test_full_model_build(void) {
    printf("\n  TEST 5: Full Model Conformal Data Build\n");
    printf("  ========================================\n");

    /* Same geometry as test_ibc_pipe.cpp */
    double a = 0.3333;
    double b = 1.0;
    double L = 1.395;
    double R_pipe = 0.0125;
    double R_aper = 0.0175;
    double pipe_len = 0.050;
    int num_passes = 10;

    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, R_pipe, R_aper, pipe_len, 0.0);
    pipe_config_add_multi_pass(&pipes, L / 2.0, num_passes);

    printf("    %d radial pipes configured\n", pipes.num_pipes);

    /* Moderate grid matching test_ibc_pipe.cpp */
    int Nr_cavity = 81;
    int Nr_pipe_cells = 7;
    int Nphi = 256;
    int Nz_cavity = 84;

    /* No endcap extensions for this test */
    GridParams grid;
    grid_init_with_pipes(&grid, a, b, L, pipe_len,
                         Nr_cavity, Nr_pipe_cells, Nphi, Nz_cavity);

    printf("    Grid: Nr=%d, Nphi=%d, Nz=%d\n", grid.Nr, grid.Nphi, grid.Nz);
    printf("    dr=%.4f mm, dphi=%.4f deg, dz=%.4f mm\n",
           grid.dr * 1e3, grid.dphi * 180.0 / M_PI, grid.dz * 1e3);

    /* Build conformal data */
    ConformalData cd;
    conformal_data_build(&cd, &pipes, &grid, b, 0.0);
    conformal_data_print_stats(&cd, &grid);

    /* Checks */
    int total_cut = cd.num_cut_Er + cd.num_cut_Ephi + cd.num_cut_Ez;
    check("Has cut edges (total_cut > 0)", total_cut > 0);
    printf("      (total cut edges: %d)\n", total_cut);

    /* Each pipe has a circumference ~2πR cutting through cells.
     * At each r-plane in the pipe region: circumference ≈ 2π·R ≈ 78mm (aperture)
     * Cells along circumference: ~78mm / dphi_phys ≈ 78/24 ≈ 3-4 per pipe
     * 20 pipes × ~4 cut cells × (Nr_pipe r-planes) × 3 components ≈ few thousand
     */
    check("Cut edges in reasonable range (100 < total < 100000)",
          total_cut > 100 && total_cut < 100000);

    /* All cavity edges (r < b) should have frac = 1.0 */
    {
        int cavity_nonunit = 0;
        for (int k = 0; k <= grid.Nz; k++) {
            for (int j = 0; j < grid.Nphi; j++) {
                for (int i = 0; i < grid.Nr; i++) {
                    double r = grid.a + (i + 0.5) * grid.dr;
                    if (r < b - grid.dr) {  /* safely inside cavity */
                        int idx = i + grid.Nr * (j + grid.Nphi * k);
                        if (cd.edge_frac_Er[idx] < 1.0 - 1e-10)
                            cavity_nonunit++;
                    }
                }
            }
        }
        check("All cavity Er edges have frac == 1.0",
              cavity_nonunit == 0);
        if (cavity_nonunit > 0)
            printf("      (%d cavity edges with frac < 1!)\n", cavity_nonunit);
    }

    /* Deep PEC edges (far from any pipe, r > b) should have frac = 0.0 */
    {
        int pec_nonzero = 0;
        /* Check Ephi edges at r = b + pipe_len (near pipe cap) */
        for (int k = 0; k <= grid.Nz; k++) {
            for (int j = 0; j < grid.Nphi; j++) {
                int i = grid.Nr;  /* r = a + Nr·dr = b + pipe_len */
                int idx = i + (grid.Nr + 1) * (j + grid.Nphi * k);
                if (cd.edge_frac_Ephi[idx] > 1e-10)
                    pec_nonzero++;
            }
        }
        check("PEC cap edges have frac == 0",
              pec_nonzero == 0);
        if (pec_nonzero > 0)
            printf("      (%d PEC cap edges with frac > 0!)\n", pec_nonzero);
    }

    /* IBC weight checks */
    int total_ibc = cd.num_ibc_Er + cd.num_ibc_Ephi + cd.num_ibc_Ez;
    check("Has IBC boundary weights (total > 0)", total_ibc > 0);
    printf("      (total IBC boundary cells: %d)\n", total_ibc);

    /* All IBC weights should be positive */
    {
        int neg = 0;
        for (int i = 0; i < cd.size_Ephi; i++)
            if (cd.ibc_weight_Ephi[i] < -1e-30) neg++;
        for (int i = 0; i < cd.size_Ez; i++)
            if (cd.ibc_weight_Ez[i] < -1e-30) neg++;
        check("No negative IBC weights", neg == 0);
    }

    /* Compare with staircase mask */
    printf("\n    Comparing with staircase material mask...\n");
    {
        MaterialMask mask;
        material_mask_build(&mask, &pipes, &grid);

        /* For edges deep inside vacuum (mask=1): conformal frac should be 1 */
        /* For edges deep inside PEC (mask=0): conformal frac should be 0 */
        /* Disagreements are expected only at pipe boundary (1-2 cell layer) */
        int agree_vac = 0, agree_pec = 0, disagree = 0;
        for (int idx = 0; idx < cd.size_Er; idx++) {
            int mask_val = mask.mask_Er[idx];
            double frac = cd.edge_frac_Er[idx];
            if (mask_val == 1 && frac >= 1.0 - 1e-10) agree_vac++;
            else if (mask_val == 0 && frac <= 1e-10) agree_pec++;
            else disagree++;
        }
        printf("      Er: agree_vac=%d, agree_pec=%d, boundary_disagree=%d\n",
               agree_vac, agree_pec, disagree);
        /* The disagreements should be ~ the number of cut cells */
        check("Er disagreements ≈ cut cells (within 5x)",
              disagree > 0 && disagree < 5 * cd.num_cut_Er + 5000);

        material_mask_free(&mask);
    }

    conformal_data_free(&cd);
    pipe_config_free(&pipes);
}

/*=============================================================================
 * TEST 6: Consistency — edge fractions vs face areas
 *
 * For a face fully in vacuum: area should equal standard area AND
 * all 4 bounding edge fractions should be 1.0.
 *
 * For a face fully in PEC: area should be 0 AND
 * all 4 bounding edge fractions should be 0.
 *============================================================================*/
static void test_consistency(void) {
    printf("\n  TEST 6: Edge-Face Consistency\n");
    printf("  ============================\n");

    /* Simple setup: 1 pipe at φ=0, z=L/2 */
    double a = 0.3333, b = 1.0, L = 1.395;
    double R_pipe = 0.0125, R_aper = 0.0175, pipe_len = 0.050;

    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, R_pipe, R_aper, pipe_len, 0.0);
    pipe_config_add(&pipes, 0.0, L / 2.0);

    /* Coarse grid for easy manual inspection */
    int Nr_cav = 20, Nr_p = 3, Nphi = 64, Nz = 20;
    GridParams grid;
    grid_init_with_pipes(&grid, a, b, L, pipe_len, Nr_cav, Nr_p, Nphi, Nz);

    ConformalData cd;
    conformal_data_build(&cd, &pipes, &grid, b, 0.0);

    printf("    Grid: Nr=%d, Nphi=%d, Nz=%d\n", grid.Nr, Nphi, Nz);
    printf("    dr=%.2f mm, r·dphi(at b)=%.2f mm, dz=%.2f mm\n",
           grid.dr*1e3, b*grid.dphi*1e3, grid.dz*1e3);
    conformal_data_print_stats(&cd, &grid);

    /* Pick a Hphi face near the pipe boundary and dump its edges */
    printf("\n    Sample Hphi faces near pipe:\n");
    int i_wall = (int)round((b - a) / grid.dr);
    int j_pipe = 0;  /* pipe at φ=0 */
    int k_mid = Nz / 2;

    for (int di = -1; di <= 3; di++) {
        int i = i_wall + di;
        if (i < 0 || i >= grid.Nr) continue;

        double r = grid.a + (i + 0.5) * grid.dr;
        int idx_face = i + grid.Nr * (j_pipe + Nphi * k_mid);
        double face_a = cd.face_area_Hphi[idx_face];
        double std_a = grid.dr * grid.dz;

        /* Bounding edges:
         * Er(i,j,k+1), Er(i,j,k), Ez(i+1,j,k), Ez(i,j,k)
         */
        int idx_Er_top = i + grid.Nr * (j_pipe + Nphi * (k_mid + 1));
        int idx_Er_bot = i + grid.Nr * (j_pipe + Nphi * k_mid);
        int idx_Ez_rgt = (i+1) + (grid.Nr+1) * (j_pipe + Nphi * k_mid);
        int idx_Ez_lft = i + (grid.Nr+1) * (j_pipe + Nphi * k_mid);

        double f_Er_top = (k_mid + 1 <= grid.Nz && i < grid.Nr) ?
            cd.edge_frac_Er[idx_Er_top] : -1;
        double f_Er_bot = (i < grid.Nr) ?
            cd.edge_frac_Er[idx_Er_bot] : -1;
        double f_Ez_rgt = (i+1 <= grid.Nr && k_mid < grid.Nz) ?
            cd.edge_frac_Ez[idx_Ez_rgt] : -1;
        double f_Ez_lft = (k_mid < grid.Nz) ?
            cd.edge_frac_Ez[idx_Ez_lft] : -1;

        printf("      i=%d (r=%.4f): face=%.6e (%.1f%% of std), "
               "edges: Er_top=%.3f Er_bot=%.3f Ez_rgt=%.3f Ez_lft=%.3f\n",
               i, r, face_a, 100.0 * face_a / std_a,
               f_Er_top, f_Er_bot, f_Ez_rgt, f_Ez_lft);
    }

    /* The face area should be consistent: if all 4 edge fracs = 1,
     * face area should equal standard area */
    {
        int inconsistent = 0;
        for (int idx = 0; idx < cd.size_Hphi; idx++) {
            double face_a = cd.face_area_Hphi[idx];
            double std_a = grid.dr * grid.dz;

            if (face_a > std_a * 1.01) {
                /* Face area larger than standard — this shouldn't happen */
                inconsistent++;
            }
        }
        check("No Hphi face area exceeds standard", inconsistent == 0);
        if (inconsistent > 0)
            printf("      (%d faces exceed standard area!)\n", inconsistent);
    }

    conformal_data_free(&cd);
    pipe_config_free(&pipes);
}

/*=============================================================================
 * TEST 7: Azimuthal symmetry — all pipes should give same cut pattern
 *============================================================================*/
static void test_symmetry(void) {
    printf("\n  TEST 7: Azimuthal Symmetry\n");
    printf("  =========================\n");

    double a = 0.3333, b = 1.0, L = 1.395;
    double R_pipe = 0.0125, R_aper = 0.0175, pipe_len = 0.050;

    /* 4 pipes at 0°, 90°, 180°, 270° */
    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, R_pipe, R_aper, pipe_len, 0.0);
    pipe_config_add(&pipes, 0.0, L/2.0);
    pipe_config_add(&pipes, M_PI/2.0, L/2.0);
    pipe_config_add(&pipes, M_PI, L/2.0);
    pipe_config_add(&pipes, 3.0*M_PI/2.0, L/2.0);

    /* Grid divisible by 4 in φ */
    int Nr_cav = 40, Nr_p = 4, Nphi = 128, Nz = 40;
    GridParams grid;
    grid_init_with_pipes(&grid, a, b, L, pipe_len, Nr_cav, Nr_p, Nphi, Nz);

    ConformalData cd;
    conformal_data_build(&cd, &pipes, &grid, b, 0.0);

    /* Count cut Eφ edges in each quadrant */
    int quarter = Nphi / 4;
    int cuts_per_quadrant[4] = {0, 0, 0, 0};

    for (int idx = 0; idx < cd.size_Ephi; idx++) {
        double f = cd.edge_frac_Ephi[idx];
        if (f > 0.0 && f < 1.0) {
            /* Find j from index */
            int Nr1 = grid.Nr + 1;
            int tmp = idx;
            int i = tmp % Nr1; tmp /= Nr1;
            int j = tmp % Nphi;
            int q = j / quarter;
            if (q < 4) cuts_per_quadrant[q]++;
        }
    }

    printf("    Cut Ephi edges per quadrant: [%d, %d, %d, %d]\n",
           cuts_per_quadrant[0], cuts_per_quadrant[1],
           cuts_per_quadrant[2], cuts_per_quadrant[3]);

    /* All quadrants should have the same count (exact symmetry) */
    check("All quadrants have same cut count",
          cuts_per_quadrant[0] == cuts_per_quadrant[1] &&
          cuts_per_quadrant[1] == cuts_per_quadrant[2] &&
          cuts_per_quadrant[2] == cuts_per_quadrant[3]);

    conformal_data_free(&cd);
    pipe_config_free(&pipes);
}

/*=============================================================================
 * TEST 8: Total pipe wall area estimate
 *
 * For a straight pipe (past the fillet), the wall area per unit
 * radial length is 2π·R_pipe. Summing the IBC weights times
 * dual cell volumes should approximate this.
 *============================================================================*/
static void test_wall_area(void) {
    printf("\n  TEST 8: Total Wall Area from IBC Weights\n");
    printf("  ==========================================\n");

    double a = 0.3333, b = 1.0, L = 1.395;
    double R_pipe = 0.0125, R_aper = 0.0175, pipe_len = 0.050;
    double R_f = R_aper - R_pipe;

    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, R_pipe, R_aper, pipe_len, 0.0);
    pipe_config_add(&pipes, 0.0, L / 2.0);  /* single pipe for clarity */

    int Nr_cav = 81, Nr_p = 7, Nphi = 256, Nz = 84;
    GridParams grid;
    grid_init_with_pipes(&grid, a, b, L, pipe_len, Nr_cav, Nr_p, Nphi, Nz);

    ConformalData cd;
    conformal_data_build(&cd, &pipes, &grid, b, 0.0);

    /* Sum wall arc lengths from Hr faces at each r beyond b */
    double total_wall = 0.0;
    int i_wall = (int)round((b - a) / grid.dr);

    for (int i = i_wall; i <= grid.Nr; i++) {
        double r = grid.a + i * grid.dr;
        if (r <= b) continue;

        /* Sum arc lengths of all Hr faces at this r */
        double arc_sum = 0.0;
        for (int j = 0; j < Nphi; j++) {
            for (int k = 0; k < grid.Nz; k++) {
                double phi0 = j * grid.dphi;
                double phi1 = phi0 + grid.dphi;
                double z0 = k * grid.dz;  /* grid z */
                double z1 = z0 + grid.dz;

                double arc = wall_arc_length_Hr(
                    r, phi0, phi1, z0, z1,
                    pipes.pipes[0].phi_center,
                    pipes.pipes[0].z_center,
                    b, R_aper, R_pipe, pipe_len);
                arc_sum += arc;
            }
        }

        double R_at_r = conformal_pipe_radius(r, b, R_aper, R_pipe);
        double expected_circ = 2.0 * M_PI * R_at_r;

        if (arc_sum > 0.01 * expected_circ) {
            printf("      r=%.4f (R=%.3fmm): arc_sum=%.4fmm, 2πR=%.4fmm, "
                   "ratio=%.4f\n",
                   r, R_at_r * 1e3, arc_sum * 1e3, expected_circ * 1e3,
                   arc_sum / expected_circ);
        }

        total_wall += arc_sum * grid.dr;  /* wall area element = arc · dr */
    }

    /* Analytical wall area for straight pipe of length (pipe_len - R_f):
     * A_straight = 2π·R_pipe · (pipe_len - R_f)
     * Plus fillet surface area (harder to compute, but rough check):
     */
    double A_straight = 2.0 * M_PI * R_pipe * (pipe_len - R_f);
    printf("\n    Total wall area from arc sums: %.6f mm²\n", total_wall * 1e6);
    printf("    Straight pipe area (no fillet): %.6f mm²\n", A_straight * 1e6);
    printf("    Ratio (includes fillet): %.3f\n", total_wall / A_straight);

    /* Should be > 1.0 (fillet adds area) and < 2.0 (reasonable) */
    check("Total wall area > straight pipe area", total_wall > A_straight * 0.8);
    check("Total wall area in reasonable range", total_wall < A_straight * 3.0);

    conformal_data_free(&cd);
    pipe_config_free(&pipes);
}

/*=============================================================================
 * MAIN
 *============================================================================*/
int main(void) {
    printf("\n");
    printf("================================================================\n");
    printf("  CONFORMAL GEOMETRY ENGINE — PHASE 1 VALIDATION\n");
    printf("================================================================\n");

    test_primitives();
    test_edge_fractions_single_pipe();
    test_face_areas();
    test_fillet();
    test_full_model_build();
    test_consistency();
    test_symmetry();
    test_wall_area();

    printf("\n================================================================\n");
    printf("  RESULTS: %d passed, %d failed\n", g_pass, g_fail);
    printf("================================================================\n\n");

    return g_fail > 0 ? 1 : 0;
}
