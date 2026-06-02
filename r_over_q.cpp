#include "r_over_q.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*=============================================================================
 * Compute gap voltage along a single radial line
 *
 * E_r lives at grid positions (i+1/2, j, k):
 *   r   = a + (i + 0.5) * dr     for i = 0 .. Nr-1
 *   phi = j * dphi                for j = 0 .. Nphi-1
 *   z   = k * dz                  for k = 0 .. Nz
 *
 * For a beam path at arbitrary (phi, z_phys), we:
 *   1. Convert z_phys to grid coordinate: z_grid = z_phys + z0_offset
 *   2. Bilinear interpolation in (phi, z) at each radial index i
 *   3. Trapezoidal integration along r from a to b_cavity
 *============================================================================*/
double compute_gap_voltage(
    const CurlCurlOperator* op,
    const double* eigenvector,
    double phi,
    double z_phys,
    double z0_offset,
    double b_cavity
) {
    const GridParams* g = &op->grid;
    int Nr = g->Nr;
    int Nphi = g->Nphi;
    double dr = g->dr;
    double dphi = g->dphi;
    double dz = g->dz;

    /* Convert physical z to grid z */
    double z_grid = z_phys + z0_offset;

    /* Find bracketing k indices for z interpolation
     * E_r lives at integer k, so z = k * dz */
    double k_float = z_grid / dz;
    int k_lo = (int)floor(k_float);
    int k_hi = k_lo + 1;
    double wz_hi = k_float - k_lo;     /* weight for k_hi */
    double wz_lo = 1.0 - wz_hi;        /* weight for k_lo */

    /* Clamp to valid range */
    if (k_lo < 0) { k_lo = 0; k_hi = 0; wz_lo = 1.0; wz_hi = 0.0; }
    if (k_hi > g->Nz) { k_hi = g->Nz; k_lo = g->Nz; wz_lo = 1.0; wz_hi = 0.0; }

    /* Find bracketing j indices for phi interpolation
     * E_r lives at phi = j * dphi */
    double phi_norm = fmod(phi, 2.0 * M_PI);
    if (phi_norm < 0) phi_norm += 2.0 * M_PI;

    double j_float = phi_norm / dphi;
    int j_lo = (int)floor(j_float);
    int j_hi = j_lo + 1;
    double wp_hi = j_float - j_lo;
    double wp_lo = 1.0 - wp_hi;

    /* Periodic wrapping in phi */
    j_lo = j_lo % Nphi;
    j_hi = j_hi % Nphi;
    if (j_lo < 0) j_lo += Nphi;
    if (j_hi < 0) j_hi += Nphi;

    /* Find the radial index range covering the cavity [a, b_cavity]
     * E_r(i) is at r = a + (i + 0.5) * dr
     * Last index with r <= b_cavity: i_max = floor((b_cavity - a)/dr - 0.5) */
    int i_max = (int)floor((b_cavity - g->a) / dr - 0.5);
    if (i_max >= Nr) i_max = Nr - 1;
    if (i_max < 0) i_max = 0;

    /* Integrate E_r along r using trapezoidal rule with bilinear interpolation */
    double voltage = 0.0;

    for (int i = 0; i <= i_max; i++) {
        /* Bilinear interpolation: 4-point stencil in (j, k) */
        double Er_ll = eigenvector[op->offset_Er + idx_Er(g, i, j_lo, k_lo)];
        double Er_lh = eigenvector[op->offset_Er + idx_Er(g, i, j_lo, k_hi)];
        double Er_hl = eigenvector[op->offset_Er + idx_Er(g, i, j_hi, k_lo)];
        double Er_hh = eigenvector[op->offset_Er + idx_Er(g, i, j_hi, k_hi)];

        double Er_interp = wp_lo * (wz_lo * Er_ll + wz_hi * Er_lh)
                         + wp_hi * (wz_lo * Er_hl + wz_hi * Er_hh);

        /* Midpoint rule: weight = 1.0 for full cells,
         * fractional for last cell if it extends beyond b_cavity */
        double weight = 1.0;
        if (i == i_max) {
            /* Cell covers [a + i*dr, a + (i+1)*dr].
             * Only integrate up to b_cavity */
            double r_cell_start = g->a + i * dr;
            double r_cell_end = g->a + (i + 1) * dr;
            if (r_cell_end > b_cavity) {
                weight = (b_cavity - r_cell_start) / dr;
                if (weight < 0.0) weight = 0.0;
                if (weight > 1.0) weight = 1.0;
            }
        }
        voltage += weight * Er_interp * dr;
    }

    return voltage;
}


/*=============================================================================
 * Main R/Q computation
 *============================================================================*/
RoverQResult compute_r_over_q(
    const CurlCurlOperator* op,
    const double* eigenvector,
    double eigenvalue,
    double b_cavity,
    double L_cavity,
    double z0_offset,
    int num_passes
) {
    RoverQResult result;
    memset(&result, 0, sizeof(RoverQResult));

    const GridParams* g = &op->grid;
    double a = g->a;

    /* Mode properties */
    result.k_squared = eigenvalue;
    result.omega = ROQ_C0 * sqrt(fabs(eigenvalue));
    result.frequency_Hz = result.omega / (2.0 * M_PI);

    /* Stored energy: W = (eps_0 / 2) * integral |E|^2 r dr dphi dz
     * The weighted inner product gives integral |E|^2 r dr dphi dz */
    double E_sq = vec_dot_product_weighted(eigenvector, eigenvector, op);
    result.W_stored = 0.5 * ROQ_EPS0 * E_sq;

    /* Allocate per-crossing and per-pass arrays */
    result.n_passes = num_passes;
    result.n_crossings = 2 * num_passes;

    result.crossings = (CrossingVoltage*)malloc(
        result.n_crossings * sizeof(CrossingVoltage));
    result.V_pass = (double*)malloc(num_passes * sizeof(double));
    result.R_over_Q_pass_linac = (double*)malloc(num_passes * sizeof(double));

    /* Beam midplane */
    double z_mid = L_cavity / 2.0;
    double dphi_pass = M_PI / num_passes;

    /* Compute gap voltage for each crossing */
    double V_sum = 0.0;
    double V_sum_sq = 0.0;
    result.V_gap_min = 1e30;
    result.V_gap_max = -1e30;

    for (int pass = 0; pass < num_passes; pass++) {
        double phi_entry = pass * dphi_pass;
        double phi_exit = phi_entry + M_PI;
        if (phi_exit >= 2.0 * M_PI) phi_exit -= 2.0 * M_PI;

        /* Entry crossing (outer -> inner, beam travels inward) */
        double V_entry = compute_gap_voltage(
            op, eigenvector, phi_entry, z_mid, z0_offset, b_cavity);

        /* Exit crossing (inner -> outer, beam travels outward) */
        double V_exit = compute_gap_voltage(
            op, eigenvector, phi_exit, z_mid, z0_offset, b_cavity);

        /* Store per-crossing results */
        int idx_entry = 2 * pass;
        int idx_exit = 2 * pass + 1;

        result.crossings[idx_entry].V_gap = V_entry;
        result.crossings[idx_entry].phi = phi_entry;
        result.crossings[idx_entry].pass = pass;
        result.crossings[idx_entry].is_exit = 0;

        result.crossings[idx_exit].V_gap = V_exit;
        result.crossings[idx_exit].phi = phi_exit;
        result.crossings[idx_exit].pass = pass;
        result.crossings[idx_exit].is_exit = 1;

        /* Per-pass voltage: entry + exit (coherent sum — beam sees both) */
        result.V_pass[pass] = fabs(V_entry) + fabs(V_exit);
        result.R_over_Q_pass_linac[pass] =
            result.V_pass[pass] * result.V_pass[pass]
            / (result.omega * result.W_stored);

        /* Accumulate statistics using |V| */
        double absV_entry = fabs(V_entry);
        double absV_exit = fabs(V_exit);

        V_sum += absV_entry + absV_exit;
        V_sum_sq += absV_entry * absV_entry + absV_exit * absV_exit;

        if (absV_entry < result.V_gap_min) result.V_gap_min = absV_entry;
        if (absV_exit < result.V_gap_min)  result.V_gap_min = absV_exit;
        if (absV_entry > result.V_gap_max) result.V_gap_max = absV_entry;
        if (absV_exit > result.V_gap_max)  result.V_gap_max = absV_exit;
    }

    /* Summary statistics */
    result.V_gap_avg = V_sum / result.n_crossings;
    result.V_gap_spread = (result.V_gap_avg > 0)
        ? (result.V_gap_max - result.V_gap_min) / result.V_gap_avg * 100.0
        : 0.0;
    result.V_total = V_sum;

    /* R/Q per single crossing (using average voltage) */
    result.R_over_Q_crossing_linac =
        result.V_gap_avg * result.V_gap_avg / (result.omega * result.W_stored);
    result.R_over_Q_crossing_circuit =
        result.R_over_Q_crossing_linac / 2.0;

    /* R/Q per pass (using average pass voltage = 2 * V_avg) */
    double V_pass_avg = 2.0 * result.V_gap_avg;
    result.R_over_Q_per_pass_linac =
        V_pass_avg * V_pass_avg / (result.omega * result.W_stored);
    result.R_over_Q_per_pass_circuit =
        result.R_over_Q_per_pass_linac / 2.0;

    /* Total machine R/Q: V_total = sum of ALL gap voltages */
    result.R_over_Q_total_linac =
        result.V_total * result.V_total / (result.omega * result.W_stored);
    result.R_over_Q_total_circuit =
        result.R_over_Q_total_linac / 2.0;

    /* Analytical comparison (TEM coaxial HWR) */
    double ln_ba = log(b_cavity / a);
    result.R_over_Q_analytical_linac = 2.0 * ROQ_ETA0 * ln_ba / (M_PI * M_PI);
    result.R_over_Q_analytical_circuit = result.R_over_Q_analytical_linac / 2.0;

    return result;
}


/*=============================================================================
 * Analytical R/Q for TEM coaxial HWR
 *
 * V_gap = A * ln(b/a)
 * W = eps_0 * pi * A^2 * L * ln(b/a) / 2
 * omega = pi * c / L
 *
 * R/Q_linac = V^2/(omega*W) = 2 * eta_0 * ln(b/a) / pi^2
 *============================================================================*/
RoverQResult compute_r_over_q_analytical(double a, double b, double L) {
    RoverQResult result;
    memset(&result, 0, sizeof(RoverQResult));

    double ln_ba = log(b / a);
    double omega = M_PI * ROQ_C0 / L;

    result.k_squared = (M_PI / L) * (M_PI / L);
    result.omega = omega;
    result.frequency_Hz = omega / (2.0 * M_PI);

    /* Analytical stored energy for unit amplitude A = 1:
     * W = eps_0 * pi * A^2 * L * ln(b/a) / 2 */
    result.W_stored = ROQ_EPS0 * M_PI * L * ln_ba / 2.0;

    /* Analytical gap voltage for A = 1: V = ln(b/a) */
    result.V_gap_avg = ln_ba;
    result.V_gap_min = ln_ba;
    result.V_gap_max = ln_ba;
    result.V_gap_spread = 0.0;

    /* R/Q per crossing */
    result.R_over_Q_crossing_linac = 2.0 * ROQ_ETA0 * ln_ba / (M_PI * M_PI);
    result.R_over_Q_crossing_circuit = result.R_over_Q_crossing_linac / 2.0;

    /* R/Q per pass (2 crossings) */
    result.R_over_Q_per_pass_linac = 4.0 * result.R_over_Q_crossing_linac;
    result.R_over_Q_per_pass_circuit = result.R_over_Q_per_pass_linac / 2.0;

    /* Analytical comparison is itself */
    result.R_over_Q_analytical_linac = result.R_over_Q_crossing_linac;
    result.R_over_Q_analytical_circuit = result.R_over_Q_crossing_circuit;

    /* No per-crossing/pass arrays for analytical */
    result.n_crossings = 0;
    result.n_passes = 0;
    result.crossings = NULL;
    result.V_pass = NULL;
    result.R_over_Q_pass_linac = NULL;

    return result;
}


/*=============================================================================
 * Free allocated memory
 *============================================================================*/
void r_over_q_free(RoverQResult* result) {
    if (result->crossings) {
        free(result->crossings);
        result->crossings = NULL;
    }
    if (result->V_pass) {
        free(result->V_pass);
        result->V_pass = NULL;
    }
    if (result->R_over_Q_pass_linac) {
        free(result->R_over_Q_pass_linac);
        result->R_over_Q_pass_linac = NULL;
    }
}


/*=============================================================================
 * Print R/Q results
 *============================================================================*/
void r_over_q_print(const RoverQResult* result, const char* label) {
    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────┐\n");
    printf("  │  R/Q: %-47s │\n", label);
    printf("  ├──────────────────────────────────────────────────────────┤\n");
    printf("  │  Mode:                                                  │\n");
    printf("  │    Frequency      = %-12.6f MHz                  │\n",
        result->frequency_Hz / 1e6);
    printf("  │    omega          = %.6e   rad/s               │\n",
        result->omega);
    printf("  │    W_stored       = %.6e   [arb]                │\n",
        result->W_stored);
    printf("  ├──────────────────────────────────────────────────────────┤\n");
    printf("  │  Gap voltage (per crossing):                            │\n");
    printf("  │    V_avg          = %.6e   [arb]                │\n",
        result->V_gap_avg);
    if (result->n_crossings > 0) {
        printf("  │    V_min          = %.6e                        │\n",
            result->V_gap_min);
        printf("  │    V_max          = %.6e                        │\n",
            result->V_gap_max);
        printf("  │    Spread         = %.4f%%                          │\n",
            result->V_gap_spread);
    }
    printf("  ├──────────────────────────────────────────────────────────┤\n");
    printf("  │  R/Q per single crossing:                               │\n");
    printf("  │    Linac   V^2/(wW) = %10.4f Ohm                 │\n",
        result->R_over_Q_crossing_linac);
    printf("  │    Circuit V^2/(2wW)= %10.4f Ohm                 │\n",
        result->R_over_Q_crossing_circuit);
    printf("  │  R/Q per pass (2 crossings):                            │\n");
    printf("  │    Linac            = %10.4f Ohm                 │\n",
        result->R_over_Q_per_pass_linac);
    printf("  │    Circuit          = %10.4f Ohm                 │\n",
        result->R_over_Q_per_pass_circuit);

    if (result->n_passes > 0) {
        printf("  │  R/Q total (%d passes, %d crossings):              │\n",
            result->n_passes, result->n_crossings);
        printf("  │    V_total         = %.6e                        │\n",
            result->V_total);
        printf("  │    Linac           = %10.2f Ohm                 │\n",
            result->R_over_Q_total_linac);
        printf("  │    Circuit         = %10.2f Ohm                 │\n",
            result->R_over_Q_total_circuit);
    }
    printf("  ├──────────────────────────────────────────────────────────┤\n");
    printf("  │  Analytical (TEM HWR):                                  │\n");
    printf("  │    R/Q_linac       = %10.4f Ohm  (per crossing)  │\n",
        result->R_over_Q_analytical_linac);
    printf("  │    R/Q_circuit     = %10.4f Ohm  (per crossing)  │\n",
        result->R_over_Q_analytical_circuit);
    printf("  └──────────────────────────────────────────────────────────┘\n");

    /* Per-pass breakdown table */
    if (result->n_passes > 0 && result->n_passes <= 20) {
        printf("\n  Per-pass breakdown:\n");
        printf("  %5s  %12s  %12s  %12s  %12s  %10s\n",
            "Pass", "phi_entry", "V_entry", "V_exit", "V_pass", "R/Q_pass");
        printf("  %5s  %12s  %12s  %12s  %12s  %10s\n",
            "-----", "----------", "----------", "----------",
            "----------", "----------");

        for (int p = 0; p < result->n_passes; p++) {
            CrossingVoltage* entry = &result->crossings[2 * p];
            CrossingVoltage* exit  = &result->crossings[2 * p + 1];

            printf("  %5d  %9.2f deg  %12.6e  %12.6e  %12.6e  %10.4f\n",
                p + 1,
                entry->phi * 180.0 / M_PI,
                entry->V_gap,
                exit->V_gap,
                result->V_pass[p],
                result->R_over_Q_pass_linac[p]);
        }
    }
}


/*=============================================================================
 * Print comparison: analytical vs numerical
 *============================================================================*/
void r_over_q_print_comparison(
    const RoverQResult* analytical,
    const RoverQResult* numerical
) {
    printf("\n");
    printf("  ┌────────────────────────────────────────────────────────────┐\n");
    printf("  │              R/Q COMPARISON                                │\n");
    printf("  ├────────────────────────────────────────────────────────────┤\n");
    printf("  │  Quantity                Analytical    Numerical          │\n");
    printf("  │  ──────────────────────  ──────────  ──────────          │\n");
    printf("  │  f (MHz)              %12.4f  %12.4f          │\n",
        analytical->frequency_Hz / 1e6, numerical->frequency_Hz / 1e6);
    printf("  │  R/Q per crossing:                                        │\n");
    printf("  │    Linac  (Ohm)       %12.4f  %12.4f          │\n",
        analytical->R_over_Q_crossing_linac,
        numerical->R_over_Q_crossing_linac);
    printf("  │    Circuit (Ohm)      %12.4f  %12.4f          │\n",
        analytical->R_over_Q_crossing_circuit,
        numerical->R_over_Q_crossing_circuit);

    /* Error */
    if (analytical->R_over_Q_crossing_linac > 0) {
        double err = (numerical->R_over_Q_crossing_linac
                      - analytical->R_over_Q_crossing_linac)
                     / analytical->R_over_Q_crossing_linac * 100.0;
        printf("  │    Error vs analytical    —         %+.2f%%           │\n",
            err);
    }

    printf("  │  R/Q per pass:                                            │\n");
    printf("  │    Linac  (Ohm)       %12.4f  %12.4f          │\n",
        analytical->R_over_Q_per_pass_linac,
        numerical->R_over_Q_per_pass_linac);

    if (numerical->n_passes > 0) {
        printf("  │  R/Q total (%d passes):                              │\n",
            numerical->n_passes);
        printf("  │    Linac  (Ohm)             —     %12.2f          │\n",
            numerical->R_over_Q_total_linac);
    }

    printf("  │  Voltage spread:                                          │\n");
    printf("  │    (Vmax-Vmin)/Vavg       0.00%%     %8.4f%%           │\n",
        numerical->V_gap_spread);
    printf("  └────────────────────────────────────────────────────────────┘\n");
}
