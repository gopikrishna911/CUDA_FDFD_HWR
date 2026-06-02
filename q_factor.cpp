#include "q_factor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>

/*=============================================================================
 * Surface integral on INNER conductor (r = a)
 *
 * LINEAR EXTRAPOLATION to the actual wall at r = a:
 *   H(a) = 1.5 * H(i=0) - 0.5 * H(i=1)
 * This eliminates the O(dr) bias and gives O(dr^2) accuracy.
 *
 * Surface element: dS = a * dphi * dz
 *============================================================================*/
double surface_integral_inner(
    const CurlE* curlE, const GridParams* grid,
    const PortConfig* apertures
) {
    double sum = 0.0;
    double a = grid->a;
    double dphi = grid->dphi;
    double dz = grid->dz;
    double dS = a * dphi * dz;
    int has_ap = (apertures != NULL && apertures->num_ports > 0);

    /* H_phi contribution: extrapolate curl_phi from i=0,1 to r=a */
    for (int k = 0; k < grid->Nz; k++) {
        double z = (k + 0.5) * dz;
        for (int j = 0; j < grid->Nphi; j++) {
            double phi = j * dphi;

            if (has_ap &&
                point_in_port_cylindrical(apertures, grid, SURFACE_INNER, phi, z))
                continue;

            double v0 = curlE->curl_phi[idx_curl_phi(grid, 0, j, k)];
            double v1 = curlE->curl_phi[idx_curl_phi(grid, 1, j, k)];
            double val = 1.5 * v0 - 0.5 * v1;
            sum += val * val * dS;
        }
    }

    /* H_z contribution: extrapolate curl_z from i=0,1 to r=a */
    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * dz;
        for (int j = 0; j < grid->Nphi; j++) {
            double phi = (j + 0.5) * dphi;

            if (has_ap &&
                point_in_port_cylindrical(apertures, grid, SURFACE_INNER, phi, z))
                continue;

            double v0 = curlE->curl_z[idx_curl_z(grid, 0, j, k)];
            double v1 = curlE->curl_z[idx_curl_z(grid, 1, j, k)];
            double val = 1.5 * v0 - 0.5 * v1;
            sum += val * val * dS;
        }
    }

    return sum;
}

/*=============================================================================
 * Surface integral on OUTER conductor (r = b_cavity)
 *
 * LINEAR EXTRAPOLATION to the actual wall at r = b_cavity:
 *   H(b) = 1.5 * H(i_outer) - 0.5 * H(i_outer-1)
 * Falls back to nearest-point sampling if i_outer == 0.
 *
 * Surface element: dS = r_outer * dphi * dz
 *============================================================================*/
double surface_integral_outer(
    const CurlE* curlE, const GridParams* grid,
    int i_outer, double r_outer,
    const PortConfig* apertures
) {
    double sum = 0.0;
    double dphi = grid->dphi;
    double dz = grid->dz;
    double dS = r_outer * dphi * dz;
    int has_ap = (apertures != NULL && apertures->num_ports > 0);
    int can_extrap = (i_outer > 0);
    int i_prev = can_extrap ? i_outer - 1 : 0;

    /* H_phi contribution: extrapolate to r=b */
    for (int k = 0; k < grid->Nz; k++) {
        double z = (k + 0.5) * dz;
        for (int j = 0; j < grid->Nphi; j++) {
            double phi = j * dphi;

            if (has_ap &&
                point_in_port_cylindrical(apertures, grid, SURFACE_OUTER, phi, z))
                continue;

            double v_out = curlE->curl_phi[idx_curl_phi(grid, i_outer, j, k)];
            double val;
            if (can_extrap) {
                double v_in = curlE->curl_phi[idx_curl_phi(grid, i_prev, j, k)];
                val = 1.5 * v_out - 0.5 * v_in;
            }
            else {
                val = v_out;
            }
            sum += val * val * dS;
        }
    }

    /* H_z contribution: extrapolate to r=b */
    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * dz;
        for (int j = 0; j < grid->Nphi; j++) {
            double phi = (j + 0.5) * dphi;

            if (has_ap &&
                point_in_port_cylindrical(apertures, grid, SURFACE_OUTER, phi, z))
                continue;

            double v_out = curlE->curl_z[idx_curl_z(grid, i_outer, j, k)];
            double val;
            if (can_extrap) {
                double v_in = curlE->curl_z[idx_curl_z(grid, i_prev, j, k)];
                val = 1.5 * v_out - 0.5 * v_in;
            }
            else {
                val = v_out;
            }
            sum += val * val * dS;
        }
    }

    return sum;
}

/*=============================================================================
 * Surface integral on z=0 ENDPLATE
 *
 * LINEAR EXTRAPOLATION to z = 0:
 *   H(z=0) = 1.5 * H(k=0) - 0.5 * H(k=1)
 *
 * Surface element: dS = r * dr * dphi
 *============================================================================*/
double surface_integral_endplate_z0(
    const CurlE* curlE, const GridParams* grid,
    int i_max_r, int i_max_half,
    const PortConfig* apertures
) {
    double sum = 0.0;
    double dr = grid->dr;
    double dphi = grid->dphi;
    int has_ap = (apertures != NULL && apertures->num_ports > 0);

    /* H_r contribution: extrapolate curl_r from k=0,1 to z=0 */
    for (int j = 0; j < grid->Nphi; j++) {
        double phi = (j + 0.5) * dphi;
        for (int i = 0; i <= i_max_r; i++) {
            double r = grid->a + i * dr;

            if (has_ap &&
                point_in_port_endplate(apertures, grid, SURFACE_ENDPLATE_Z0, r, phi))
                continue;

            double v0 = curlE->curl_r[idx_curl_r(grid, i, j, 0)];
            double v1 = curlE->curl_r[idx_curl_r(grid, i, j, 1)];
            double val = 1.5 * v0 - 0.5 * v1;
            double dS = r * dr * dphi;
            sum += val * val * dS;
        }
    }

    /* H_phi contribution: extrapolate curl_phi from k=0,1 to z=0 */
    for (int j = 0; j < grid->Nphi; j++) {
        double phi = j * dphi;
        for (int i = 0; i <= i_max_half; i++) {
            double r = grid->a + (i + 0.5) * dr;

            if (has_ap &&
                point_in_port_endplate(apertures, grid, SURFACE_ENDPLATE_Z0, r, phi))
                continue;

            double v0 = curlE->curl_phi[idx_curl_phi(grid, i, j, 0)];
            double v1 = curlE->curl_phi[idx_curl_phi(grid, i, j, 1)];
            double val = 1.5 * v0 - 0.5 * v1;
            double dS = r * dr * dphi;
            sum += val * val * dS;
        }
    }

    return sum;
}

/*=============================================================================
 * Surface integral on z=L ENDPLATE
 *
 * LINEAR EXTRAPOLATION to z = L:
 *   H(z=L) = 1.5 * H(k=Nz-1) - 0.5 * H(k=Nz-2)
 *============================================================================*/
double surface_integral_endplate_zL(
    const CurlE* curlE, const GridParams* grid,
    int i_max_r, int i_max_half,
    const PortConfig* apertures
) {
    double sum = 0.0;
    double dr = grid->dr;
    double dphi = grid->dphi;
    int k_last = grid->Nz - 1;
    int k_prev = grid->Nz - 2;
    int has_ap = (apertures != NULL && apertures->num_ports > 0);

    /* H_r contribution: extrapolate to z=L */
    for (int j = 0; j < grid->Nphi; j++) {
        double phi = (j + 0.5) * dphi;
        for (int i = 0; i <= i_max_r; i++) {
            double r = grid->a + i * dr;

            if (has_ap &&
                point_in_port_endplate(apertures, grid, SURFACE_ENDPLATE_ZL, r, phi))
                continue;

            double v0 = curlE->curl_r[idx_curl_r(grid, i, j, k_last)];
            double v1 = curlE->curl_r[idx_curl_r(grid, i, j, k_prev)];
            double val = 1.5 * v0 - 0.5 * v1;
            double dS = r * dr * dphi;
            sum += val * val * dS;
        }
    }

    /* H_phi contribution: extrapolate to z=L */
    for (int j = 0; j < grid->Nphi; j++) {
        double phi = j * dphi;
        for (int i = 0; i <= i_max_half; i++) {
            double r = grid->a + (i + 0.5) * dr;

            if (has_ap &&
                point_in_port_endplate(apertures, grid, SURFACE_ENDPLATE_ZL, r, phi))
                continue;

            double v0 = curlE->curl_phi[idx_curl_phi(grid, i, j, k_last)];
            double v1 = curlE->curl_phi[idx_curl_phi(grid, i, j, k_prev)];
            double val = 1.5 * v0 - 0.5 * v1;
            double dS = r * dr * dphi;
            sum += val * val * dS;
        }
    }

    return sum;
}

static double surface_integral_endplate_at_k(
    const CurlE* curlE, const GridParams* grid,
    int i_max_r, int i_max_half,
    const PortConfig* apertures,
    PortSurface surface,
    int k_wall,         /* k-index of the wall */
    int k_interior      /* k-index one cell into cavity */
) {
    double sum = 0.0;
    double dr = grid->dr;
    double dphi = grid->dphi;
    int has_ap = (apertures != NULL && apertures->num_ports > 0);

    /* H_r contribution: extrapolate from k_wall, k_interior */
    for (int j = 0; j < grid->Nphi; j++) {
        double phi = (j + 0.5) * dphi;
        for (int i = 0; i <= i_max_r; i++) {
            double r = grid->a + i * dr;

            if (has_ap &&
                point_in_port_endplate(apertures, grid, surface, r, phi))
                continue;

            double v0 = curlE->curl_r[idx_curl_r(grid, i, j, k_wall)];
            double v1 = curlE->curl_r[idx_curl_r(grid, i, j, k_interior)];
            double val = 1.5 * v0 - 0.5 * v1;
            double dS = r * dr * dphi;
            sum += val * val * dS;
        }
    }

    /* H_phi contribution */
    for (int j = 0; j < grid->Nphi; j++) {
        double phi = j * dphi;
        for (int i = 0; i <= i_max_half; i++) {
            double r = grid->a + (i + 0.5) * dr;

            if (has_ap &&
                point_in_port_endplate(apertures, grid, surface, r, phi))
                continue;

            double v0 = curlE->curl_phi[idx_curl_phi(grid, i, j, k_wall)];
            double v1 = curlE->curl_phi[idx_curl_phi(grid, i, j, k_interior)];
            double val = 1.5 * v0 - 0.5 * v1;
            double dS = r * dr * dphi;
            sum += val * val * dS;
        }
    }

    return sum;
}

/*=============================================================================
 * Main Q factor computation (wall losses)
 *============================================================================*/
QFactorResult compute_q_factor(
    const CurlCurlOperator* op,
    const double* eigenvector,
    double eigenvalue,
    double conductivity,
    double b_cavity,
    const PortConfig* apertures
) {
    QFactorResult result;
    const GridParams* grid = &op->grid;

    const PortConfig* ap = apertures;
    if (!ap && op->has_ports) {
        ap = &op->ports;
    }

    result.k_squared = eigenvalue;
    result.omega = Q_C0 * sqrt(fabs(eigenvalue));
    result.frequency_Hz = result.omega / (2.0 * M_PI);
    result.wavelength = Q_C0 / result.frequency_Hz;
    result.conductivity = conductivity;

    result.R_s = sqrt(result.omega * Q_MU0 / (2.0 * conductivity));
    result.skin_depth = sqrt(2.0 / (result.omega * Q_MU0 * conductivity));

    double E_sq = vec_dot_product_weighted(eigenvector, eigenvector, op);
    result.W_stored = 0.5 * Q_EPS0 * E_sq;

    EField E_temp;
    efield_alloc(&E_temp, grid);
    unpack_field(eigenvector, &E_temp, op);

    CurlE curlE;
    curl_alloc(&curlE, grid);
    compute_curl_E(&E_temp, &curlE, grid);

    int i_outer = (int)round((b_cavity - grid->a) / grid->dr - 0.5);
    if (i_outer < 0) i_outer = 0;
    if (i_outer >= grid->Nr) i_outer = grid->Nr - 1;
    double r_outer = b_cavity;

    int i_max_r = (int)floor((b_cavity - grid->a) / grid->dr);
    if (i_max_r > grid->Nr) i_max_r = grid->Nr;

    int i_max_half = (int)floor((b_cavity - grid->a) / grid->dr - 0.5);
    if (i_max_half >= grid->Nr) i_max_half = grid->Nr - 1;

    double S_inner = surface_integral_inner(&curlE, grid, ap);
    double S_outer = surface_integral_outer(&curlE, grid, i_outer, r_outer, ap);
    double S_z0 = surface_integral_endplate_z0(&curlE, grid, i_max_r, i_max_half, ap);
    double S_zL = surface_integral_endplate_zL(&curlE, grid, i_max_r, i_max_half, ap);
    double S_total = S_inner + S_outer + S_z0 + S_zL;

    double loss_scale = result.R_s /
        (2.0 * result.omega * result.omega * Q_MU0 * Q_MU0);

    result.P_inner = loss_scale * S_inner;
    result.P_outer = loss_scale * S_outer;
    result.P_endplate_z0 = loss_scale * S_z0;
    result.P_endplate_zL = loss_scale * S_zL;
    result.P_wall_total = loss_scale * S_total;

    if (result.P_wall_total > 0.0) {
        result.Q_0 = result.omega * result.W_stored / result.P_wall_total;
    }
    else {
        result.Q_0 = 0.0;
    }

    result.G_factor = result.Q_0 * result.R_s;

    efield_free(&E_temp);
    curl_free(&curlE);

    return result;
}

/*=============================================================================
 * Q factor for extended grid with endcap z-pipes
 *
 * k_endplate_z0: k-index of the physical z=0 endplate
 * k_endplate_zL: k-index of the physical z=L endplate
 *
 * If k_endplate_z0 < 0, uses default (k=0)
 * If k_endplate_zL < 0, uses default (k=Nz)
 *============================================================================*/
QFactorResult compute_q_factor_extended(
    const CurlCurlOperator* op,
    const double* eigenvector,
    double eigenvalue,
    double conductivity,
    double b_cavity,
    const PortConfig* apertures,
    int k_endplate_z0,
    int k_endplate_zL
) {
    QFactorResult result;
    const GridParams* grid = &op->grid;

    const PortConfig* ap = apertures;
    if (!ap && op->has_ports) {
        ap = &op->ports;
    }

    /* Use defaults if negative */
    if (k_endplate_z0 < 0) k_endplate_z0 = 0;
    if (k_endplate_zL < 0) k_endplate_zL = grid->Nz;

    result.k_squared = eigenvalue;
    result.omega = Q_C0 * sqrt(fabs(eigenvalue));
    result.frequency_Hz = result.omega / (2.0 * M_PI);
    result.wavelength = Q_C0 / result.frequency_Hz;
    result.conductivity = conductivity;

    result.R_s = sqrt(result.omega * Q_MU0 / (2.0 * conductivity));
    result.skin_depth = sqrt(2.0 / (result.omega * Q_MU0 * conductivity));

    double E_sq = vec_dot_product_weighted(eigenvector, eigenvector, op);
    result.W_stored = 0.5 * Q_EPS0 * E_sq;

    EField E_temp;
    efield_alloc(&E_temp, grid);
    unpack_field(eigenvector, &E_temp, op);

    CurlE curlE;
    curl_alloc(&curlE, grid);
    compute_curl_E(&E_temp, &curlE, grid);

    int i_outer = (int)round((b_cavity - grid->a) / grid->dr - 0.5);
    if (i_outer < 0) i_outer = 0;
    if (i_outer >= grid->Nr) i_outer = grid->Nr - 1;
    double r_outer = b_cavity;

    int i_max_r = (int)floor((b_cavity - grid->a) / grid->dr);
    if (i_max_r > grid->Nr) i_max_r = grid->Nr;

    int i_max_half = (int)floor((b_cavity - grid->a) / grid->dr - 0.5);
    if (i_max_half >= grid->Nr) i_max_half = grid->Nr - 1;

    double S_inner = surface_integral_inner(&curlE, grid, ap);
    double S_outer = surface_integral_outer(&curlE, grid, i_outer, r_outer, ap);

    /* Endplate integrals at correct k-indices */
    double S_z0, S_zL;

    if (k_endplate_z0 == 0) {
        /* Standard: endplate at grid boundary */
        S_z0 = surface_integral_endplate_z0(&curlE, grid,
            i_max_r, i_max_half, ap);
    }
    else {
        /* Extended grid: endplate at interior k-index
         * Extrapolate from cavity side: k_wall, k_wall+1 */
        S_z0 = surface_integral_endplate_at_k(&curlE, grid,
            i_max_r, i_max_half, ap,
            SURFACE_ENDPLATE_Z0,
            k_endplate_z0, k_endplate_z0 + 1);
    }

    if (k_endplate_zL == grid->Nz) {
        S_zL = surface_integral_endplate_zL(&curlE, grid,
            i_max_r, i_max_half, ap);
    }
    else {
        /* Extrapolate from cavity side: k_wall, k_wall-1 */
        S_zL = surface_integral_endplate_at_k(&curlE, grid,
            i_max_r, i_max_half, ap,
            SURFACE_ENDPLATE_ZL,
            k_endplate_zL - 1, k_endplate_zL - 2);
    }

    double S_total = S_inner + S_outer + S_z0 + S_zL;

    double loss_scale = result.R_s /
        (2.0 * result.omega * result.omega * Q_MU0 * Q_MU0);

    result.P_inner = loss_scale * S_inner;
    result.P_outer = loss_scale * S_outer;
    result.P_endplate_z0 = loss_scale * S_z0;
    result.P_endplate_zL = loss_scale * S_zL;
    result.P_wall_total = loss_scale * S_total;

    if (result.P_wall_total > 0.0) {
        result.Q_0 = result.omega * result.W_stored / result.P_wall_total;
    }
    else {
        result.Q_0 = 0.0;
    }

    result.G_factor = result.Q_0 * result.R_s;

    efield_free(&E_temp);
    curl_free(&curlE);

    return result;
}


/*=============================================================================
 * Analytical Q_0 for coaxial half-wave resonator (TEM mode)
 *============================================================================*/
QFactorResult compute_q_analytical_coaxial_hwr(
    double a,
    double b,
    double L,
    double conductivity
) {
    QFactorResult result;

    double ln_ba = log(b / a);

    result.k_squared = (M_PI / L) * (M_PI / L);
    result.omega = Q_C0 * M_PI / L;
    result.frequency_Hz = Q_C0 / (2.0 * L);
    result.wavelength = 2.0 * L;
    result.conductivity = conductivity;

    result.R_s = sqrt(result.omega * Q_MU0 / (2.0 * conductivity));
    result.skin_depth = sqrt(2.0 / (result.omega * Q_MU0 * conductivity));

    double P_cyl_norm = L / (2.0);
    double P_end_norm = ln_ba;
    double common = 2.0 * M_PI / (Q_ETA0 * Q_ETA0);

    result.P_inner = (result.R_s / 2.0) * common * P_cyl_norm / a;
    result.P_outer = (result.R_s / 2.0) * common * P_cyl_norm / b;
    result.P_endplate_z0 = (result.R_s / 2.0) * common * P_end_norm;
    result.P_endplate_zL = (result.R_s / 2.0) * common * P_end_norm;
    result.P_wall_total = result.P_inner + result.P_outer + result.P_endplate_z0 + result.P_endplate_zL;

    result.W_stored = 0.5 * Q_EPS0 * 2.0 * M_PI * (L / 2.0) * ln_ba;

    double numerator = M_PI * ln_ba;
    double denominator = (L / 2.0) * (1.0 / a + 1.0 / b) + 2.0 * ln_ba;
    result.Q_0 = (Q_ETA0 / result.R_s) * numerator / (2.0 * denominator);

    result.G_factor = result.Q_0 * result.R_s;

    return result;
}

/*=============================================================================
 * Print single Q factor result
 *============================================================================*/
void q_factor_print(const QFactorResult* result, const char* label) {
    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────┐\n");
    printf("  │  Q Factor: %-44s│\n", label ? label : "");
    printf("  ├──────────────────────────────────────────────────────────┤\n");

    printf("  │  Mode:                                                  │\n");
    printf("  │    k^2            = %-14.8e  1/m^2            │\n", result->k_squared);
    printf("  │    Frequency      = %-14.6f  MHz              │\n", result->frequency_Hz / 1.0e6);
    printf("  │    Wavelength     = %-14.4f  m                │\n", result->wavelength);

    printf("  │  Material (copper):                                     │\n");
    printf("  │    Conductivity   = %-14.2e  S/m              │\n", result->conductivity);
    printf("  │    R_s            = %-14.4e  Ohm              │\n", result->R_s);
    printf("  │    Skin depth     = %-14.2f  um               │\n", result->skin_depth * 1.0e6);

    printf("  │  Power dissipation breakdown:                           │\n");
    if (result->P_wall_total > 0.0) {
        printf("  │    P_inner  (r=a) = %-14.6e  (%5.1f%%)          │\n",
            result->P_inner, 100.0 * result->P_inner / result->P_wall_total);
        printf("  │    P_outer  (r=b) = %-14.6e  (%5.1f%%)          │\n",
            result->P_outer, 100.0 * result->P_outer / result->P_wall_total);
        printf("  │    P_z0    (z=0)  = %-14.6e  (%5.1f%%)          │\n",
            result->P_endplate_z0, 100.0 * result->P_endplate_z0 / result->P_wall_total);
        printf("  │    P_zL    (z=L)  = %-14.6e  (%5.1f%%)          │\n",
            result->P_endplate_zL, 100.0 * result->P_endplate_zL / result->P_wall_total);
        printf("  │    P_total        = %-14.6e                    │\n",
            result->P_wall_total);
    }

    printf("  │  Result:                                                │\n");
    printf("  │    Q_0            = %-14.0f                    │\n", result->Q_0);
    printf("  │    G factor       = %-14.2f  Ohm              │\n", result->G_factor);
    printf("  └──────────────────────────────────────────────────────────┘\n");
}


/*=============================================================================
 * Print comparison table
 *============================================================================*/
void q_factor_print_comparison(
    const QFactorResult* analytical,
    const QFactorResult* reference,
    const QFactorResult* full_model
) {
    printf("\n");
    printf("  ┌────────────────────────────────────────────────────────────┐\n");
    printf("  │              Q FACTOR COMPARISON                           │\n");
    printf("  ├────────────────────────────────────────────────────────────┤\n");

    printf("  │  %-20s  %12s", "Parameter", "Analytical");
    if (reference)  printf("  %12s", "Reference");
    if (full_model) printf("  %12s", "Full Model");
    printf("  │\n");

    printf("  │  %-20s  %12s", "────────────────────", "────────────");
    if (reference)  printf("  %12s", "────────────");
    if (full_model) printf("  %12s", "────────────");
    printf("  │\n");

    printf("  │  %-20s  %10.4f  ", "f (MHz)", analytical->frequency_Hz / 1e6);
    if (reference)  printf("  %10.4f  ", reference->frequency_Hz / 1e6);
    if (full_model) printf("  %10.4f  ", full_model->frequency_Hz / 1e6);
    printf("│\n");

    printf("  │  %-20s  %10.4f  ", "R_s (mOhm)", analytical->R_s * 1e3);
    if (reference)  printf("  %10.4f  ", reference->R_s * 1e3);
    if (full_model) printf("  %10.4f  ", full_model->R_s * 1e3);
    printf("│\n");

    printf("  │  %-20s  %10.2f  ", "Skin depth (um)", analytical->skin_depth * 1e6);
    if (reference)  printf("  %10.2f  ", reference->skin_depth * 1e6);
    if (full_model) printf("  %10.2f  ", full_model->skin_depth * 1e6);
    printf("│\n");

    printf("  │  %-20s  %10.0f  ", "Q_0", analytical->Q_0);
    if (reference)  printf("  %10.0f  ", reference->Q_0);
    if (full_model) printf("  %10.0f  ", full_model->Q_0);
    printf("│\n");

    if (reference) {
        double err_ref = (reference->Q_0 - analytical->Q_0) / analytical->Q_0 * 100.0;
        printf("  │  %-20s  %10s  ", "Error vs analytical", "—");
        printf("  %+9.2f%%  ", err_ref);
        if (full_model) {
            double err_full = (full_model->Q_0 - analytical->Q_0) / analytical->Q_0 * 100.0;
            printf("  %+9.2f%%  ", err_full);
        }
        printf("│\n");
    }

    printf("  │  %-20s  %10.2f  ", "G (Ohm)", analytical->G_factor);
    if (reference)  printf("  %10.2f  ", reference->G_factor);
    if (full_model) printf("  %10.2f  ", full_model->G_factor);
    printf("│\n");

    printf("  │                                                            │\n");
    printf("  │  Power breakdown (%%):                                      │\n");

    if (analytical->P_wall_total > 0) {
        printf("  │  %-20s  %9.1f%%  ", "P_inner",
            100.0 * analytical->P_inner / analytical->P_wall_total);
        if (reference && reference->P_wall_total > 0)
            printf("  %9.1f%%  ", 100.0 * reference->P_inner / reference->P_wall_total);
        if (full_model && full_model->P_wall_total > 0)
            printf("  %9.1f%%  ", 100.0 * full_model->P_inner / full_model->P_wall_total);
        printf("│\n");

        printf("  │  %-20s  %9.1f%%  ", "P_outer",
            100.0 * analytical->P_outer / analytical->P_wall_total);
        if (reference && reference->P_wall_total > 0)
            printf("  %9.1f%%  ", 100.0 * reference->P_outer / reference->P_wall_total);
        if (full_model && full_model->P_wall_total > 0)
            printf("  %9.1f%%  ", 100.0 * full_model->P_outer / full_model->P_wall_total);
        printf("│\n");

        printf("  │  %-20s  %9.1f%%  ", "P_endplates",
            100.0 * (analytical->P_endplate_z0 + analytical->P_endplate_zL)
            / analytical->P_wall_total);
        if (reference && reference->P_wall_total > 0)
            printf("  %9.1f%%  ",
                100.0 * (reference->P_endplate_z0 + reference->P_endplate_zL)
                / reference->P_wall_total);
        if (full_model && full_model->P_wall_total > 0)
            printf("  %9.1f%%  ",
                100.0 * (full_model->P_endplate_z0 + full_model->P_endplate_zL)
                / full_model->P_wall_total);
        printf("│\n");
    }

    if (reference && full_model) {
        double dQ = full_model->Q_0 - reference->Q_0;
        double dQ_pct = dQ / reference->Q_0 * 100.0;
        printf("  │                                                            │\n");
        printf("  │  Q_0 change (full vs ref):  %+.0f  (%+.2f%%)             │\n",
            dQ, dQ_pct);
    }

    printf("  └────────────────────────────────────────────────────────────┘\n");
}