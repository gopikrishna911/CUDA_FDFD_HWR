#include "field_export.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*=============================================================================
 * Helper: physical z from grid k-index
 *============================================================================*/
static inline double z_physical(int k, double dz, double z0_offset) {
    return k * dz - z0_offset;
}

/*=============================================================================
 * Helper: find grid k-index closest to a physical z value
 *============================================================================*/
static int k_from_z_physical(double z_phys, double dz, double z0_offset) {
    double k_float = (z_phys + z0_offset) / dz;
    int k = (int)round(k_float);
    return k;
}

/*=============================================================================
 * Helper: find peak |Er| for normalization
 *============================================================================*/
static double find_peak_Er(const EField* E, const GridParams* g) {
    double peak = 0.0;
    for (int k = 0; k <= g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i < g->Nr; i++) {
                double val = fabs(E->Er[idx_Er(g, i, j, k)]);
                if (val > peak) peak = val;
            }
        }
    }
    return peak;
}


/*=============================================================================
 * 2D SLICE: Er(r,z) at phi = 0
 *
 * Outputs the E_r component on the r-z plane at the phi=0 grid line.
 * This shows the classic half-wave structure: sin(pi*z/L) / r.
 *
 * CSV format: r_m, z_m, Er, Er_analytical
 * Only cavity region (k_start to k_end).
 *============================================================================*/
void export_Er_rz(
    const CurlCurlOperator* op,
    const EField* E,
    const FieldExportConfig* config,
    const char* filename
) {
    const GridParams* g = &op->grid;
    int j = 0;  /* phi = 0 */

    /* Find normalization: peak Er in cavity */
    double peak = find_peak_Er(E, g);
    if (peak < 1e-30) peak = 1.0;

    /* Analytical normalization: peak is at r=a, z=L/2 → 1/a */
    double a = g->a;
    double L = config->L_cavity;
    double norm_ana = 1.0 / a;  /* peak of sin(pi*z/L)/r at r=a, z=L/2 */

    FILE* fp = fopen(filename, "w");
    if (!fp) { printf("  Error: cannot open %s\n", filename); return; }

    fprintf(fp, "# Er(r,z) at phi=0\n");
    fprintf(fp, "# Nr=%d, Nz_cavity=%d\n",
        g->Nr, config->k_cavity_end - config->k_cavity_start);
    fprintf(fp, "# Normalized to peak=1\n");
    fprintf(fp, "r_m,z_m,Er,Er_analytical\n");

    for (int k = config->k_cavity_start; k <= config->k_cavity_end; k++) {
        double z = z_physical(k, g->dz, config->z0_offset);
        for (int i = 0; i < g->Nr; i++) {
            double r = a + (i + 0.5) * g->dr;
            if (r > config->b_cavity) continue;

            double Er_val = E->Er[idx_Er(g, i, j, k)] / peak;
            double Er_ana = (sin(M_PI * z / L) / r) / norm_ana;

            fprintf(fp, "%.6f,%.6f,%.6e,%.6e\n", r, z, Er_val, Er_ana);
        }
    }

    fclose(fp);
    printf("  Exported: %s\n", filename);
}


/*=============================================================================
 * 2D SLICE: Er(r,phi) at z = L/2
 *
 * Midplane slice showing azimuthal structure and pipe perturbations.
 *
 * CSV format: r_m, phi_deg, Er
 *============================================================================*/
void export_Er_rphi(
    const CurlCurlOperator* op,
    const EField* E,
    const FieldExportConfig* config,
    const char* filename
) {
    const GridParams* g = &op->grid;

    /* Find k closest to z = L/2 */
    int k_mid = k_from_z_physical(config->L_cavity / 2.0,
        g->dz, config->z0_offset);
    if (k_mid < 0) k_mid = 0;
    if (k_mid > g->Nz) k_mid = g->Nz;

    double peak = find_peak_Er(E, g);
    if (peak < 1e-30) peak = 1.0;

    FILE* fp = fopen(filename, "w");
    if (!fp) { printf("  Error: cannot open %s\n", filename); return; }

    double z_actual = z_physical(k_mid, g->dz, config->z0_offset);
    fprintf(fp, "# Er(r,phi) at z=%.4f m (k=%d, target=%.4f m)\n",
        z_actual, k_mid, config->L_cavity / 2.0);
    fprintf(fp, "# Nr=%d, Nphi=%d\n", g->Nr, g->Nphi);
    fprintf(fp, "# Normalized to peak=1\n");
    fprintf(fp, "r_m,phi_deg,Er\n");

    for (int j = 0; j < g->Nphi; j++) {
        double phi_deg = j * g->dphi * 180.0 / M_PI;
        for (int i = 0; i < g->Nr; i++) {
            double r = g->a + (i + 0.5) * g->dr;
            if (r > config->b_cavity) continue;

            double Er_val = E->Er[idx_Er(g, i, j, k_mid)] / peak;
            fprintf(fp, "%.6f,%.4f,%.6e\n", r, phi_deg, Er_val);
        }
    }

    fclose(fp);
    printf("  Exported: %s\n", filename);
}


/*=============================================================================
 * 2D SURFACE: |H_tan|^2(phi,z) on inner conductor (r = a)
 *
 * H_phi and H_z extrapolated to wall using linear extrapolation.
 * This is the loss density that drives P_inner.
 *
 * CSV format: phi_deg, z_m, Hphi_sq, Hz_sq, Htotal_sq
 * Note: H_phi lives at (i+1/2, j, k+1/2), H_z at (i+1/2, j+1/2, k).
 * We evaluate both on a common (phi_j, z_k) grid by using the available
 * staggered values — H_phi at each (j, k+0.5) and H_z at each (j+0.5, k).
 * For the combined loss density map, we average to cell centers.
 *============================================================================*/
void export_Hloss_inner(
    const CurlCurlOperator* op,
    const CurlE* curlE,
    const FieldExportConfig* config,
    const char* filename
) {
    const GridParams* g = &op->grid;
    int k_start = config->k_cavity_start;
    int k_end = config->k_cavity_end;

    /* Find peak for normalization */
    double peak_sq = 0.0;
    for (int k = k_start; k < k_end; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            double v0 = curlE->curl_phi[idx_curl_phi(g, 0, j, k)];
            double v1 = curlE->curl_phi[idx_curl_phi(g, 1, j, k)];
            double hp = 1.5 * v0 - 0.5 * v1;
            if (hp * hp > peak_sq) peak_sq = hp * hp;
        }
    }
    if (peak_sq < 1e-60) peak_sq = 1.0;

    FILE* fp = fopen(filename, "w");
    if (!fp) { printf("  Error: cannot open %s\n", filename); return; }

    fprintf(fp, "# |H_tan|^2 on inner conductor (r=a=%.4f m)\n", g->a);
    fprintf(fp, "# Nphi=%d, Nz_cavity=%d\n", g->Nphi, k_end - k_start);
    fprintf(fp, "# Normalized to peak |H_phi|^2 = 1\n");
    fprintf(fp, "phi_deg,z_m,Hphi_sq,Hz_sq,Htotal_sq\n");

    for (int k = k_start; k < k_end; k++) {
        double z = z_physical(k, g->dz, config->z0_offset) + 0.5 * g->dz;
        for (int j = 0; j < g->Nphi; j++) {
            double phi_deg = j * g->dphi * 180.0 / M_PI;

            /* H_phi: curl_phi at (i+1/2, j, k+1/2), extrapolate i=0,1 to wall */
            double v0 = curlE->curl_phi[idx_curl_phi(g, 0, j, k)];
            double v1 = curlE->curl_phi[idx_curl_phi(g, 1, j, k)];
            double hp = 1.5 * v0 - 0.5 * v1;

            /* H_z: curl_z at (i+1/2, j+1/2, k), extrapolate i=0,1 to wall.
             * Average over k and k+1 to center at k+1/2 */
            double hz0_k0 = curlE->curl_z[idx_curl_z(g, 0, j, k)];
            double hz1_k0 = curlE->curl_z[idx_curl_z(g, 1, j, k)];
            double hz0_k1 = curlE->curl_z[idx_curl_z(g, 0, j, k + 1)];
            double hz1_k1 = curlE->curl_z[idx_curl_z(g, 1, j, k + 1)];
            double hz_k0 = 1.5 * hz0_k0 - 0.5 * hz1_k0;
            double hz_k1 = 1.5 * hz0_k1 - 0.5 * hz1_k1;
            double hz = 0.5 * (hz_k0 + hz_k1);

            double hp_sq = (hp * hp) / peak_sq;
            double hz_sq = (hz * hz) / peak_sq;
            double ht_sq = hp_sq + hz_sq;

            fprintf(fp, "%.4f,%.6f,%.6e,%.6e,%.6e\n",
                phi_deg, z, hp_sq, hz_sq, ht_sq);
        }
    }

    fclose(fp);
    printf("  Exported: %s\n", filename);
}


/*=============================================================================
 * 2D SURFACE: |H_tan|^2(phi,z) on outer conductor (r = b)
 *============================================================================*/
void export_Hloss_outer(
    const CurlCurlOperator* op,
    const CurlE* curlE,
    const FieldExportConfig* config,
    const char* filename
) {
    const GridParams* g = &op->grid;
    int k_start = config->k_cavity_start;
    int k_end = config->k_cavity_end;

    /* Find outermost radial index for cavity (r = b) */
    int i_outer = (int)round((config->b_cavity - g->a) / g->dr - 0.5);
    if (i_outer < 1) i_outer = 1;
    if (i_outer >= g->Nr) i_outer = g->Nr - 1;
    int i_prev = i_outer - 1;

    /* Find peak for normalization */
    double peak_sq = 0.0;
    for (int k = k_start; k < k_end; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            double v_out = curlE->curl_phi[idx_curl_phi(g, i_outer, j, k)];
            double v_in = curlE->curl_phi[idx_curl_phi(g, i_prev, j, k)];
            double hp = 1.5 * v_out - 0.5 * v_in;
            if (hp * hp > peak_sq) peak_sq = hp * hp;
        }
    }
    if (peak_sq < 1e-60) peak_sq = 1.0;

    FILE* fp = fopen(filename, "w");
    if (!fp) { printf("  Error: cannot open %s\n", filename); return; }

    fprintf(fp, "# |H_tan|^2 on outer conductor (r=b=%.4f m)\n",
        config->b_cavity);
    fprintf(fp, "# i_outer=%d, Nphi=%d, Nz_cavity=%d\n",
        i_outer, g->Nphi, k_end - k_start);
    fprintf(fp, "# Normalized to peak |H_phi|^2 = 1\n");
    fprintf(fp, "phi_deg,z_m,Hphi_sq,Hz_sq,Htotal_sq\n");

    for (int k = k_start; k < k_end; k++) {
        double z = z_physical(k, g->dz, config->z0_offset) + 0.5 * g->dz;
        for (int j = 0; j < g->Nphi; j++) {
            double phi_deg = j * g->dphi * 180.0 / M_PI;

            /* H_phi extrapolated to r=b */
            double v_out = curlE->curl_phi[idx_curl_phi(g, i_outer, j, k)];
            double v_in = curlE->curl_phi[idx_curl_phi(g, i_prev, j, k)];
            double hp = 1.5 * v_out - 0.5 * v_in;

            /* H_z extrapolated to r=b, averaged to k+1/2 */
            double hz_out_k0 = curlE->curl_z[idx_curl_z(g, i_outer, j, k)];
            double hz_in_k0 = curlE->curl_z[idx_curl_z(g, i_prev, j, k)];
            double hz_out_k1 = curlE->curl_z[idx_curl_z(g, i_outer, j, k + 1)];
            double hz_in_k1 = curlE->curl_z[idx_curl_z(g, i_prev, j, k + 1)];
            double hz_k0 = 1.5 * hz_out_k0 - 0.5 * hz_in_k0;
            double hz_k1 = 1.5 * hz_out_k1 - 0.5 * hz_in_k1;
            double hz = 0.5 * (hz_k0 + hz_k1);

            double hp_sq = (hp * hp) / peak_sq;
            double hz_sq = (hz * hz) / peak_sq;

            fprintf(fp, "%.4f,%.6f,%.6e,%.6e,%.6e\n",
                phi_deg, z, hp_sq, hz_sq, hp_sq + hz_sq);
        }
    }

    fclose(fp);
    printf("  Exported: %s\n", filename);
}


/*=============================================================================
 * 2D SURFACE: |H_tan|^2(r,phi) on z=0 endplate
 *
 * Tangential components are H_r and H_phi.
 *============================================================================*/
void export_Hloss_endplate_z0(
    const CurlCurlOperator* op,
    const CurlE* curlE,
    const FieldExportConfig* config,
    const char* filename
) {
    const GridParams* g = &op->grid;
    int k0 = config->k_cavity_start;  /* k-index of z=0 endplate */

    int i_max = (int)floor((config->b_cavity - g->a) / g->dr);
    if (i_max > g->Nr) i_max = g->Nr;

    /* Find peak for normalization */
    double peak_sq = 0.0;
    for (int j = 0; j < g->Nphi; j++) {
        for (int i = 0; i < i_max; i++) {
            double v0 = curlE->curl_phi[idx_curl_phi(g, i, j, k0)];
            double v1 = curlE->curl_phi[idx_curl_phi(g, i, j, k0 + 1)];
            double hp = 1.5 * v0 - 0.5 * v1;
            if (hp * hp > peak_sq) peak_sq = hp * hp;
        }
    }
    if (peak_sq < 1e-60) peak_sq = 1.0;

    FILE* fp = fopen(filename, "w");
    if (!fp) { printf("  Error: cannot open %s\n", filename); return; }

    fprintf(fp, "# |H_tan|^2 on z=0 endplate (k=%d)\n", k0);
    fprintf(fp, "# Nr=%d, Nphi=%d\n", i_max, g->Nphi);
    fprintf(fp, "# Normalized to peak |H_phi|^2 = 1\n");
    fprintf(fp, "r_m,phi_deg,Hr_sq,Hphi_sq,Htotal_sq\n");

    for (int j = 0; j < g->Nphi; j++) {
        double phi_deg = j * g->dphi * 180.0 / M_PI;
        for (int i = 0; i < i_max; i++) {
            double r = g->a + (i + 0.5) * g->dr;

            /* H_r: curl_r at (i, j+1/2, k+1/2), extrapolate k=k0,k0+1 to endplate
             * Note: curl_r is at integer i, so use i and i+1 averaged for i+1/2 */
            double hr0_i = curlE->curl_r[idx_curl_r(g, i, j, k0)];
            double hr1_i = curlE->curl_r[idx_curl_r(g, i, j, k0 + 1)];
            double hr0_ip = curlE->curl_r[idx_curl_r(g, i + 1, j, k0)];
            double hr1_ip = curlE->curl_r[idx_curl_r(g, i + 1, j, k0 + 1)];
            double hr_i = 1.5 * hr0_i - 0.5 * hr1_i;
            double hr_ip = 1.5 * hr0_ip - 0.5 * hr1_ip;
            double hr = 0.5 * (hr_i + hr_ip);

            /* H_phi: curl_phi at (i+1/2, j, k+1/2), extrapolate k=k0,k0+1 */
            double v0 = curlE->curl_phi[idx_curl_phi(g, i, j, k0)];
            double v1 = curlE->curl_phi[idx_curl_phi(g, i, j, k0 + 1)];
            double hp = 1.5 * v0 - 0.5 * v1;

            double hr_sq = (hr * hr) / peak_sq;
            double hp_sq = (hp * hp) / peak_sq;

            fprintf(fp, "%.6f,%.4f,%.6e,%.6e,%.6e\n",
                r, phi_deg, hr_sq, hp_sq, hr_sq + hp_sq);
        }
    }

    fclose(fp);
    printf("  Exported: %s\n", filename);
}


/*=============================================================================
 * 2D SURFACE: |H_tan|^2(r,phi) on z=L endplate
 *============================================================================*/
void export_Hloss_endplate_zL(
    const CurlCurlOperator* op,
    const CurlE* curlE,
    const FieldExportConfig* config,
    const char* filename
) {
    const GridParams* g = &op->grid;
    int kL = config->k_cavity_end;  /* k-index of z=L endplate */

    int i_max = (int)floor((config->b_cavity - g->a) / g->dr);
    if (i_max > g->Nr) i_max = g->Nr;

    /* Find peak for normalization */
    double peak_sq = 0.0;
    for (int j = 0; j < g->Nphi; j++) {
        for (int i = 0; i < i_max; i++) {
            double v0 = curlE->curl_phi[idx_curl_phi(g, i, j, kL - 1)];
            double v1 = curlE->curl_phi[idx_curl_phi(g, i, j, kL - 2)];
            double hp = 1.5 * v0 - 0.5 * v1;
            if (hp * hp > peak_sq) peak_sq = hp * hp;
        }
    }
    if (peak_sq < 1e-60) peak_sq = 1.0;

    FILE* fp = fopen(filename, "w");
    if (!fp) { printf("  Error: cannot open %s\n", filename); return; }

    fprintf(fp, "# |H_tan|^2 on z=L endplate (k=%d)\n", kL);
    fprintf(fp, "# Nr=%d, Nphi=%d\n", i_max, g->Nphi);
    fprintf(fp, "# Normalized to peak |H_phi|^2 = 1\n");
    fprintf(fp, "r_m,phi_deg,Hr_sq,Hphi_sq,Htotal_sq\n");

    for (int j = 0; j < g->Nphi; j++) {
        double phi_deg = j * g->dphi * 180.0 / M_PI;
        for (int i = 0; i < i_max; i++) {
            double r = g->a + (i + 0.5) * g->dr;

            /* H_r: extrapolate k=kL-1, kL-2 to endplate */
            double hr0_i = curlE->curl_r[idx_curl_r(g, i, j, kL - 1)];
            double hr1_i = curlE->curl_r[idx_curl_r(g, i, j, kL - 2)];
            double hr0_ip = curlE->curl_r[idx_curl_r(g, i + 1, j, kL - 1)];
            double hr1_ip = curlE->curl_r[idx_curl_r(g, i + 1, j, kL - 2)];
            double hr_i = 1.5 * hr0_i - 0.5 * hr1_i;
            double hr_ip = 1.5 * hr0_ip - 0.5 * hr1_ip;
            double hr = 0.5 * (hr_i + hr_ip);

            /* H_phi: extrapolate k=kL-1, kL-2 */
            double v0 = curlE->curl_phi[idx_curl_phi(g, i, j, kL - 1)];
            double v1 = curlE->curl_phi[idx_curl_phi(g, i, j, kL - 2)];
            double hp = 1.5 * v0 - 0.5 * v1;

            double hr_sq = (hr * hr) / peak_sq;
            double hp_sq = (hp * hp) / peak_sq;

            fprintf(fp, "%.6f,%.4f,%.6e,%.6e,%.6e\n",
                r, phi_deg, hr_sq, hp_sq, hr_sq + hp_sq);
        }
    }

    fclose(fp);
    printf("  Exported: %s\n", filename);
}


/*=============================================================================
 * 1D PROFILE: Er(r) at z = L/2, phi = 0
 *
 * Includes analytical comparison: Er_ana = 1/r (normalized to peak)
 *============================================================================*/
void export_Er_vs_r(
    const CurlCurlOperator* op,
    const EField* E,
    const FieldExportConfig* config,
    const char* filename
) {
    const GridParams* g = &op->grid;
    int j = 0;  /* phi = 0 */

    int k_mid = k_from_z_physical(config->L_cavity / 2.0,
        g->dz, config->z0_offset);
    if (k_mid < 0) k_mid = 0;
    if (k_mid > g->Nz) k_mid = g->Nz;

    /* Find peak Er on this line for normalization */
    double peak = 0.0;
    for (int i = 0; i < g->Nr; i++) {
        double r = g->a + (i + 0.5) * g->dr;
        if (r > config->b_cavity) break;
        double val = fabs(E->Er[idx_Er(g, i, j, k_mid)]);
        if (val > peak) peak = val;
    }
    if (peak < 1e-30) peak = 1.0;
    double norm_ana = 1.0 / g->a;  /* peak of 1/r */

    FILE* fp = fopen(filename, "w");
    if (!fp) { printf("  Error: cannot open %s\n", filename); return; }

    double z_actual = z_physical(k_mid, g->dz, config->z0_offset);
    fprintf(fp, "# Er(r) at z=%.4f m, phi=0 deg\n", z_actual);
    fprintf(fp, "# Normalized to peak=1\n");
    fprintf(fp, "r_m,Er,Er_analytical\n");

    for (int i = 0; i < g->Nr; i++) {
        double r = g->a + (i + 0.5) * g->dr;
        if (r > config->b_cavity) break;

        double Er_val = E->Er[idx_Er(g, i, j, k_mid)] / peak;
        double Er_ana = (1.0 / r) / norm_ana;

        fprintf(fp, "%.6f,%.6e,%.6e\n", r, Er_val, Er_ana);
    }

    fclose(fp);
    printf("  Exported: %s\n", filename);
}


/*=============================================================================
 * 1D PROFILE: Er(z) at r = r_mid, phi = 0
 *
 * r_mid = (a + b) / 2.
 * Includes analytical comparison: sin(pi*z/L).
 *============================================================================*/
void export_Er_vs_z(
    const CurlCurlOperator* op,
    const EField* E,
    const FieldExportConfig* config,
    const char* filename
) {
    const GridParams* g = &op->grid;
    int j = 0;

    /* Find radial index closest to r_mid */
    double r_mid = (g->a + config->b_cavity) / 2.0;
    int i_mid = (int)round((r_mid - g->a) / g->dr - 0.5);
    if (i_mid < 0) i_mid = 0;
    if (i_mid >= g->Nr) i_mid = g->Nr - 1;
    double r_actual = g->a + (i_mid + 0.5) * g->dr;

    /* Find peak for normalization */
    double peak = 0.0;
    for (int k = config->k_cavity_start; k <= config->k_cavity_end; k++) {
        double val = fabs(E->Er[idx_Er(g, i_mid, j, k)]);
        if (val > peak) peak = val;
    }
    if (peak < 1e-30) peak = 1.0;

    FILE* fp = fopen(filename, "w");
    if (!fp) { printf("  Error: cannot open %s\n", filename); return; }

    double L = config->L_cavity;
    fprintf(fp, "# Er(z) at r=%.4f m, phi=0 deg\n", r_actual);
    fprintf(fp, "# Normalized to peak=1\n");
    fprintf(fp, "z_m,Er,Er_analytical\n");

    for (int k = config->k_cavity_start; k <= config->k_cavity_end; k++) {
        double z = z_physical(k, g->dz, config->z0_offset);

        double Er_val = E->Er[idx_Er(g, i_mid, j, k)] / peak;
        double Er_ana = sin(M_PI * z / L);

        fprintf(fp, "%.6f,%.6e,%.6e\n", z, Er_val, Er_ana);
    }

    fclose(fp);
    printf("  Exported: %s\n", filename);
}


/*=============================================================================
 * 1D PROFILE: Er(phi) at r = r_mid, z = L/2
 *
 * Shows azimuthal uniformity / pipe perturbation.
 *============================================================================*/
void export_Er_vs_phi(
    const CurlCurlOperator* op,
    const EField* E,
    const FieldExportConfig* config,
    const char* filename
) {
    const GridParams* g = &op->grid;

    double r_mid = (g->a + config->b_cavity) / 2.0;
    int i_mid = (int)round((r_mid - g->a) / g->dr - 0.5);
    if (i_mid < 0) i_mid = 0;
    if (i_mid >= g->Nr) i_mid = g->Nr - 1;
    double r_actual = g->a + (i_mid + 0.5) * g->dr;

    int k_mid = k_from_z_physical(config->L_cavity / 2.0,
        g->dz, config->z0_offset);
    if (k_mid < 0) k_mid = 0;
    if (k_mid > g->Nz) k_mid = g->Nz;

    /* Normalization: average value (should be ~constant for TEM) */
    double sum = 0.0;
    for (int j = 0; j < g->Nphi; j++)
        sum += E->Er[idx_Er(g, i_mid, j, k_mid)];
    double avg = sum / g->Nphi;
    if (fabs(avg) < 1e-30) avg = 1.0;

    FILE* fp = fopen(filename, "w");
    if (!fp) { printf("  Error: cannot open %s\n", filename); return; }

    double z_actual = z_physical(k_mid, g->dz, config->z0_offset);
    fprintf(fp, "# Er(phi) at r=%.4f m, z=%.4f m\n", r_actual, z_actual);
    fprintf(fp, "# Normalized to azimuthal average = 1\n");
    fprintf(fp, "phi_deg,Er,Er_deviation_pct\n");

    for (int j = 0; j < g->Nphi; j++) {
        double phi_deg = j * g->dphi * 180.0 / M_PI;
        double Er_val = E->Er[idx_Er(g, i_mid, j, k_mid)];
        double Er_norm = Er_val / avg;
        double dev_pct = (Er_norm - 1.0) * 100.0;

        fprintf(fp, "%.4f,%.6e,%.4e\n", phi_deg, Er_norm, dev_pct);
    }

    fclose(fp);
    printf("  Exported: %s\n", filename);
}


/*=============================================================================
 * 1D PROFILE: |H_phi|(z) on inner wall at phi = 0
 *
 * Analytical comparison: cos(pi*z/L) (normalized to peak).
 * This profile shows the axial loss distribution.
 *============================================================================*/
void export_Hphi_vs_z(
    const CurlCurlOperator* op,
    const CurlE* curlE,
    const FieldExportConfig* config,
    const char* filename
) {
    const GridParams* g = &op->grid;
    int j = 0;  /* phi = 0 */
    int k_start = config->k_cavity_start;
    int k_end = config->k_cavity_end;

    /* Find peak for normalization */
    double peak = 0.0;
    for (int k = k_start; k < k_end; k++) {
        double v0 = curlE->curl_phi[idx_curl_phi(g, 0, j, k)];
        double v1 = curlE->curl_phi[idx_curl_phi(g, 1, j, k)];
        double hp = fabs(1.5 * v0 - 0.5 * v1);
        if (hp > peak) peak = hp;
    }
    if (peak < 1e-30) peak = 1.0;

    FILE* fp = fopen(filename, "w");
    if (!fp) { printf("  Error: cannot open %s\n", filename); return; }

    double L = config->L_cavity;
    fprintf(fp, "# |H_phi|(z) on inner wall (r=a), phi=0 deg\n");
    fprintf(fp, "# Normalized to peak=1\n");
    fprintf(fp, "z_m,Hphi_abs,Hphi_analytical\n");

    for (int k = k_start; k < k_end; k++) {
        double z = z_physical(k, g->dz, config->z0_offset) + 0.5 * g->dz;

        double v0 = curlE->curl_phi[idx_curl_phi(g, 0, j, k)];
        double v1 = curlE->curl_phi[idx_curl_phi(g, 1, j, k)];
        double hp = (1.5 * v0 - 0.5 * v1) / peak;
        double hp_ana = cos(M_PI * z / L);

        fprintf(fp, "%.6f,%.6e,%.6e\n", z, fabs(hp), fabs(hp_ana));
    }

    fclose(fp);
    printf("  Exported: %s\n", filename);
}


/*=============================================================================
 * Export ALL field data in one call
 *============================================================================*/
void export_all_field_data(
    const CurlCurlOperator* op,
    const double* eigenvector,
    const FieldExportConfig* config
) {
    const GridParams* g = &op->grid;
    char filename[512];

    printf("\n  ┌──────────────────────────────────────────────────────┐\n");
    printf("  │  FIELD DATA EXPORT                                   │\n");
    printf("  │  Prefix: %-42s │\n", config->prefix);
    printf("  ├──────────────────────────────────────────────────────┤\n");

    /* Unpack eigenvector into E-field */
    EField E;
    efield_alloc(&E, g);
    unpack_field(eigenvector, &E, op);

    /* Ensure positive Er convention (eigenvectors are ±ambiguous) */
    {
        double peak_val = 0.0;
        double peak_abs = 0.0;
        int k_mid = config->k_cavity_start
            + (config->k_cavity_end - config->k_cavity_start) / 2;
        for (int i = 0; i < g->Nr; i++) {
            double val = E.Er[idx_Er(g, i, 0, k_mid)];
            if (fabs(val) > peak_abs) {
                peak_abs = fabs(val);
                peak_val = val;
            }
        }
        if (peak_val < 0.0) {
            printf("  Note: flipping eigenvector sign (Er was negative)\n");
            for (int n = 0; n < E.size_Er; n++) E.Er[n] = -E.Er[n];
            for (int n = 0; n < E.size_Ephi; n++) E.Ephi[n] = -E.Ephi[n];
            for (int n = 0; n < E.size_Ez; n++) E.Ez[n] = -E.Ez[n];
        }
    }

    /* Compute curl(E) for H-field data */
    CurlE curlE;
    curl_alloc(&curlE, g);
    compute_curl_E(&E, &curlE, g);

    /* --- 2D slices --- */
    printf("  │  2D slices:                                          │\n");

    snprintf(filename, sizeof(filename), "%s_Er_rz.csv", config->prefix);
    export_Er_rz(op, &E, config, filename);

    snprintf(filename, sizeof(filename), "%s_Er_rphi.csv", config->prefix);
    export_Er_rphi(op, &E, config, filename);

    snprintf(filename, sizeof(filename), "%s_Hloss_inner.csv", config->prefix);
    export_Hloss_inner(op, &curlE, config, filename);

    snprintf(filename, sizeof(filename), "%s_Hloss_outer.csv", config->prefix);
    export_Hloss_outer(op, &curlE, config, filename);

    snprintf(filename, sizeof(filename), "%s_Hloss_z0.csv", config->prefix);
    export_Hloss_endplate_z0(op, &curlE, config, filename);

    snprintf(filename, sizeof(filename), "%s_Hloss_zL.csv", config->prefix);
    export_Hloss_endplate_zL(op, &curlE, config, filename);

    /* --- 1D profiles --- */
    printf("  │  1D profiles:                                        │\n");

    snprintf(filename, sizeof(filename), "%s_Er_vs_r.csv", config->prefix);
    export_Er_vs_r(op, &E, config, filename);

    snprintf(filename, sizeof(filename), "%s_Er_vs_z.csv", config->prefix);
    export_Er_vs_z(op, &E, config, filename);

    snprintf(filename, sizeof(filename), "%s_Er_vs_phi.csv", config->prefix);
    export_Er_vs_phi(op, &E, config, filename);

    snprintf(filename, sizeof(filename), "%s_Hphi_vs_z.csv", config->prefix);
    export_Hphi_vs_z(op, &curlE, config, filename);

    printf("  └──────────────────────────────────────────────────────┘\n");

    /* Cleanup */
    efield_free(&E);
    curl_free(&curlE);
}
