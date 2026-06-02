/*=============================================================================
 * field_map_export.cpp   v2
 *
 * Differences vs v1:
 *   (1) Optional complex-phase rotation.  RQI returns a complex eigenvector
 *       v_cx = e^{i*theta} * v_true with arbitrary global phase theta.  Raw
 *       Re(v_cx) therefore can look nothing like the physical TEM mode.
 *       If cfg->eigenvector_imag is non-NULL we find the dominant Er cell,
 *       compute (cos_theta, sin_theta) at that cell, and rotate so the
 *       reference value becomes purely real positive.  Then Re(v) is the
 *       physical mode and Im(v) is the small wall-loss correction.
 *   (2) Writes h.L = Nz_cav * dz (cavity-only), not g->L (extended grid).
 *   (3) Reads the file back and re-prints the on-disk peaks to confirm
 *       what is on disk matches the in-memory diagnostic.
 *============================================================================*/

#include "field_map_export.h"
#include "curl_E.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* (i, j, k) -> linear, i fastest, k slowest -- must match tracker loader */
static inline size_t cc_idx(int i, int j, int k, int Nr, int Nphi) {
    return (size_t)i + (size_t)Nr * ((size_t)j + (size_t)Nphi * (size_t)k);
}

/*--- E-component averages to cell centre ----------------------------------*/

static double Er_cc(const EField* E, const GridParams* g,
                    int i, int j, int k)
{
    int jp = (j + 1) % g->Nphi;
    double a = E->Er[idx_Er(g, i, j,  k    )];
    double b = E->Er[idx_Er(g, i, jp, k    )];
    double c = E->Er[idx_Er(g, i, j,  k + 1)];
    double d = E->Er[idx_Er(g, i, jp, k + 1)];
    return 0.25 * (a + b + c + d);
}

static double Ephi_cc(const EField* E, const GridParams* g,
                      int i, int j, int k)
{
    double a = E->Ephi[idx_Ephi(g, i,     j, k    )];
    double b = E->Ephi[idx_Ephi(g, i + 1, j, k    )];
    double c = E->Ephi[idx_Ephi(g, i,     j, k + 1)];
    double d = E->Ephi[idx_Ephi(g, i + 1, j, k + 1)];
    return 0.25 * (a + b + c + d);
}

static double Ez_cc(const EField* E, const GridParams* g,
                    int i, int j, int k)
{
    int jp = (j + 1) % g->Nphi;
    double a = E->Ez[idx_Ez(g, i,     j,  k)];
    double b = E->Ez[idx_Ez(g, i + 1, j,  k)];
    double c = E->Ez[idx_Ez(g, i,     jp, k)];
    double d = E->Ez[idx_Ez(g, i + 1, jp, k)];
    return 0.25 * (a + b + c + d);
}

/*--- B-component averages (B = -curl(E)/omega) ----------------------------*/

static double Br_cc(const CurlE* curl, const GridParams* g,
                    int i, int j, int k, double inv_omega)
{
    double a = curl->curl_r[idx_curl_r(g, i,     j, k)];
    double b = curl->curl_r[idx_curl_r(g, i + 1, j, k)];
    return -0.5 * (a + b) * inv_omega;
}

static double Bphi_cc(const CurlE* curl, const GridParams* g,
                      int i, int j, int k, double inv_omega)
{
    int jp = (j + 1) % g->Nphi;
    double a = curl->curl_phi[idx_curl_phi(g, i, j,  k)];
    double b = curl->curl_phi[idx_curl_phi(g, i, jp, k)];
    return -0.5 * (a + b) * inv_omega;
}

static double Bz_cc(const CurlE* curl, const GridParams* g,
                    int i, int j, int k, double inv_omega)
{
    double a = curl->curl_z[idx_curl_z(g, i, j, k    )];
    double b = curl->curl_z[idx_curl_z(g, i, j, k + 1)];
    return -0.5 * (a + b) * inv_omega;
}

/*--- in-place complex phase rotation --------------------------------------
 * Multiplies (re + i*im) by e^{-i*theta} = cos(theta) - i*sin(theta), i.e.
 *     new_re = re*cos_t + im*sin_t
 *     new_im = im*cos_t - re*sin_t
 * Operates on N elements of two parallel arrays.                            */
static void phase_rotate_inplace(double* re, double* im, int N,
                                 double cos_t, double sin_t)
{
    for (int n = 0; n < N; n++) {
        double r = re[n];
        double q = im[n];
        re[n] = r * cos_t + q * sin_t;
        im[n] = q * cos_t - r * sin_t;
    }
}

/*=============================================================================
 * Public entry point
 *============================================================================*/
void export_cavity_field_map(
    const CurlCurlOperator*       op,
    const double*                 eigenvector,
    const FieldMapExportConfig*   cfg,
    const char*                   filename)
{
    if (!op || !eigenvector || !cfg || !filename) {
        fprintf(stderr, "[field_map] NULL argument(s)\n");
        return;
    }
    if (cfg->omega <= 0.0) {
        fprintf(stderr, "[field_map] omega must be > 0 (got %.3e)\n", cfg->omega);
        return;
    }

    const GridParams* g = &op->grid;
    const int Nr     = g->Nr;
    const int Nphi   = g->Nphi;
    const int Nz_cav = cfg->k_cavity_end - cfg->k_cavity_start;

    if (Nz_cav <= 0) {
        fprintf(stderr, "[field_map] invalid cavity range: k_start=%d k_end=%d\n",
                cfg->k_cavity_start, cfg->k_cavity_end);
        return;
    }
    if (cfg->k_cavity_end > g->Nz) {
        fprintf(stderr, "[field_map] k_cavity_end=%d exceeds grid Nz=%d\n",
                cfg->k_cavity_end, g->Nz);
        return;
    }

    const size_t Ncells = (size_t)Nr * (size_t)Nphi * (size_t)Nz_cav;

    /*--- 1. Unpack eigenvector (and optionally its imag part) -------------*/
    EField E;
    efield_alloc(&E, g);
    unpack_field(eigenvector, &E, op);

    EField Eim;
    int have_imag = (cfg->eigenvector_imag != NULL);
    if (have_imag) {
        efield_alloc(&Eim, g);
        unpack_field(cfg->eigenvector_imag, &Eim, op);
    }

    /*--- 2. Phase rotation (complex eigenvector only) ---------------------
     * RQI gives v_cx = e^{i*theta} * v_true.  We find the dominant Er cell
     * inside the cavity slab using |v|^2 = re^2 + im^2 (phase-invariant),
     * compute theta there, then rotate so the reference becomes purely real.
     * After rotation, Re(v) is the physical mode shape.                    */
    if (have_imag) {
        double best_mag2 = 0.0;
        double ref_re = 1.0, ref_im = 0.0;
        int    best_i = -1, best_j = -1, best_k = -1;
        for (int k = cfg->k_cavity_start; k < cfg->k_cavity_end; k++) {
            for (int j = 0; j < Nphi; j++) {
                for (int i = 0; i < Nr; i++) {
                    int idx = idx_Er(g, i, j, k);
                    double r = E.Er[idx];
                    double q = Eim.Er[idx];
                    double m2 = r*r + q*q;
                    if (m2 > best_mag2) {
                        best_mag2 = m2;
                        ref_re = r; ref_im = q;
                        best_i = i; best_j = j; best_k = k;
                    }
                }
            }
        }
        double mag_ref = sqrt(best_mag2);
        double cos_t = 1.0, sin_t = 0.0;
        if (mag_ref > 1e-300) {
            cos_t = ref_re / mag_ref;
            sin_t = ref_im / mag_ref;
        }
        double theta_deg = atan2(ref_im, ref_re) * 180.0 / M_PI;
        printf("[field_map] phase rotation: ref Er at (i=%d, j=%d, k=%d), |v|=%.3e, theta=%.2f deg\n",
               best_i, best_j, best_k, mag_ref, theta_deg);

        phase_rotate_inplace(E.Er,   Eim.Er,   E.size_Er,   cos_t, sin_t);
        phase_rotate_inplace(E.Ephi, Eim.Ephi, E.size_Ephi, cos_t, sin_t);
        phase_rotate_inplace(E.Ez,   Eim.Ez,   E.size_Ez,   cos_t, sin_t);

        /* Diagnostic: peak |Im(v)| / peak |Re(v)| after rotation
         * should be ~ 1/(2Q) for a high-Q cavity.                         */
        double maxIm = 0.0, maxRe = 0.0;
        for (int n = 0; n < E.size_Er; n++) {
            double a = fabs(E.Er[n]);   if (a > maxRe) maxRe = a;
            double b = fabs(Eim.Er[n]); if (b > maxIm) maxIm = b;
        }
        printf("[field_map] after rotation: max|Re Er|=%.3e  max|Im Er|=%.3e  ratio=%.2e\n",
               maxRe, maxIm, (maxRe > 0) ? maxIm/maxRe : 0.0);
    }

    /*--- 3. Optional sign flip (eigenvectors are +/- ambiguous) ----------*/
    if (cfg->flip_sign_for_positive_Er) {
        int k_mid = (cfg->k_cavity_start + cfg->k_cavity_end) / 2;
        double peak_abs = 0.0, peak_signed = 0.0;
        for (int i = 0; i < Nr; i++) {
            double v = E.Er[idx_Er(g, i, 0, k_mid)];
            if (fabs(v) > peak_abs) { peak_abs = fabs(v); peak_signed = v; }
        }
        if (peak_signed < 0.0) {
            for (int n = 0; n < E.size_Er; n++)   E.Er[n]   = -E.Er[n];
            for (int n = 0; n < E.size_Ephi; n++) E.Ephi[n] = -E.Ephi[n];
            for (int n = 0; n < E.size_Ez; n++)   E.Ez[n]   = -E.Ez[n];
            printf("[field_map] flipped eigenvector sign (Re part was negative at mid-z)\n");
        }
    }

    /*--- 4. Optional normalization to peak |Er| = 1 ----------------------*/
    double peak_Er = 0.0;
    if (cfg->normalize_to_peak) {
        for (int k = cfg->k_cavity_start; k <= cfg->k_cavity_end; k++) {
            for (int j = 0; j < Nphi; j++) {
                for (int i = 0; i < Nr; i++) {
                    double v = fabs(E.Er[idx_Er(g, i, j, k)]);
                    if (v > peak_Er) peak_Er = v;
                }
            }
        }
        if (peak_Er > 1.0e-30) {
            double inv = 1.0 / peak_Er;
            for (int n = 0; n < E.size_Er; n++)   E.Er[n]   *= inv;
            for (int n = 0; n < E.size_Ephi; n++) E.Ephi[n] *= inv;
            for (int n = 0; n < E.size_Ez; n++)   E.Ez[n]   *= inv;
            printf("[field_map] normalized: divisor (peak |Er|) = %.6e\n", peak_Er);
        } else {
            printf("[field_map] WARNING: peak |Er| ~ 0, skipping normalization\n");
            peak_Er = 0.0;
        }
    }

    /*--- 5. Compute curl(E) for the B-field ------------------------------*/
    CurlE curlE;
    curl_alloc(&curlE, g);
    compute_curl_E(&E, &curlE, g);

    /*--- 6. Build cell-centred arrays ------------------------------------*/
    double* Er_out   = (double*)malloc(Ncells * sizeof(double));
    double* Ephi_out = (double*)malloc(Ncells * sizeof(double));
    double* Ez_out   = (double*)malloc(Ncells * sizeof(double));
    double* Br_out   = (double*)malloc(Ncells * sizeof(double));
    double* Bphi_out = (double*)malloc(Ncells * sizeof(double));
    double* Bz_out   = (double*)malloc(Ncells * sizeof(double));

    if (!Er_out || !Ephi_out || !Ez_out || !Br_out || !Bphi_out || !Bz_out) {
        fprintf(stderr, "[field_map] out of memory\n");
        goto cleanup;
    }

    {
        double inv_omega = 1.0 / cfg->omega;
        for (int k_cav = 0; k_cav < Nz_cav; k_cav++) {
            int k = cfg->k_cavity_start + k_cav;   /* extended-grid k */
            for (int j = 0; j < Nphi; j++) {
                for (int i = 0; i < Nr; i++) {
                    size_t n = cc_idx(i, j, k_cav, Nr, Nphi);
                    Er_out  [n] = Er_cc(  &E,     g, i, j, k);
                    Ephi_out[n] = Ephi_cc(&E,     g, i, j, k);
                    Ez_out  [n] = Ez_cc(  &E,     g, i, j, k);
                    Br_out  [n] = Br_cc(  &curlE, g, i, j, k, inv_omega);
                    Bphi_out[n] = Bphi_cc(&curlE, g, i, j, k, inv_omega);
                    Bz_out  [n] = Bz_cc(  &curlE, g, i, j, k, inv_omega);
                }
            }
        }
    }

    /*--- 7. Diagnostics: peak locations and ratios -----------------------*/
    {
        double pE = 0.0, pB = 0.0;
        size_t  argE = 0, argB = 0;
        const char* compE = "Er"; const char* compB = "Br";
        const char* names[6] = {"Er","Ephi","Ez","Br","Bphi","Bz"};
        double* arrs[6] = {Er_out, Ephi_out, Ez_out, Br_out, Bphi_out, Bz_out};

        for (int c = 0; c < 6; c++) {
            double cm = 0.0; size_t ci = 0;
            for (size_t n = 0; n < Ncells; n++) {
                double v = fabs(arrs[c][n]);
                if (v > cm) { cm = v; ci = n; }
            }
            int k_cav = (int)(ci / ((size_t)Nr * Nphi));
            int rem   = (int)(ci % ((size_t)Nr * Nphi));
            int j     = rem / Nr;
            int i     = rem % Nr;
            printf("[field_map]  |%s|max = %.4e at (i=%d, j=%d, k_cav=%d)\n",
                   names[c], cm, i, j, k_cav);
            if (c < 3 && cm > pE) { pE = cm; argE = ci; compE = names[c]; }
            if (c >=3 && cm > pB) { pB = cm; argB = ci; compB = names[c]; }
        }
        (void)argE; (void)argB; (void)compE; (void)compB;
        printf("[field_map] |E|max / |B|max  = %.4e  (vacuum: 2.998e+08)\n",
               pE / (pB > 0 ? pB : 1.0));
    }

    /*--- 8. Write file ----------------------------------------------------*/
    {
        FILE* fp = fopen(filename, "wb");
        if (!fp) {
            fprintf(stderr, "[field_map] cannot open %s for writing\n", filename);
            goto cleanup;
        }

        FieldMapHeader h;
        memset(&h, 0, sizeof(h));
        h.magic                = FIELD_MAP_MAGIC;
        h.version              = FIELD_MAP_VERSION;
        h.Nr                   = Nr;
        h.Nphi                 = Nphi;
        h.Nz_cav               = Nz_cav;
        h.a                    = g->a;
        h.b                    = g->b;                /* extended outer radius */
        h.L                    = (double)Nz_cav * g->dz;   /* CAVITY-only L */
        h.dr                   = g->dr;
        h.dphi                 = g->dphi;
        h.dz                   = g->dz;
        h.omega                = cfg->omega;
        h.freq_Hz              = cfg->omega / (2.0 * M_PI);
        h.peak_Er_before_norm  = cfg->normalize_to_peak ? peak_Er : 0.0;

        size_t nh = fwrite(&h, sizeof(h), 1, fp);
        size_t n1 = fwrite(Er_out,   sizeof(double), Ncells, fp);
        size_t n2 = fwrite(Ephi_out, sizeof(double), Ncells, fp);
        size_t n3 = fwrite(Ez_out,   sizeof(double), Ncells, fp);
        size_t n4 = fwrite(Br_out,   sizeof(double), Ncells, fp);
        size_t n5 = fwrite(Bphi_out, sizeof(double), Ncells, fp);
        size_t n6 = fwrite(Bz_out,   sizeof(double), Ncells, fp);
        fclose(fp);

        if (nh != 1 || n1 != Ncells || n2 != Ncells || n3 != Ncells
                    || n4 != Ncells || n5 != Ncells || n6 != Ncells) {
            fprintf(stderr, "[field_map] short write to %s\n", filename);
            goto cleanup;
        }

        double mb = (sizeof(h) + 6.0 * (double)Ncells * sizeof(double))
                    / (1024.0 * 1024.0);
        printf("[field_map] wrote %s\n", filename);
        printf("            grid = %d x %d x %d cells, %.2f MB\n",
               Nr, Nphi, Nz_cav, mb);
        printf("            header: a=%.4f m, b=%.4f m, L=%.4f m (cavity-only)\n",
               h.a, h.b, h.L);
        printf("            freq = %.4f MHz, omega = %.6e rad/s\n",
               h.freq_Hz / 1.0e6, h.omega);
    }

    /*--- 9. Read-back verification ---------------------------------------*/
    {
        FILE* fp = fopen(filename, "rb");
        if (fp) {
            FieldMapHeader hr;
            if (fread(&hr, sizeof(hr), 1, fp) == 1
                && hr.magic == FIELD_MAP_MAGIC) {
                /* Skip Er and look at peak of file-stored Er */
                double maxRe = 0.0;
                double buf;
                for (size_t n = 0; n < Ncells; n++) {
                    if (fread(&buf, sizeof(double), 1, fp) != 1) break;
                    if (fabs(buf) > maxRe) maxRe = fabs(buf);
                }
                printf("[field_map] verify: on-disk peak |Er| = %.4e (expect ~%.4e)\n",
                       maxRe, cfg->normalize_to_peak ? 1.0 : peak_Er);
            }
            fclose(fp);
        }
    }

cleanup:
    free(Er_out);
    free(Ephi_out);
    free(Ez_out);
    free(Br_out);
    free(Bphi_out);
    free(Bz_out);
    efield_free(&E);
    if (have_imag) efield_free(&Eim);
    curl_free(&curlE);
}
