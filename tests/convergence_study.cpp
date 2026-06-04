/*=============================================================================
 * convergence_study.cpp
 *
 * Grid convergence study for the UNPERTURBED coaxial half-wave resonator
 * (HWR) cavity.  No radial pipes, no endcap pipes, no inner-conductor
 * ports — just the bare coaxial pillbox so the underlying FDFD bulk
 * discretization can be characterized in isolation.
 *
 *   Phase A — PEC:
 *       Real eigenproblem, RQI + MINRES inner solve (library workspace).
 *       Reports f_PEC and Q_PEC_surf (surface integral with copper R_s).
 *
 *   Phase B — IBC (grid-plane Leontovich on all four cavity walls):
 *       Complex eigenproblem with finite Z_s = R_s(1+j); complex RQI
 *       with GMRES inner solve, seeded by the converged PEC eigenvector.
 *       Reports f_IBC and Q_IBC = -Re(k^2)/Im(k^2) directly from the
 *       eigenvalue.
 *
 * Richardson extrapolation is applied to the two finest converged levels
 * assuming second-order accuracy.
 *
 * Grid sizing per level (target dr):
 *     r*dphi ~ 1.5 * dr   (at r = b)
 *     dz     ~ 2.0 * dr
 *
 * Convergence levels: dr = 14, 12, 10, 8, 6, 4 mm.
 *============================================================================*/

#include "cuda_operator.h"
#include "cuda_eigensolver.h"
#include "cuda_fields.h"
#include "cuda_vector_ops.h"
#include "curlcurl_operator.h"
#include "q_factor.h"
#include "r_over_q.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <time.h>

#define SIGMA_CU    5.8e7   /* copper conductivity (S/m) */


 /*=============================================================================
  * Convergence level result
  *============================================================================*/
typedef struct {
    /* Grid parameters */
    double dr_target_mm;
    int    Nr, Nphi, Nz;
    double dr_mm, dphi_deg, dz_mm;
    double rdphi_b_mm;
    long long n_dofs;
    double memory_est_GB;

    /* Execution */
    int    ran;
    double wall_pec_sec;
    double wall_ibc_sec;
    double wall_total_sec;

    /* Phase A — PEC results (real eigenvalue, surface-integral Q) */
    int    pec_converged;
    double pec_k_squared;
    double pec_frequency_Hz;
    double pec_Q_surf;          /* Q from surface integral with copper R_s */
    double pec_G_factor;
    double pec_R_over_Q;

    /* Phase B — IBC results (complex eigenvalue, Q from -Re/Im) */
    int    ibc_converged;
    double ibc_k2_re;
    double ibc_k2_im;
    double ibc_frequency_Hz;
    double ibc_Q_eig;           /* Q from eigenvalue, paper's number */
    int    ibc_iterations;
    double ibc_residual;
    double ibc_R_over_Q;
} ConvLevel;


/*=============================================================================
 * Main
 *============================================================================*/
int main() {
    printf("\n");
    printf("****************************************************************\n");
    printf("*   CONVERGENCE STUDY — UNPERTURBED COAXIAL HWR CAVITY        *\n");
    printf("****************************************************************\n\n");

    cuda_print_device_info();

    /*=========================================================================
     * Geometry — unperturbed cavity only (no pipes, no ports)
     *========================================================================*/
    double a = 1.0 / 3.0;
    double b = 1.0;
    double L = 1.395;
    int    num_passes = 10;     /* for R/Q geometric averaging */

    double c0 = 299792458.0;

    printf("  Geometry: a = %.4f m, b = %.4f m, L = %.4f m\n\n", a, b, L);

    /*=========================================================================
     * Analytical reference values
     *========================================================================*/
    QFactorResult qf_analytical = compute_q_analytical_coaxial_hwr(
        a, b, L, Q_SIGMA_CU);
    RoverQResult  roq_analytical = compute_r_over_q_analytical(a, b, L);

    printf("  Analytical (unperturbed cavity):\n");
    printf("    f       = %.6f MHz\n", qf_analytical.frequency_Hz / 1e6);
    printf("    Q_0     = %.0f\n", qf_analytical.Q_0);
    printf("    R/Q     = %.4f Ohm\n\n", roq_analytical.R_over_Q_crossing_linac);

    /*=========================================================================
     * Define convergence levels by target dr (mm)
     *========================================================================*/
    double dr_targets[] = { 14.0, 12.0, 10.0, 8.0, 6.0, 4.0 };
    int n_levels = sizeof(dr_targets) / sizeof(dr_targets[0]);

    double ratio_rdphi_dr = 1.5;
    double ratio_dz_dr = 2.0;
    double bytes_per_dof = 40.0;

    /* GPU memory */
    size_t gpu_free_bytes, gpu_total_bytes;
    cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
    double gpu_available_GB = gpu_free_bytes / (1024.0 * 1024.0 * 1024.0);
    double gpu_usable_GB = gpu_available_GB * 0.85;

    printf("  GPU memory: %.1f GB free / %.1f GB total (using %.1f GB)\n\n",
        gpu_available_GB, gpu_total_bytes / (1024.0 * 1024.0 * 1024.0),
        gpu_usable_GB);

    /*=========================================================================
     * Compute grid parameters for each level
     *========================================================================*/
    ConvLevel* levels = (ConvLevel*)calloc(n_levels, sizeof(ConvLevel));

    printf("  ┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("  │ Lvl  dr(mm) r·dφ(mm) dz(mm)   Nr   Nphi   Nz     DOFs    Memory     │\n");
    printf("  ├──────────────────────────────────────────────────────────────────────┤\n");

    for (int lvl = 0; lvl < n_levels; lvl++) {
        ConvLevel* Lv = &levels[lvl];
        double dr_m = dr_targets[lvl] * 1e-3;
        Lv->dr_target_mm = dr_targets[lvl];

        /* Radial */
        Lv->Nr = (int)round((b - a) / dr_m);
        if (Lv->Nr < 4) Lv->Nr = 4;

        /* Azimuthal */
        double dphi_target = ratio_rdphi_dr * dr_m / b;
        Lv->Nphi = (int)round(2.0 * M_PI / dphi_target);
        if (Lv->Nphi < 16) Lv->Nphi = 16;
        if (Lv->Nphi % 2 != 0) Lv->Nphi++;

        /* Axial */
        double dz_target = ratio_dz_dr * dr_m;
        Lv->Nz = (int)round(L / dz_target);
        if (Lv->Nz < 4) Lv->Nz = 4;

        /* Actual cell sizes */
        Lv->dr_mm = (b - a) / Lv->Nr * 1000.0;
        Lv->dphi_deg = 360.0 / Lv->Nphi;
        Lv->dz_mm = L / Lv->Nz * 1000.0;
        Lv->rdphi_b_mm = b * (2.0 * M_PI / Lv->Nphi) * 1000.0;

        /* DOF estimate (Yee cylindrical) */
        long long Nr_ll = Lv->Nr;
        long long Nphi_ll = Lv->Nphi;
        long long Nz_ll = Lv->Nz;
        Lv->n_dofs = Nr_ll * Nphi_ll * (Nz_ll + 1)
            + (Nr_ll + 1) * Nphi_ll * (Nz_ll + 1)
            + (Nr_ll + 1) * Nphi_ll * Nz_ll;
        Lv->memory_est_GB = Lv->n_dofs * bytes_per_dof
            / (1024.0 * 1024.0 * 1024.0);

        const char* status = (Lv->memory_est_GB <= gpu_usable_GB) ? "" : " SKIP";
        printf("  │ %2d   %5.1f  %6.1f   %5.1f  %4d  %5d  %4d  %7.1fM  %5.1f GB%s │\n",
            lvl + 1, Lv->dr_target_mm, Lv->rdphi_b_mm, Lv->dz_mm,
            Lv->Nr, Lv->Nphi, Lv->Nz,
            Lv->n_dofs / 1e6, Lv->memory_est_GB, status);
    }
    printf("  └──────────────────────────────────────────────────────────────────────┘\n\n");

    /*=========================================================================
     * Run each level
     *========================================================================*/
    for (int lvl = 0; lvl < n_levels; lvl++) {
        ConvLevel* Lv = &levels[lvl];

        if (Lv->memory_est_GB > gpu_usable_GB) {
            printf("  --- Level %d (dr=%.1f mm): SKIPPED (%.1f GB > %.1f GB) ---\n\n",
                lvl + 1, Lv->dr_target_mm, Lv->memory_est_GB, gpu_usable_GB);
            Lv->ran = 0;
            continue;
        }

        printf("  === Level %d: dr=%.1f mm, dphi=%.2f°, dz=%.1f mm, %.1fM DOFs ===\n",
            lvl + 1, Lv->dr_target_mm, Lv->dphi_deg, Lv->dz_mm,
            Lv->n_dofs / 1e6);

        clock_t t_level_start = clock();

        /* Bare-cavity grid */
        GridParams grid;
        grid_init(&grid, a, b, L, Lv->Nr, Lv->Nphi, Lv->Nz);
        cuda_grid_init(&grid);

        /* CPU operator (no ports) */
        CurlCurlOperator cpu_op;
        curlcurl_op_init(&cpu_op, &grid);
        int n = cpu_op.n_total;

        /* TEM initial guess: E_r = sin(pi z / L) / r */
        double* h_x = (double*)calloc(n, sizeof(double));
        for (int k = 0; k <= grid.Nz; k++) {
            double z = k * grid.dz;
            for (int j = 0; j < grid.Nphi; j++) {
                for (int i = 0; i < grid.Nr; i++) {
                    double r = grid.a + (i + 0.5) * grid.dr;
                    int idx = cpu_op.offset_Er + idx_Er(&grid, i, j, k);
                    h_x[idx] = sin(M_PI * z / L) / r;
                }
            }
        }

        double* d_x;
        gpu_vector_alloc(&d_x, n);
        gpu_vector_to_device(d_x, h_x, n);

        /*-----------------------------------------------------------------
         * Phase A : PEC eigensolve (real, RQI + MINRES)
         *-----------------------------------------------------------------*/
        printf("    [phase A] PEC eigensolve...\n"); fflush(stdout);
        clock_t t_pec0 = clock();

        GPU_Operator gpu_op_pec;
        gpu_operator_init(&gpu_op_pec, &cpu_op);

        EigensolverWorkspace ws;
        eigensolver_workspace_init(&ws, n);

        double k2_target = (M_PI / L) * (M_PI / L);

        GPU_EigenResult pec_res = gpu_rqi_ws(
            &gpu_op_pec, d_x, k2_target, 20, 1e-8, &ws);

        eigensolver_workspace_free(&ws);
        Lv->wall_pec_sec = (double)(clock() - t_pec0) / CLOCKS_PER_SEC;

        double k2_pec = pec_res.eigenvalue;
        gpu_vector_to_host(h_x, d_x, n);

        /* Surface-integral Q (no apertures → pass NULL) */
        QFactorResult qf_pec = compute_q_factor(
            &cpu_op, h_x, k2_pec, Q_SIGMA_CU, b, NULL);
        RoverQResult roq_pec = compute_r_over_q(
            &cpu_op, h_x, k2_pec, b, L, 0.0, num_passes);

        Lv->pec_converged = pec_res.converged;
        Lv->pec_k_squared = k2_pec;
        Lv->pec_frequency_Hz = sqrt(fabs(k2_pec)) * c0 / (2.0 * M_PI);
        Lv->pec_Q_surf = qf_pec.Q_0;
        Lv->pec_G_factor = qf_pec.G_factor;
        Lv->pec_R_over_Q = roq_pec.R_over_Q_crossing_linac;

        printf("    [phase A] f_PEC=%.6f MHz, Q_surf=%.0f, R/Q=%.4f Ohm  (%.1f s)\n",
            Lv->pec_frequency_Hz / 1e6, Lv->pec_Q_surf, Lv->pec_R_over_Q,
            Lv->wall_pec_sec);

        gpu_operator_free(&gpu_op_pec);

        /*-----------------------------------------------------------------
         * Phase B : IBC eigensolve (complex, RQI + GMRES,
         *           seeded by PEC eigenvector)
         *-----------------------------------------------------------------*/
        printf("    [phase B] initializing complex IBC operator...\n"); fflush(stdout);

        GPU_Operator gpu_op_ibc;
        gpu_operator_init_complex(&gpu_op_ibc, &cpu_op);

        /* Complex seed: [PEC eigenvector | zeros] */
        int n2 = 2 * n;
        double* h_x_cx = (double*)calloc(n2, sizeof(double));
        memcpy(h_x_cx, h_x, n * sizeof(double));

        double* d_x_cx;
        gpu_vector_alloc(&d_x_cx, n2);
        gpu_vector_to_device(d_x_cx, h_x_cx, n2);

        printf("    [phase B] complex RQI + GMRES(50), seeded from PEC...\n"); fflush(stdout);
        clock_t t_ibc0 = clock();
        GPU_ComplexEigenResult ibc_res = gpu_rqi_complex(
            &gpu_op_ibc, d_x_cx, k2_pec, SIGMA_CU, 30, 1e-8, 50);
        Lv->wall_ibc_sec = (double)(clock() - t_ibc0) / CLOCKS_PER_SEC;

        /* Pull complex eigenvector back; real part is what R/Q uses */
        gpu_vector_to_host(h_x_cx, d_x_cx, n2);
        double* h_x_ibc_re = h_x_cx;   /* first n doubles */

        RoverQResult roq_ibc = compute_r_over_q(
            &cpu_op, h_x_ibc_re, ibc_res.k2_re, b, L, 0.0, num_passes);

        Lv->ibc_converged = ibc_res.converged;
        Lv->ibc_k2_re = ibc_res.k2_re;
        Lv->ibc_k2_im = ibc_res.k2_im;
        Lv->ibc_frequency_Hz = ibc_res.frequency_Hz;
        Lv->ibc_Q_eig = ibc_res.Q_factor;
        Lv->ibc_iterations = ibc_res.iterations;
        Lv->ibc_residual = ibc_res.residual;
        Lv->ibc_R_over_Q = roq_ibc.R_over_Q_crossing_linac;

        printf("    [phase B] f_IBC=%.6f MHz, Q_IBC=%.0f, R/Q=%.4f Ohm  "
            "(iters=%d, res=%.2e, %.1f s)%s\n",
            Lv->ibc_frequency_Hz / 1e6, Lv->ibc_Q_eig, Lv->ibc_R_over_Q,
            Lv->ibc_iterations, Lv->ibc_residual, Lv->wall_ibc_sec,
            Lv->ibc_converged ? "" : "  [NOT CONVERGED]");

        clock_t t_level_end = clock();
        Lv->wall_total_sec = (double)(t_level_end - t_level_start) / CLOCKS_PER_SEC;
        Lv->ran = 1;
        printf("    level total: %.1f s\n\n", Lv->wall_total_sec);

        /* Cleanup this level */
        gpu_vector_free(d_x_cx);
        free(h_x_cx);
        gpu_operator_free(&gpu_op_ibc);
        r_over_q_free(&roq_pec);
        r_over_q_free(&roq_ibc);
        free(h_x);
        gpu_vector_free(d_x);
        curlcurl_op_free(&cpu_op);
    }

    /*=========================================================================
     * Convergence summary tables
     *========================================================================*/
     /* Use finest converged level as the reference for both methods */
    int i_finest = -1;
    for (int i = n_levels - 1; i >= 0; i--) {
        if (levels[i].ran) { i_finest = i; break; }
    }

    if (i_finest < 0) {
        printf("\n  WARNING: no levels ran successfully.\n");
        free(levels);
        return 1;
    }

    ConvLevel* Lref = &levels[i_finest];
    double f_pec_ref = Lref->pec_frequency_Hz;
    double Q_pec_ref = Lref->pec_Q_surf;
    double f_ibc_ref = Lref->ibc_frequency_Hz;
    double Q_ibc_ref = Lref->ibc_Q_eig;
    double RoQ_ibc_ref = Lref->ibc_R_over_Q;

    /* ----- Table 1: PEC results ----- */
    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │              PHASE A — PEC CONVERGENCE (unperturbed cavity)                      │\n");
    printf("  │  Reference (lvl %d):  f=%.4f MHz, Q_surf=%.0f                                  │\n",
        i_finest + 1, f_pec_ref / 1e6, Q_pec_ref);
    printf("  │  Analytical:          f=%.4f MHz, Q_0=%.0f                                    │\n",
        qf_analytical.frequency_Hz / 1e6, qf_analytical.Q_0);
    printf("  ├──────────────────────────────────────────────────────────────────────────────────┤\n");
    printf("  │ Lvl  dr(mm)  dz(mm)     DOFs   f(MHz)     Δf(Hz)    Q_surf  R/Q(Ω)   T(s)      │\n");
    printf("  │ ───  ──────  ──────  ────────  ─────────  ────────  ──────  ───────  ──────    │\n");

    for (int lvl = 0; lvl < n_levels; lvl++) {
        ConvLevel* Lv = &levels[lvl];
        if (!Lv->ran) {
            printf("  │ %2d   %5.1f   %5.1f   %6.1fM  %58s│\n",
                lvl + 1, Lv->dr_target_mm, Lv->dz_mm, Lv->n_dofs / 1e6,
                "— SKIPPED —");
            continue;
        }
        double df = Lv->pec_frequency_Hz - f_pec_ref;
        printf("  │ %2d   %5.1f   %5.1f   %6.1fM  %9.4f  %+8.1f  %6.0f  %7.4f  %6.1f%s   │\n",
            lvl + 1, Lv->dr_target_mm, Lv->dz_mm, Lv->n_dofs / 1e6,
            Lv->pec_frequency_Hz / 1e6, df, Lv->pec_Q_surf,
            Lv->pec_R_over_Q, Lv->wall_pec_sec,
            (lvl == i_finest) ? "*" : " ");
    }
    printf("  └──────────────────────────────────────────────────────────────────────────────────┘\n");

    /* ----- Table 2: IBC results ----- */
    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │              PHASE B — IBC CONVERGENCE (unperturbed cavity)                      │\n");
    printf("  │  Reference (lvl %d):  f=%.4f MHz, Q_eig=%.0f, R/Q=%.4f Ω                     │\n",
        i_finest + 1, f_ibc_ref / 1e6, Q_ibc_ref, RoQ_ibc_ref);
    printf("  ├──────────────────────────────────────────────────────────────────────────────────┤\n");
    printf("  │ Lvl  dr(mm)  dz(mm)     DOFs   f(MHz)     Δf(Hz)    Q_eig   R/Q(Ω)   T(s)      │\n");
    printf("  │ ───  ──────  ──────  ────────  ─────────  ────────  ──────  ───────  ──────    │\n");

    for (int lvl = 0; lvl < n_levels; lvl++) {
        ConvLevel* Lv = &levels[lvl];
        if (!Lv->ran) {
            printf("  │ %2d   %5.1f   %5.1f   %6.1fM  %58s│\n",
                lvl + 1, Lv->dr_target_mm, Lv->dz_mm, Lv->n_dofs / 1e6,
                "— SKIPPED —");
            continue;
        }
        double df = Lv->ibc_frequency_Hz - f_ibc_ref;
        printf("  │ %2d   %5.1f   %5.1f   %6.1fM  %9.4f  %+8.1f  %6.0f  %7.4f  %6.1f%s%s │\n",
            lvl + 1, Lv->dr_target_mm, Lv->dz_mm, Lv->n_dofs / 1e6,
            Lv->ibc_frequency_Hz / 1e6, df, Lv->ibc_Q_eig,
            Lv->ibc_R_over_Q, Lv->wall_ibc_sec,
            (lvl == i_finest) ? "*" : " ",
            Lv->ibc_converged ? "" : "!");
    }
    printf("  └──────────────────────────────────────────────────────────────────────────────────┘\n");
    printf("    * = finest grid (reference);  ! = IBC did not converge\n");

    /*=========================================================================
     * Richardson extrapolation (PEC and IBC, two finest converged levels)
     *========================================================================*/
    int i_fine = -1, i_coarse = -1;
    for (int i = n_levels - 1; i >= 0; i--) {
        if (levels[i].ran) {
            if (i_fine < 0) i_fine = i;
            else if (i_coarse < 0) { i_coarse = i; break; }
        }
    }

    if (i_fine >= 0 && i_coarse >= 0) {
        ConvLevel* Lf = &levels[i_fine];
        ConvLevel* Lc = &levels[i_coarse];
        double r = Lc->dr_target_mm / Lf->dr_target_mm;
        double p = 2.0;        /* assumed second-order */
        double rp = pow(r, p);

        printf("\n  Richardson extrapolation (levels %d & %d, h-ratio = %.2f, p = %.1f):\n",
            i_coarse + 1, i_fine + 1, r, p);

        /* PEC */
        double f_pec_rich = Lf->pec_frequency_Hz
            + (Lf->pec_frequency_Hz - Lc->pec_frequency_Hz) / (rp - 1.0);
        double Q_pec_rich = Lf->pec_Q_surf
            + (Lf->pec_Q_surf - Lc->pec_Q_surf) / (rp - 1.0);
        double RoQp_rich = Lf->pec_R_over_Q
            + (Lf->pec_R_over_Q - Lc->pec_R_over_Q) / (rp - 1.0);

        /* IBC */
        double f_ibc_rich = Lf->ibc_frequency_Hz
            + (Lf->ibc_frequency_Hz - Lc->ibc_frequency_Hz) / (rp - 1.0);
        double Q_ibc_rich = Lf->ibc_Q_eig
            + (Lf->ibc_Q_eig - Lc->ibc_Q_eig) / (rp - 1.0);
        double RoQi_rich = Lf->ibc_R_over_Q
            + (Lf->ibc_R_over_Q - Lc->ibc_R_over_Q) / (rp - 1.0);

        printf("    ┌──────────────────────────────────────────────────────────────────────────┐\n");
        printf("    │  Quantity            L%d (coarse)   L%d (fine)    Richardson (h→0)   │\n",
            i_coarse + 1, i_fine + 1);
        printf("    │  ──────────────────  ───────────   ──────────   ────────────────       │\n");
        printf("    │  PEC  f (MHz)        %11.6f   %11.6f   %11.6f         │\n",
            Lc->pec_frequency_Hz / 1e6, Lf->pec_frequency_Hz / 1e6, f_pec_rich / 1e6);
        printf("    │  PEC  Q_surf         %11.1f   %11.1f   %11.1f         │\n",
            Lc->pec_Q_surf, Lf->pec_Q_surf, Q_pec_rich);
        printf("    │  PEC  R/Q (Ω)        %11.4f   %11.4f   %11.4f         │\n",
            Lc->pec_R_over_Q, Lf->pec_R_over_Q, RoQp_rich);
        printf("    │  ──────────────────  ───────────   ──────────   ────────────────       │\n");
        printf("    │  IBC  f (MHz)        %11.6f   %11.6f   %11.6f         │\n",
            Lc->ibc_frequency_Hz / 1e6, Lf->ibc_frequency_Hz / 1e6, f_ibc_rich / 1e6);
        printf("    │  IBC  Q_eig          %11.1f   %11.1f   %11.1f         │\n",
            Lc->ibc_Q_eig, Lf->ibc_Q_eig, Q_ibc_rich);
        printf("    │  IBC  R/Q (Ω)        %11.4f   %11.4f   %11.4f         │\n",
            Lc->ibc_R_over_Q, Lf->ibc_R_over_Q, RoQi_rich);
        printf("    └──────────────────────────────────────────────────────────────────────────┘\n");
    }
    else {
        printf("\n  (Richardson extrapolation needs >=2 converged levels — skipped.)\n");
    }

    /*=========================================================================
     * Export CSV (one row per level, both methods)
     *========================================================================*/
    {
        FILE* fp = fopen("convergence_unperturbed.csv", "w");
        if (fp) {
            fprintf(fp,
                "level,dr_mm,dphi_deg,dz_mm,n_dofs,"
                "pec_f_Hz,pec_Q_surf,pec_R_over_Q,pec_wall_sec,"
                "ibc_f_Hz,ibc_Q_eig,ibc_R_over_Q,ibc_iters,ibc_residual,ibc_wall_sec,"
                "total_wall_sec\n");
            for (int i = 0; i < n_levels; i++) {
                ConvLevel* Lv = &levels[i];
                if (!Lv->ran) continue;
                fprintf(fp,
                    "%d,%.4f,%.6f,%.4f,%lld,"
                    "%.6f,%.4f,%.6f,%.2f,"
                    "%.6f,%.4f,%.6f,%d,%.6e,%.2f,"
                    "%.2f\n",
                    i + 1, Lv->dr_mm, Lv->dphi_deg, Lv->dz_mm, Lv->n_dofs,
                    Lv->pec_frequency_Hz, Lv->pec_Q_surf, Lv->pec_R_over_Q, Lv->wall_pec_sec,
                    Lv->ibc_frequency_Hz, Lv->ibc_Q_eig, Lv->ibc_R_over_Q,
                    Lv->ibc_iterations, Lv->ibc_residual, Lv->wall_ibc_sec,
                    Lv->wall_total_sec);
            }
            fclose(fp);
            printf("\n  Data written to convergence_unperturbed.csv\n");
        }
    }

    /*=========================================================================
     * Cleanup
     *========================================================================*/
    free(levels);

    printf("\n****************************************************************\n");
    printf("*    UNPERTURBED CAVITY CONVERGENCE STUDY COMPLETE             *\n");
    printf("****************************************************************\n\n");

    return 0;
}