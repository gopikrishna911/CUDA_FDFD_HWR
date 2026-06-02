/*=============================================================================
 * test_convergence.cpp
 *
 * Grid convergence study for the Rhodotron coaxial HWR eigensolver.
 *
 * Runs the REFERENCE cavity (no pipes, no apertures) at multiple grid
 * resolutions and compares eigenfrequency, Q_0, and R/Q against
 * analytical formulas. Reports convergence order and Richardson
 * extrapolation.
 *
 * Levels are defined by target dr (radial cell size).  Azimuthal and
 * axial cell sizes maintain fixed aspect ratios:
 *     r*dphi ≈ 1.5 * dr   (at r = b)
 *     dz     ≈ 2.0 * dr
 *
 * Levels that exceed available GPU memory are automatically skipped.
 *============================================================================*/

#include "cuda_pipe_model.h"
#include "cuda_eigensolver.h"
#include "pipe_model.h"
#include "curlcurl_operator.h"
#include "q_factor.h"
#include "r_over_q.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <time.h>

/*=============================================================================
 * Eigensolver helper (same RQI + MINRES as test_pipe_model.cpp)
 *============================================================================*/
static int run_eigensolver(
    GPU_PipeOperator* pipe_op,
    const CurlCurlOperator* cpu_op,
    const GridParams* grid,
    double* d_x,
    int max_iter,
    int minres_max,
    double tol,
    double* eigenvalue_out
) {
    int n = cpu_op->n_total;

    EigensolverWorkspace ws;
    eigensolver_workspace_init(&ws, n);

    double sigma;

    gpu_vec_normalize_weighted(d_x, cpu_op);

    double* d_Ax;
    cudaMalloc(&d_Ax, n * sizeof(double));
    gpu_pipe_matvec(pipe_op, d_x, d_Ax);

    double xAx, xx;
    gpu_vec_dot_weighted_ws(d_x, d_Ax, &xAx, cpu_op,
        &pipe_op->base.reduction_ws);
    gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op,
        &pipe_op->base.reduction_ws);
    sigma = xAx / xx;

    double k2_target = (M_PI / grid->L) * (M_PI / grid->L);
    if (fabs(sigma) < 1e-10 || sigma < 0) sigma = k2_target;

    printf("    RQI: ");

    int blocks = (n + 256 - 1) / 256;

    for (int iter = 0; iter < max_iter; iter++) {
        gpu_vec_zero(ws.d_y, n);
        gpu_vec_zero(ws.minres_ws.d_v_old, n);
        gpu_vec_zero(ws.minres_ws.d_w_old, n);
        gpu_vec_zero(ws.minres_ws.d_w_cur, n);

        double b_norm;
        gpu_vec_dot_weighted_ws(d_x, d_x, &b_norm, cpu_op,
            &pipe_op->base.reduction_ws);
        b_norm = sqrt(b_norm);

        gpu_vec_copy(d_x, ws.minres_ws.d_v_cur, n);
        gpu_vec_scale(ws.minres_ws.d_v_cur, 1.0 / b_norm, n);

        double beta_cur = b_norm;
        double eta = b_norm;
        double c_old = 1.0, c_cur = 1.0;
        double s_old = 0.0, s_cur = 0.0;

        int ls_iters = 0;
        double ls_residual = 1.0;

        for (int ls = 0; ls < minres_max; ls++) {
            gpu_pipe_matvec(pipe_op, ws.minres_ws.d_v_cur, ws.minres_ws.d_Av);

            extern __global__ void shift_kernel(double*, const double*, double, int);
            shift_kernel<<<blocks, 256>>>(ws.minres_ws.d_Av, ws.minres_ws.d_v_cur, sigma, n);

            double alpha;
            gpu_vec_dot_weighted_ws(ws.minres_ws.d_v_cur, ws.minres_ws.d_Av,
                &alpha, cpu_op, &pipe_op->base.reduction_ws);

            gpu_vec_copy(ws.minres_ws.d_Av, ws.minres_ws.d_v_new, n);
            gpu_vec_axpy(-alpha, ws.minres_ws.d_v_cur, ws.minres_ws.d_v_new, n);
            gpu_vec_axpy(-beta_cur, ws.minres_ws.d_v_old, ws.minres_ws.d_v_new, n);

            double beta_new;
            gpu_vec_dot_weighted_ws(ws.minres_ws.d_v_new, ws.minres_ws.d_v_new,
                &beta_new, cpu_op, &pipe_op->base.reduction_ws);
            beta_new = sqrt(beta_new);

            if (beta_new > 1e-14)
                gpu_vec_scale(ws.minres_ws.d_v_new, 1.0 / beta_new, n);

            double rho1 = s_old * beta_cur;
            double rho2 = c_old * c_cur * beta_cur + s_cur * alpha;
            double rho3_bar = c_cur * alpha - c_old * s_cur * beta_cur;
            double gamma = sqrt(rho3_bar * rho3_bar + beta_new * beta_new);

            double c_new, s_new;
            if (gamma > 1e-14) { c_new = rho3_bar / gamma; s_new = beta_new / gamma; }
            else { c_new = 1.0; s_new = 0.0; }

            gpu_vec_copy(ws.minres_ws.d_v_cur, ws.minres_ws.d_w_new, n);
            gpu_vec_axpy(-rho2, ws.minres_ws.d_w_cur, ws.minres_ws.d_w_new, n);
            gpu_vec_axpy(-rho1, ws.minres_ws.d_w_old, ws.minres_ws.d_w_new, n);
            if (fabs(gamma) > 1e-14)
                gpu_vec_scale(ws.minres_ws.d_w_new, 1.0 / gamma, n);

            gpu_vec_axpy(c_new * eta, ws.minres_ws.d_w_new, ws.d_y, n);

            eta = -s_new * eta;
            ls_residual = fabs(eta) / b_norm;
            ls_iters = ls + 1;

            if (ls_residual < 1e-4) break;

            double* temp;
            temp = ws.minres_ws.d_v_old; ws.minres_ws.d_v_old = ws.minres_ws.d_v_cur;
            ws.minres_ws.d_v_cur = ws.minres_ws.d_v_new; ws.minres_ws.d_v_new = temp;
            temp = ws.minres_ws.d_w_old; ws.minres_ws.d_w_old = ws.minres_ws.d_w_cur;
            ws.minres_ws.d_w_cur = ws.minres_ws.d_w_new; ws.minres_ws.d_w_new = temp;

            beta_cur = beta_new;
            c_old = c_cur; c_cur = c_new;
            s_old = s_cur; s_cur = s_new;
        }

        gpu_vec_normalize_weighted(ws.d_y, cpu_op);
        gpu_vec_copy(ws.d_y, d_x, n);

        gpu_pipe_matvec(pipe_op, d_x, d_Ax);
        gpu_vec_dot_weighted_ws(d_x, d_Ax, &xAx, cpu_op, &pipe_op->base.reduction_ws);
        gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op, &pipe_op->base.reduction_ws);
        double sigma_new = xAx / xx;

        extern __global__ void shift_kernel(double*, const double*, double, int);
        shift_kernel<<<blocks, 256>>>(d_Ax, d_x, sigma_new, n);
        double res_sq;
        gpu_vec_dot_weighted_ws(d_Ax, d_Ax, &res_sq, cpu_op,
            &pipe_op->base.reduction_ws);
        double residual = sqrt(res_sq);

        printf("iter %d: k²=%.10f res=%.2e ls=%d  ",
            iter, sigma_new, residual, ls_iters);

        if (residual < tol) {
            printf("✓\n");
            *eigenvalue_out = sigma_new;
            break;
        }

        *eigenvalue_out = sigma_new;
        sigma = sigma_new;
    }

    cudaFree(d_Ax);
    eigensolver_workspace_free(&ws);
    return 0;
}


/*=============================================================================
 * Convergence level result
 *============================================================================*/
typedef struct {
    /* Grid parameters */
    double dr_target_mm;
    int Nr_cavity, Nr_pipe, Nphi, Nz;
    double dr_mm, dphi_deg, dz_mm;
    double rdphi_b_mm;      /* r*dphi at r = b */
    long long n_dofs;
    double memory_est_GB;

    /* Execution */
    int ran;
    int converged;
    double wall_sec;

    /* Results */
    double k_squared;
    double frequency_Hz;
    double Q_0;
    double G_factor;
    double R_over_Q_linac;

    /* Errors vs analytical */
    double err_f_pct;
    double err_Q_pct;
    double err_RoQ_pct;
} ConvLevel;


/*=============================================================================
 * Main
 *============================================================================*/
int main() {
    printf("\n");
    printf("****************************************************************\n");
    printf("*      CONVERGENCE STUDY — COAXIAL HWR REFERENCE CAVITY       *\n");
    printf("****************************************************************\n\n");

    cuda_print_device_info();

    /*=========================================================================
     * Geometry (same as test_pipe_model)
     *========================================================================*/
    double a = 1.0 / 3.0;
    double b = 1.0;
    double L = 1.395;
    double pipe_radius = 0.0125;
    double aperture_radius = 0.0175;
    double pipe_length = 0.050;
    double taper_length = 0.005;
    int num_passes = 10;

    double ln_ba = log(b / a);
    double c0 = 299792458.0;

    printf("  Geometry: a = %.4f m, b = %.4f m, L = %.4f m\n", a, b, L);
    printf("  ln(b/a) = %.6f\n\n", ln_ba);

    /*=========================================================================
     * Analytical reference values
     *========================================================================*/
    QFactorResult qf_analytical = compute_q_analytical_coaxial_hwr(
        a, b, L, Q_SIGMA_CU);
    RoverQResult roq_analytical = compute_r_over_q_analytical(a, b, L);

    double f_exact = qf_analytical.frequency_Hz;
    double Q_exact = qf_analytical.Q_0;
    double RoQ_exact = roq_analytical.R_over_Q_crossing_linac;

    printf("  Analytical targets:\n");
    printf("    f       = %.6f MHz\n", f_exact / 1e6);
    printf("    Q_0     = %.0f\n", Q_exact);
    printf("    R/Q     = %.4f Ohm (linac, per crossing)\n\n", RoQ_exact);

    /*=========================================================================
     * Define convergence levels by target dr
     *
     * Aspect ratios (from standard grid):
     *   r*dphi(r=b) ≈ 1.5 * dr
     *   dz          ≈ 2.0 * dr
     *========================================================================*/
    double dr_targets[] = {16.0, 10.0, 8.0, 5.5, 4.1, 3.3, 2.7};
    int n_levels = sizeof(dr_targets) / sizeof(dr_targets[0]);

    double ratio_rdphi_dr = 1.5;
    double ratio_dz_dr = 2.0;

    /* Bytes per DOF for memory estimation (empirical from solver runs) */
    double bytes_per_dof = 40.0;

    /* Query available GPU memory */
    size_t gpu_free_bytes, gpu_total_bytes;
    cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
    double gpu_available_GB = gpu_free_bytes / (1024.0 * 1024.0 * 1024.0);
    /* Leave 15% headroom for fragmentation and driver */
    double gpu_usable_GB = gpu_available_GB * 0.85;

    printf("  GPU memory: %.1f GB free / %.1f GB total (using %.1f GB)\n\n",
        gpu_available_GB, gpu_total_bytes / (1024.0*1024.0*1024.0), gpu_usable_GB);

    /*=========================================================================
     * Compute grid parameters for each level
     *========================================================================*/
    ConvLevel* levels = (ConvLevel*)calloc(n_levels, sizeof(ConvLevel));

    printf("  ┌─────────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  Level  dr(mm)  r·dφ(mm)  dz(mm)   Nr   Nphi    Nz      DOFs    Memory     │\n");
    printf("  ├─────────────────────────────────────────────────────────────────────────────┤\n");

    for (int lvl = 0; lvl < n_levels; lvl++) {
        ConvLevel* L_ = &levels[lvl];
        double dr_m = dr_targets[lvl] * 1e-3;

        L_->dr_target_mm = dr_targets[lvl];

        /* Radial: cavity + pipe extension */
        L_->Nr_cavity = (int)round((b - a) / dr_m);
        if (L_->Nr_cavity < 4) L_->Nr_cavity = 4;
        L_->Nr_pipe = (int)round(pipe_length / dr_m);
        if (L_->Nr_pipe < 3) L_->Nr_pipe = 3;
        int Nr_total = L_->Nr_cavity + L_->Nr_pipe;

        /* Azimuthal: r*dphi ≈ ratio * dr at r = b */
        double dphi_target = ratio_rdphi_dr * dr_m / b;
        L_->Nphi = (int)round(2.0 * M_PI / dphi_target);
        if (L_->Nphi < 16) L_->Nphi = 16;
        /* Round to nearest even for symmetry */
        if (L_->Nphi % 2 != 0) L_->Nphi++;

        /* Axial: dz ≈ ratio * dr */
        double dz_target = ratio_dz_dr * dr_m;
        L_->Nz = (int)round(L / dz_target);
        if (L_->Nz < 4) L_->Nz = 4;

        /* Actual cell sizes from rounded N values */
        double r_max = b + pipe_length;
        L_->dr_mm = (r_max - a) / Nr_total * 1000.0;
        L_->dphi_deg = 360.0 / L_->Nphi;
        L_->dz_mm = L / L_->Nz * 1000.0;
        L_->rdphi_b_mm = b * (2.0 * M_PI / L_->Nphi) * 1000.0;

        /* DOF estimate: 3 components × Nr × Nphi × Nz (approximate) */
        long long Nr_ll = Nr_total;
        long long Nphi_ll = L_->Nphi;
        long long Nz_ll = L_->Nz;
        L_->n_dofs = Nr_ll * Nphi_ll * (Nz_ll + 1)          /* Er */
                   + (Nr_ll + 1) * Nphi_ll * (Nz_ll + 1)    /* Ephi */
                   + (Nr_ll + 1) * Nphi_ll * Nz_ll;          /* Ez */

        L_->memory_est_GB = L_->n_dofs * bytes_per_dof / (1024.0*1024.0*1024.0);

        const char* status = (L_->memory_est_GB <= gpu_usable_GB) ? "" : " (skip)";
        printf("  │  %3d   %5.1f   %6.1f    %5.1f   %4d  %5d  %4d  %8.1fM  %5.1f GB%s │\n",
            lvl + 1, L_->dr_target_mm, L_->rdphi_b_mm, L_->dz_mm,
            Nr_total, L_->Nphi, L_->Nz,
            L_->n_dofs / 1e6, L_->memory_est_GB, status);
    }
    printf("  └─────────────────────────────────────────────────────────────────────────────┘\n\n");

    /*=========================================================================
     * Run each level
     *========================================================================*/
    for (int lvl = 0; lvl < n_levels; lvl++) {
        ConvLevel* L_ = &levels[lvl];

        if (L_->memory_est_GB > gpu_usable_GB) {
            printf("  --- Level %d (dr=%.1f mm): SKIPPED — needs %.1f GB, "
                   "only %.1f GB available ---\n\n",
                lvl + 1, L_->dr_target_mm, L_->memory_est_GB, gpu_usable_GB);
            L_->ran = 0;
            continue;
        }

        printf("  === Level %d: dr=%.1f mm, dphi=%.2f°, dz=%.1f mm, "
               "%.1fM DOFs ===\n",
            lvl + 1, L_->dr_target_mm, L_->dphi_deg, L_->dz_mm,
            L_->n_dofs / 1e6);

        clock_t t_start = clock();

        /* Grid: radial pipe extension, no z extension */
        GridParams grid;
        grid_init_with_pipes(&grid, a, b, L, pipe_length,
            L_->Nr_cavity, L_->Nr_pipe, L_->Nphi, L_->Nz);

        cuda_grid_init(&grid);

        /* CPU operator (no ports for reference cavity) */
        CurlCurlOperator cpu_op;
        curlcurl_op_init(&cpu_op, &grid);

        int n = cpu_op.n_total;

        /* Material mask: no pipes → all cavity cells are vacuum */
        PipeConfig no_pipes;
        pipe_config_init(&no_pipes, a, b, pipe_radius, aperture_radius,
            pipe_length, taper_length);
        /* Don't add any pipes — pure reference cavity */

        MaterialMask mask;
        material_mask_build(&mask, &no_pipes, &grid);

        /* GPU operator */
        GPU_PipeOperator gpu_op;
        gpu_pipe_operator_init(&gpu_op, &cpu_op, &mask);

        /* Initial guess: TEM mode Er = sin(pi*z/L) / r */
        double* h_x = (double*)calloc(n, sizeof(double));
        for (int k = 0; k <= grid.Nz; k++) {
            double z = k * grid.dz;
            for (int j = 0; j < grid.Nphi; j++) {
                for (int i = 0; i < grid.Nr; i++) {
                    double r = grid.a + (i + 0.5) * grid.dr;
                    if (r <= b) {
                        int idx = cpu_op.offset_Er + idx_Er(&grid, i, j, k);
                        h_x[idx] = sin(M_PI * z / L) / r;
                    }
                }
            }
        }

        double* d_x;
        gpu_vector_alloc(&d_x, n);
        gpu_vector_to_device(d_x, h_x, n);

        /* Solve */
        double k2 = 0.0;
        int minres_max = (n < 20000000) ? 1000 : 3000;
        run_eigensolver(&gpu_op, &cpu_op, &grid, d_x,
            10, minres_max, 1e-4, &k2);

        /* Retrieve eigenvector */
        gpu_vector_to_host(h_x, d_x, n);

        /* Compute Q factor */
        QFactorResult qf = compute_q_factor(
            &cpu_op, h_x, k2, Q_SIGMA_CU, b, NULL);

        /* Compute R/Q */
        RoverQResult roq = compute_r_over_q(
            &cpu_op, h_x, k2, b, L, 0.0, num_passes);

        clock_t t_end = clock();
        L_->wall_sec = (double)(t_end - t_start) / CLOCKS_PER_SEC;

        /* Store results */
        L_->ran = 1;
        L_->converged = 1;
        L_->k_squared = k2;
        L_->frequency_Hz = qf.frequency_Hz;
        L_->Q_0 = qf.Q_0;
        L_->G_factor = qf.G_factor;
        L_->R_over_Q_linac = roq.R_over_Q_crossing_linac;

        L_->err_f_pct = (L_->frequency_Hz - f_exact) / f_exact * 100.0;
        L_->err_Q_pct = (L_->Q_0 - Q_exact) / Q_exact * 100.0;
        L_->err_RoQ_pct = (L_->R_over_Q_linac - RoQ_exact) / RoQ_exact * 100.0;

        printf("    f = %.6f MHz (Δ = %+.1f Hz),  Q = %.0f (%+.2f%%),  "
               "R/Q = %.4f Ω (%+.2f%%),  %.1f s\n\n",
            L_->frequency_Hz / 1e6,
            L_->frequency_Hz - f_exact,
            L_->Q_0, L_->err_Q_pct,
            L_->R_over_Q_linac, L_->err_RoQ_pct,
            L_->wall_sec);

        /* Cleanup this level */
        r_over_q_free(&roq);
        free(h_x);
        gpu_vector_free(d_x);
        gpu_pipe_operator_free(&gpu_op);
        material_mask_free(&mask);
        pipe_config_free(&no_pipes);
        curlcurl_op_free(&cpu_op);
    }

    /*=========================================================================
     * Convergence summary table
     *========================================================================*/
    printf("\n");
    printf("  ┌────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │                           CONVERGENCE RESULTS                                         │\n");
    printf("  │  Analytical:  f = %.4f MHz,  Q_0 = %.0f,  R/Q = %.4f Ohm (linac)              │\n",
        f_exact / 1e6, Q_exact, RoQ_exact);
    printf("  ├────────────────────────────────────────────────────────────────────────────────────────┤\n");
    printf("  │ Lvl  dr(mm)  dz(mm)     DOFs   f(MHz)       Δf(Hz)  err_f   Q_0    err_Q   R/Q(Ω)  err_RoQ  Time │\n");
    printf("  │ ───  ──────  ──────  ────────  ──────────  ───────  ──────  ─────  ──────  ───────  ───────  ──── │\n");

    for (int lvl = 0; lvl < n_levels; lvl++) {
        ConvLevel* L_ = &levels[lvl];
        if (!L_->ran) {
            printf("  │ %2d   %5.1f   %5.1f   %6.1fM  %43s │\n",
                lvl + 1, L_->dr_target_mm, L_->dz_mm,
                L_->n_dofs / 1e6, "— SKIPPED —");
            continue;
        }

        printf("  │ %2d   %5.1f   %5.1f   %6.1fM  %10.4f  %+8.1f  %+.2f%%  %5.0f  %+.2f%%  %7.3f  %+.2f%%  %5.1fs │\n",
            lvl + 1,
            L_->dr_target_mm, L_->dz_mm,
            L_->n_dofs / 1e6,
            L_->frequency_Hz / 1e6,
            L_->frequency_Hz - f_exact,
            L_->err_f_pct,
            L_->Q_0,
            L_->err_Q_pct,
            L_->R_over_Q_linac,
            L_->err_RoQ_pct,
            L_->wall_sec);
    }
    printf("  └────────────────────────────────────────────────────────────────────────────────────────┘\n");

    /*=========================================================================
     * Convergence order (from consecutive level pairs)
     *========================================================================*/
    printf("\n  Convergence order (log-log slope between consecutive levels):\n");
    printf("  %5s → %5s    h_ratio    p(freq)    p(Q_0)    p(R/Q)\n",
        "Lvl_i", "Lvl_j");
    printf("  ───────────────────────────────────────────────────────\n");

    for (int i = 0; i < n_levels - 1; i++) {
        ConvLevel* Li = &levels[i];
        ConvLevel* Lj = &levels[i + 1];
        if (!Li->ran || !Lj->ran) continue;

        double h_ratio = Li->dr_target_mm / Lj->dr_target_mm;
        double log_h = log(h_ratio);

        double p_f = 0.0, p_Q = 0.0, p_RoQ = 0.0;

        double ef_i = fabs(Li->frequency_Hz - f_exact);
        double ef_j = fabs(Lj->frequency_Hz - f_exact);
        if (ef_i > 0 && ef_j > 0)
            p_f = log(ef_i / ef_j) / log_h;

        double eQ_i = fabs(Li->Q_0 - Q_exact);
        double eQ_j = fabs(Lj->Q_0 - Q_exact);
        if (eQ_i > 0 && eQ_j > 0)
            p_Q = log(eQ_i / eQ_j) / log_h;

        double eR_i = fabs(Li->R_over_Q_linac - RoQ_exact);
        double eR_j = fabs(Lj->R_over_Q_linac - RoQ_exact);
        if (eR_i > 0 && eR_j > 0)
            p_RoQ = log(eR_i / eR_j) / log_h;

        printf("  %3d   → %3d      %.2f     %5.2f      %5.2f     %5.2f\n",
            i + 1, i + 2, h_ratio, p_f, p_Q, p_RoQ);
    }

    /*=========================================================================
     * Richardson extrapolation from two finest converged levels
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

        printf("\n  Richardson extrapolation (levels %d & %d, h-ratio = %.2f):\n",
            i_coarse + 1, i_fine + 1, r);

        /* Use observed convergence order or assume p=2 */
        double log_r = log(r);

        /* Frequency */
        double ef_c = Lc->frequency_Hz - f_exact;
        double ef_f = Lf->frequency_Hz - f_exact;
        double p_f = 2.0;
        if (fabs(ef_c) > 0 && fabs(ef_f) > 0 && (ef_c * ef_f > 0))
            p_f = log(fabs(ef_c / ef_f)) / log_r;
        double f_rich = Lf->frequency_Hz
            + (Lf->frequency_Hz - Lc->frequency_Hz) / (pow(r, p_f) - 1.0);

        /* Q factor */
        double eQ_c = Lc->Q_0 - Q_exact;
        double eQ_f = Lf->Q_0 - Q_exact;
        double p_Q = 2.0;
        if (fabs(eQ_c) > 0 && fabs(eQ_f) > 0 && (eQ_c * eQ_f > 0))
            p_Q = log(fabs(eQ_c / eQ_f)) / log_r;
        double Q_rich = Lf->Q_0
            + (Lf->Q_0 - Lc->Q_0) / (pow(r, p_Q) - 1.0);

        /* R/Q */
        double eR_c = Lc->R_over_Q_linac - RoQ_exact;
        double eR_f = Lf->R_over_Q_linac - RoQ_exact;
        double p_RoQ = 2.0;
        if (fabs(eR_c) > 0 && fabs(eR_f) > 0 && (eR_c * eR_f > 0))
            p_RoQ = log(fabs(eR_c / eR_f)) / log_r;
        double RoQ_rich = Lf->R_over_Q_linac
            + (Lf->R_over_Q_linac - Lc->R_over_Q_linac) / (pow(r, p_RoQ) - 1.0);

        printf("    ┌──────────────────────────────────────────────────────────────┐\n");
        printf("    │  Quantity       Analytical   Finest grid   Richardson extrap │\n");
        printf("    │  ────────────   ──────────   ───────────   ──────────────── │\n");
        printf("    │  f (MHz)       %11.4f   %11.4f   %11.4f              │\n",
            f_exact / 1e6, Lf->frequency_Hz / 1e6, f_rich / 1e6);
        printf("    │  Δf (Hz)              0.0   %+11.1f   %+11.1f              │\n",
            Lf->frequency_Hz - f_exact, f_rich - f_exact);
        printf("    │  Q_0          %11.0f   %11.0f   %11.0f              │\n",
            Q_exact, Lf->Q_0, Q_rich);
        printf("    │  R/Q (Ω)      %11.4f   %11.4f   %11.4f              │\n",
            RoQ_exact, Lf->R_over_Q_linac, RoQ_rich);
        printf("    │  p observed         —         —         f:%.1f Q:%.1f R/Q:%.1f   │\n",
            p_f, p_Q, p_RoQ);
        printf("    └──────────────────────────────────────────────────────────────┘\n");
    }

    /*=========================================================================
     * Export convergence data to CSV
     *========================================================================*/
    {
        FILE* fp = fopen("convergence_data.csv", "w");
        if (fp) {
            fprintf(fp, "level,dr_mm,dphi_deg,dz_mm,n_dofs,"
                        "frequency_Hz,err_f_pct,"
                        "Q_0,err_Q_pct,"
                        "R_over_Q_linac,err_RoQ_pct,"
                        "wall_sec\n");
            for (int i = 0; i < n_levels; i++) {
                ConvLevel* L_ = &levels[i];
                if (!L_->ran) continue;
                fprintf(fp, "%d,%.4f,%.6f,%.4f,%lld,"
                            "%.6f,%.6f,"
                            "%.2f,%.6f,"
                            "%.6f,%.6f,"
                            "%.2f\n",
                    i + 1, L_->dr_mm, L_->dphi_deg, L_->dz_mm, L_->n_dofs,
                    L_->frequency_Hz, L_->err_f_pct,
                    L_->Q_0, L_->err_Q_pct,
                    L_->R_over_Q_linac, L_->err_RoQ_pct,
                    L_->wall_sec);
            }
            fclose(fp);
            printf("\n  Convergence data written to convergence_data.csv\n");
        }
    }

    /*=========================================================================
     * Cleanup
     *========================================================================*/
    free(levels);

    printf("\n****************************************************************\n");
    printf("*              CONVERGENCE STUDY COMPLETE                      *\n");
    printf("****************************************************************\n\n");

    return 0;
}
