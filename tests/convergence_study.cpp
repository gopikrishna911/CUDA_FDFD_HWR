/*=============================================================================
 * convergence_study.cpp
 *
 * Grid convergence study for the FULL HWR model — TWO methods reported at
 * every grid level so the paper's conformal-IBC contribution is
 * characterized, not just the bulk discretization:
 *
 *   Phase A — PEC (staircased pipes):
 *       Real eigenproblem, RQI + MINRES inner solve.
 *       Reports f_PEC and Q_PEC_surf (surface integral with copper R_s).
 *
 *   Phase B — Conformal IBC (Dey-Mittra + grid-plane endplate IBC):
 *       Complex eigenproblem with finite Z_s = R_s(1+j); complex RQI
 *       with GMRES inner solve, seeded by the converged PEC eigenvector.
 *       Reports f_IBC and Q_IBC = -Re(k^2)/Im(k^2) directly from the
 *       eigenvalue.  This is the paper's headline method.
 *
 * Richardson extrapolation is applied to the two finest IBC results
 * since that is where the paper's claimed convergence behaviour lives.
 *
 * The full cavity (20 radial pipes + 2 endcap pipes + inner ports) is
 * used at every level.  Levels exceeding GPU memory are skipped.
 *
 * Grid sizing per level (target dr):
 *     r*dphi ~ 1.5 * dr   (at r = b)
 *     dz     ~ 2.0 * dr
 *============================================================================*/

#include "cuda_pipe_model.h"
#include "cuda_eigensolver.h"
#include "cuda_conformal_pipe.h"
#include "conformal_geometry.h"
#include "pipe_model.h"
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
  * Eigensolver (RQI + MINRES, same as other test files)
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
    gpu_vec_dot_weighted_ws(d_x, d_Ax, &xAx, cpu_op, &pipe_op->base.reduction_ws);
    gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op, &pipe_op->base.reduction_ws);
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
        gpu_vec_dot_weighted_ws(d_x, d_x, &b_norm, cpu_op, &pipe_op->base.reduction_ws);
        b_norm = sqrt(b_norm);

        gpu_vec_copy(d_x, ws.minres_ws.d_v_cur, n);
        gpu_vec_scale(ws.minres_ws.d_v_cur, 1.0 / b_norm, n);

        double beta_cur = b_norm, eta = b_norm;
        double c_old = 1.0, c_cur = 1.0, s_old = 0.0, s_cur = 0.0;
        int ls_iters = 0;
        double ls_residual = 1.0;

        for (int ls = 0; ls < minres_max; ls++) {
            gpu_pipe_matvec(pipe_op, ws.minres_ws.d_v_cur, ws.minres_ws.d_Av);

            extern __global__ void shift_kernel(double*, const double*, double, int);
            shift_kernel << <blocks, 256 >> > (ws.minres_ws.d_Av, ws.minres_ws.d_v_cur, sigma, n);

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

            if (ls_residual < 1e-6) break;

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
        shift_kernel << <blocks, 256 >> > (d_Ax, d_x, sigma_new, n);
        double res_sq;
        gpu_vec_dot_weighted_ws(d_Ax, d_Ax, &res_sq, cpu_op, &pipe_op->base.reduction_ws);
        double residual = sqrt(res_sq);

        printf("iter %d: k²=%.10f res=%.2e ls=%d  ", iter, sigma_new, residual, ls_iters);

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
    int Nr_cavity, Nr_pipe, Nphi, Nz_cavity, Nz_pipe_z0, Nz_pipe_zL;
    double dr_mm, dphi_deg, dz_mm;
    double rdphi_b_mm;
    long long n_dofs;
    double memory_est_GB;

    /* Execution */
    int ran;
    double wall_pec_sec;
    double wall_ibc_sec;
    double wall_total_sec;

    /* Phase A — PEC results (real eigenvalue, surface-integral Q) */
    int      pec_converged;
    double   pec_k_squared;
    double   pec_frequency_Hz;
    double   pec_Q_surf;        /* Q from surface integral with copper R_s */
    double   pec_G_factor;
    double   pec_R_over_Q;

    /* Phase B — Conformal IBC results (complex eigenvalue, Q from -Re/Im) */
    int      ibc_converged;
    double   ibc_k2_re;
    double   ibc_k2_im;
    double   ibc_frequency_Hz;
    double   ibc_Q_eig;         /* Q from eigenvalue, paper's number */
    int      ibc_iterations;
    double   ibc_residual;
    double   ibc_R_over_Q;
} ConvLevel;


/*=============================================================================
 * Main
 *============================================================================*/
int main() {
    printf("\n");
    printf("****************************************************************\n");
    printf("*   CONVERGENCE STUDY — FULL MODEL (all pipes)                *\n");
    printf("****************************************************************\n\n");

    cuda_print_device_info();

    /*=========================================================================
     * Geometry (matches test_pipe_model.cpp exactly)
     *========================================================================*/
    double a = 0.3333;
    double b = 1.0;
    double L = 1.395;
    double pipe_radius = 0.0125;
    double aperture_radius = 0.0175;
    double pipe_length = 0.050;
    double taper_length = 0.0;
    int num_passes = 10;

    /* Endcap pipes */
    double endcap_z0_r_center = 0.85;
    double endcap_z0_phi = M_PI / 2.0;
    double endcap_z0_aperture_radius = 0.105;
    double endcap_z0_pipe_radius = 0.100;
    double endcap_z0_pipe_length = 0.28;

    double endcap_zL_r_center = 0.85;
    double endcap_zL_phi = 3.0 * M_PI / 2.0;
    double endcap_zL_aperture_radius = 0.095;
    double endcap_zL_pipe_radius = 0.090;
    double endcap_zL_pipe_length = 0.25;

    double c0 = 299792458.0;
    double ln_ba = log(b / a);

    printf("  Geometry: a = %.4f m, b = %.4f m, L = %.4f m\n", a, b, L);
    printf("  Radial pipes: %d × %.1f mm diam, %.1f mm long\n",
        2 * num_passes, pipe_radius * 2000, pipe_length * 1000);
    printf("  Endcap z=0: %.0f mm diam aperture, %.0f mm pipe\n",
        endcap_z0_aperture_radius * 2000, endcap_z0_pipe_length * 1000);
    printf("  Endcap z=L: %.0f mm diam aperture, %.0f mm pipe\n\n",
        endcap_zL_aperture_radius * 2000, endcap_zL_pipe_length * 1000);

    /*=========================================================================
     * Configure pipe structures (shared across all levels)
     *========================================================================*/
    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, pipe_radius, aperture_radius,
        pipe_length, taper_length);
    pipe_config_add_multi_pass(&pipes, L / 2.0, num_passes);

    EndcapPipeConfig endcap_pipes;
    endcap_pipe_config_init(&endcap_pipes);
    endcap_pipe_config_add(&endcap_pipes,
        endcap_z0_r_center, endcap_z0_phi,
        endcap_z0_aperture_radius, endcap_z0_pipe_radius,
        endcap_z0_pipe_length, 1);
    endcap_pipe_config_add(&endcap_pipes,
        endcap_zL_r_center, endcap_zL_phi,
        endcap_zL_aperture_radius, endcap_zL_pipe_radius,
        endcap_zL_pipe_length, 0);

    /*=========================================================================
     * Analytical reference values (unperturbed cavity)
     *========================================================================*/
    QFactorResult qf_analytical = compute_q_analytical_coaxial_hwr(
        a, b, L, Q_SIGMA_CU);
    RoverQResult roq_analytical = compute_r_over_q_analytical(a, b, L);

    printf("  Analytical (unperturbed, no pipes):\n");
    printf("    f       = %.6f MHz\n", qf_analytical.frequency_Hz / 1e6);
    printf("    Q_0     = %.0f\n", qf_analytical.Q_0);
    printf("    R/Q     = %.4f Ohm\n\n", roq_analytical.R_over_Q_crossing_linac);

    /*=========================================================================
     * Define convergence levels by target dr
     *========================================================================*/
    double dr_targets[] = { 16.0, 10.0, 8.0, 5.5, 4.1 };
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
        gpu_available_GB, gpu_total_bytes / (1024.0 * 1024.0 * 1024.0), gpu_usable_GB);

    /*=========================================================================
     * Compute grid parameters for each level
     *========================================================================*/
    ConvLevel* levels = (ConvLevel*)calloc(n_levels, sizeof(ConvLevel));

    printf("  ┌──────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │ Lvl  dr(mm) r·dφ(mm) dz(mm)  Nr  Nphi   Nz(cav) Nz0  NzL   DOFs    Memory         │\n");
    printf("  ├──────────────────────────────────────────────────────────────────────────────────────┤\n");

    for (int lvl = 0; lvl < n_levels; lvl++) {
        ConvLevel* Lv = &levels[lvl];
        double dr_m = dr_targets[lvl] * 1e-3;
        Lv->dr_target_mm = dr_targets[lvl];

        /* Radial: cavity + radial pipe extension */
        Lv->Nr_cavity = (int)round((b - a) / dr_m);
        if (Lv->Nr_cavity < 4) Lv->Nr_cavity = 4;
        Lv->Nr_pipe = (int)round(pipe_length / dr_m);
        if (Lv->Nr_pipe < 3) Lv->Nr_pipe = 3;
        int Nr_total = Lv->Nr_cavity + Lv->Nr_pipe;

        /* Azimuthal */
        double dphi_target = ratio_rdphi_dr * dr_m / b;
        Lv->Nphi = (int)round(2.0 * M_PI / dphi_target);
        if (Lv->Nphi < 16) Lv->Nphi = 16;
        if (Lv->Nphi % 2 != 0) Lv->Nphi++;

        /* Axial: cavity cells */
        double dz_target = ratio_dz_dr * dr_m;
        Lv->Nz_cavity = (int)round(L / dz_target);
        if (Lv->Nz_cavity < 4) Lv->Nz_cavity = 4;

        /* z-pipe cells (match dz from cavity) */
        double dz_actual = L / Lv->Nz_cavity;
        Lv->Nz_pipe_z0 = (int)ceil(endcap_pipes.z0_extension / dz_actual);
        Lv->Nz_pipe_zL = (int)ceil(endcap_pipes.zL_extension / dz_actual);
        int Nz_total = Lv->Nz_pipe_z0 + Lv->Nz_cavity + Lv->Nz_pipe_zL;

        /* Actual cell sizes */
        double r_max = b + pipe_length;
        Lv->dr_mm = (r_max - a) / Nr_total * 1000.0;
        Lv->dphi_deg = 360.0 / Lv->Nphi;
        Lv->dz_mm = dz_actual * 1000.0;
        Lv->rdphi_b_mm = b * (2.0 * M_PI / Lv->Nphi) * 1000.0;

        /* DOF estimate */
        long long Nr_ll = Nr_total;
        long long Nphi_ll = Lv->Nphi;
        long long Nz_ll = Nz_total;
        Lv->n_dofs = Nr_ll * Nphi_ll * (Nz_ll + 1)
            + (Nr_ll + 1) * Nphi_ll * (Nz_ll + 1)
            + (Nr_ll + 1) * Nphi_ll * Nz_ll;
        Lv->memory_est_GB = Lv->n_dofs * bytes_per_dof / (1024.0 * 1024.0 * 1024.0);

        const char* status = (Lv->memory_est_GB <= gpu_usable_GB) ? "" : " SKIP";
        printf("  │ %2d   %5.1f  %6.1f   %5.1f  %4d %5d  %5d   %3d  %3d  %7.1fM  %5.1f GB%s │\n",
            lvl + 1, Lv->dr_target_mm, Lv->rdphi_b_mm, Lv->dz_mm,
            Nr_total, Lv->Nphi, Lv->Nz_cavity,
            Lv->Nz_pipe_z0, Lv->Nz_pipe_zL,
            Lv->n_dofs / 1e6, Lv->memory_est_GB, status);
    }
    printf("  └──────────────────────────────────────────────────────────────────────────────────────┘\n\n");

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
            lvl + 1, Lv->dr_target_mm, Lv->dphi_deg, Lv->dz_mm, Lv->n_dofs / 1e6);

        clock_t t_level_start = clock();

        /* Grid: extended in r and z */
        GridParams grid;
        grid_init_with_all_pipes(&grid, a, b, L,
            pipe_length,
            endcap_pipes.z0_extension,
            endcap_pipes.zL_extension,
            Lv->Nr_cavity, Lv->Nr_pipe, Lv->Nphi,
            Lv->Nz_cavity, Lv->Nz_pipe_z0, Lv->Nz_pipe_zL);

        double z0_offset = Lv->Nz_pipe_z0 * grid.dz;
        int k_endplate_z0 = Lv->Nz_pipe_z0;
        int k_endplate_zL = Lv->Nz_pipe_z0 + Lv->Nz_cavity;

        cuda_grid_init(&grid);

        /* Inner conductor ports (with z0_offset applied) */
        PortConfig inner_ports;
        port_config_init(&inner_ports);
        double dphi_pass = M_PI / num_passes;
        for (int pass = 0; pass < num_passes; pass++) {
            double phi_entry = pass * dphi_pass;
            double phi_exit = phi_entry + M_PI;
            if (phi_exit >= 2.0 * M_PI) phi_exit -= 2.0 * M_PI;

            CavityPort port;
            port.type = PORT_BEAM;
            port.surface = SURFACE_INNER;
            port.radius = aperture_radius;
            port.pos2 = L / 2.0 + z0_offset;

            port.pos1 = phi_entry;
            port.name = "Inner Entry";
            if (inner_ports.num_ports >= inner_ports.capacity) {
                inner_ports.capacity *= 2;
                inner_ports.ports = (CavityPort*)realloc(
                    inner_ports.ports, inner_ports.capacity * sizeof(CavityPort));
            }
            inner_ports.ports[inner_ports.num_ports++] = port;

            port.pos1 = phi_exit;
            port.name = "Inner Exit";
            if (inner_ports.num_ports >= inner_ports.capacity) {
                inner_ports.capacity *= 2;
                inner_ports.ports = (CavityPort*)realloc(
                    inner_ports.ports, inner_ports.capacity * sizeof(CavityPort));
            }
            inner_ports.ports[inner_ports.num_ports++] = port;
        }

        /* CPU operator with inner conductor ports */
        printf("    [setup] building CPU operator with ports...\n"); fflush(stdout);
        CurlCurlOperator cpu_op;
        curlcurl_op_init_with_ports(&cpu_op, &grid, &inner_ports);

        int n = cpu_op.n_total;

        /* Material mask: full model (radial + endcap pipes) */
        printf("    [setup] building material mask (%d radial pipes, %d endcap pipes)...\n",
            pipes.num_pipes, endcap_pipes.num_pipes); fflush(stdout);
        MaterialMask mask;
        material_mask_build_full(&mask, &pipes, &endcap_pipes, &grid, L, z0_offset);

        /* Conformal data (radial pipes) + IBC unmask of surface cells.
         * Built up-front so both Phase A and Phase B see the same geometry,
         * and so Phase B can re-use the same ConformalData. */
        printf("    [setup] building conformal data + IBC unmask...\n"); fflush(stdout);
        ConformalData cd;
        conformal_data_build(&cd, &pipes, &grid, b, z0_offset);
        {
            MaterialMask ibc_mask;
            material_mask_build_ibc(&ibc_mask, &mask, &grid);
            conformal_data_apply_ibc_unmask(&cd, &mask, &ibc_mask, &grid);
            material_mask_free(&ibc_mask);
        }

        /* GPU PEC operator with endplate pipe holes (Phase A seed) */
        printf("    [setup] initializing GPU PEC operator (n=%d DOFs)...\n", n); fflush(stdout);
        GPU_PipeOperator gpu_op;
        gpu_pipe_operator_init(&gpu_op, &cpu_op, &mask);
        gpu_pipe_operator_set_endplates(&gpu_op,
            k_endplate_z0, k_endplate_zL,
            &endcap_pipes, &grid, z0_offset);

        /* TEM initial guess (shifted z coordinate) */
        double* h_x = (double*)calloc(n, sizeof(double));
        for (int k = 0; k <= grid.Nz; k++) {
            double z_grid = k * grid.dz;
            double z_phys = z_grid - z0_offset;
            for (int j = 0; j < grid.Nphi; j++) {
                for (int i = 0; i < grid.Nr; i++) {
                    double r = grid.a + (i + 0.5) * grid.dr;
                    if (r <= b && z_phys >= 0.0 && z_phys <= L) {
                        int idx = cpu_op.offset_Er + idx_Er(&grid, i, j, k);
                        h_x[idx] = sin(M_PI * z_phys / L) / r;
                    }
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
        double k2_pec = 0.0;
        int minres_max = (n < 20000000) ? 1000 : 3000;
        run_eigensolver(&gpu_op, &cpu_op, &grid, d_x,
            20, minres_max, 1e-4, &k2_pec);
        Lv->wall_pec_sec = (double)(clock() - t_pec0) / CLOCKS_PER_SEC;

        /* Retrieve PEC eigenvector for Q_surf, R/Q, and as IBC seed */
        gpu_vector_to_host(h_x, d_x, n);

        /* Surface-integral Q (extended, with endplate k-indices) */
        PortConfig all_apertures;
        port_config_init(&all_apertures);
        for (int i = 0; i < inner_ports.num_ports; i++) {
            if (all_apertures.num_ports >= all_apertures.capacity) {
                all_apertures.capacity *= 2;
                all_apertures.ports = (CavityPort*)realloc(
                    all_apertures.ports,
                    all_apertures.capacity * sizeof(CavityPort));
            }
            all_apertures.ports[all_apertures.num_ports++] = inner_ports.ports[i];
        }
        for (int pass = 0; pass < num_passes; pass++) {
            double phi_entry = pass * dphi_pass;
            double phi_exit = phi_entry + M_PI;
            if (phi_exit >= 2.0 * M_PI) phi_exit -= 2.0 * M_PI;

            CavityPort port;
            port.type = PORT_BEAM;
            port.surface = SURFACE_OUTER;
            port.radius = aperture_radius;
            port.pos2 = L / 2.0 + z0_offset;

            port.pos1 = phi_entry;
            port.name = "Outer Entry";
            if (all_apertures.num_ports >= all_apertures.capacity) {
                all_apertures.capacity *= 2;
                all_apertures.ports = (CavityPort*)realloc(
                    all_apertures.ports,
                    all_apertures.capacity * sizeof(CavityPort));
            }
            all_apertures.ports[all_apertures.num_ports++] = port;

            port.pos1 = phi_exit;
            port.name = "Outer Exit";
            if (all_apertures.num_ports >= all_apertures.capacity) {
                all_apertures.capacity *= 2;
                all_apertures.ports = (CavityPort*)realloc(
                    all_apertures.ports,
                    all_apertures.capacity * sizeof(CavityPort));
            }
            all_apertures.ports[all_apertures.num_ports++] = port;
        }

        QFactorResult qf_pec = compute_q_factor_extended(
            &cpu_op, h_x, k2_pec, Q_SIGMA_CU, b, &all_apertures,
            k_endplate_z0, k_endplate_zL);
        RoverQResult roq_pec = compute_r_over_q(
            &cpu_op, h_x, k2_pec, b, L, z0_offset, num_passes);

        Lv->pec_converged = 1;
        Lv->pec_k_squared = k2_pec;
        Lv->pec_frequency_Hz = sqrt(fabs(k2_pec)) * c0 / (2.0 * M_PI);
        Lv->pec_Q_surf = qf_pec.Q_0;
        Lv->pec_G_factor = qf_pec.G_factor;
        Lv->pec_R_over_Q = roq_pec.R_over_Q_crossing_linac;

        printf("    [phase A] f_PEC=%.6f MHz, Q_surf=%.0f, R/Q=%.4f Ohm  (%.1f s)\n",
            Lv->pec_frequency_Hz / 1e6, Lv->pec_Q_surf, Lv->pec_R_over_Q,
            Lv->wall_pec_sec);

        /*-----------------------------------------------------------------
         * Phase B : Conformal IBC eigensolve (complex, RQI + GMRES,
         *           seeded by PEC eigenvector)
         *-----------------------------------------------------------------*/
        printf("    [phase B] initializing conformal IBC operator...\n"); fflush(stdout);
        GPU_ConformalPipeOperator gpu_cfm;
        gpu_conformal_pipe_operator_init(&gpu_cfm, &cpu_op, &mask, &cd,
            k_endplate_z0, k_endplate_zL, &endcap_pipes, &grid, z0_offset);

        /* Complex seed: [PEC eigenvector | zeros] */
        int n2 = 2 * n;
        double* h_x_cx = (double*)calloc(n2, sizeof(double));
        memcpy(h_x_cx, h_x, n * sizeof(double));

        double* d_x_cx;
        gpu_vector_alloc(&d_x_cx, n2);
        gpu_vector_to_device(d_x_cx, h_x_cx, n2);

        printf("    [phase B] complex RQI + GMRES(50), seeded from PEC...\n"); fflush(stdout);
        clock_t t_ibc0 = clock();
        GPU_ComplexEigenResult ibc_res = gpu_rqi_complex_conformal_pipe(
            &gpu_cfm, d_x_cx, k2_pec, SIGMA_CU, 30, 1e-6, 50);
        Lv->wall_ibc_sec = (double)(clock() - t_ibc0) / CLOCKS_PER_SEC;

        /* Pull complex eigenvector back; the real part is what R/Q uses */
        gpu_vector_to_host(h_x_cx, d_x_cx, n2);
        double* h_x_ibc_re = h_x_cx;   /* first n doubles */

        RoverQResult roq_ibc = compute_r_over_q(
            &cpu_op, h_x_ibc_re, ibc_res.k2_re, b, L, z0_offset, num_passes);

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
        gpu_conformal_pipe_operator_free(&gpu_cfm);
        port_config_free(&all_apertures);
        r_over_q_free(&roq_pec);
        r_over_q_free(&roq_ibc);
        free(h_x);
        gpu_vector_free(d_x);
        gpu_pipe_operator_free(&gpu_op);
        conformal_data_free(&cd);
        material_mask_free(&mask);
        port_config_free(&inner_ports);
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
        pipe_config_free(&pipes);
        endcap_pipe_config_free(&endcap_pipes);
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
    printf("  │              PHASE A — PEC CONVERGENCE (staircased pipes)                        │\n");
    printf("  │  Reference (lvl %d):  f=%.4f MHz, Q_surf=%.0f                                  │\n",
        i_finest + 1, f_pec_ref / 1e6, Q_pec_ref);
    printf("  │  Analytical (no pipes): f=%.4f MHz, Q_0=%.0f                                  │\n",
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

    /* ----- Table 2: Conformal IBC results (the paper's headline) ----- */
    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │              PHASE B — CONFORMAL IBC CONVERGENCE (paper's method)               │\n");
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
        double p = 2.0;        /* assumed second-order in the bulk */
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

        /* IBC (paper's numbers) */
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
        FILE* fp = fopen("convergence_full_model.csv", "w");
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
            printf("\n  Data written to convergence_full_model.csv\n");
        }
    }

    /*=========================================================================
     * Cleanup
     *========================================================================*/
    free(levels);
    pipe_config_free(&pipes);
    endcap_pipe_config_free(&endcap_pipes);

    printf("\n****************************************************************\n");
    printf("*         FULL MODEL CONVERGENCE STUDY COMPLETE                *\n");
    printf("****************************************************************\n\n");

    return 0;
}