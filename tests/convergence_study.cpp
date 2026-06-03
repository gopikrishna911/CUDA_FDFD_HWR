/*=============================================================================
 * test_convergence_full.cpp
 *
 * Grid convergence study for the FULL Rhodotron model (all pipes).
 *
 * Runs the complete cavity (20 radial pipes + 2 endcap pipes) at multiple
 * grid resolutions. Reports f, Q_0, and R/Q at each level, convergence
 * order, and Richardson extrapolation.
 *
 * The reference comparison is the finest-grid FDFD result, since there is
 * no closed-form analytical solution for the perturbed cavity. The
 * analytical (unperturbed) values are also shown for context.
 *
 * Grid levels defined by target dr. Aspect ratios:
 *     r*dphi ≈ 1.5 * dr   (at r = b)
 *     dz     ≈ 2.0 * dr
 *
 * Levels exceeding GPU memory are automatically skipped.
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
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <time.h>

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
    int converged;
    double wall_sec;

    /* Results */
    double k_squared;
    double frequency_Hz;
    double Q_0;
    double G_factor;
    double R_over_Q_linac;
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
    double dr_targets[] = { 16.0, 10.0, 8.0, 5.5 };
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

        clock_t t_start = clock();

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

        /* GPU operator with endplate pipe holes */
        printf("    [setup] initializing GPU pipe operator (n=%d DOFs)...\n", n); fflush(stdout);
        GPU_PipeOperator gpu_op;
        gpu_pipe_operator_init(&gpu_op, &cpu_op, &mask);

        printf("    [setup] setting up endplate pipe-hole masks...\n"); fflush(stdout);
        gpu_pipe_operator_set_endplates(&gpu_op,
            k_endplate_z0, k_endplate_zL,
            &endcap_pipes, &grid, z0_offset);

        printf("    [setup] done, entering solver.\n"); fflush(stdout);

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

        /* Solve */
        double k2 = 0.0;
        int minres_max = (n < 20000000) ? 1000 : 3000;
        run_eigensolver(&gpu_op, &cpu_op, &grid, d_x,
            20, minres_max, 1e-4, &k2);

        /* Retrieve eigenvector */
        gpu_vector_to_host(h_x, d_x, n);

        /* Q factor (extended, with endplate k-indices) */
        /* Build outer aperture exclusion list */
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

        QFactorResult qf = compute_q_factor_extended(
            &cpu_op, h_x, k2, Q_SIGMA_CU, b, &all_apertures,
            k_endplate_z0, k_endplate_zL);

        /* R/Q */
        RoverQResult roq = compute_r_over_q(
            &cpu_op, h_x, k2, b, L, z0_offset, num_passes);

        clock_t t_end = clock();
        Lv->wall_sec = (double)(t_end - t_start) / CLOCKS_PER_SEC;

        /* Store results */
        Lv->ran = 1;
        Lv->converged = 1;
        Lv->k_squared = k2;
        Lv->frequency_Hz = sqrt(fabs(k2)) * c0 / (2.0 * M_PI);
        Lv->Q_0 = qf.Q_0;
        Lv->G_factor = qf.G_factor;
        Lv->R_over_Q_linac = roq.R_over_Q_crossing_linac;

        printf("    f = %.6f MHz,  Q0 = %.0f,  R/Q = %.4f Ohm,  %.1f s\n\n",
            Lv->frequency_Hz / 1e6, Lv->Q_0, Lv->R_over_Q_linac, Lv->wall_sec);

        /* Cleanup this level */
        port_config_free(&all_apertures);
        r_over_q_free(&roq);
        free(h_x);
        gpu_vector_free(d_x);
        gpu_pipe_operator_free(&gpu_op);
        material_mask_free(&mask);
        port_config_free(&inner_ports);
        curlcurl_op_free(&cpu_op);
    }

    /*=========================================================================
     * Convergence summary table
     *========================================================================*/
     /* Use finest-grid result as reference */
    int i_finest = -1;
    for (int i = n_levels - 1; i >= 0; i--) {
        if (levels[i].ran) { i_finest = i; break; }
    }

    double f_ref = (i_finest >= 0) ? levels[i_finest].frequency_Hz : 0;
    double Q_ref = (i_finest >= 0) ? levels[i_finest].Q_0 : 0;
    double RoQ_ref = (i_finest >= 0) ? levels[i_finest].R_over_Q_linac : 0;

    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │                    FULL MODEL CONVERGENCE RESULTS                                │\n");
    printf("  │  Finest grid (lvl %d):  f=%.4f MHz, Q0=%.0f, R/Q=%.4f Ohm                  │\n",
        i_finest + 1, f_ref / 1e6, Q_ref, RoQ_ref);
    printf("  │  Analytical (no pipes): f=%.4f MHz, Q0=%.0f, R/Q=%.4f Ohm                  │\n",
        qf_analytical.frequency_Hz / 1e6, qf_analytical.Q_0,
        roq_analytical.R_over_Q_crossing_linac);
    printf("  ├──────────────────────────────────────────────────────────────────────────────────┤\n");
    printf("  │ Lvl  dr(mm)  dz(mm)     DOFs   f(MHz)      Δf(Hz)   Q_0    R/Q(Ω)   Time     │\n");
    printf("  │                                            vs finest                            │\n");
    printf("  │ ───  ──────  ──────  ────────  ──────────  ───────  ─────  ───────  ──────     │\n");

    for (int lvl = 0; lvl < n_levels; lvl++) {
        ConvLevel* Lv = &levels[lvl];
        if (!Lv->ran) {
            printf("  │ %2d   %5.1f   %5.1f   %6.1fM  %45s │\n",
                lvl + 1, Lv->dr_target_mm, Lv->dz_mm, Lv->n_dofs / 1e6,
                "— SKIPPED —");
            continue;
        }

        double df_vs_finest = Lv->frequency_Hz - f_ref;
        printf("  │ %2d   %5.1f   %5.1f   %6.1fM  %10.4f  %+8.1f  %5.0f  %7.4f  %6.1fs%s │\n",
            lvl + 1, Lv->dr_target_mm, Lv->dz_mm,
            Lv->n_dofs / 1e6,
            Lv->frequency_Hz / 1e6,
            df_vs_finest,
            Lv->Q_0,
            Lv->R_over_Q_linac,
            Lv->wall_sec,
            (lvl == i_finest) ? " *" : "  ");
    }
    printf("  └──────────────────────────────────────────────────────────────────────────────────┘\n");

    /*=========================================================================
     * Convergence rates (consecutive pairs)
     *========================================================================*/
    if (i_finest >= 0) {
        printf("\n  Convergence toward finest grid:\n");
        printf("  %5s  %8s  %12s  %10s  %10s\n",
            "Level", "dr(mm)", "Δf(Hz)", "ΔQ", "ΔR/Q(Ω)");
        printf("  ───── ──────── ──────────── ────────── ──────────\n");

        for (int i = 0; i < n_levels; i++) {
            if (!levels[i].ran) continue;
            printf("  %5d  %8.1f  %+12.1f  %+10.0f  %+10.4f\n",
                i + 1, levels[i].dr_target_mm,
                levels[i].frequency_Hz - f_ref,
                levels[i].Q_0 - Q_ref,
                levels[i].R_over_Q_linac - RoQ_ref);
        }
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

        /* Assume p=2 (second-order FD) */
        double p = 2.0;
        double rp = pow(r, p);

        double f_rich = Lf->frequency_Hz
            + (Lf->frequency_Hz - Lc->frequency_Hz) / (rp - 1.0);
        double Q_rich = Lf->Q_0
            + (Lf->Q_0 - Lc->Q_0) / (rp - 1.0);
        double RoQ_rich = Lf->R_over_Q_linac
            + (Lf->R_over_Q_linac - Lc->R_over_Q_linac) / (rp - 1.0);

        printf("    Assuming p=2 (second-order convergence):\n");
        printf("    ┌──────────────────────────────────────────────────────┐\n");
        printf("    │  Quantity     Coarse (L%d)  Fine (L%d)   Richardson  │\n",
            i_coarse + 1, i_fine + 1);
        printf("    │  ──────────  ───────────  ──────────  ──────────── │\n");
        printf("    │  f (MHz)     %11.4f  %11.4f  %11.4f     │\n",
            Lc->frequency_Hz / 1e6, Lf->frequency_Hz / 1e6, f_rich / 1e6);
        printf("    │  Q_0         %11.0f  %11.0f  %11.0f     │\n",
            Lc->Q_0, Lf->Q_0, Q_rich);
        printf("    │  R/Q (Ω)     %11.4f  %11.4f  %11.4f     │\n",
            Lc->R_over_Q_linac, Lf->R_over_Q_linac, RoQ_rich);
        printf("    └──────────────────────────────────────────────────────┘\n");
    }

    /*=========================================================================
     * Export CSV
     *========================================================================*/
    {
        FILE* fp = fopen("convergence_full_model.csv", "w");
        if (fp) {
            fprintf(fp, "level,dr_mm,dphi_deg,dz_mm,n_dofs,"
                "frequency_Hz,Q_0,R_over_Q_linac,wall_sec\n");
            for (int i = 0; i < n_levels; i++) {
                ConvLevel* Lv = &levels[i];
                if (!Lv->ran) continue;
                fprintf(fp, "%d,%.4f,%.6f,%.4f,%lld,"
                    "%.6f,%.2f,%.6f,%.2f\n",
                    i + 1, Lv->dr_mm, Lv->dphi_deg, Lv->dz_mm, Lv->n_dofs,
                    Lv->frequency_Hz, Lv->Q_0, Lv->R_over_Q_linac, Lv->wall_sec);
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