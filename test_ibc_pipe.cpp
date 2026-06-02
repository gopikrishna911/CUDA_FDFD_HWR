/*=============================================================================
 * test_ibc_pipe.cpp
 *
 * IBC test for the full rhodotron model: 10-pass with radial beam pipes
 * and 2 endcap ports (vacuum + RF coupler).
 *
 * Phase 1: PEC solve (full model) — establishes reference f, Q
 * Phase 2: IBC solve (full model) — compares IBC f,Q vs PEC f,Q
 *
 * Uses a moderate grid (coarser than test_pipe_model.cpp) to fit in
 * 8 GB GPU memory with the doubled complex working arrays.
 *
 * Usage:
 *   make test_ibc_pipe
 *   ./test_ibc_pipe
 *============================================================================*/

#include "cuda_pipe_model.h"
#include "cuda_eigensolver.h"
#include "pipe_model.h"
#include "curlcurl_operator.h"
#include "q_factor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <time.h>

#define C0          299792458.0
#define MU0         (4.0e-7 * M_PI)
#define SIGMA_CU    5.8e7

/*=============================================================================
 * Helper: PEC pipe eigensolver (same pattern as test_pipe_model.cpp)
 *============================================================================*/

extern __global__ void shift_kernel(double*, const double*, double, int);

static double run_pec_pipe_solver(
    GPU_PipeOperator* pipe_op,
    const CurlCurlOperator* cpu_op,
    const GridParams* grid,
    double* d_x,
    int rqi_max,
    int minres_max,
    double tol
) {
    int n = cpu_op->n_total;
    int blocks = (n + 256 - 1) / 256;

    EigensolverWorkspace ws;
    eigensolver_workspace_init(&ws, n);

    gpu_vec_normalize_weighted(d_x, cpu_op);

    double* d_Ax;
    cudaMalloc(&d_Ax, n * sizeof(double));
    gpu_pipe_matvec(pipe_op, d_x, d_Ax);

    double xAx, xx;
    gpu_vec_dot_weighted_ws(d_x, d_Ax, &xAx, cpu_op,
        &pipe_op->base.reduction_ws);
    gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op,
        &pipe_op->base.reduction_ws);
    double sigma = xAx / xx;

    double k2_target = (M_PI / grid->L) * (M_PI / grid->L);
    if (fabs(sigma) < 1e-10 || sigma < 0) sigma = k2_target;

    printf("    PEC Pipe RQI: sigma0 = %.6f\n", sigma);

    for (int iter = 0; iter < rqi_max; iter++) {
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

        double beta_cur = b_norm, eta = b_norm;
        double c_old = 1.0, c_cur = 1.0, s_old = 0.0, s_cur = 0.0;
        int ls_iters = 0;

        for (int ls = 0; ls < minres_max; ls++) {
            gpu_pipe_matvec(pipe_op, ws.minres_ws.d_v_cur, ws.minres_ws.d_Av);
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

            double c_new = 1.0, s_new = 0.0;
            if (gamma > 1e-14) { c_new = rho3_bar / gamma; s_new = beta_new / gamma; }

            gpu_vec_copy(ws.minres_ws.d_v_cur, ws.minres_ws.d_w_new, n);
            gpu_vec_axpy(-rho2, ws.minres_ws.d_w_cur, ws.minres_ws.d_w_new, n);
            gpu_vec_axpy(-rho1, ws.minres_ws.d_w_old, ws.minres_ws.d_w_new, n);
            if (fabs(gamma) > 1e-14)
                gpu_vec_scale(ws.minres_ws.d_w_new, 1.0 / gamma, n);

            gpu_vec_axpy(c_new * eta, ws.minres_ws.d_w_new, ws.d_y, n);
            eta = -s_new * eta;
            ls_iters = ls + 1;

            if (fabs(eta) / b_norm < 1e-6) break;

            double* temp;
            temp = ws.minres_ws.d_v_old; ws.minres_ws.d_v_old = ws.minres_ws.d_v_cur;
            ws.minres_ws.d_v_cur = ws.minres_ws.d_v_new; ws.minres_ws.d_v_new = temp;
            temp = ws.minres_ws.d_w_old; ws.minres_ws.d_w_old = ws.minres_ws.d_w_cur;
            ws.minres_ws.d_w_cur = ws.minres_ws.d_w_new; ws.minres_ws.d_w_new = temp;
            beta_cur = beta_new;
            c_old = c_cur; c_cur = c_new; s_old = s_cur; s_cur = s_new;
        }

        gpu_vec_normalize_weighted(ws.d_y, cpu_op);
        gpu_vec_copy(ws.d_y, d_x, n);

        gpu_pipe_matvec(pipe_op, d_x, d_Ax);
        gpu_vec_dot_weighted_ws(d_x, d_Ax, &xAx, cpu_op,
            &pipe_op->base.reduction_ws);
        gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op,
            &pipe_op->base.reduction_ws);
        double sigma_new = xAx / xx;

        printf("      iter %d: k2 = %.10f, MINRES iters = %d\n",
            iter, sigma_new, ls_iters);

        if (fabs(sigma_new - sigma) / fabs(sigma) < tol && iter > 0) {
            printf("    PEC converged.\n");
            sigma = sigma_new;
            break;
        }
        sigma = sigma_new;
    }

    cudaFree(d_Ax);
    eigensolver_workspace_free(&ws);
    return sigma;
}

/*=============================================================================
 * MAIN
 *============================================================================*/
int main(void) {
    printf("\n");
    printf("================================================================\n");
    printf("  IBC FULL PIPE MODEL TEST\n");
    printf("================================================================\n\n");

    cuda_print_device_info();
    cuda_print_memory_info("startup");

    /*=========================================================================
     * Geometry (same as test_pipe_model.cpp)
     *========================================================================*/
    double a = 0.3333;
    double b = 1.0;
    double L = 1.395;
    double pipe_radius = 0.0125;
    double aperture_radius = 0.0175;
    double pipe_length = 0.050;
    double taper_length = 0.0;
    int num_passes = 10;

    /* Endcap pipe geometry */
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

    printf("  Geometry: a=%.4f, b=%.4f, L=%.4f m\n", a, b, L);
    printf("  Radial pipes: %d passes, r_pipe=%.1f mm, l_pipe=%.1f mm\n",
        num_passes, pipe_radius*1e3, pipe_length*1e3);
    printf("  Endcap z0: r=%.3f m, R=%.1f mm, L=%.0f mm\n",
        endcap_z0_r_center, endcap_z0_pipe_radius*1e3, endcap_z0_pipe_length*1e3);
    printf("  Endcap zL: r=%.3f m, R=%.1f mm, L=%.0f mm\n\n",
        endcap_zL_r_center, endcap_zL_pipe_radius*1e3, endcap_zL_pipe_length*1e3);

    /* Configure radial pipes */
    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, pipe_radius, aperture_radius,
        pipe_length, taper_length);
    pipe_config_add_multi_pass(&pipes, L / 2.0, num_passes);

    /* Configure endcap pipes */
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

    /* Configure inner conductor apertures */
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
        port.pos2 = L / 2.0;

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

    /*=========================================================================
     * Grid — MODERATE resolution for IBC memory budget
     * Full test_pipe_model uses Nr_cavity=162, Nphi=1024, Nz=167
     * We halve each → ~8× fewer DOFs, fits in 8GB with complex arrays
     *========================================================================*/
    int Nr_cavity = 81;
    int Nr_pipe = 7;
    int Nphi = 512;
    int Nz_cavity = 84;

    double dz_target = L / Nz_cavity;
    int Nz_pipe_z0 = (int)ceil(endcap_pipes.z0_extension / dz_target);
    int Nz_pipe_zL = (int)ceil(endcap_pipes.zL_extension / dz_target);

    GridParams grid;
    grid_init_with_all_pipes(&grid, a, b, L,
        pipe_length,
        endcap_pipes.z0_extension,
        endcap_pipes.zL_extension,
        Nr_cavity, Nr_pipe, Nphi,
        Nz_cavity, Nz_pipe_z0, Nz_pipe_zL);

    double z0_offset = endcap_pipes.z0_extension;
    int k_endplate_z0 = Nz_pipe_z0;
    int k_endplate_zL = Nz_pipe_z0 + Nz_cavity;

    printf("  Endplate k-indices: z0 at k=%d, zL at k=%d (of Nz=%d)\n",
        k_endplate_z0, k_endplate_zL, grid.Nz);

    /* Build combined material mask */
    printf("\n  Building material mask...\n");
    MaterialMask mask;
    material_mask_build_full(&mask, &pipes, &endcap_pipes,
        &grid, L, z0_offset);
    material_mask_print_stats(&mask, &grid);

    /* CPU operator with inner conductor ports */
    CurlCurlOperator cpu_op;
    curlcurl_op_init_with_ports(&cpu_op, &grid, &inner_ports);
    int n = cpu_op.n_total;

    printf("\n  Total DOFs: %d (%.1f M)\n", n, n / 1e6);
    cuda_grid_init(&grid);

    /* Memory check */
    size_t gpu_free, gpu_total;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    double free_GB = gpu_free / (1024.0 * 1024.0 * 1024.0);
    /* IBC needs ~80 bytes/DOF (real fields + complex fields + workspace) */
    double ibc_est_GB = n * 80.0 / (1024.0 * 1024.0 * 1024.0);
    printf("  GPU: %.1f GB free, IBC est: %.1f GB\n\n", free_GB, ibc_est_GB);

    /* TEM initial guess (z-shifted for extended grid) */
    double* h_x = (double*)calloc(n, sizeof(double));
    for (int k = 0; k <= grid.Nz; k++) {
        double z_phys = k * grid.dz - z0_offset;
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

    /* Analytical reference */
    QFactorResult qf_analytical = compute_q_analytical_coaxial_hwr(
        a, b, L, SIGMA_CU);

    /*=========================================================================
     * PHASE 1: PEC solve (full model)
     *========================================================================*/
    printf("================================================================\n");
    printf("  PHASE 1: PEC FULL MODEL\n");
    printf("================================================================\n\n");

    double k2_pec = 0.0;
    double f_pec = 0.0, Q_pec = 0.0;
    {
        GPU_PipeOperator gpu_pec;
        gpu_pipe_operator_init(&gpu_pec, &cpu_op, &mask);
        gpu_pipe_operator_set_endplates(&gpu_pec,
            k_endplate_z0, k_endplate_zL,
            &endcap_pipes, &grid, z0_offset);

        double* d_x;
        gpu_vector_alloc(&d_x, n);
        gpu_vector_to_device(d_x, h_x, n);

        clock_t t0 = clock();
        k2_pec = run_pec_pipe_solver(&gpu_pec, &cpu_op, &grid, d_x,
            20, 3000, 1e-8);
        double t_pec = (double)(clock() - t0) / CLOCKS_PER_SEC;

        /* Retrieve eigenvector for Q computation */
        /* Save PEC eigenvector into h_x for Phase 2 IBC initial guess */
        gpu_vector_to_host(h_x, d_x, n);

        /* Also copy for Q computation */
        double* h_eigvec = (double*)malloc(n * sizeof(double));
        gpu_vector_to_host(h_eigvec, d_x, n);

        QFactorResult qf_pec = compute_q_factor_extended(
            &cpu_op, h_eigvec, k2_pec, SIGMA_CU, b, NULL,
            k_endplate_z0, k_endplate_zL);

        f_pec = qf_pec.frequency_Hz;
        Q_pec = qf_pec.Q_0;

        printf("\n  PEC Results:\n");
        printf("    k2      = %.10f\n", k2_pec);
        printf("    f       = %.6f MHz\n", f_pec / 1e6);
        printf("    Q(surf) = %.1f\n", Q_pec);
        printf("    Time    = %.1f s\n\n", t_pec);

        free(h_eigvec);
        gpu_vector_free(d_x);
        gpu_pipe_operator_free(&gpu_pec);
    }

    /*=========================================================================
     * PHASE 2: IBC Perturbative (surface correction only — instant)
     *========================================================================*/
    printf("================================================================\n");
    printf("  PHASE 2: IBC PERTURBATIVE\n");
    printf("================================================================\n\n");

    {
        GPU_PipeOperator gpu_ibc;
        gpu_pipe_operator_init_complex(&gpu_ibc, &cpu_op, &mask);

    //    /* Upload PEC eigenvector (real, size n) */
        double* d_x_pec;
        gpu_vector_alloc(&d_x_pec, n);
        gpu_vector_to_device(d_x_pec, h_x, n);

        clock_t t0 = clock();
        GPU_ComplexEigenResult ibc_result = gpu_ibc_perturbative_pipe(
            &gpu_ibc, d_x_pec, k2_pec, SIGMA_CU);
        double t_ibc = (double)(clock() - t0) / CLOCKS_PER_SEC;

        printf("    Time    = %.1f s\n\n", t_ibc);

        double df_pct = (ibc_result.frequency_Hz - f_pec) / f_pec * 100.0;
        double dQ_pct = (Q_pec > 0) ?
            (ibc_result.Q_factor - Q_pec) / Q_pec * 100.0 : 0.0;

        printf("  ================================================================\n");
        printf("  COMPARISON\n");
        printf("  ================================================================\n");
        printf("    Analytical (ref cavity): f=%.4f MHz, Q=%.0f\n",
            qf_analytical.frequency_Hz / 1e6, qf_analytical.Q_0);
        printf("    PEC (full model):        f=%.4f MHz, Q=%.0f (surf integral)\n",
            f_pec / 1e6, Q_pec);
        printf("    IBC perturbative:        f=%.4f MHz, Q=%.0f\n",
            ibc_result.frequency_Hz / 1e6, ibc_result.Q_factor);
        printf("    f diff (IBC vs PEC):     %+.4f%%\n", df_pct);
        printf("    Q diff (IBC vs PEC):     %+.2f%%\n", dQ_pct);
        printf("  ================================================================\n\n");

        gpu_vector_free(d_x_pec);
        gpu_pipe_operator_free(&gpu_ibc);
    }



    /*=========================================================================
     * PHASE 3: IBC Full Iterative Solve
     *========================================================================*/
    printf("================================================================\n");
    printf("  PHASE 3: IBC FULL ITERATIVE (Complex RQI + GMRES)\n");
    printf("================================================================\n\n");

    {
        GPU_PipeOperator gpu_ibc_full;
        gpu_pipe_operator_init_complex(&gpu_ibc_full, &cpu_op, &mask);

        /* Build complex initial guess: [PEC eigenvector | zeros] */
        int n2 = 2 * n;
        double* h_x_cx = (double*)calloc(n2, sizeof(double));
        memcpy(h_x_cx, h_x, n * sizeof(double));  /* real part = PEC eigvec */
        /* h_x_cx[n..2n-1] = 0 already from calloc */

        double* d_x_cx;
        gpu_vector_alloc(&d_x_cx, n2);
        gpu_vector_to_device(d_x_cx, h_x_cx, n2);

        clock_t t0 = clock();
        GPU_ComplexEigenResult ibc_full = gpu_rqi_complex_pipe(
            &gpu_ibc_full, d_x_cx,
            k2_pec,         /* initial shift from PEC */
            SIGMA_CU,       /* copper conductivity */
            30,             /* max RQI iterations */
            1e-6,           /* convergence tolerance */
            120              /* GMRES restart m */
        );
        double t_ibc = (double)(clock() - t0) / CLOCKS_PER_SEC;

        printf("\n  IBC Full Iterative Results:\n");
        printf("    k2_re  = %.10f\n", ibc_full.k2_re);
        printf("    k2_im  = %.6e\n", ibc_full.k2_im);
        printf("    f(IBC) = %.6f MHz\n", ibc_full.frequency_Hz / 1e6);
        printf("    Q(IBC) = %.1f\n", ibc_full.Q_factor);
        printf("    Converged: %s (residual = %.2e, iters = %d)\n",
            ibc_full.converged ? "YES" : "NO",
            ibc_full.residual, ibc_full.iterations);
        printf("    Time   = %.1f s\n\n", t_ibc);

        printf("  ================================================================\n");
        printf("  FULL COMPARISON\n");
        printf("  ================================================================\n");
        printf("    Analytical (ref cavity): f=%.4f MHz, Q=%.0f\n",
            qf_analytical.frequency_Hz / 1e6, qf_analytical.Q_0);
        printf("    PEC (full model):        f=%.4f MHz, Q=%.0f (surf integral)\n",
            f_pec / 1e6, Q_pec);
        printf("    IBC iterative:           f=%.4f MHz, Q=%.0f\n",
            ibc_full.frequency_Hz / 1e6, ibc_full.Q_factor);

        double df_pct = (ibc_full.frequency_Hz - f_pec) / f_pec * 100.0;
        double dQ_pct = (Q_pec > 0) ?
            (ibc_full.Q_factor - Q_pec) / Q_pec * 100.0 : 0.0;
        printf("    f diff (IBC vs PEC):     %+.4f%%\n", df_pct);
        printf("    Q diff (IBC vs PEC):     %+.2f%%\n", dQ_pct);
        printf("  ================================================================\n\n");

        gpu_vector_free(d_x_cx);
        free(h_x_cx);
        gpu_pipe_operator_free(&gpu_ibc_full);
    }



    /* Cleanup */
    free(h_x);
    port_config_free(&inner_ports);
    endcap_pipe_config_free(&endcap_pipes);
    pipe_config_free(&pipes);
    material_mask_free(&mask);
    curlcurl_op_free(&cpu_op);

    printf("================================================================\n");
    printf("  IBC FULL PIPE MODEL TEST COMPLETE\n");
    printf("================================================================\n\n");

    return 0;
}
