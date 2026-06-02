/*=============================================================================
 * test_conformal_ibc.cpp
 *
 * Phase 3 validation: Full IBC eigensolver with Dey-Mittra conformal
 * subcell method on the rhodotron pipe model.
 *
 * Phase 1: PEC eigensolver (full model with 20 radial pipes)
 *          → establishes reference f, Q, eigenvector
 *
 * Phase 2: Staircase IBC perturbative (existing code, for comparison)
 *
 * Phase 3: Conformal IBC perturbative (new — should fix the O(dr) error)
 *
 * Phase 4: Conformal IBC full iterative (RQI + GMRES)
 *
 * The key test: Phase 3 conformal perturbative Δk² should be O(α) ≈ 10⁻⁴,
 * NOT the O(1) or O(dr) values that the staircase gives.
 *
 * Compile:
 *   nvcc -O2 -o test_conformal_ibc test_conformal_ibc.cpp \
 *        cuda_conformal_pipe_cu.cpp cuda_conformal_curls_cu.cpp \
 *        conformal_geometry.c \
 *        cuda_curls_cu.cpp cuda_fields_cu.cpp cuda_operator_cu.cpp \
 *        cuda_vector_ops_cu.cpp cuda_pipe_model_cu.cpp \
 *        cuda_eigensolver_cu.cpp \
 *        curlcurl_operator.cpp curl_E.cpp curl_H.cpp pipe_model.cpp \
 *        -lm
 *============================================================================*/

#include "cuda_conformal_pipe.h"
#include "cuda_pipe_model.h"
#include "cuda_eigensolver.h"
#include "conformal_geometry.h"
#include "pipe_model.h"
#include "curlcurl_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>
#include <device_launch_parameters.h>

#define C0          299792458.0
#define MU0         (4.0e-7 * M_PI)
#define SIGMA_CU    5.8e7

/*=============================================================================
 * PEC Pipe Eigensolver (same as test_ibc_pipe.cpp)
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
            shift_kernel<<<blocks, 256>>>(ws.minres_ws.d_Av,
                ws.minres_ws.d_v_cur, sigma, n);

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
            if (gamma > 1e-14) {
                c_new = rho3_bar / gamma;
                s_new = beta_new / gamma;
            }

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
            temp = ws.minres_ws.d_v_old;
            ws.minres_ws.d_v_old = ws.minres_ws.d_v_cur;
            ws.minres_ws.d_v_cur = ws.minres_ws.d_v_new;
            ws.minres_ws.d_v_new = temp;

            temp = ws.minres_ws.d_w_old;
            ws.minres_ws.d_w_old = ws.minres_ws.d_w_cur;
            ws.minres_ws.d_w_cur = ws.minres_ws.d_w_new;
            ws.minres_ws.d_w_new = temp;

            beta_cur = beta_new;
            c_old = c_cur; c_cur = c_new;
            s_old = s_cur; s_cur = s_new;
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
    printf("  CONFORMAL IBC — PHASE 3 VALIDATION\n");
    printf("================================================================\n\n");

    cuda_print_device_info();
    cuda_print_memory_info("startup");

    /*=========================================================================
     * Geometry (same as test_ibc_pipe.cpp)
     *========================================================================*/
    double a = 0.3333;
    double b = 1.0;
    double L = 1.395;
    double pipe_radius = 0.0125;
    double aperture_radius = 0.0175;
    double pipe_length = 0.050;
    int num_passes = 10;

    printf("  Geometry: a=%.4f, b=%.4f, L=%.4f m\n", a, b, L);
    printf("  Radial pipes: %d passes, r_pipe=%.1f mm, aperture=%.1f mm\n",
           num_passes, pipe_radius * 1e3, aperture_radius * 1e3);

    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, pipe_radius, aperture_radius,
                     pipe_length, 0.0);
    pipe_config_add_multi_pass(&pipes, L / 2.0, num_passes);

    /* Inner conductor apertures for IBC port masking */
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
     * Grid
     *========================================================================*/
    int Nr_cavity = 81;
    int Nr_pipe = 7;
    int Nphi = 256;
    int Nz = 84;

    GridParams grid;
    grid_init_with_pipes(&grid, a, b, L, pipe_length,
                         Nr_cavity, Nr_pipe, Nphi, Nz);

    printf("  Grid: Nr=%d, Nphi=%d, Nz=%d\n", grid.Nr, Nphi, Nz);
    printf("  dr=%.2f mm, dz=%.2f mm\n", grid.dr * 1e3, grid.dz * 1e3);

    /* Build material mask (staircase) */
    MaterialMask mask;
    material_mask_build(&mask, &pipes, &grid);
    material_mask_print_stats(&mask, &grid);

    /* Build conformal data */
    printf("\n  Building conformal geometry data...\n");
    ConformalData cd;
    conformal_data_build(&cd, &pipes, &grid, b, 0.0);
    conformal_data_print_stats(&cd, &grid);

    /* CPU operator with inner conductor ports */
    CurlCurlOperator cpu_op;
    curlcurl_op_init_with_ports(&cpu_op, &grid, &inner_ports);
    int n = cpu_op.n_total;

    printf("\n  Total DOFs: %d (%.1f M)\n", n, n / 1e6);
    cuda_grid_init(&grid);

    /* TEM initial guess */
    double* h_x = (double*)calloc(n, sizeof(double));
    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = grid.a + (i + 0.5) * grid.dr;
                if (r <= b && z >= 0.0 && z <= L) {
                    int idx = cpu_op.offset_Er + idx_Er(&grid, i, j, k);
                    h_x[idx] = sin(M_PI * z / L) / r;
                }
            }
        }
    }

    /*=========================================================================
     * PHASE 1: PEC Eigensolver
     *========================================================================*/
    printf("\n================================================================\n");
    printf("  PHASE 1: PEC FULL MODEL\n");
    printf("================================================================\n\n");

    double k2_pec, f_pec;
    {
        GPU_PipeOperator gpu_pec;
        gpu_pipe_operator_init(&gpu_pec, &cpu_op, &mask);

        double* d_x;
        gpu_vector_alloc(&d_x, n);
        gpu_vector_to_device(d_x, h_x, n);

        clock_t t0 = clock();
        k2_pec = run_pec_pipe_solver(&gpu_pec, &cpu_op, &grid, d_x,
                                     20, 3000, 1e-8);
        double t_pec = (double)(clock() - t0) / CLOCKS_PER_SEC;

        f_pec = C0 * sqrt(fabs(k2_pec)) / (2.0 * M_PI);

        /* Save PEC eigenvector for IBC phases */
        gpu_vector_to_host(h_x, d_x, n);

        printf("\n  PEC Results:\n");
        printf("    k2      = %.10f\n", k2_pec);
        printf("    f       = %.6f MHz\n", f_pec / 1e6);
        printf("    Time    = %.1f s\n\n", t_pec);

        gpu_vector_free(d_x);
        gpu_pipe_operator_free(&gpu_pec);
    }

    /*=========================================================================
     * PHASE 2: Staircase IBC Perturbative (existing code, for comparison)
     *========================================================================*/
    printf("================================================================\n");
    printf("  PHASE 2: STAIRCASE IBC PERTURBATIVE (existing)\n");
    printf("================================================================\n\n");

    double f_staircase = 0.0, Q_staircase = 0.0;
    {
        GPU_PipeOperator gpu_ibc;
        gpu_pipe_operator_init_complex(&gpu_ibc, &cpu_op, &mask);

        double* d_x_pec;
        gpu_vector_alloc(&d_x_pec, n);
        gpu_vector_to_device(d_x_pec, h_x, n);

        clock_t t0 = clock();
        GPU_ComplexEigenResult ibc_result = gpu_ibc_perturbative_pipe(
            &gpu_ibc, d_x_pec, k2_pec, SIGMA_CU);
        double t_ibc = (double)(clock() - t0) / CLOCKS_PER_SEC;

        f_staircase = ibc_result.frequency_Hz;
        Q_staircase = ibc_result.Q_factor;

        printf("    Time = %.2f s\n\n", t_ibc);

        gpu_vector_free(d_x_pec);
        gpu_pipe_operator_free(&gpu_ibc);
    }

    /*=========================================================================
     * PHASE 3: Conformal IBC Perturbative (new!)
     *========================================================================*/
    printf("================================================================\n");
    printf("  PHASE 3: CONFORMAL IBC PERTURBATIVE (Dey-Mittra)\n");
    printf("================================================================\n\n");

    double f_conformal = 0.0, Q_conformal = 0.0;
    {
        GPU_ConformalPipeOperator gpu_cfm;
        gpu_conformal_pipe_operator_init(&gpu_cfm, &cpu_op, &mask, &cd);

        double* d_x_pec;
        gpu_vector_alloc(&d_x_pec, n);
        gpu_vector_to_device(d_x_pec, h_x, n);

        clock_t t0 = clock();
        GPU_ComplexEigenResult cfm_result = gpu_ibc_perturbative_conformal_pipe(
            &gpu_cfm, d_x_pec, k2_pec, SIGMA_CU);
        double t_cfm = (double)(clock() - t0) / CLOCKS_PER_SEC;

        f_conformal = cfm_result.frequency_Hz;
        Q_conformal = cfm_result.Q_factor;

        printf("    Time = %.2f s\n\n", t_cfm);

        gpu_vector_free(d_x_pec);
        gpu_conformal_pipe_operator_free(&gpu_cfm);
    }

    /*=========================================================================
     * PHASE 4: Conformal IBC Full Iterative (RQI + GMRES)
     *========================================================================*/
    printf("================================================================\n");
    printf("  PHASE 4: CONFORMAL IBC FULL ITERATIVE (RQI + GMRES)\n");
    printf("================================================================\n\n");

    double f_cfm_full = 0.0, Q_cfm_full = 0.0;
    {
        GPU_ConformalPipeOperator gpu_cfm;
        gpu_conformal_pipe_operator_init(&gpu_cfm, &cpu_op, &mask, &cd);

        /* Complex initial guess: [PEC eigvec | zeros] */
        int n2 = 2 * n;
        double* h_x_cx = (double*)calloc(n2, sizeof(double));
        memcpy(h_x_cx, h_x, n * sizeof(double));

        double* d_x_cx;
        gpu_vector_alloc(&d_x_cx, n2);
        gpu_vector_to_device(d_x_cx, h_x_cx, n2);

        clock_t t0 = clock();
        GPU_ComplexEigenResult cfm_full = gpu_rqi_complex_conformal_pipe(
            &gpu_cfm, d_x_cx,
            k2_pec,         /* initial shift from PEC */
            SIGMA_CU,       /* copper conductivity */
            30,             /* max RQI iterations */
            1e-6,           /* convergence tolerance */
            50             /* GMRES restart m */
        );
        double t_cfm = (double)(clock() - t0) / CLOCKS_PER_SEC;

        f_cfm_full = cfm_full.frequency_Hz;
        Q_cfm_full = cfm_full.Q_factor;

        printf("\n  Conformal IBC Full Iterative Results:\n");
        printf("    k2_re  = %.10f\n", cfm_full.k2_re);
        printf("    k2_im  = %.6e\n", cfm_full.k2_im);
        printf("    f(IBC) = %.6f MHz\n", f_cfm_full / 1e6);
        printf("    Q(IBC) = %.1f\n", Q_cfm_full);
        printf("    Converged: %s (residual = %.2e, iters = %d)\n",
               cfm_full.converged ? "YES" : "NO",
               cfm_full.residual, cfm_full.iterations);
        printf("    Time   = %.1f s\n\n", t_cfm);

        gpu_vector_free(d_x_cx);
        free(h_x_cx);
        gpu_conformal_pipe_operator_free(&gpu_cfm);
    }

    /*=========================================================================
     * COMPARISON TABLE
     *========================================================================*/
    printf("================================================================\n");
    printf("  COMPARISON\n");
    printf("================================================================\n\n");

    double k2_analytical = (M_PI / L) * (M_PI / L);
    double f_analytical = C0 * sqrt(k2_analytical) / (2.0 * M_PI);

    printf("    %-30s  %12s  %12s\n", "Method", "f (MHz)", "Q");
    printf("    %-30s  %12s  %12s\n", "------", "-------", "---");
    printf("    %-30s  %12.4f  %12s\n", "Analytical (no pipes)",
           f_analytical / 1e6, "—");
    printf("    %-30s  %12.4f  %12s\n", "PEC (full model)",
           f_pec / 1e6, "—");
    printf("    %-30s  %12.4f  %12.0f\n", "Staircase IBC perturbative",
           f_staircase / 1e6, Q_staircase);
    printf("    %-30s  %12.4f  %12.0f\n", "Conformal IBC perturbative",
           f_conformal / 1e6, Q_conformal);
    printf("    %-30s  %12.4f  %12.0f\n", "Conformal IBC iterative",
           f_cfm_full / 1e6, Q_cfm_full);

    printf("\n    Frequency shifts from PEC:\n");
    printf("      Staircase perturbative: %+.4f%%\n",
           (f_staircase - f_pec) / f_pec * 100.0);
    printf("      Conformal perturbative: %+.4f%%\n",
           (f_conformal - f_pec) / f_pec * 100.0);
    printf("      Conformal iterative:    %+.4f%%\n",
           (f_cfm_full - f_pec) / f_pec * 100.0);

    printf("\n    Expected IBC frequency shift: ~0.001%% (O(α) ≈ O(skin_depth))\n");
    printf("    Expected Q for copper at ~107 MHz: ~48000\n");

    printf("\n================================================================\n\n");

    /* Cleanup */
    free(h_x);
    port_config_free(&inner_ports);
    pipe_config_free(&pipes);
    material_mask_free(&mask);
    conformal_data_free(&cd);
    curlcurl_op_free(&cpu_op);

    return 0;
}
