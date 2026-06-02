/*=============================================================================
 * test_ibc.cpp
 *
 * Three-stage test of the IBC (Impedance Boundary Condition) implementation.
 *
 * Stage 1: PEC REGRESSION
 *   Run the existing PEC solver on the reference cavity (no pipes).
 *   Verify it still matches analytical values.
 *   → If this fails, the struct changes broke backward compatibility.
 *
 * Stage 2: COMPLEX INFRASTRUCTURE
 *   Test complex alloc, pack/unpack, dot products, curl on a small grid.
 *   Feed a real-only vector (imag=0) through the complex path.
 *   → If this fails, the Phase 1-3 plumbing has a bug.
 *
 * Stage 3: IBC EIGENSOLVER
 *   Run the full IBC complex RQI on the reference cavity.
 *   Compare IBC-f vs PEC-f (should agree to ~0.01%)
 *   Compare IBC-Q vs surface-integral-Q (should agree to ~1-5% at moderate grid)
 *   → This is the real physics validation.
 *
 * Usage:
 *   make test_ibc
 *   ./test_ibc
 *
 * Uses a single moderate grid (dr ≈ 10 mm) for speed.
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

/*=============================================================================
 * Constants
 *============================================================================*/
#define C0          299792458.0
#define MU0         (4.0e-7 * M_PI)
#define SIGMA_CU    5.8e7

/*=============================================================================
 * Helper: fill initial guess with TEM mode Er = sin(π z/L) / r
 *============================================================================*/
static void fill_tem_guess(
    double* h_x,
    const CurlCurlOperator* op,
    const GridParams* grid,
    double b_cavity,
    double L_cavity
) {
    int n = op->n_total;
    for (int i = 0; i < n; i++) h_x[i] = 0.0;

    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                double r = grid->a + (i + 0.5) * grid->dr;
                if (r <= b_cavity) {
                    int idx = op->offset_Er + idx_Er(grid, i, j, k);
                    h_x[idx] = sin(M_PI * z / L_cavity) / r;
                }
            }
        }
    }
}

/*=============================================================================
 * Helper: run PEC eigensolver (same as existing test code)
 *============================================================================*/

/* Forward-declare the shift kernel from cuda_eigensolver.cu */
extern __global__ void shift_kernel(double* Av, const double* v, double sigma, int n);

static double run_pec_solver(
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

    /* Initial Rayleigh quotient */
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

    printf("    PEC RQI: sigma0 = %.6f\n", sigma);

    for (int iter = 0; iter < rqi_max; iter++) {
        /* MINRES solve (A - σI)y = x */
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

        /* Normalize, update */
        gpu_vec_normalize_weighted(ws.d_y, cpu_op);
        gpu_vec_copy(ws.d_y, d_x, n);

        /* New Rayleigh quotient */
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
int main() {
    printf("\n");
    printf("================================================================\n");
    printf("  IBC IMPLEMENTATION TEST\n");
    printf("================================================================\n\n");

    cuda_print_device_info();

    /*=========================================================================
     * Geometry: reference cavity (no pipes)
     *========================================================================*/
    double a = 1.0 / 3.0;
    double b = 1.0;
    double L = 1.395;
    double pipe_length = 0.050;

    printf("  Geometry: a = %.4f m, b = %.4f m, L = %.4f m\n", a, b, L);

    /* Analytical reference */
    QFactorResult qf_analytical = compute_q_analytical_coaxial_hwr(
        a, b, L, SIGMA_CU);
    double f_exact = qf_analytical.frequency_Hz;
    double Q_exact = qf_analytical.Q_0;

    printf("  Analytical: f = %.6f MHz, Q = %.1f\n\n",
        f_exact / 1e6, Q_exact);

    /*=========================================================================
     * Grid: moderate resolution (dr ≈ 10 mm)
     *========================================================================*/
    double dr_target = 10.0e-3;
    int Nr_cavity = (int)round((b - a) / dr_target);
    int Nr_pipe = (int)round(pipe_length / dr_target);
    double dphi_target = 1.5 * dr_target / b;
    int Nphi = (int)round(2.0 * M_PI / dphi_target);
    if (Nphi % 2 != 0) Nphi++;
    int Nz = (int)round(L / (2.0 * dr_target));

    GridParams grid;
    grid_init_with_pipes(&grid, a, b, L, pipe_length,
        Nr_cavity, Nr_pipe, Nphi, Nz);
    cuda_grid_init(&grid);

    CurlCurlOperator cpu_op;
    curlcurl_op_init(&cpu_op, &grid);
    int n = cpu_op.n_total;

    printf("  Grid: Nr=%d+%d, Nphi=%d, Nz=%d, DOFs=%d (%.1fM)\n",
        Nr_cavity, Nr_pipe, Nphi, Nz, n, n / 1e6);
    printf("  dr=%.1f mm, r*dphi=%.1f mm, dz=%.1f mm\n\n",
        grid.dr * 1e3, b * grid.dphi * 1e3, grid.dz * 1e3);

    /* Material mask: no pipes → all cavity */
    PipeConfig no_pipes;
    pipe_config_init(&no_pipes, a, b, 0.0125, 0.0175,
        pipe_length, 0.005);

    MaterialMask mask;
    material_mask_build(&mask, &no_pipes, &grid);

    /*=========================================================================
     * STAGE 1: PEC REGRESSION
     *========================================================================*/
    printf("================================================================\n");
    printf("  STAGE 1: PEC REGRESSION TEST\n");
    printf("================================================================\n\n");

    {
        GPU_PipeOperator gpu_op;
        gpu_pipe_operator_init(&gpu_op, &cpu_op, &mask);

        double* h_x = (double*)calloc(n, sizeof(double));
        fill_tem_guess(h_x, &cpu_op, &grid, b, L);

        double* d_x;
        gpu_vector_alloc(&d_x, n);
        gpu_vector_to_device(d_x, h_x, n);

        clock_t t0 = clock();
        double k2_pec = run_pec_solver(&gpu_op, &cpu_op, &grid, d_x,
            20, 1000, 1e-8);
        double t_pec = (double)(clock() - t0) / CLOCKS_PER_SEC;

        /* Retrieve eigenvector for Q computation */
        gpu_vector_to_host(h_x, d_x, n);

        QFactorResult qf_pec = compute_q_factor(
            &cpu_op, h_x, k2_pec, SIGMA_CU, b, NULL);

        double f_pec = qf_pec.frequency_Hz;
        double Q_pec = qf_pec.Q_0;
        double err_f = (f_pec - f_exact) / f_exact * 100.0;
        double err_Q = (Q_pec - Q_exact) / Q_exact * 100.0;

        printf("\n  PEC Results:\n");
        printf("    k2     = %.10f\n", k2_pec);
        printf("    f      = %.6f MHz  (err = %+.3f%%)\n", f_pec / 1e6, err_f);
        printf("    Q(surf)= %.1f        (err = %+.3f%%)\n", Q_pec, err_Q);
        printf("    Time   = %.1f s\n\n", t_pec);

        int pec_pass = (fabs(err_f) < 1.0 && fabs(err_Q) < 5.0);
        printf("  STAGE 1: %s (f within 1%%, Q within 5%%)\n\n",
            pec_pass ? "PASS" : "FAIL *** CHECK BACKWARD COMPATIBILITY ***");

        /* Save PEC results for comparison */
        double saved_f_pec = f_pec;
        double saved_Q_pec = Q_pec;
        double saved_k2_pec = k2_pec;

        gpu_vector_free(d_x);
        free(h_x);
        gpu_pipe_operator_free(&gpu_op);

        /*=====================================================================
         * STAGE 2: COMPLEX INFRASTRUCTURE TEST
         *====================================================================*/
        printf("================================================================\n");
        printf("  STAGE 2: COMPLEX INFRASTRUCTURE TEST\n");
        printf("================================================================\n\n");

        {
            /* Allocate complex E-field */
            GPU_EField E_test;
            int rc = gpu_efield_alloc_complex(&E_test, &grid);
            printf("  gpu_efield_alloc_complex: %s\n", rc == 0 ? "OK" : "FAIL");

            /* Allocate complex H-field */
            GPU_HField H_test;
            rc = gpu_hfield_alloc_complex(&H_test, &grid);
            printf("  gpu_hfield_alloc_complex: %s\n", rc == 0 ? "OK" : "FAIL");

            /* Test complex pack/unpack round-trip */
            int n2 = 2 * n;
            double* d_packed;
            gpu_vector_alloc(&d_packed, n2);

            /* Fill real part with TEM guess, imag = 0 */
            double* h_test = (double*)calloc(n2, sizeof(double));
            fill_tem_guess(h_test, &cpu_op, &grid, b, L);
            /* h_test[n..2n-1] stays zero (imaginary part) */

            gpu_vector_to_device(d_packed, h_test, n2);

            /* Unpack complex → E_test */
            gpu_unpack_field_complex(d_packed, &E_test,
                cpu_op.offset_Er, cpu_op.offset_Ephi, cpu_op.offset_Ez, n);

            /* Pack back */
            double* d_packed2;
            gpu_vector_alloc(&d_packed2, n2);
            gpu_pack_field_complex(&E_test, d_packed2,
                cpu_op.offset_Er, cpu_op.offset_Ephi, cpu_op.offset_Ez, n);

            /* Compare on host */
            double* h_roundtrip = (double*)calloc(n2, sizeof(double));
            gpu_vector_to_host(h_roundtrip, d_packed2, n2);

            double max_diff = 0.0;
            for (int i = 0; i < n2; i++) {
                double d = fabs(h_test[i] - h_roundtrip[i]);
                if (d > max_diff) max_diff = d;
            }
            printf("  Complex pack/unpack round-trip max error: %.2e  %s\n",
                max_diff, max_diff < 1e-14 ? "OK" : "FAIL");

            /* Test complex weighted norm (imag=0 → should equal real norm) */
            double norm_real, norm_complex;
            gpu_vec_norm_weighted(d_packed, &norm_real, &cpu_op);
            gpu_vec_norm_weighted_complex(d_packed, &norm_complex, &cpu_op, n);
            double norm_diff = fabs(norm_real - norm_complex) /
                              (norm_real > 0 ? norm_real : 1.0);
            printf("  Complex norm (imag=0) vs real norm: rel diff = %.2e  %s\n",
                norm_diff, norm_diff < 1e-12 ? "OK" : "FAIL");

            /* Test complex curl: with imag=0, should match real-only curl */
            gpu_compute_curl_E_complex(&E_test, &H_test, &grid);

            /* Check that imaginary part of curl result is zero */
            double* h_curl_im = (double*)calloc(H_test.size_Hr, sizeof(double));
            cudaMemcpy(h_curl_im, H_test.Hr_im,
                H_test.size_Hr * sizeof(double), cudaMemcpyDeviceToHost);
            double max_im = 0.0;
            for (int i = 0; i < H_test.size_Hr; i++) {
                if (fabs(h_curl_im[i]) > max_im) max_im = fabs(h_curl_im[i]);
            }
            printf("  Complex curl_E (imag=0 input) → max |Hr_im| = %.2e  %s\n",
                max_im, max_im < 1e-14 ? "OK" : "FAIL");

            int stage2_pass = (max_diff < 1e-14 && norm_diff < 1e-12 && max_im < 1e-14);
            printf("\n  STAGE 2: %s\n\n",
                stage2_pass ? "PASS" : "FAIL *** CHECK COMPLEX INFRASTRUCTURE ***");

            free(h_test);
            free(h_roundtrip);
            free(h_curl_im);
            gpu_vector_free(d_packed);
            gpu_vector_free(d_packed2);
            gpu_efield_free(&E_test);
            gpu_hfield_free(&H_test);
        }

        /*=====================================================================
         * STAGE 3: IBC EIGENSOLVER
         *====================================================================*/
        printf("================================================================\n");
        printf("  STAGE 3: IBC COMPLEX EIGENSOLVER\n");
        printf("================================================================\n\n");

        {
            /* For IBC, use a SIMPLE grid: r from a to b (no pipe extension).
             * This ensures the grid boundaries ARE the physical cavity walls,
             * so IBC at i=0 (r=a) and i=Nr (r=b) is correct. */
            GridParams ibc_grid;
            grid_init(&ibc_grid, a, b, L, Nr_cavity, Nphi, Nz);
            cuda_grid_init(&ibc_grid);

            CurlCurlOperator ibc_cpu_op;
            curlcurl_op_init(&ibc_cpu_op, &ibc_grid);
            int n_ibc = ibc_cpu_op.n_total;

            printf("  IBC grid: Nr=%d, Nphi=%d, Nz=%d, DOFs=%d (%.1fM)\n",
                ibc_grid.Nr, Nphi, Nz, n_ibc, n_ibc / 1e6);
            printf("  r range: [%.4f, %.4f] m (cavity walls only)\n\n",
                ibc_grid.a, ibc_grid.a + ibc_grid.Nr * ibc_grid.dr);

            /* Initialize complex GPU operator */
            GPU_Operator gpu_op_ibc;
            int rc = gpu_operator_init_complex(&gpu_op_ibc, &ibc_cpu_op);
            printf("  gpu_operator_init_complex: %s\n\n", rc == 0 ? "OK" : "FAIL");

            if (rc != 0) {
                printf("  STAGE 3: FAIL (could not initialize complex operator)\n");
            } else {
                /* Initial guess: TEM mode in real part, zero imaginary */
                int n2 = 2 * n_ibc;
                double* h_x_ibc = (double*)calloc(n2, sizeof(double));
                fill_tem_guess(h_x_ibc, &ibc_cpu_op, &ibc_grid, b, L);
                /* Imaginary part h_x_ibc[n_ibc..2*n_ibc-1] = 0 */

                double* d_x_ibc;
                gpu_vector_alloc(&d_x_ibc, n2);
                gpu_vector_to_device(d_x_ibc, h_x_ibc, n2);

                /* Run IBC solver */
                clock_t t0 = clock();
                GPU_ComplexEigenResult ibc_result = gpu_rqi_complex(
                    &gpu_op_ibc, d_x_ibc,
                    saved_k2_pec,       /* Use PEC k2 as initial guess */
                    SIGMA_CU,           /* Copper conductivity */
                    30,                 /* Max RQI iterations */
                    1e-8,               /* Tolerance */
                    50                  /* GMRES restart m */
                );
                double t_ibc = (double)(clock() - t0) / CLOCKS_PER_SEC;

                printf("\n  IBC Results:\n");
                printf("    k2_re  = %.10f\n", ibc_result.k2_re);
                printf("    k2_im  = %.6e\n", ibc_result.k2_im);
                printf("    f(IBC) = %.6f MHz\n", ibc_result.frequency_Hz / 1e6);
                printf("    Q(IBC) = %.1f\n", ibc_result.Q_factor);
                printf("    Converged: %s (residual = %.2e, iters = %d)\n",
                    ibc_result.converged ? "YES" : "NO",
                    ibc_result.residual, ibc_result.iterations);
                printf("    Time   = %.1f s\n\n", t_ibc);

                /* Compare with PEC results */
                double df_pct = (ibc_result.frequency_Hz - saved_f_pec) / saved_f_pec * 100.0;
                double dQ_pct = (ibc_result.Q_factor - saved_Q_pec) / saved_Q_pec * 100.0;

                printf("  Comparison:\n");
                printf("    PEC f = %.6f MHz, IBC f = %.6f MHz (diff = %+.4f%%)\n",
                    saved_f_pec / 1e6, ibc_result.frequency_Hz / 1e6, df_pct);
                printf("    PEC Q(surf integral) = %.1f, IBC Q = %.1f (diff = %+.2f%%)\n",
                    saved_Q_pec, ibc_result.Q_factor, dQ_pct);
                printf("    Analytical Q = %.1f\n\n", Q_exact);

                int f_ok = (fabs(df_pct) < 0.1);
                int Q_ok = (fabs(dQ_pct) < 10.0);

                printf("  STAGE 3: f match: %s (< 0.1%% required, got %.4f%%)\n",
                    f_ok ? "PASS" : "FAIL", fabs(df_pct));
                printf("           Q match: %s (< 10%% required, got %.2f%%)\n",
                    Q_ok ? "PASS" : "FAIL", fabs(dQ_pct));
                printf("           Overall: %s\n\n",
                    (f_ok && Q_ok && ibc_result.converged) ? "PASS" : "FAIL");

                gpu_vector_free(d_x_ibc);
                free(h_x_ibc);
            }

            gpu_operator_free(&gpu_op_ibc);
            curlcurl_op_free(&ibc_cpu_op);
        }
    }

    /* Final cleanup */
    material_mask_free(&mask);
    pipe_config_free(&no_pipes);
    curlcurl_op_free(&cpu_op);

    printf("================================================================\n");
    printf("  IBC TEST COMPLETE\n");
    printf("================================================================\n\n");

    return 0;
}
