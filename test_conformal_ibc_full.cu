/*=============================================================================
 * test_conformal_ibc_full.cpp
 *
 * Full rhodotron model: PEC eigensolver + Conformal IBC iterative.
 * All results (frequency, Q, R/Q) saved to a summary text file.
 * Field data exported to CSV for plotting.
 *
 * Geometry:
 *   - Coaxial HWR: a=0.3333m, b=1.0m, L=1.395m
 *   - 20 radial beam pipes at z=L/2 (10-pass Rhodotron)
 *   - 1 endcap pipe at z=0 (vacuum port)
 *   - 1 endcap pipe at z=L (RF coupler)
 *   - Grid extended in z for endcap pipes
 *   - Interior endplates at k_z0, k_zL
 *
 * Phases:
 *   1. PEC eigensolver → k², f, eigenvector, Q(surface integral), R/Q, fields
 *   2. Conformal IBC iterative (RQI + GMRES) → complex k², f, Q, R/Q, fields
 *
 * Field exports use the REAL PART of the IBC eigenvector.
 * At Q~48000, |E| and Re(E) differ by O(1/Q²) ~ 10^{-10}.
 *
 * Usage:
 *   make test_conformal_ibc_full
 *   ./test_conformal_ibc_full
 *============================================================================*/

#include "cuda_conformal_pipe.h"
#include "cuda_pipe_model.h"
#include "cuda_eigensolver.h"
#include "conformal_geometry.h"
#include "pipe_model.h"
#include "curlcurl_operator.h"
#include "q_factor.h"
#include "r_over_q.h"
#include "field_export.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

#define C0          299792458.0
#define MU0         (4.0e-7 * M_PI)
#define EPS0        (1.0 / (MU0 * C0 * C0))
#define ETA0        (MU0 * C0)
#define SIGMA_CU    5.8e7

 /*=============================================================================
  * PEC Pipe Eigensolver (Rayleigh Quotient Iteration + MINRES)
  *============================================================================*/
extern __global__ void shift_kernel(double*, const double*, double, int);

static double run_pec_pipe_solver(
    GPU_PipeOperator* pipe_op,
    const CurlCurlOperator* cpu_op,
    const GridParams* grid,
    double* d_x,
    int rqi_max, int minres_max, double tol
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
    gpu_vec_dot_weighted_ws(d_x, d_Ax, &xAx, cpu_op, &pipe_op->base.reduction_ws);
    gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op, &pipe_op->base.reduction_ws);
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
        gpu_vec_dot_weighted_ws(d_x, d_x, &b_norm, cpu_op, &pipe_op->base.reduction_ws);
        b_norm = sqrt(b_norm);
        gpu_vec_copy(d_x, ws.minres_ws.d_v_cur, n);
        gpu_vec_scale(ws.minres_ws.d_v_cur, 1.0 / b_norm, n);

        double beta_cur = b_norm, eta = b_norm;
        double c_old = 1.0, c_cur = 1.0, s_old = 0.0, s_cur = 0.0;
        int ls_iters = 0;

        for (int ls = 0; ls < minres_max; ls++) {
            gpu_pipe_matvec(pipe_op, ws.minres_ws.d_v_cur, ws.minres_ws.d_Av);
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
            if (beta_new > 1e-14) gpu_vec_scale(ws.minres_ws.d_v_new, 1.0 / beta_new, n);

            double rho1 = s_old * beta_cur;
            double rho2 = c_old * c_cur * beta_cur + s_cur * alpha;
            double rho3_bar = c_cur * alpha - c_old * s_cur * beta_cur;
            double gamma = sqrt(rho3_bar * rho3_bar + beta_new * beta_new);
            double c_new = 1.0, s_new = 0.0;
            if (gamma > 1e-14) { c_new = rho3_bar / gamma; s_new = beta_new / gamma; }

            gpu_vec_copy(ws.minres_ws.d_v_cur, ws.minres_ws.d_w_new, n);
            gpu_vec_axpy(-rho2, ws.minres_ws.d_w_cur, ws.minres_ws.d_w_new, n);
            gpu_vec_axpy(-rho1, ws.minres_ws.d_w_old, ws.minres_ws.d_w_new, n);
            if (fabs(gamma) > 1e-14) gpu_vec_scale(ws.minres_ws.d_w_new, 1.0 / gamma, n);

            gpu_vec_axpy(c_new * eta, ws.minres_ws.d_w_new, ws.d_y, n);
            eta = -s_new * eta;
            ls_iters = ls + 1;
            if (fabs(eta) / b_norm < 1e-6) break;

            double* temp;
            temp = ws.minres_ws.d_v_old; ws.minres_ws.d_v_old = ws.minres_ws.d_v_cur;
            ws.minres_ws.d_v_cur = ws.minres_ws.d_v_new; ws.minres_ws.d_v_new = temp;
            temp = ws.minres_ws.d_w_old; ws.minres_ws.d_w_old = ws.minres_ws.d_w_cur;
            ws.minres_ws.d_w_cur = ws.minres_ws.d_w_new; ws.minres_ws.d_w_new = temp;
            beta_cur = beta_new; c_old = c_cur; c_cur = c_new; s_old = s_cur; s_cur = s_new;
        }

        gpu_vec_normalize_weighted(ws.d_y, cpu_op);
        gpu_vec_copy(ws.d_y, d_x, n);
        gpu_pipe_matvec(pipe_op, d_x, d_Ax);
        gpu_vec_dot_weighted_ws(d_x, d_Ax, &xAx, cpu_op, &pipe_op->base.reduction_ws);
        gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op, &pipe_op->base.reduction_ws);
        double sigma_new = xAx / xx;
        printf("      iter %d: k2 = %.10f, MINRES iters = %d\n", iter, sigma_new, ls_iters);
        if (fabs(sigma_new - sigma) / fabs(sigma) < tol && iter > 0) {
            printf("    PEC converged.\n"); sigma = sigma_new; break;
        }
        sigma = sigma_new;
    }

    cudaFree(d_Ax);
    eigensolver_workspace_free(&ws);
    return sigma;
}


/*=============================================================================
 * Write summary text file
 *============================================================================*/
static void write_summary_file(
    const char* filename,
    /* Geometry */
    double a, double b, double L, int num_passes,
    double pipe_radius, double aperture_radius, double pipe_length,
    /* Grid */
    const GridParams* grid, int k_z0, int k_zL,
    /* Analytical */
    const QFactorResult* qf_analytical,
    const RoverQResult* roq_analytical,
    /* PEC results */
    double k2_pec, double f_pec, double Q_pec_surf, double t_pec,
    const RoverQResult* roq_pec,
    /* IBC results */
    const GPU_ComplexEigenResult* ibc_result, double t_ibc,
    const RoverQResult* roq_ibc
) {
    FILE* fp = fopen(filename, "w");
    if (!fp) { printf("  ERROR: cannot open %s\n", filename); return; }

    fprintf(fp, "================================================================\n");
    fprintf(fp, "  RHODOTRON FDFD EIGENSOLVER — CONFORMAL IBC RESULTS\n");
    fprintf(fp, "================================================================\n\n");

    /* Timestamp */
    time_t now = time(NULL);
    fprintf(fp, "  Date: %s\n", ctime(&now));

    /* Geometry */
    fprintf(fp, "  GEOMETRY\n");
    fprintf(fp, "  --------\n");
    fprintf(fp, "    Inner radius  a       = %.4f m\n", a);
    fprintf(fp, "    Outer radius  b       = %.4f m\n", b);
    fprintf(fp, "    Cavity length L       = %.4f m\n", L);
    fprintf(fp, "    Beam passes           = %d\n", num_passes);
    fprintf(fp, "    Pipe radius           = %.1f mm\n", pipe_radius * 1e3);
    fprintf(fp, "    Aperture radius       = %.1f mm\n", aperture_radius * 1e3);
    fprintf(fp, "    Pipe length           = %.1f mm\n", pipe_length * 1e3);
    fprintf(fp, "    Wall conductivity     = %.2e S/m (copper)\n", SIGMA_CU);
    fprintf(fp, "\n");

    /* Grid */
    fprintf(fp, "  GRID\n");
    fprintf(fp, "  ----\n");
    fprintf(fp, "    Nr=%d, Nphi=%d, Nz=%d\n", grid->Nr, grid->Nphi, grid->Nz);
    fprintf(fp, "    dr=%.6f m, dphi=%.6f rad (%.3f deg), dz=%.6f m\n",
        grid->dr, grid->dphi, grid->dphi * 180.0 / M_PI, grid->dz);
    fprintf(fp, "    Total DOFs            = %d\n",
        grid->Nr * grid->Nphi * (grid->Nz + 1)
        + (grid->Nr + 1) * grid->Nphi * (grid->Nz + 1)
        + (grid->Nr + 1) * grid->Nphi * grid->Nz);
    fprintf(fp, "    Endplate indices      = k_z0=%d, k_zL=%d\n", k_z0, k_zL);
    fprintf(fp, "\n");

    /* Analytical reference */
    fprintf(fp, "  ANALYTICAL REFERENCE (TEM coaxial HWR, no pipes)\n");
    fprintf(fp, "  ------------------------------------------------\n");
    fprintf(fp, "    k²                    = %.10f 1/m²\n",
        (M_PI / L) * (M_PI / L));
    fprintf(fp, "    Frequency             = %.6f MHz\n",
        qf_analytical->frequency_Hz / 1e6);
    fprintf(fp, "    Q₀ (copper)           = %.0f\n", qf_analytical->Q_0);
    fprintf(fp, "    Geometry factor G     = %.2f Ohm\n", qf_analytical->G_factor);
    fprintf(fp, "    R/Q per crossing      = %.4f Ohm (linac)\n",
        roq_analytical->R_over_Q_crossing_linac);
    fprintf(fp, "    R/Q per crossing      = %.4f Ohm (circuit)\n",
        roq_analytical->R_over_Q_crossing_circuit);
    fprintf(fp, "\n");

    /* PEC results */
    fprintf(fp, "  PEC RESULTS (full model with pipes)\n");
    fprintf(fp, "  -----------------------------------\n");
    fprintf(fp, "    k²                    = %.10f 1/m²\n", k2_pec);
    fprintf(fp, "    Frequency             = %.6f MHz\n", f_pec / 1e6);
    fprintf(fp, "    Q₀ (surface integral) = %.0f\n", Q_pec_surf);
    fprintf(fp, "    Solve time            = %.1f s\n", t_pec);
    fprintf(fp, "    R/Q per crossing      = %.4f Ohm (linac)\n",
        roq_pec->R_over_Q_crossing_linac);
    fprintf(fp, "    R/Q per crossing      = %.4f Ohm (circuit)\n",
        roq_pec->R_over_Q_crossing_circuit);
    fprintf(fp, "    R/Q per pass          = %.4f Ohm (linac)\n",
        roq_pec->R_over_Q_per_pass_linac);
    fprintf(fp, "    R/Q total (%d passes) = %.2f Ohm (linac)\n",
        roq_pec->n_passes, roq_pec->R_over_Q_total_linac);
    fprintf(fp, "    V_gap spread          = %.4f%%\n", roq_pec->V_gap_spread);
    fprintf(fp, "\n");

    /* IBC results */
    fprintf(fp, "  CONFORMAL IBC RESULTS (Dey-Mittra, iterative RQI+GMRES)\n");
    fprintf(fp, "  --------------------------------------------------------\n");
    fprintf(fp, "    Re(k²)                = %.10f 1/m²\n", ibc_result->k2_re);
    fprintf(fp, "    Im(k²)                = %.6e 1/m²\n", ibc_result->k2_im);
    fprintf(fp, "    Frequency             = %.6f MHz\n",
        ibc_result->frequency_Hz / 1e6);
    fprintf(fp, "    Q (from eigenvalue)   = %.0f\n", ibc_result->Q_factor);
    fprintf(fp, "    Converged             = %s\n",
        ibc_result->converged ? "YES" : "NO");
    fprintf(fp, "    Residual              = %.2e\n", ibc_result->residual);
    fprintf(fp, "    RQI iterations        = %d\n", ibc_result->iterations);
    fprintf(fp, "    Solve time            = %.1f s\n", t_ibc);
    fprintf(fp, "    R/Q per crossing      = %.4f Ohm (linac)\n",
        roq_ibc->R_over_Q_crossing_linac);
    fprintf(fp, "    R/Q per crossing      = %.4f Ohm (circuit)\n",
        roq_ibc->R_over_Q_crossing_circuit);
    fprintf(fp, "    R/Q per pass          = %.4f Ohm (linac)\n",
        roq_ibc->R_over_Q_per_pass_linac);
    fprintf(fp, "    R/Q total (%d passes) = %.2f Ohm (linac)\n",
        roq_ibc->n_passes, roq_ibc->R_over_Q_total_linac);
    fprintf(fp, "    V_gap spread          = %.4f%%\n", roq_ibc->V_gap_spread);
    fprintf(fp, "\n");

    /* Comparison */
    fprintf(fp, "  COMPARISON\n");
    fprintf(fp, "  ----------\n");
    fprintf(fp, "    %-30s  %12s  %12s  %12s\n",
        "", "Analytical", "PEC", "IBC Conformal");
    fprintf(fp, "    %-30s  %12s  %12s  %12s\n",
        "", "----------", "---", "-------------");
    fprintf(fp, "    %-30s  %12.4f  %12.4f  %12.4f\n", "Frequency (MHz)",
        qf_analytical->frequency_Hz / 1e6, f_pec / 1e6,
        ibc_result->frequency_Hz / 1e6);
    fprintf(fp, "    %-30s  %12.0f  %12.0f  %12.0f\n", "Q factor",
        qf_analytical->Q_0, Q_pec_surf, ibc_result->Q_factor);
    fprintf(fp, "    %-30s  %12.4f  %12.4f  %12.4f\n",
        "R/Q per crossing (linac, Ohm)",
        roq_analytical->R_over_Q_crossing_linac,
        roq_pec->R_over_Q_crossing_linac,
        roq_ibc->R_over_Q_crossing_linac);
    fprintf(fp, "    %-30s  %12s  %12.2f  %12.2f\n",
        "R/Q total (linac, Ohm)", "—",
        roq_pec->R_over_Q_total_linac,
        roq_ibc->R_over_Q_total_linac);
    fprintf(fp, "\n");

    double df_ibc_pec = (ibc_result->frequency_Hz - f_pec) / f_pec * 100.0;
    double dQ_ibc_pec = (Q_pec_surf > 0) ?
        (ibc_result->Q_factor - Q_pec_surf) / Q_pec_surf * 100.0 : 0.0;
    double dRoQ_pec_ana = (roq_analytical->R_over_Q_crossing_linac > 0) ?
        (roq_pec->R_over_Q_crossing_linac - roq_analytical->R_over_Q_crossing_linac)
        / roq_analytical->R_over_Q_crossing_linac * 100.0 : 0.0;
    double dRoQ_ibc_ana = (roq_analytical->R_over_Q_crossing_linac > 0) ?
        (roq_ibc->R_over_Q_crossing_linac - roq_analytical->R_over_Q_crossing_linac)
        / roq_analytical->R_over_Q_crossing_linac * 100.0 : 0.0;

    fprintf(fp, "    f shift (IBC vs PEC)      = %+.4f%%\n", df_ibc_pec);
    fprintf(fp, "    Q shift (IBC vs PEC surf) = %+.2f%%\n", dQ_ibc_pec);
    fprintf(fp, "    R/Q error (PEC vs ana)    = %+.2f%%\n", dRoQ_pec_ana);
    fprintf(fp, "    R/Q error (IBC vs ana)    = %+.2f%%\n", dRoQ_ibc_ana);
    fprintf(fp, "\n");

    /* Per-pass R/Q breakdown for IBC */
    if (roq_ibc->n_passes > 0 && roq_ibc->n_passes <= 20) {
        fprintf(fp, "  PER-PASS R/Q BREAKDOWN (IBC, linac convention)\n");
        fprintf(fp, "  -----------------------------------------------\n");
        fprintf(fp, "    %5s  %12s  %12s  %12s  %12s  %10s\n",
            "Pass", "phi_entry", "V_entry", "V_exit", "V_pass", "R/Q_pass");
        for (int p = 0; p < roq_ibc->n_passes; p++) {
            CrossingVoltage* entry = &roq_ibc->crossings[2 * p];
            CrossingVoltage* exit = &roq_ibc->crossings[2 * p + 1];
            fprintf(fp, "    %5d  %9.2f deg  %12.6e  %12.6e  %12.6e  %10.4f\n",
                p + 1, entry->phi * 180.0 / M_PI,
                entry->V_gap, exit->V_gap,
                roq_ibc->V_pass[p], roq_ibc->R_over_Q_pass_linac[p]);
        }
        fprintf(fp, "\n");
    }

    /* IBC physics */
    double omega_ibc = C0 * sqrt(fabs(ibc_result->k2_re));
    double Rs = 1.0 / sqrt(2.0 * SIGMA_CU * omega_ibc * MU0) * omega_ibc * MU0;
    /* actually R_s = sqrt(omega*mu0/(2*sigma)) */
    Rs = sqrt(omega_ibc * MU0 / (2.0 * SIGMA_CU));
    double delta = sqrt(2.0 / (SIGMA_CU * omega_ibc * MU0));

    fprintf(fp, "  IBC PHYSICS\n");
    fprintf(fp, "  -----------\n");
    fprintf(fp, "    Surface resistance Rs = %.4f mOhm\n", Rs * 1e3);
    fprintf(fp, "    Skin depth delta      = %.3f um\n", delta * 1e6);
    fprintf(fp, "    alpha = Rs/(w*mu0)    = %.6e\n",
        Rs / (omega_ibc * MU0));
    fprintf(fp, "    |Im(E)/Re(E)|         ~ 1/(2Q) = %.2e\n",
        0.5 / ibc_result->Q_factor);
    fprintf(fp, "\n");

    fprintf(fp, "  EXPORTED FIELD FILES\n");
    fprintf(fp, "  --------------------\n");
    fprintf(fp, "    PEC fields:   pec_*  (10 CSV files)\n");
    fprintf(fp, "    IBC fields:   ibc_*  (10 CSV files)\n");
    fprintf(fp, "    See file headers for dimensions and units.\n");
    fprintf(fp, "\n");

    fprintf(fp, "================================================================\n");

    fclose(fp);
    printf("  Summary written to: %s\n", filename);
}


/*=============================================================================
 * MAIN
 *============================================================================*/
int main(void) {
    printf("\n");
    printf("================================================================\n");
    printf("  RHODOTRON FDFD — PEC + CONFORMAL IBC FULL SOLVE\n");
    printf("================================================================\n\n");

    cuda_print_device_info();
    cuda_print_memory_info("startup");

    /*=========================================================================
     * Geometry
     *========================================================================*/
    double a = 0.3333, b = 1.0, L = 1.395;
    double pipe_radius = 0.0125, aperture_radius = 0.0175;
    double pipe_length = 0.050;
    int num_passes = 10;

    /* Endcap pipe geometry */
    double endcap_z0_r_center = 0.85, endcap_z0_phi = M_PI / 2.0;
    double endcap_z0_aperture_radius = 0.105, endcap_z0_pipe_radius = 0.100;
    double endcap_z0_pipe_length = 0.28;

    double endcap_zL_r_center = 0.85, endcap_zL_phi = 3.0 * M_PI / 2.0;
    double endcap_zL_aperture_radius = 0.095, endcap_zL_pipe_radius = 0.090;
    double endcap_zL_pipe_length = 0.25;

    printf("  Geometry: a=%.4f, b=%.4f, L=%.4f m\n", a, b, L);
    printf("  Radial pipes: %d passes, r_pipe=%.1f mm, l_pipe=%.1f mm\n",
        num_passes, pipe_radius * 1e3, pipe_length * 1e3);
    printf("  Endcap z0: r=%.3f m, R=%.1f mm, L=%.0f mm\n",
        endcap_z0_r_center, endcap_z0_pipe_radius * 1e3,
        endcap_z0_pipe_length * 1e3);
    printf("  Endcap zL: r=%.3f m, R=%.1f mm, L=%.0f mm\n\n",
        endcap_zL_r_center, endcap_zL_pipe_radius * 1e3,
        endcap_zL_pipe_length * 1e3);

    /* Radial pipes */
    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, pipe_radius, aperture_radius, pipe_length, 0.0);
    pipe_config_add_multi_pass(&pipes, L / 2.0, num_passes);

    /* Endcap pipes */
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

    /* Inner conductor port apertures */
    PortConfig inner_ports;
    port_config_init(&inner_ports);
    double dphi_pass = M_PI / num_passes;
    for (int pass = 0; pass < num_passes; pass++) {
        double phi_entry = pass * dphi_pass;
        double phi_exit = phi_entry + M_PI;
        if (phi_exit >= 2.0 * M_PI) phi_exit -= 2.0 * M_PI;
        CavityPort port;
        port.type = PORT_BEAM; port.surface = SURFACE_INNER;
        port.radius = aperture_radius; port.pos2 = L / 2.0;

        port.pos1 = phi_entry; port.name = "Inner Entry";
        if (inner_ports.num_ports >= inner_ports.capacity) {
            inner_ports.capacity *= 2;
            inner_ports.ports = (CavityPort*)realloc(inner_ports.ports,
                inner_ports.capacity * sizeof(CavityPort));
        }
        inner_ports.ports[inner_ports.num_ports++] = port;

        port.pos1 = phi_exit; port.name = "Inner Exit";
        if (inner_ports.num_ports >= inner_ports.capacity) {
            inner_ports.capacity *= 2;
            inner_ports.ports = (CavityPort*)realloc(inner_ports.ports,
                inner_ports.capacity * sizeof(CavityPort));
        }
        inner_ports.ports[inner_ports.num_ports++] = port;
    }

    /*=========================================================================
     * Grid — z-extended for endcap pipes
     *========================================================================*/
    int Nr_cavity = 81, Nr_pipe = 7, Nphi = 256, Nz_cavity = 84;

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

    printf("  Grid: Nr=%d, Nphi=%d, Nz=%d\n", grid.Nr, grid.Nphi, grid.Nz);
    printf("  dr=%.6f m, dphi=%.6f rad, dz=%.6f m\n", grid.dr, grid.dphi, grid.dz);
    printf("  Endplate k-indices: z0 at k=%d, zL at k=%d (of Nz=%d)\n",
        k_endplate_z0, k_endplate_zL, grid.Nz);
    printf("  z0_offset = %.4f m\n", z0_offset);

    /* Build combined material mask */
    printf("\n  Building material mask...\n");
    MaterialMask mask;
    material_mask_build_full(&mask, &pipes, &endcap_pipes, &grid, L, z0_offset);
    material_mask_print_stats(&mask, &grid);

    /* Build conformal data (radial pipes only — endcap pipes use staircase) */
    printf("\n  Building conformal geometry data...\n");
    ConformalData cd;
    conformal_data_build(&cd, &pipes, &grid, b, z0_offset);
    conformal_data_print_stats(&cd, &grid);

    /* CPU operator with inner conductor ports */
    CurlCurlOperator cpu_op;
    curlcurl_op_init_with_ports(&cpu_op, &grid, &inner_ports);
    int n = cpu_op.n_total;
    printf("\n  Total DOFs: %d (%.1f M)\n", n, n / 1e6);
    cuda_grid_init(&grid);

    /* GPU memory check */
    size_t gpu_free, gpu_total;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    printf("  GPU: %.1f GB free of %.1f GB\n\n",
        gpu_free / (1024.0 * 1024.0 * 1024.0), gpu_total / (1024.0 * 1024.0 * 1024.0));

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

    /* Analytical references */
    QFactorResult qf_analytical = compute_q_analytical_coaxial_hwr(a, b, L, SIGMA_CU);
    RoverQResult roq_analytical = compute_r_over_q_analytical(a, b, L);

    /* Field export configuration */
    FieldExportConfig pec_fconfig = {
        b,          /* b_cavity */
        L,          /* L_cavity */
        z0_offset,  /* z0_offset */
        k_endplate_z0,   /* k_cavity_start */
        k_endplate_zL,   /* k_cavity_end */
        "pec"       /* prefix */
    };
    FieldExportConfig ibc_fconfig = {
        b, L, z0_offset,
        k_endplate_z0, k_endplate_zL,
        "ibc"
    };


    /*=========================================================================
     * PHASE 1: PEC Eigensolver
     *========================================================================*/
    printf("================================================================\n");
    printf("  PHASE 1: PEC FULL MODEL (conformal stencil, PEC walls)\n");
    printf("================================================================\n\n");

    double k2_pec = 0.0, f_pec = 0.0, Q_pec_surf = 0.0;
    double t_pec = 0.0;
    RoverQResult roq_pec;
    memset(&roq_pec, 0, sizeof(roq_pec));

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
        k2_pec = run_pec_pipe_solver(&gpu_pec, &cpu_op, &grid, d_x, 20, 3000, 1e-8);
        t_pec = (double)(clock() - t0) / CLOCKS_PER_SEC;

        f_pec = C0 * sqrt(fabs(k2_pec)) / (2.0 * M_PI);

        /* Copy PEC eigenvector to host (used as IBC initial guess later) */
        gpu_vector_to_host(h_x, d_x, n);

        /* Q factor from surface integrals */
        QFactorResult qf_pec = compute_q_factor_extended(
            &cpu_op, h_x, k2_pec, SIGMA_CU, b, NULL,
            k_endplate_z0, k_endplate_zL);
        Q_pec_surf = qf_pec.Q_0;

        printf("\n  PEC Results:\n");
        printf("    k²      = %.10f 1/m²\n", k2_pec);
        printf("    f       = %.6f MHz\n", f_pec / 1e6);
        printf("    Q(surf) = %.0f\n", Q_pec_surf);
        printf("    Time    = %.1f s\n\n", t_pec);

        /* R/Q for PEC */
        roq_pec = compute_r_over_q(
            &cpu_op, h_x, k2_pec, b, L, z0_offset, num_passes);
        r_over_q_print(&roq_pec, "PEC (full model)");

        /* Field export for PEC */
        printf("\n  Exporting PEC field data...\n");
        export_all_field_data(&cpu_op, h_x, &pec_fconfig);

        gpu_vector_free(d_x);
        gpu_pipe_operator_free(&gpu_pec);
    }


    /*=========================================================================
     * PHASE 2: Conformal IBC Full Iterative (RQI + GMRES)
     *========================================================================*/
    printf("\n================================================================\n");
    printf("  PHASE 2: CONFORMAL IBC ITERATIVE (RQI + GMRES)\n");
    printf("================================================================\n\n");

    GPU_ComplexEigenResult ibc_result;
    memset(&ibc_result, 0, sizeof(ibc_result));
    double t_ibc = 0.0;
    RoverQResult roq_ibc;
    memset(&roq_ibc, 0, sizeof(roq_ibc));

    {
        GPU_ConformalPipeOperator gpu_cfm;
        gpu_conformal_pipe_operator_init(&gpu_cfm, &cpu_op, &mask, &cd, &pipes,
            k_endplate_z0, k_endplate_zL, &endcap_pipes, z0_offset);

        /* Complex initial guess: [PEC eigenvector | zeros] */
        int n2 = 2 * n;
        double* h_x_cx = (double*)calloc(n2, sizeof(double));
        memcpy(h_x_cx, h_x, n * sizeof(double));

        double* d_x_cx;
        gpu_vector_alloc(&d_x_cx, n2);
        gpu_vector_to_device(d_x_cx, h_x_cx, n2);

        clock_t t0 = clock();
        ibc_result = gpu_rqi_complex_conformal_pipe(
            &gpu_cfm, d_x_cx, k2_pec, SIGMA_CU, 30, 1e-6, 50);
        t_ibc = (double)(clock() - t0) / CLOCKS_PER_SEC;

        printf("\n  Conformal IBC Results:\n");
        printf("    Re(k²)  = %.10f 1/m²\n", ibc_result.k2_re);
        printf("    Im(k²)  = %.6e 1/m²\n", ibc_result.k2_im);
        printf("    f       = %.6f MHz\n", ibc_result.frequency_Hz / 1e6);
        printf("    Q       = %.0f (from eigenvalue: -Re(k²)/Im(k²))\n",
            ibc_result.Q_factor);
        printf("    Converged: %s (residual=%.2e, iters=%d)\n",
            ibc_result.converged ? "YES" : "NO",
            ibc_result.residual, ibc_result.iterations);
        printf("    Time    = %.1f s\n\n", t_ibc);

        /* Copy complex eigenvector back to host */
        gpu_vector_to_host(h_x_cx, d_x_cx, n2);

        /*-----------------------------------------------------------------
         * Post-processing uses the REAL PART of the IBC eigenvector.
         *
         * Justification: Im(E)/Re(E) ~ 1/(2Q) ~ 10^{-5}.
         * Error in |V|^2 from ignoring Im(E): O(1/Q^2) ~ 10^{-10}.
         * This is far below grid discretization error O(dr^2).
         *-----------------------------------------------------------------*/
        double* h_x_ibc_re = h_x_cx;  /* first n doubles = real part */

        /* R/Q for IBC (using real part of eigenvector) */
        roq_ibc = compute_r_over_q(
            &cpu_op, h_x_ibc_re, ibc_result.k2_re,
            b, L, z0_offset, num_passes);
        r_over_q_print(&roq_ibc, "Conformal IBC (iterative)");

        /* Field export for IBC (using real part of eigenvector) */
        printf("\n  Exporting IBC field data...\n");
        export_all_field_data(&cpu_op, h_x_ibc_re, &ibc_fconfig);

        gpu_vector_free(d_x_cx);
        free(h_x_cx);
        gpu_conformal_pipe_operator_free(&gpu_cfm);
    }


    /*=========================================================================
     * COMPARISON (console)
     *========================================================================*/
    printf("\n================================================================\n");
    printf("  COMPARISON\n");
    printf("================================================================\n\n");

    printf("    %-30s  %12s  %12s  %12s\n",
        "", "Analytical", "PEC", "IBC Conformal");
    printf("    %-30s  %12s  %12s  %12s\n",
        "", "----------", "---", "-------------");
    printf("    %-30s  %12.4f  %12.4f  %12.4f\n", "Frequency (MHz)",
        qf_analytical.frequency_Hz / 1e6, f_pec / 1e6,
        ibc_result.frequency_Hz / 1e6);
    printf("    %-30s  %12.0f  %12.0f  %12.0f\n", "Q factor",
        qf_analytical.Q_0, Q_pec_surf, ibc_result.Q_factor);
    printf("    %-30s  %12.4f  %12.4f  %12.4f\n", "R/Q crossing (linac, Ohm)",
        roq_analytical.R_over_Q_crossing_linac,
        roq_pec.R_over_Q_crossing_linac,
        roq_ibc.R_over_Q_crossing_linac);
    printf("    %-30s  %12s  %12.2f  %12.2f\n", "R/Q total (linac, Ohm)",
        "—",
        roq_pec.R_over_Q_total_linac,
        roq_ibc.R_over_Q_total_linac);

    double df = (ibc_result.frequency_Hz - f_pec) / f_pec * 100.0;
    double dQ = (Q_pec_surf > 0) ?
        (ibc_result.Q_factor - Q_pec_surf) / Q_pec_surf * 100.0 : 0.0;
    printf("\n    f shift (IBC vs PEC):  %+.4f%%\n", df);
    printf("    Q shift (IBC vs PEC):  %+.2f%%\n", dQ);


    /*=========================================================================
     * Save all results to text file
     *========================================================================*/
    printf("\n  Writing summary file...\n");
    write_summary_file("results_summary.txt",
        a, b, L, num_passes, pipe_radius, aperture_radius, pipe_length,
        &grid, k_endplate_z0, k_endplate_zL,
        &qf_analytical, &roq_analytical,
        k2_pec, f_pec, Q_pec_surf, t_pec, &roq_pec,
        &ibc_result, t_ibc, &roq_ibc);


    /*=========================================================================
     * Cleanup
     *========================================================================*/
    r_over_q_free(&roq_pec);
    r_over_q_free(&roq_ibc);
    free(h_x);
    port_config_free(&inner_ports);
    endcap_pipe_config_free(&endcap_pipes);
    pipe_config_free(&pipes);
    material_mask_free(&mask);
    conformal_data_free(&cd);
    curlcurl_op_free(&cpu_op);

    printf("\n================================================================\n");
    printf("  DONE — Results in results_summary.txt\n");
    printf("  PEC fields:  pec_Er_rz.csv, pec_Er_rphi.csv, ...\n");
    printf("  IBC fields:  ibc_Er_rz.csv, ibc_Er_rphi.csv, ...\n");
    printf("================================================================\n\n");

    return 0;
}