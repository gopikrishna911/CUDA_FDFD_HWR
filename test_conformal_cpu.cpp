/* test_conformal_cpu.cpp — Stage 2 CPU conformal-pipe test.
 *
 * Builds the coaxial HWR with radial beam pipes on a Z-EXTENDED grid so the
 * end walls are INTERIOR endplates (not the grid edge). This is the
 * physically-correct configuration:
 *   - endplate grid-plane IBC is applied at interior planes k_z0, k_zL,
 *     restoring the ~44% of wall loss that the endplates carry (Q -> ~48k),
 *   - the endplate conductor-side re-zero runs in-bounds (k_z0-1 >= 0,
 *     k_zL+1 <= Nz), so the GPU OOB bug is never triggered.
 *
 * A minimal empty endcap config with small z-extensions makes the regions
 * beyond the endplates fully PEC (solid endplates, no apertures).
 *
 * Usage: ./test_conformal_cpu [Nz_cavity]   (default 32)
 */
#include "cpu_conformal.h"
#include "curl_E.h"
#include "curlcurl_operator.h"
#include "pipe_model.h"
#include "conformal_geometry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <chrono>

static double now_seconds() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
        steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    double a = 0.3333, b = 1.0, L = 1.395;
    int Nz_cavity = (argc > 1) ? atoi(argv[1]) : 32;
    double sigma_cu = 5.8e7;

    /* Radial beam pipes — production geometry (12.5 mm pipes, 10 passes). */
    double pipe_radius = 0.0125, aperture_radius = 0.0175, pipe_length = 0.050;
    int num_passes = 10;
    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, pipe_radius, aperture_radius, pipe_length, 0.0);
    pipe_config_add_multi_pass(&pipes, L / 2.0, num_passes);

    /* Endcap pipes — PRODUCTION geometry (matches test_conformal_ibc_full):
     * z0 endcap at r=0.85, phi=pi/2, R=100mm, L=280mm;
     * zL endcap at r=0.85, phi=3pi/2, R=90mm, L=250mm. The grid z-extension
     * is the endcap pipe lengths, so the endplates sit at interior planes. */
    double ec_z0_r = 0.85, ec_z0_phi = M_PI / 2.0;
    double ec_z0_ap = 0.105, ec_z0_R = 0.100, ec_z0_L = 0.28;
    double ec_zL_r = 0.85, ec_zL_phi = 3.0 * M_PI / 2.0;
    double ec_zL_ap = 0.095, ec_zL_R = 0.090, ec_zL_L = 0.25;
    EndcapPipeConfig endcap;
    endcap_pipe_config_init(&endcap);
    endcap_pipe_config_add(&endcap, ec_z0_r, ec_z0_phi, ec_z0_ap, ec_z0_R, ec_z0_L, 1);
    endcap_pipe_config_add(&endcap, ec_zL_r, ec_zL_phi, ec_zL_ap, ec_zL_R, ec_zL_L, 0);
    endcap.z0_extension = ec_z0_L;
    endcap.zL_extension = ec_zL_L;

    /* Z-extended grid (extension = endcap pipe lengths) */
    double dz_cav = L / Nz_cavity;
    int Nr_cavity = Nz_cavity;          /* keep roughly cubic in r,z */
    int Nr_pipe   = (Nr_cavity >= 16) ? Nr_cavity / 4 : 4;
    int Nphi      = Nz_cavity;
    int Nz_pipe_z0 = (int)ceil(endcap.z0_extension / dz_cav);
    int Nz_pipe_zL = (int)ceil(endcap.zL_extension / dz_cav);

    GridParams grid;
    grid_init_with_all_pipes(&grid, a, b, L,
        pipe_length, endcap.z0_extension, endcap.zL_extension,
        Nr_cavity, Nr_pipe, Nphi, Nz_cavity, Nz_pipe_z0, Nz_pipe_zL);

    double z0_offset = endcap.z0_extension;
    int k_z0 = Nz_pipe_z0;                 /* interior endplate plane (>=1)   */
    int k_zL = Nz_pipe_z0 + Nz_cavity;     /* interior endplate plane (<Nz)   */

    /* Inner-conductor beam-port apertures (Phase B1): entry/exit holes in the
     * inner conductor for each pass, matching test_conformal_ibc_full. The
     * port z-position is in GRID coordinates (cavity center + z0_offset),
     * because build_port_masks evaluates z = k*dz directly. */
    PortConfig inner_ports;
    port_config_init(&inner_ports);
    {
        double dphi_pass = M_PI / num_passes;
        double z_grid = L / 2.0 + z0_offset;
        for (int pass = 0; pass < num_passes; pass++) {
            double phi_entry = pass * dphi_pass;
            double phi_exit  = phi_entry + M_PI;
            if (phi_exit >= 2.0 * M_PI) phi_exit -= 2.0 * M_PI;
            CavityPort port;
            port.type = PORT_BEAM; port.surface = SURFACE_INNER;
            port.radius = aperture_radius; port.pos2 = z_grid;
            port.pos1 = phi_entry; port.name = "Inner Entry";
            if (inner_ports.num_ports >= inner_ports.capacity) {
                inner_ports.capacity = inner_ports.capacity ? inner_ports.capacity*2 : 4;
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
    }

    CurlCurlOperator op;
    curlcurl_op_init_with_ports(&op, &grid, &inner_ports);

    printf("=== Stage 2 conformal CPU test (z-extended, interior endplates) ===\n");
    printf("Grid %dx%dx%d  DOFs=%d  a=%.3f b=%.3f L=%.3f\n",
           grid.Nr, grid.Nphi, grid.Nz, op.n_total, a, b, L);
    printf("Endplates: k_z0=%d, k_zL=%d (of Nz=%d)  z0_offset=%.4f m\n",
           k_z0, k_zL, grid.Nz, z0_offset);

    /* Full material mask (radial pipes in cavity z-range; solid PEC beyond
     * the endplates since the endcap config has no apertures). */
    MaterialMask pec_mask;
    material_mask_build_full(&pec_mask, &pipes, &endcap, &grid, L, z0_offset);

    MaterialMask ibc_mask;
    material_mask_build_ibc(&ibc_mask, &pec_mask, &grid);

    /* Conformal Dey-Mittra data for the radial pipes (uses z0_offset) */
    ConformalData cd;
    conformal_data_build(&cd, &pipes, &grid, b, z0_offset);
    conformal_data_apply_ibc_unmask(&cd, &pec_mask, &ibc_mask, &grid);

    /* Operator with INTERIOR endplates + real endcap apertures (B2). The
     * endcap config supplies the endplate hole masks; IBC is skipped where
     * the beam pipe passes through the endplate. */
    CpuConformalPipeOperator cop;
    if (cpu_conformal_pipe_operator_init(&cop, &op, &cd, &pec_mask, &ibc_mask,
                                         k_z0, k_zL, &endcap, &grid) != 0) {
        printf("operator init FAILED\n"); return 1;
    }

    /* TEM seed in the cavity z-range: Er = sin(pi (z-z0)/L)/r */
    int n_real = op.n_total;
    std::vector<double> x(2 * n_real, 0.0);
    for (int k = 0; k <= grid.Nz; k++) {
        double z_phys = k * grid.dz - z0_offset;
        if (z_phys < 0.0 || z_phys > L) continue;
        for (int j = 0; j < grid.Nphi; j++)
            for (int i = 0; i < grid.Nr; i++) {
                double r = grid.a + (i + 0.5) * grid.dr;
                if (r <= b) {
                    int idx = op.offset_Er + idx_Er(&grid, i, j, k);
                    x[idx] = sin(M_PI * z_phys / L) / r;
                }
            }
    }

    double sigma0 = (M_PI / L) * (M_PI / L);

    /* Single matvec checksum (operator-level comparison vs GPU). */
    std::vector<double> y(2 * n_real, 0.0);
    double alpha0 = 1.0 / sqrt(2.0 * sigma_cu * (299792458.0 * sqrt(sigma0))
                               * (4.0e-7 * M_PI));
    cpu_conformal_pipe_matvec_complex_omp(&cop, x.data(), y.data(), alpha0);
    double ynorm = 0.0, xdotAx = 0.0;
    for (int i = 0; i < 2 * n_real; i++) ynorm  += y[i] * y[i];
    for (int i = 0; i < 2 * n_real; i++) xdotAx += x[i] * y[i];
    ynorm = sqrt(ynorm);
    printf("MATVEC_CHECK: ||Ax|| = %.13e   <x,Ax> = %.13e   alpha = %.13e\n",
           ynorm, xdotAx, alpha0);
    if (!(ynorm > 0.0 && ynorm < 1e30)) { printf("MATVEC FAILED\n"); return 1; }

    /* -- matvec throughput timing (PERFORMANCE BENCHMARK) -- */
    int matvec_iters = 50;
    cpu_conformal_pipe_matvec_complex_omp(&cop, x.data(), y.data(), alpha0); /* warm */
    double tm0 = now_seconds();
    for (int it = 0; it < matvec_iters; it++)
        cpu_conformal_pipe_matvec_complex_omp(&cop, x.data(), y.data(), alpha0);
    double tm1 = now_seconds();
    double matvec_us = (tm1 - tm0) * 1.0e6 / matvec_iters;
    printf("TIMING: matvec = %.3f us/call (%d iters, %d threads)\n",
           matvec_us, matvec_iters, cpu_omp_num_threads());

    /* -- eigensolve (re-seed since the matvec loop left x untouched; the TEM
     *    seed in x is intact, RQI normalizes internally) -- */
    double te0 = now_seconds();
    CpuComplexEigenResult r = cpu_rqi_complex_conformal_pipe_omp(
        &cop, x.data(), sigma0, sigma_cu, /*max_iter*/12, /*tol*/1e-6,
        /*gmres_restart*/30);
    double te1 = now_seconds();
    double eig_s = te1 - te0;

    printf("\nRESULT: f = %.4f MHz, Q = %.1f, k2_re = %.10f, k2_im = %.6e, "
           "conv=%d, iters=%d\n",
           r.frequency_Hz / 1e6, r.Q_factor, r.k2_re, r.k2_im,
           r.converged, r.iterations);
    printf("TIMING: eigensolve = %.4f s\n", eig_s);

    /* Optional CSV append: --csv=path */
    const char* csv_path = NULL;
    for (int ai = 1; ai < argc; ai++)
        if (!strncmp(argv[ai], "--csv=", 6)) csv_path = argv[ai] + 6;
    if (csv_path) {
        FILE* f = fopen(csv_path, "a");
        if (f) {
            fprintf(f, "conformal,%d,%d,%d,%d,%d,%.6e,%.6e,%d,"
                       "%.13f,%.6e,%.6f,%.1f,%d\n",
                    grid.Nr, grid.Nphi, grid.Nz, op.n_total,
                    cpu_omp_num_threads(), matvec_us, eig_s, r.iterations,
                    r.k2_re, r.k2_im, r.frequency_Hz / 1e6, r.Q_factor,
                    r.converged);
            fclose(f);
        }
    }

    int ok = (r.frequency_Hz / 1e6 > 95.0 && r.frequency_Hz / 1e6 < 120.0
              && r.Q_factor > 0.0);
    printf("%s\n", ok ? "PASS (frequency in physical range, Q positive)"
                      : "CHECK (frequency outside expected band)");

    cpu_conformal_pipe_operator_free(&cop);
    conformal_data_free(&cd);
    material_mask_free(&ibc_mask);
    material_mask_free(&pec_mask);
    curlcurl_op_free(&op);
    endcap_pipe_config_free(&endcap);
    port_config_free(&inner_ports);
    pipe_config_free(&pipes);
    return ok ? 0 : 2;
}
