/*============================================================================
 * bench_gpu_conformal_xcheck.cu
 *
 * GPU cross-check for the Stage 2 CPU conformal-pipe path. Mirrors
 * test_conformal_cpu.cpp EXACTLY, now with the Z-EXTENDED grid and INTERIOR
 * endplates (the physically-correct configuration that restores endplate
 * loss -> Q ~ 48k and keeps the endplate re-zero in-bounds).
 *
 * Any k2 / matvec difference vs the CPU run is then pure reduction order.
 *
 * Build: see Makefile.conformal_xcheck
 * Run:   ./bench_gpu_conformal_xcheck 24
 *==========================================================================*/
#include "curl_E.h"
#include "curlcurl_operator.h"
#include "pipe_model.h"
#include "conformal_geometry.h"
#include "cuda_conformal_pipe.h"
#include "cuda_pipe_model.h"
#include "cuda_conformal.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <chrono>

static double g_matvec_us = 0.0;
static double now_seconds_gpu() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
        steady_clock::now().time_since_epoch()).count();
}

#define CUDA_CHECK(call) do {                                            \
    cudaError_t _e = (call);                                             \
    if (_e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(_e));                                 \
        exit(1);                                                         \
    }                                                                    \
} while (0)

int main(int argc, char** argv) {
    double a = 0.3333, b = 1.0, L = 1.395;
    int Nz_cavity = (argc > 1) ? atoi(argv[1]) : 32;
    double sigma_cu = 5.8e7;

    double pipe_radius = 0.0125, aperture_radius = 0.0175, pipe_length = 0.050;
    int num_passes = 10;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        printf("  GPU: %s\n", prop.name);

    /* Radial beam pipes — production geometry (matches test_conformal_cpu) */
    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, pipe_radius, aperture_radius, pipe_length, 0.0);
    pipe_config_add_multi_pass(&pipes, L / 2.0, num_passes);

    /* Endcap pipes — PRODUCTION geometry (matches test_conformal_cpu) */
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

    double dz_cav = L / Nz_cavity;
    int Nr_cavity = Nz_cavity;
    int Nr_pipe   = (Nr_cavity >= 16) ? Nr_cavity / 4 : 4;
    int Nphi      = Nz_cavity;
    int Nz_pipe_z0 = (int)ceil(endcap.z0_extension / dz_cav);
    int Nz_pipe_zL = (int)ceil(endcap.zL_extension / dz_cav);

    GridParams grid;
    grid_init_with_all_pipes(&grid, a, b, L,
        pipe_length, endcap.z0_extension, endcap.zL_extension,
        Nr_cavity, Nr_pipe, Nphi, Nz_cavity, Nz_pipe_z0, Nz_pipe_zL);
    cuda_grid_init(&grid);

    double z0_offset = endcap.z0_extension;
    int k_z0 = Nz_pipe_z0;
    int k_zL = Nz_pipe_z0 + Nz_cavity;

    /* Inner-conductor beam-port apertures (Phase B1), identical to
     * test_conformal_cpu. The GPU operator's build_port_masks will derive the
     * same inner masks from these. Port z in GRID coordinates. */
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

    CurlCurlOperator cpu_op;
    curlcurl_op_init_with_ports(&cpu_op, &grid, &inner_ports);
    int n = cpu_op.n_total;

    printf("=== Stage 2 conformal GPU cross-check (z-extended) ===\n");
    printf("Grid %dx%dx%d  DOFs=%d  a=%.3f b=%.3f L=%.3f\n",
           grid.Nr, grid.Nphi, grid.Nz, n, a, b, L);
    printf("Endplates: k_z0=%d, k_zL=%d (of Nz=%d)  z0_offset=%.4f m\n",
           k_z0, k_zL, grid.Nz, z0_offset);

    /* Full material mask + conformal data (uses z0_offset) */
    MaterialMask pec_mask;
    material_mask_build_full(&pec_mask, &pipes, &endcap, &grid, L, z0_offset);

    ConformalData cd;
    conformal_data_build(&cd, &pipes, &grid, b, z0_offset);
    /* Note: the GPU operator builds the IBC mask + weights internally from the
     * PEC mask, then unmasks the conformal data the same way. We pass the PEC
     * mask; the GPU init handles the rest (mirrors gpu_pipe_operator_init_complex). */
    {
        MaterialMask ibc_mask;
        material_mask_build_ibc(&ibc_mask, &pec_mask, &grid);
        conformal_data_apply_ibc_unmask(&cd, &pec_mask, &ibc_mask, &grid);
        material_mask_free(&ibc_mask);
    }

    /* GPU conformal-pipe operator: INTERIOR endplates + real endcap apertures. */
    GPU_ConformalPipeOperator gpu_cop;
    if (gpu_conformal_pipe_operator_init(&gpu_cop, &cpu_op, &pec_mask, &cd,
                                         k_z0, k_zL, &endcap,
                                         &grid, z0_offset) != 0) {
        printf("GPU operator init FAILED\n"); return 1;
    }

    /* TEM seed in the cavity z-range: Er = sin(pi (z-z0)/L)/r */
    std::vector<double> h_x(2 * n, 0.0);
    for (int k = 0; k <= grid.Nz; k++) {
        double z_phys = k * grid.dz - z0_offset;
        if (z_phys < 0.0 || z_phys > L) continue;
        for (int j = 0; j < grid.Nphi; j++)
            for (int i = 0; i < grid.Nr; i++) {
                double r = grid.a + (i + 0.5) * grid.dr;
                if (r <= b) {
                    int idx = cpu_op.offset_Er + idx_Er(&grid, i, j, k);
                    h_x[idx] = sin(M_PI * z_phys / L) / r;
                }
            }
    }

    double* d_x = NULL;
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)2 * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)2 * n * sizeof(double),
                          cudaMemcpyHostToDevice));

    double sigma0 = (M_PI / L) * (M_PI / L);

    /* Single matvec checksum (operator-level comparison) */
    {
        double alpha0 = 1.0 / sqrt(2.0 * sigma_cu * (299792458.0 * sqrt(sigma0))
                                   * (4.0e-7 * M_PI));
        double* d_y = NULL;
        CUDA_CHECK(cudaMalloc(&d_y, (size_t)2 * n * sizeof(double)));
        gpu_conformal_pipe_matvec_complex(&gpu_cop, d_x, d_y, alpha0);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> h_y(2 * n);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, (size_t)2 * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
        double ynorm = 0.0, xdotAx = 0.0;
        for (int i = 0; i < 2 * n; i++) ynorm  += h_y[i] * h_y[i];
        for (int i = 0; i < 2 * n; i++) xdotAx += h_x[i] * h_y[i];
        ynorm = sqrt(ynorm);
        printf("MATVEC_CHECK: ||Ax|| = %.13e   <x,Ax> = %.13e   alpha = %.13e\n",
               ynorm, xdotAx, alpha0);

        /* -- matvec throughput timing via CUDA events (PERFORMANCE) -- */
        int matvec_iters = 50;
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int it = 0; it < matvec_iters; it++)
            gpu_conformal_pipe_matvec_complex(&gpu_cop, d_x, d_y, alpha0);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        g_matvec_us = (double)ms * 1.0e3 / matvec_iters;
        printf("TIMING: matvec = %.3f us/call (%d iters, GPU)\n",
               g_matvec_us, matvec_iters);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaFree(d_y);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double te0 = now_seconds_gpu();
    GPU_ComplexEigenResult r = gpu_rqi_complex_conformal_pipe(
        &gpu_cop, d_x, sigma0, sigma_cu, /*max_iter*/12, /*tol*/1e-6,
        /*gmres_restart*/30);
    CUDA_CHECK(cudaDeviceSynchronize());
    double eig_s = now_seconds_gpu() - te0;

    printf("\nRESULT(GPU): f = %.4f MHz, Q = %.1f, k2_re = %.10f, k2_im = %.6e, "
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
            fprintf(f, "conformal_gpu,%d,%d,%d,%d,gpu,%.6e,%.6e,%d,"
                       "%.13f,%.6e,%.6f,%.1f,%d\n",
                    grid.Nr, grid.Nphi, grid.Nz, n,
                    g_matvec_us, eig_s, r.iterations,
                    r.k2_re, r.k2_im, r.frequency_Hz / 1e6, r.Q_factor,
                    r.converged);
            fclose(f);
        }
    }

    cudaFree(d_x);
    gpu_conformal_pipe_operator_free(&gpu_cop);
    conformal_data_free(&cd);
    material_mask_free(&pec_mask);
    curlcurl_op_free(&cpu_op);
    endcap_pipe_config_free(&endcap);
    port_config_free(&inner_ports);
    pipe_config_free(&pipes);
    return 0;
}
