/*=============================================================================
 * test_rhodotron_gpu.cpp
 *
 * GPU-accelerated Rhodotron cavity simulation
 * Now with grids fine enough to resolve beam apertures!
 *============================================================================*/

#include "cuda_eigensolver.h"
#include "cuda_operator.h"
#include "curlcurl_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

 /*=============================================================================
  * Grid resolution calculator
  * Returns minimum grid sizes to resolve a given aperture
  *============================================================================*/
static void compute_grid_for_aperture(
    double a, double b, double L,
    double aperture_radius,
    int cells_per_aperture,
    int* Nr, int* Nphi, int* Nz
) {
    double target_size = (2.0 * aperture_radius) / cells_per_aperture;

    *Nr = (int)ceil((b - a) / target_size);
    *Nz = (int)ceil(L / target_size);

    /* phi resolution: need target_size arc at inner conductor */
    double dphi_target = target_size / a;
    *Nphi = (int)ceil(2.0 * M_PI / dphi_target);

    /* Round Nphi up to multiple of 32 (for GPU warp alignment) */
    *Nphi = ((*Nphi + 31) / 32) * 32;

    /* Ensure minimum sizes */
    if (*Nr < 16) *Nr = 16;
    if (*Nphi < 64) *Nphi = 64;
    if (*Nz < 16) *Nz = 16;
}

/*=============================================================================
 * Print memory estimate
 *============================================================================*/
static void print_memory_estimate(int Nr, int Nphi, int Nz) {
    long size_Er = (long)Nr * Nphi * (Nz + 1);
    long size_Ephi = (long)(Nr + 1) * Nphi * (Nz + 1);
    long size_Ez = (long)(Nr + 1) * Nphi * Nz;
    long n_total = size_Er + size_Ephi + size_Ez;

    /* GPU needs: E_work, result_work, H_temp, plus MINRES vectors (7) + Ax, y */
    double field_MB = (size_Er + size_Ephi + size_Ez) * 8.0 / 1e6;
    double hfield_MB = field_MB;  /* roughly same */
    double vector_MB = n_total * 8.0 / 1e6;

    /* Total: 2 EFields + 1 HField + ~10 packed vectors */
    double total_MB = 2 * field_MB + hfield_MB + 10 * vector_MB;

    printf("  Memory estimate:\n");
    printf("    E-field:  %.1f MB\n", field_MB);
    printf("    H-field:  %.1f MB\n", hfield_MB);
    printf("    Vectors:  %.1f MB (×10)\n", vector_MB);
    printf("    Total GPU: %.1f MB\n", total_MB);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("    GPU free:  %.1f MB\n", free_mem / 1e6);

    if (total_MB > free_mem / 1e6 * 0.9) {
        printf("    ⚠ WARNING: May exceed GPU memory!\n");
    }
    else {
        printf("    ✓ Fits in GPU memory\n");
    }
}

/*=============================================================================
 * Run a single simulation
 *============================================================================*/
static int run_simulation(
    double a, double b, double L,
    double aperture_radius,
    int Nr, int Nphi, int Nz,
    int num_passes,
    int rqi_max_iter,
    double rqi_tol,
    const char* vtk_filename
) {
    printf("\n");
    printf("================================================================\n");
    printf("  Aperture: %.1f mm, Grid: %d × %d × %d\n",
        aperture_radius * 1000, Nr, Nphi, Nz);
    printf("================================================================\n");

    /* Grid setup */
    GridParams grid;
    grid_init(&grid, a, b, L, Nr, Nphi, Nz);
    cuda_grid_init(&grid);

    printf("  dr = %.2f mm, a*dphi = %.2f mm, dz = %.2f mm\n",
        grid.dr * 1000, a * grid.dphi * 1000, grid.dz * 1000);
    printf("  Aperture diameter / dr = %.1f cells\n",
        2.0 * aperture_radius / grid.dr);
    printf("  Aperture diameter / a*dphi = %.1f cells\n",
        2.0 * aperture_radius / (a * grid.dphi));
    printf("  Aperture diameter / dz = %.1f cells\n",
        2.0 * aperture_radius / grid.dz);

    /* Reference: no ports */
    CurlCurlOperator ref_op;
    curlcurl_op_init(&ref_op, &grid);

    /* With ports */
    PortConfig ports;
    port_config_init(&ports);

    double z_beam = L / 2.0;
    if (num_passes == 1) {
        port_config_add_beam_apertures_single_pass(&ports, z_beam, aperture_radius);
    }
    else {
        port_config_add_beam_apertures_multi_pass(&ports, z_beam, aperture_radius, num_passes);
    }

    CurlCurlOperator port_op;
    curlcurl_op_init_with_ports(&port_op, &grid, &ports);

    int n = port_op.n_total;
    printf("  Total DOFs: %d (%.1f M)\n", n, n / 1.0e6);

    print_memory_estimate(Nr, Nphi, Nz);

    /* Initialize GPU operators */
    GPU_Operator gpu_ref, gpu_port;
    printf("\n  Initializing GPU operators...\n");
    gpu_operator_init(&gpu_ref, &ref_op);
    gpu_operator_init(&gpu_port, &port_op);
    cuda_print_memory_info("after GPU init");

    /* Allocate eigensolver workspace */
    EigensolverWorkspace eigen_ws;
    eigensolver_workspace_init(&eigen_ws, n);

    /* Create TEM initial guess */
    double* h_x = (double*)malloc(n * sizeof(double));
    vec_zero(h_x, n);

    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        double sin_piz_L = sin(M_PI * z / L);
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                int idx = ref_op.offset_Er + idx_Er(&grid, i, j, k);
                h_x[idx] = sin_piz_L / r;
            }
        }
    }

    double* d_x_ref, * d_x_port;
    gpu_vector_alloc(&d_x_ref, n);
    gpu_vector_alloc(&d_x_port, n);
    gpu_vector_to_device(d_x_ref, h_x, n);
    gpu_vector_to_device(d_x_port, h_x, n);

    /* Add small perturbation to port version */
    double* h_pert = (double*)malloc(n * sizeof(double));
    srand(12345);
    for (int i = 0; i < n; i++) {
        h_pert[i] = h_x[i] * (1.0 + 0.01 * ((double)rand() / RAND_MAX - 0.5));
    }
    gpu_vector_to_device(d_x_port, h_pert, n);

    double k2_target = (M_PI / L) * (M_PI / L);

    /* Reference eigensolve */
    printf("\n  --- Reference (ideal cavity) ---\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    GPU_EigenResult ref_result = gpu_rqi_ws(
        &gpu_ref, d_x_ref, k2_target, rqi_max_iter, rqi_tol, &eigen_ws);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ref_ms;
    cudaEventElapsedTime(&ref_ms, start, stop);

    /* Port eigensolve */
    printf("\n  --- With ports ---\n");

    cudaEventRecord(start);
    GPU_EigenResult port_result = gpu_rqi_ws(
        &gpu_port, d_x_port, k2_target, rqi_max_iter, rqi_tol, &eigen_ws);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float port_ms;
    cudaEventElapsedTime(&port_ms, start, stop);

    /* Results */
    double c = 299792458.0;
    double f_ref = sqrt(fabs(ref_result.eigenvalue)) * c / (2.0 * M_PI);
    double f_port = sqrt(fabs(port_result.eigenvalue)) * c / (2.0 * M_PI);
    double k2_exact = k2_target;

    printf("\n");
    printf("  ┌────────────────────────────────────────────────────┐\n");
    printf("  │                    RESULTS                         │\n");
    printf("  ├────────────────────────────────────────────────────┤\n");
    printf("  │ k² analytical:  %15.10f                 │\n", k2_exact);
    printf("  │ k² reference:   %15.10f  (%.4f%% err)   │\n",
        ref_result.eigenvalue,
        fabs(ref_result.eigenvalue - k2_exact) / k2_exact * 100);
    printf("  │ k² with ports:  %15.10f  (%.4f%% err)   │\n",
        port_result.eigenvalue,
        fabs(port_result.eigenvalue - k2_exact) / k2_exact * 100);
    printf("  │ k² shift:       %15.6e  (%.6f%%)       │\n",
        port_result.eigenvalue - ref_result.eigenvalue,
        (port_result.eigenvalue - ref_result.eigenvalue) / ref_result.eigenvalue * 100);
    printf("  ├────────────────────────────────────────────────────┤\n");
    printf("  │ f reference:    %12.6f MHz                 │\n", f_ref / 1e6);
    printf("  │ f with ports:   %12.6f MHz                 │\n", f_port / 1e6);
    printf("  │ Δf:             %12.3f kHz                  │\n", (f_port - f_ref) / 1e3);
    printf("  ├────────────────────────────────────────────────────┤\n");
    printf("  │ Reference:  %d iters, res=%.1e, %.0f ms       │\n",
        ref_result.iterations, ref_result.residual, ref_ms);
    printf("  │ With ports: %d iters, res=%.1e, %.0f ms       │\n",
        port_result.iterations, port_result.residual, port_ms);
    printf("  └────────────────────────────────────────────────────┘\n");

    /* Export VTK if requested */
    if (vtk_filename) {
        gpu_vector_to_host(h_x, d_x_port, n);

        double max_val = 0.0;
        for (int i = 0; i < n; i++)
            if (fabs(h_x[i]) > max_val) max_val = fabs(h_x[i]);
        if (max_val > 0) vec_scale(h_x, 1.0 / max_val, n);

        export_field_vtk(&port_op, h_x, vtk_filename);
    }

    /* Cleanup */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_x);
    free(h_pert);
    gpu_vector_free(d_x_ref);
    gpu_vector_free(d_x_port);
    eigensolver_workspace_free(&eigen_ws);
    gpu_operator_free(&gpu_ref);
    gpu_operator_free(&gpu_port);
    port_config_free(&ports);
    curlcurl_op_free(&ref_op);
    curlcurl_op_free(&port_op);

    return (ref_result.converged && port_result.converged) ? 0 : -1;
}

/*=============================================================================
 * Main
 *============================================================================*/
int main(int /*argc*/, char ** /*argv[]*/) {
    printf("\n");
    printf("****************************************************************\n");
    printf("*         GPU RHODOTRON CAVITY SIMULATION                     *\n");
    printf("****************************************************************\n");

    cuda_print_device_info();

    /* Geometry */
    double a = 0.3333;
    double b = 1.0;
    double L = 1.39;

    printf("=== GEOMETRY ===\n");
    printf("  Inner radius a = %.4f m\n", a);
    printf("  Outer radius b = %.4f m\n", b);
    printf("  Length L = %.4f m\n", L);
    printf("  k² analytical = %.10f\n", (M_PI / L) * (M_PI / L));
    printf("  f analytical = %.6f MHz\n",
        sqrt((M_PI / L) * (M_PI / L)) * 299792458.0 / (2.0 * M_PI) / 1e6);

    /*=========================================================================
     * Run 1: Small aperture, coarse grid (sanity check)
     *========================================================================*/
    printf("\n\n===== RUN 1: COARSE GRID (sanity check) =====\n");
    run_simulation(a, b, L,
        0.005,           /* 5mm aperture */
        32, 128, 128,    /* same as CPU test */
        1,               /* single pass */
        20, 1e-10,
        NULL);

    /*=========================================================================
     * Run 2: 25mm aperture, medium grid (aperture barely resolved)
     *========================================================================*/
    printf("\n\n===== RUN 2: 25mm APERTURE, MEDIUM GRID =====\n");
    {
        int Nr, Nphi, Nz;
        compute_grid_for_aperture(a, b, L, 0.025, 4, &Nr, &Nphi, &Nz);
        printf("  Auto grid for 25mm aperture (4 cells): %d × %d × %d\n",
            Nr, Nphi, Nz);

        run_simulation(a, b, L,
            0.025,       /* 25mm aperture */
            Nr, Nphi, Nz,
            1,           /* single pass */
            20, 1e-8,
            "rhodotron_25mm_medium.vtk");
    }

    /*=========================================================================
     * Run 3: 25mm aperture, fine grid (aperture well resolved)
     *========================================================================*/
    printf("\n\n===== RUN 3: 25mm APERTURE, FINE GRID =====\n");
    {
        int Nr, Nphi, Nz;
        compute_grid_for_aperture(a, b, L, 0.025, 6, &Nr, &Nphi, &Nz);
        printf("  Auto grid for 25mm aperture (6 cells): %d × %d × %d\n",
            Nr, Nphi, Nz);

        run_simulation(a, b, L,
            0.025,
            Nr, Nphi, Nz,
            1,
            20, 1e-8,
            "rhodotron_25mm_fine.vtk");
    }

    /*=========================================================================
    * Run 4: 25mm aperture, very fine grid (convergence check)
    *========================================================================*/
    printf("\n\n===== RUN 4: 25mm APERTURE, VERY FINE GRID =====\n");
    {
        int Nr, Nphi, Nz;
        compute_grid_for_aperture(a, b, L, 0.025, 8, &Nr, &Nphi, &Nz);
        printf("  Auto grid for 25mm aperture (8 cells): %d × %d × %d\n",
            Nr, Nphi, Nz);

        run_simulation(a, b, L,
            0.025,
            Nr, Nphi, Nz,
            1,
            20, 1e-8,
            "rhodotron_25mm_vfine.vtk");
    }

    printf("\n");
    printf("****************************************************************\n");
    printf("*                    ALL RUNS COMPLETE                        *\n");
    printf("****************************************************************\n\n");

    return 0;
}