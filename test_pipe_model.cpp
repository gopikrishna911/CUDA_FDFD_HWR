/*=============================================================================
 * test_pipe_model.cpp
 *
 * Test beam pipe model with GPU eigensolver
 *============================================================================*/

#include "cuda_pipe_model.h"
#include "cuda_eigensolver.h"
#include "pipe_model.h"
#include "curlcurl_operator.h"
#include "q_factor.h"                  /* <── NEW */
#include "r_over_q.h"                  /* R/Q computation */
#include "field_export.h"              /* CSV field data export */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

 /*=============================================================================
  * Helper: run eigensolver with pipe operator
  *============================================================================*/
static int run_pipe_eigensolver(
    GPU_PipeOperator* pipe_op,
    const CurlCurlOperator* cpu_op,
    const GridParams* grid,
    double* d_x,
    int max_iter,
    double tol,
    double* eigenvalue_out
) {
    int n = cpu_op->n_total;

    EigensolverWorkspace ws;
    eigensolver_workspace_init(&ws, n);

    /* Custom RQI loop using pipe matvec */
    double sigma;

    /* Normalize initial guess */
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
    sigma = xAx / xx;

    double k2_target = (M_PI / grid->L) * (M_PI / grid->L);
    if (fabs(sigma) < 1e-10 || sigma < 0) sigma = k2_target;

    printf("  Pipe RQI:\n");
    printf("  %5s %15s %15s %8s\n", "Iter", "Eigenvalue", "Residual", "LS its");
    printf("  -------------------------------------------------\n");

    int blocks = (n + 256 - 1) / 256;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Solve (A_pipe - σI)y = x using MINRES
         * We need a shifted matvec: (A_pipe - σI)v
         * Strategy: use MINRES but with pipe_matvec instead of regular matvec
         *
         * Manual MINRES with pipe operator */

         /* Zero the solution */
        gpu_vec_zero(ws.d_y, n);
        gpu_vec_zero(ws.minres_ws.d_v_old, n);
        gpu_vec_zero(ws.minres_ws.d_w_old, n);
        gpu_vec_zero(ws.minres_ws.d_w_cur, n);

        /* v_cur = x / ||x|| */
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

        for (int ls = 0; ls < 3000; ls++) {
            /* Av = A_pipe * v_cur - sigma * v_cur */
            gpu_pipe_matvec(pipe_op, ws.minres_ws.d_v_cur, ws.minres_ws.d_Av);

            /* shift: Av -= sigma * v_cur */
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

        /* Normalize y → x */
        gpu_vec_normalize_weighted(ws.d_y, cpu_op);
        gpu_vec_copy(ws.d_y, d_x, n);

        /* New Rayleigh quotient */
        gpu_pipe_matvec(pipe_op, d_x, d_Ax);
        gpu_vec_dot_weighted_ws(d_x, d_Ax, &xAx, cpu_op, &pipe_op->base.reduction_ws);
        gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op, &pipe_op->base.reduction_ws);
        double sigma_new = xAx / xx;

        /* Residual: ||Ax - λx|| (direct — avoids catastrophic cancellation) */
        /* d_Ax currently holds A*x. Shift in-place: d_Ax ← Ax - σ*x        */
        /* Safe: d_Ax is recomputed via matvec at the start of next iteration */
        extern __global__ void shift_kernel(double*, const double*, double, int);
        shift_kernel << <blocks, 256 >> > (d_Ax, d_x, sigma_new, n);
        double res_sq;
        gpu_vec_dot_weighted_ws(d_Ax, d_Ax, &res_sq, cpu_op,
            &pipe_op->base.reduction_ws);
        double residual = sqrt(res_sq);

        printf("  %5d %15.10f %15.8e %8d\n", iter, sigma_new, residual, ls_iters);

        if (residual < tol) {
            printf("  Converged!\n");
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
 * Enhanced VTK export with material mask and full field data
 *
 * Exports structured grid in cylindrical → Cartesian coordinates
 * with stride-based subsampling for manageable file sizes.
 *
 * ParaView usage:
 *   - Threshold on "material" > 0.5 to show only vacuum region
 *   - Use "E_field" for arrow/glyph/streamline plots
 *   - Use "E_magnitude" or "Er" for contour/color plots
 *============================================================================*/
static void export_pipe_field_vtk(
    const CurlCurlOperator* op,
    const double* x,
    const PipeConfig* pipes,
    const char* filename,
    int stride,
    double z0_offset
) {
    const GridParams* g = &op->grid;

    /*--- Build subsampled index arrays (always include boundaries) ---*/
    int max_r = g->Nr;
    int max_p = g->Nphi;   /* extra point to close the cylinder */
    int max_z = g->Nz;

    /* Count points per dimension */
    int Ni = 0, Nj = 0, Nk = 0;
    for (int i = 0; i <= max_r; i += stride) Ni++;
    if ((max_r % stride) != 0) Ni++;   /* ensure r=r_max included */

    for (int j = 0; j <= max_p; j += stride) Nj++;
    if ((max_p % stride) != 0) Nj++;   /* ensure phi=2π included */

    for (int k = 0; k <= max_z; k += stride) Nk++;
    if ((max_z % stride) != 0) Nk++;   /* ensure z=L included */

    /* Build index lists */
    int* ri = (int*)malloc(Ni * sizeof(int));
    int* pj = (int*)malloc(Nj * sizeof(int));
    int* zk = (int*)malloc(Nk * sizeof(int));

    int idx = 0;
    for (int i = 0; i <= max_r; i += stride) ri[idx++] = i;
    if (ri[idx - 1] != max_r) ri[idx++] = max_r;
    Ni = idx;

    idx = 0;
    for (int j = 0; j <= max_p; j += stride) pj[idx++] = j;
    if (pj[idx - 1] != max_p) pj[idx++] = max_p;
    Nj = idx;

    idx = 0;
    for (int k = 0; k <= max_z; k += stride) zk[idx++] = k;
    if (zk[idx - 1] != max_z) zk[idx++] = max_z;
    Nk = idx;

    int npoints = Ni * Nj * Nk;

    printf("\n  VTK Export: %d × %d × %d = %d vertices (stride=%d)\n",
        Ni, Nj, Nk, npoints, stride);

    /*--- Allocate interpolated field arrays ---*/
    double* Er_v = (double*)calloc(npoints, sizeof(double));
    double* Ep_v = (double*)calloc(npoints, sizeof(double));
    double* Ez_v = (double*)calloc(npoints, sizeof(double));
    float* mat_v = (float*)calloc(npoints, sizeof(float));

    /*--- Interpolate fields from staggered grid to vertices ---*/
    int vid = 0;
    for (int kk = 0; kk < Nk; kk++) {
        int k = zk[kk];
        double z = k * g->dz;

        for (int jj = 0; jj < Nj; jj++) {
            int j = pj[jj];
            int jw = j % g->Nphi;                          /* wrap for periodic */
            int jm = (j - 1 + g->Nphi) % g->Nphi;         /* j-1 wrapped */

            for (int ii = 0; ii < Ni; ii++) {
                int i = ri[ii];
                double r = g->a + i * g->dr;
                double phi = j * g->dphi;

                /* Material: 1=vacuum, 0=PEC */
                double z_phys = z - z0_offset;
                mat_v[vid] = (float)point_is_vacuum(pipes, r, phi, z_phys);

                /*--- Er at vertex (i, j, k) ---
                 * Er lives at (i+½, j, k) for i=0..Nr-1
                 * Interpolate: average Er[i-1,j,k] and Er[i,j,k]
                 */
                if (i == 0) {
                    Er_v[vid] = x[op->offset_Er + idx_Er(g, 0, jw, k)];
                }
                else if (i >= g->Nr) {
                    Er_v[vid] = x[op->offset_Er + idx_Er(g, g->Nr - 1, jw, k)];
                }
                else {
                    Er_v[vid] = 0.5 * (
                        x[op->offset_Er + idx_Er(g, i - 1, jw, k)] +
                        x[op->offset_Er + idx_Er(g, i, jw, k)]);
                }

                /*--- Ephi at vertex (i, j, k) ---
                 * Ephi lives at (i, j+½, k) for j=0..Nphi-1
                 * Interpolate: average Ephi[i,j-1,k] and Ephi[i,j,k]
                 * Periodic in phi: j-1 wraps around
                 */
                Ep_v[vid] = 0.5 * (
                    x[op->offset_Ephi + idx_Ephi(g, i, jm, k)] +
                    x[op->offset_Ephi + idx_Ephi(g, i, jw, k)]);

                /*--- Ez at vertex (i, j, k) ---
                 * Ez lives at (i, j, k+½) for k=0..Nz-1
                 * Interpolate: average Ez[i,j,k-1] and Ez[i,j,k]
                 */
                if (k == 0) {
                    Ez_v[vid] = x[op->offset_Ez + idx_Ez(g, i, jw, 0)];
                }
                else if (k >= g->Nz) {
                    Ez_v[vid] = x[op->offset_Ez + idx_Ez(g, i, jw, g->Nz - 1)];
                }
                else {
                    Ez_v[vid] = 0.5 * (
                        x[op->offset_Ez + idx_Ez(g, i, jw, k - 1)] +
                        x[op->offset_Ez + idx_Ez(g, i, jw, k)]);
                }

                vid++;
            }
        }
    }

    /*--- Write VTK file ---*/
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open %s\n", filename);
        free(ri); free(pj); free(zk);
        free(Er_v); free(Ep_v); free(Ez_v); free(mat_v);
        return;
    }

    /* Header */
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "Rhodotron cavity with beam pipes\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_GRID\n");
    fprintf(fp, "DIMENSIONS %d %d %d\n", Ni, Nj, Nk);

    /* Points: cylindrical → Cartesian */
    fprintf(fp, "POINTS %d float\n", npoints);
    for (int kk = 0; kk < Nk; kk++) {
        double z = zk[kk] * g->dz;
        for (int jj = 0; jj < Nj; jj++) {
            double phi = pj[jj] * g->dphi;
            double cphi = cos(phi);
            double sphi = sin(phi);
            for (int ii = 0; ii < Ni; ii++) {
                double r = g->a + ri[ii] * g->dr;
                fprintf(fp, "%.6g %.6g %.6g\n",
                    r * cphi, r * sphi, z);
            }
        }
    }

    /* ---- Point Data ---- */
    fprintf(fp, "POINT_DATA %d\n", npoints);

    /* 1. Material type: 1=vacuum, 0=PEC */
    fprintf(fp, "SCALARS material float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int n = 0; n < npoints; n++) {
        fprintf(fp, "%.0f\n", mat_v[n]);
    }

    /* 2. Er component (scalar) */
    fprintf(fp, "SCALARS Er float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int n = 0; n < npoints; n++) {
        fprintf(fp, "%.6e\n", Er_v[n]);
    }

    /* 3. Ephi component (scalar) */
    fprintf(fp, "SCALARS Ephi float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int n = 0; n < npoints; n++) {
        fprintf(fp, "%.6e\n", Ep_v[n]);
    }

    /* 4. Ez component (scalar) */
    fprintf(fp, "SCALARS Ez float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int n = 0; n < npoints; n++) {
        fprintf(fp, "%.6e\n", Ez_v[n]);
    }

    /* 5. E magnitude */
    fprintf(fp, "SCALARS E_magnitude float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int n = 0; n < npoints; n++) {
        double mag = sqrt(Er_v[n] * Er_v[n] + Ep_v[n] * Ep_v[n] + Ez_v[n] * Ez_v[n]);
        fprintf(fp, "%.6e\n", mag);
    }

    /* 6. E-field as Cartesian vector (for arrows/streamlines in ParaView) */
    fprintf(fp, "VECTORS E_field float\n");
    vid = 0;
    for (int kk = 0; kk < Nk; kk++) {
        for (int jj = 0; jj < Nj; jj++) {
            double phi = pj[jj] * g->dphi;
            double cphi = cos(phi);
            double sphi = sin(phi);
            for (int ii = 0; ii < Ni; ii++) {
                double er = Er_v[vid];
                double ep = Ep_v[vid];
                double ez = Ez_v[vid];

                /* Cylindrical → Cartesian */
                double Ex = er * cphi - ep * sphi;
                double Ey = er * sphi + ep * cphi;

                fprintf(fp, "%.6e %.6e %.6e\n", Ex, Ey, ez);
                vid++;
            }
        }
    }

    fclose(fp);
    printf("  VTK written: %s (%d points)\n", filename, npoints);

    /* Cleanup */
    free(ri); free(pj); free(zk);
    free(Er_v); free(Ep_v); free(Ez_v); free(mat_v);
}

int main(void) {
    printf("\n");
    printf("****************************************************************\n");
    printf("*  RHODOTRON CAVITY — 10-PASS WITH ALL PIPES                  *\n");
    printf("****************************************************************\n");

    cuda_print_device_info();
    cuda_print_memory_info("startup");

    /* Geometry */
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
    double endcap_z0_pipe_length = 0.28;    /* 28 cm */

    double endcap_zL_r_center = 0.85;
    double endcap_zL_phi = 3.0 * M_PI / 2.0;
    double endcap_zL_aperture_radius = 0.095;
    double endcap_zL_pipe_radius = 0.090;
    double endcap_zL_pipe_length = 0.25;    /* 25 cm */

    printf("\n=== GEOMETRY ===\n");
    printf("  Inner radius a = %.4f m\n", a);
    printf("  Outer radius b = %.4f m\n", b);
    printf("  Length L = %.4f m\n", L);
    printf("  Pipe radius = %.1f mm (diameter %.1f mm)\n",
        pipe_radius * 1000, pipe_radius * 2000);
    printf("  Aperture radius = %.1f mm (diameter %.1f mm)\n",
        aperture_radius * 1000, aperture_radius * 2000);
    printf("  Pipe length = %.1f mm\n", pipe_length * 1000);
    printf("  Number of passes = %d\n", num_passes);

    /* Configure radial pipes (same as before) */
    PipeConfig pipes;
    pipe_config_init(&pipes, a, b, pipe_radius, aperture_radius,
        pipe_length, taper_length);
    pipe_config_add_multi_pass(&pipes, L / 2.0, num_passes);
    pipe_config_print(&pipes);

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

    endcap_pipe_config_print(&endcap_pipes);

    /* Configure inner conductor apertures (same as before) */
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

    /* NO endplate port BCs — material mask handles everything */

    /* =====================================================================
     * Grid — extended in both r and z
     * ====================================================================*/
    int Nr_cavity = 162;
    int Nr_pipe = 13;
    int Nphi = 1024;
    int Nz_cavity = 167;

    /* z-pipe cells: match dz ≈ L/Nz_cavity = 8.35 mm */
    double dz_target = L / Nz_cavity;
    int Nz_pipe_z0 = (int)ceil(endcap_pipes.z0_extension / dz_target);
    int Nz_pipe_zL = (int)ceil(endcap_pipes.zL_extension / dz_target);

    printf("\n  z-pipe cells: z0 = %d, zL = %d (dz_target = %.2f mm)\n",
        Nz_pipe_z0, Nz_pipe_zL, dz_target * 1000);

    GridParams grid;
    grid_init_with_all_pipes(&grid, a, b, L,
        pipe_length,
        endcap_pipes.z0_extension,
        endcap_pipes.zL_extension,
        Nr_cavity, Nr_pipe, Nphi,
        Nz_cavity, Nz_pipe_z0, Nz_pipe_zL);

    double z0_offset = Nz_pipe_z0 * grid.dz;  /* Use grid-snapped value, not original pipe length */
    for (int i = 0; i < inner_ports.num_ports; i++) {
        inner_ports.ports[i].pos2 += z0_offset;
    }

    double actual_z0_pipe = z0_offset;
    double actual_zL_pipe = Nz_pipe_zL * grid.dz;

    int k_endplate_z0 = Nz_pipe_z0;
    int k_endplate_zL = Nz_pipe_z0 + Nz_cavity;

    printf("  Endplate k-indices: z0 at k=%d, zL at k=%d (of Nz=%d)\n",
        k_endplate_z0, k_endplate_zL, grid.Nz);

    /* Build combined material mask */
    printf("\n  Building combined material mask...\n");
    MaterialMask mask;
    material_mask_build_full(&mask, &pipes, &endcap_pipes,
        &grid, L, z0_offset);
    material_mask_print_stats(&mask, &grid);

    /* Create operator — inner conductor ports only, no endplate ports */
    CurlCurlOperator cpu_op;
    curlcurl_op_init_with_ports(&cpu_op, &grid, &inner_ports);

    int n = cpu_op.n_total;
    printf("\n  Total DOFs: %d (%.1f M)\n", n, n / 1.0e6);

    cuda_grid_init(&grid);

    /* TEM initial guess — shifted z coordinate */
    double* h_x = (double*)malloc(n * sizeof(double));
    vec_zero(h_x, n);

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

    double k2_ref = 0.0;
    double k2_radial = 0.0;
    double k2_pipe = 0.0;

    QFactorResult qf_analytical = compute_q_analytical_coaxial_hwr(
        a, b, L, Q_SIGMA_CU);
    QFactorResult qf_ref;
    QFactorResult qf_radial;
    QFactorResult qf_full;

    RoverQResult roq_analytical = compute_r_over_q_analytical(a, b, L);
    RoverQResult roq_ref;
    RoverQResult roq_radial;
    RoverQResult roq_full;

    /*=========================================================================
     * PHASE 1: Reference solve (no holes)
     *========================================================================*/
    printf("\n=== PHASE 1: REFERENCE SOLVE (no holes) ===\n");
    {
        CurlCurlOperator ref_op;
        curlcurl_op_init(&ref_op, &grid);

        /* Reference mask: no radial pipes, no endcap pipes */
        PipeConfig no_pipes;
        pipe_config_init(&no_pipes, a, b, pipe_radius, aperture_radius,
            pipe_length, taper_length);

        MaterialMask ref_mask;
        material_mask_build_full(&ref_mask, &no_pipes, NULL,
            &grid, L, z0_offset);
        material_mask_print_stats(&ref_mask, &grid);

        GPU_PipeOperator gpu_ref;
        gpu_pipe_operator_init(&gpu_ref, &ref_op, &ref_mask);
        gpu_pipe_operator_set_endplates(&gpu_ref,
            Nz_pipe_z0, Nz_pipe_z0 + Nz_cavity,
            NULL,  /* NULL = full PEC, no holes */
            &grid, z0_offset);

        double* d_x_ref;
        gpu_vector_alloc(&d_x_ref, n);
        gpu_vector_to_device(d_x_ref, h_x, n);

        run_pipe_eigensolver(&gpu_ref, &ref_op, &grid, d_x_ref,
            10, 1e-4, &k2_ref);

        double* h_ref = (double*)malloc(n * sizeof(double));
        gpu_vector_to_host(h_ref, d_x_ref, n);

        //qf_ref = compute_q_factor(
        //    &ref_op, h_ref, k2_ref, Q_SIGMA_CU, b, NULL);
        qf_ref = compute_q_factor_extended(
            &ref_op, h_ref, k2_ref, Q_SIGMA_CU, b, NULL,
            k_endplate_z0, k_endplate_zL);
        q_factor_print(&qf_ref, "Reference cavity (no holes)");
        q_factor_print(&qf_analytical, "Analytical coaxial HWR");
        q_factor_print_comparison(&qf_analytical, &qf_ref, NULL);

        /* R/Q for reference solve */
        roq_ref = compute_r_over_q(
            &ref_op, h_ref, k2_ref, b, L, z0_offset, num_passes);
        r_over_q_print(&roq_ref, "Reference cavity (no holes)");
        r_over_q_print_comparison(&roq_analytical, &roq_ref);

        /* Export reference field data */
        {
            FieldExportConfig fxcfg;
            fxcfg.b_cavity = b;
            fxcfg.L_cavity = L;
            fxcfg.z0_offset = z0_offset;
            fxcfg.k_cavity_start = k_endplate_z0;
            fxcfg.k_cavity_end = k_endplate_zL;
            fxcfg.prefix = "ref";
            export_all_field_data(&ref_op, h_ref, &fxcfg);
        }

        free(h_ref);
        gpu_vector_free(d_x_ref);
        gpu_pipe_operator_free(&gpu_ref);
        material_mask_free(&ref_mask);
        pipe_config_free(&no_pipes);
        curlcurl_op_free(&ref_op);
    }

    /*=========================================================================
     * PHASE 1.5: Radial pipes only (no endcap pipes)
     *
     * Isolates the frequency shift from the 20 radial beam pipes.
     * Comparison with Phase 2 reveals the endcap contribution.
     *========================================================================*/
    printf("\n=== PHASE 1.5: RADIAL PIPES ONLY (no endcap pipes) ===\n");
    {
        /* Operator with inner conductor ports (same as full model) */
        CurlCurlOperator radial_op;
        curlcurl_op_init_with_ports(&radial_op, &grid, &inner_ports);

        /* Material mask: radial pipes only, no endcap pipes */
        MaterialMask radial_mask;
        material_mask_build_full(&radial_mask, &pipes, NULL,
            &grid, L, z0_offset);
        material_mask_print_stats(&radial_mask, &grid);

        GPU_PipeOperator gpu_radial;
        gpu_pipe_operator_init(&gpu_radial, &radial_op, &radial_mask);
        gpu_pipe_operator_set_endplates(&gpu_radial,
            Nz_pipe_z0, Nz_pipe_z0 + Nz_cavity,
            NULL,  /* No endcap pipe holes in endplates */
            &grid, z0_offset);

        double* d_x_radial;
        gpu_vector_alloc(&d_x_radial, n);
        gpu_vector_to_device(d_x_radial, h_x, n);

        run_pipe_eigensolver(&gpu_radial, &radial_op, &grid, d_x_radial,
            10, 1e-4, &k2_radial);

        double* h_radial = (double*)malloc(n * sizeof(double));
        gpu_vector_to_host(h_radial, d_x_radial, n);

        /* Q factor — build outer aperture exclusion list for radial pipes */
        PortConfig radial_apertures;
        port_config_init(&radial_apertures);

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
            port.name = "Outer Pipe Entry";
            if (radial_apertures.num_ports >= radial_apertures.capacity) {
                radial_apertures.capacity *= 2;
                radial_apertures.ports = (CavityPort*)realloc(
                    radial_apertures.ports,
                    radial_apertures.capacity * sizeof(CavityPort));
            }
            radial_apertures.ports[radial_apertures.num_ports++] = port;

            port.pos1 = phi_exit;
            port.name = "Outer Pipe Exit";
            if (radial_apertures.num_ports >= radial_apertures.capacity) {
                radial_apertures.capacity *= 2;
                radial_apertures.ports = (CavityPort*)realloc(
                    radial_apertures.ports,
                    radial_apertures.capacity * sizeof(CavityPort));
            }
            radial_apertures.ports[radial_apertures.num_ports++] = port;
        }

        qf_radial = compute_q_factor_extended(
            &radial_op, h_radial, k2_radial, Q_SIGMA_CU, b, &radial_apertures,
            k_endplate_z0, k_endplate_zL);
        q_factor_print(&qf_radial, "Radial pipes only (no endcap)");

        /* R/Q */
        roq_radial = compute_r_over_q(
            &radial_op, h_radial, k2_radial, b, L, z0_offset, num_passes);
        r_over_q_print(&roq_radial, "Radial pipes only (no endcap)");

        /* Export radial-only field data */
        {
            FieldExportConfig fxcfg;
            fxcfg.b_cavity = b;
            fxcfg.L_cavity = L;
            fxcfg.z0_offset = z0_offset;
            fxcfg.k_cavity_start = k_endplate_z0;
            fxcfg.k_cavity_end = k_endplate_zL;
            fxcfg.prefix = "radial";
            export_all_field_data(&radial_op, h_radial, &fxcfg);
        }

        port_config_free(&radial_apertures);
        free(h_radial);
        gpu_vector_free(d_x_radial);
        gpu_pipe_operator_free(&gpu_radial);
        material_mask_free(&radial_mask);
        curlcurl_op_free(&radial_op);
    }

    /*=========================================================================
     * PHASE 2: Full model (radial + endcap pipes)
     *========================================================================*/
    printf("\n=== PHASE 2: FULL MODEL (radial + endcap pipes) ===\n");
    {
        GPU_PipeOperator gpu_pipe;
        gpu_pipe_operator_init(&gpu_pipe, &cpu_op, &mask);
        gpu_pipe_operator_set_endplates(&gpu_pipe,
            Nz_pipe_z0, Nz_pipe_z0 + Nz_cavity,
            &endcap_pipes,  /* Has the pipe holes */
            &grid, z0_offset);

        double* d_x_pipe;
        gpu_vector_alloc(&d_x_pipe, n);
        gpu_vector_to_device(d_x_pipe, h_x, n);

        run_pipe_eigensolver(&gpu_pipe, &cpu_op, &grid, d_x_pipe,
            10, 1e-4, &k2_pipe);

        gpu_vector_to_host(h_x, d_x_pipe, n);

        gpu_vector_free(d_x_pipe);
        gpu_pipe_operator_free(&gpu_pipe);
    }

    /* Q-factor for full model */
    PortConfig all_apertures;
    port_config_init(&all_apertures);

    /* Copy inner apertures */
    for (int i = 0; i < inner_ports.num_ports; i++) {
        if (all_apertures.num_ports >= all_apertures.capacity) {
            all_apertures.capacity *= 2;
            all_apertures.ports = (CavityPort*)realloc(
                all_apertures.ports,
                all_apertures.capacity * sizeof(CavityPort));
        }
        all_apertures.ports[all_apertures.num_ports++] = inner_ports.ports[i];
    }

    /* Add outer pipe openings */
    {
        double dphi_pass_q = M_PI / num_passes;
        for (int pass = 0; pass < num_passes; pass++) {
            double phi_entry = pass * dphi_pass_q;
            double phi_exit = phi_entry + M_PI;
            if (phi_exit >= 2.0 * M_PI) phi_exit -= 2.0 * M_PI;

            CavityPort port;
            port.type = PORT_BEAM;
            port.surface = SURFACE_OUTER;
            port.radius = aperture_radius;
            port.pos2 = L / 2.0 + z0_offset;

            port.pos1 = phi_entry;
            port.name = "Outer Pipe Entry";
            if (all_apertures.num_ports >= all_apertures.capacity) {
                all_apertures.capacity *= 2;
                all_apertures.ports = (CavityPort*)realloc(
                    all_apertures.ports,
                    all_apertures.capacity * sizeof(CavityPort));
            }
            all_apertures.ports[all_apertures.num_ports++] = port;

            port.pos1 = phi_exit;
            port.name = "Outer Pipe Exit";
            if (all_apertures.num_ports >= all_apertures.capacity) {
                all_apertures.capacity *= 2;
                all_apertures.ports = (CavityPort*)realloc(
                    all_apertures.ports,
                    all_apertures.capacity * sizeof(CavityPort));
            }
            all_apertures.ports[all_apertures.num_ports++] = port;
        }
    }

    /* NOTE: Endcap apertures handled by material mask, not Q exclusion list.
     * The endcap pipes are interior structure — the Q integration boundaries
     * are the PEC caps at the ends of the pipes, which are the grid boundaries
     * at z=0 and z=Nz. The endplate surface integral now correctly integrates
     * over the PEC cap area, which naturally excludes the pipe interior. */

     //qf_full = compute_q_factor(
     //    &cpu_op, h_x, k2_pipe, Q_SIGMA_CU, b, &all_apertures);
    qf_full = compute_q_factor_extended(
        &cpu_op, h_x, k2_pipe, Q_SIGMA_CU, b, &all_apertures,
        k_endplate_z0, k_endplate_zL);
    q_factor_print(&qf_full, "Full model (all pipes)");
    q_factor_print_comparison(&qf_analytical, &qf_ref, &qf_full);

    /* R/Q for full model */
    roq_full = compute_r_over_q(
        &cpu_op, h_x, k2_pipe, b, L, z0_offset, num_passes);
    r_over_q_print(&roq_full, "Full model (all pipes)");
    r_over_q_print_comparison(&roq_analytical, &roq_full);

    /* Export full model field data */
    {
        FieldExportConfig fxcfg;
        fxcfg.b_cavity = b;
        fxcfg.L_cavity = L;
        fxcfg.z0_offset = z0_offset;
        fxcfg.k_cavity_start = k_endplate_z0;
        fxcfg.k_cavity_end = k_endplate_zL;
        fxcfg.prefix = "full";
        export_all_field_data(&cpu_op, h_x, &fxcfg);
    }

    /*=========================================================================
     * Results Summary
     *========================================================================*/
    double c = 299792458.0;
    double k2_exact = (M_PI / L) * (M_PI / L);
    double f_ref = sqrt(fabs(k2_ref)) * c / (2.0 * M_PI);
    double f_radial = sqrt(fabs(k2_radial)) * c / (2.0 * M_PI);
    double f_pipe = sqrt(fabs(k2_pipe)) * c / (2.0 * M_PI);

    double df_radial = f_radial - f_ref;       /* Shift from radial pipes only */
    double df_full = f_pipe - f_ref;          /* Total shift */
    double df_endcap = df_full - df_radial;     /* Endcap contribution */

    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────────┐\n");
    printf("  │         RHODOTRON CAVITY — RESULTS SUMMARY                   │\n");
    printf("  │         %d passes + 2 endcap pipes                           │\n",
        num_passes);
    printf("  ├──────────────────────────────────────────────────────────────┤\n");
    printf("  │ Geometry:                                                    │\n");
    printf("  │   Radial: %d × %.0fmm diam, %.0fmm long                    │\n",
        2 * num_passes, pipe_radius * 2000, pipe_length * 1000);
    printf("  │   Endcap z=0: %.0fmm diam, %.1f cm pipe                    │\n",
        endcap_z0_aperture_radius * 2000, actual_z0_pipe * 100);
    printf("  │   Endcap z=L: %.0fmm diam, %.1f cm pipe                    │\n",
        endcap_zL_aperture_radius * 2000, actual_zL_pipe * 100);
    printf("  │   Grid: %d × %d × %d = %.1fM DOFs                    │\n",
        grid.Nr, Nphi, grid.Nz, n / 1e6);
    printf("  ├──────────────────────────────────────────────────────────────┤\n");
    printf("  │ Eigenvalues (k²):                                           │\n");
    printf("  │   Analytical:    %15.10f                            │\n", k2_exact);
    printf("  │   Reference:     %15.10f  (no holes)                │\n", k2_ref);
    printf("  │   Radial only:   %15.10f  (20 radial pipes)        │\n", k2_radial);
    printf("  │   Full model:    %15.10f  (all pipes)               │\n", k2_pipe);
    printf("  ├──────────────────────────────────────────────────────────────┤\n");
    printf("  │ Frequency:                                                   │\n");
    printf("  │   f reference:     %12.6f MHz                          │\n", f_ref / 1e6);
    printf("  │   f radial only:   %12.6f MHz                          │\n", f_radial / 1e6);
    printf("  │   f full model:    %12.6f MHz                          │\n", f_pipe / 1e6);
    printf("  │                                                              │\n");
    printf("  │ Frequency shift breakdown:                                   │\n");
    printf("  │   Δf (radial pipes):  %+12.3f kHz                      │\n", df_radial / 1e3);
    printf("  │   Δf (endcap pipes):  %+12.3f kHz                      │\n", df_endcap / 1e3);
    printf("  │   Δf (total):         %+12.3f kHz                      │\n", df_full / 1e3);
    if (fabs(df_full) > 0) {
        printf("  │   Radial fraction:    %11.1f%%                          │\n",
            100.0 * df_radial / df_full);
        printf("  │   Endcap fraction:    %11.1f%%                          │\n",
            100.0 * df_endcap / df_full);
    }
    printf("  ├──────────────────────────────────────────────────────────────┤\n");
    printf("  │ Q factor (Q_0):                                              │\n");
    printf("  │   Analytical:       %12.0f                              │\n",
        qf_analytical.Q_0);
    printf("  │   Reference:        %12.0f  (no holes)                  │\n", qf_ref.Q_0);
    printf("  │   Radial only:      %12.0f  (20 radial pipes)          │\n", qf_radial.Q_0);
    printf("  │   Full model:       %12.0f  (all pipes)                 │\n", qf_full.Q_0);
    if (qf_ref.Q_0 > 0) {
        printf("  │   ΔQ (radial vs ref):  %+10.0f  (%+.2f%%)               │\n",
            qf_radial.Q_0 - qf_ref.Q_0,
            (qf_radial.Q_0 - qf_ref.Q_0) / qf_ref.Q_0 * 100.0);
        printf("  │   ΔQ (full vs ref):    %+10.0f  (%+.2f%%)               │\n",
            qf_full.Q_0 - qf_ref.Q_0,
            (qf_full.Q_0 - qf_ref.Q_0) / qf_ref.Q_0 * 100.0);
    }
    printf("  ├──────────────────────────────────────────────────────────────┤\n");
    printf("  │ R/Q (linac convention, per crossing):                        │\n");
    printf("  │   Analytical:       %12.4f Ohm                         │\n",
        roq_analytical.R_over_Q_crossing_linac);
    printf("  │   Reference:        %12.4f Ohm                         │\n",
        roq_ref.R_over_Q_crossing_linac);
    printf("  │   Radial only:      %12.4f Ohm                         │\n",
        roq_radial.R_over_Q_crossing_linac);
    printf("  │   Full model:       %12.4f Ohm                         │\n",
        roq_full.R_over_Q_crossing_linac);
    if (roq_analytical.R_over_Q_crossing_linac > 0) {
        printf("  │   Error (full vs analytical): %+.2f%%                     │\n",
            (roq_full.R_over_Q_crossing_linac
                - roq_analytical.R_over_Q_crossing_linac)
            / roq_analytical.R_over_Q_crossing_linac * 100.0);
    }
    printf("  │ R/Q per pass:       %12.4f Ohm                         │\n",
        roq_full.R_over_Q_per_pass_linac);
    printf("  │ R/Q total (%d passes): %10.2f Ohm                       │\n",
        num_passes, roq_full.R_over_Q_total_linac);
    printf("  │ Voltage spread:     %11.4f%%                            │\n",
        roq_full.V_gap_spread);
    printf("  ├──────────────────────────────────────────────────────────────┤\n");

    double k0 = sqrt(fabs(k2_pipe));
    double alpha_z0 = sqrt((1.8412 / endcap_z0_pipe_radius)
        * (1.8412 / endcap_z0_pipe_radius) - k0 * k0);
    double alpha_zL = sqrt((1.8412 / endcap_zL_pipe_radius)
        * (1.8412 / endcap_zL_pipe_radius) - k0 * k0);

    printf("  │ Endcap evanescent attenuation at PEC caps:                   │\n");
    printf("  │   z=0: e^(-αL) = %.2e  (R=%.0fmm, L=%.1fcm, α=%.1f/m)  │\n",
        exp(-alpha_z0 * actual_z0_pipe),
        endcap_z0_pipe_radius * 1000, actual_z0_pipe * 100, alpha_z0);
    printf("  │   z=L: e^(-αL) = %.2e  (R=%.0fmm, L=%.1fcm, α=%.1f/m)  │\n",
        exp(-alpha_zL * actual_zL_pipe),
        endcap_zL_pipe_radius * 1000, actual_zL_pipe * 100, alpha_zL);
    printf("  └──────────────────────────────────────────────────────────────┘\n");

    /*=========================================================================
     * VTK Export — shifted z coordinate
     *========================================================================*/
    {
        double max_val = 0.0;
        for (int i = 0; i < n; i++)
            if (fabs(h_x[i]) > max_val) max_val = fabs(h_x[i]);
        if (max_val > 0) vec_scale(h_x, 1.0 / max_val, n);

        int vtk_stride = 4;
        export_pipe_field_vtk(&cpu_op, h_x, &pipes,
            "rhodotron_with_pipes.vtk", vtk_stride, z0_offset);
    }

    /*=========================================================================
     * Cleanup
     *========================================================================*/
    port_config_free(&all_apertures);
    r_over_q_free(&roq_ref);
    r_over_q_free(&roq_radial);
    r_over_q_free(&roq_full);
    free(h_x);
    material_mask_free(&mask);
    pipe_config_free(&pipes);
    endcap_pipe_config_free(&endcap_pipes);
    port_config_free(&inner_ports);
    curlcurl_op_free(&cpu_op);

    printf("\n****************************************************************\n");
    printf("*                    SIMULATION COMPLETE                       *\n");
    printf("****************************************************************\n\n");

    return 0;
}