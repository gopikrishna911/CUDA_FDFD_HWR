#include "cuda_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/*=============================================================================
 * Error checking macro
 *============================================================================*/
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        return -1;                                                      \
    }                                                                   \
} while(0)

#define BLOCK_SIZE 256

 /*=============================================================================
  * PEC Boundary Kernels (UNCHANGED from original)
  *============================================================================*/

typedef struct {
    double a, dr, dphi, dz;
    int Nr, Nphi, Nz;
} BndGridParams;

static BndGridParams make_bnd_params(const GridParams* grid) {
    BndGridParams p;
    p.a = grid->a; p.dr = grid->dr; p.dphi = grid->dphi; p.dz = grid->dz;
    p.Nr = grid->Nr; p.Nphi = grid->Nphi; p.Nz = grid->Nz;
    return p;
}

/* Er = 0 at z=0 (k=0) and z=L (k=Nz) */
__global__ void pec_Er_endplates_kernel(double* Er, BndGridParams gp) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face_size = gp.Nr * gp.Nphi;

    if (gid < face_size) {
        int i = gid % gp.Nr;
        int j = gid / gp.Nr;

        Er[i + gp.Nr * (j + gp.Nphi * 0)] = 0.0;
        Er[i + gp.Nr * (j + gp.Nphi * gp.Nz)] = 0.0;
    }
}

/* Ephi = 0 at r=a (i=0) and r=b (i=Nr) */
__global__ void pec_Ephi_conductors_kernel(double* Ephi, BndGridParams gp) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face_size = gp.Nphi * (gp.Nz + 1);

    if (gid < face_size) {
        int j = gid % gp.Nphi;
        int k = gid / gp.Nphi;
        int Nr1 = gp.Nr + 1;

        Ephi[0 + Nr1 * (j + gp.Nphi * k)] = 0.0;
        Ephi[gp.Nr + Nr1 * (j + gp.Nphi * k)] = 0.0;
    }
}

/* Ephi = 0 at z=0 (k=0) and z=L (k=Nz) */
__global__ void pec_Ephi_endplates_kernel(double* Ephi, BndGridParams gp) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face_size = (gp.Nr + 1) * gp.Nphi;

    if (gid < face_size) {
        int i = gid % (gp.Nr + 1);
        int j = gid / (gp.Nr + 1);
        int Nr1 = gp.Nr + 1;

        Ephi[i + Nr1 * (j + gp.Nphi * 0)] = 0.0;
        Ephi[i + Nr1 * (j + gp.Nphi * gp.Nz)] = 0.0;
    }
}

/* Ez = 0 at r=a (i=0) and r=b (i=Nr) */
__global__ void pec_Ez_conductors_kernel(double* Ez, BndGridParams gp) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face_size = gp.Nphi * gp.Nz;

    if (gid < face_size) {
        int j = gid % gp.Nphi;
        int k = gid / gp.Nphi;
        int Nr1 = gp.Nr + 1;

        Ez[0 + Nr1 * (j + gp.Nphi * k)] = 0.0;
        Ez[gp.Nr + Nr1 * (j + gp.Nphi * k)] = 0.0;
    }
}

/*=============================================================================
 * PEC Boundary with Port Masks (UNCHANGED from original)
 *============================================================================*/

__global__ void pec_Er_endplates_masked_kernel(
    double* Er, BndGridParams gp,
    const int* mask_z0, const int* mask_zL
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face_size = gp.Nr * gp.Nphi;

    if (gid < face_size) {
        int i = gid % gp.Nr;
        int j = gid / gp.Nr;

        if (!mask_z0 || !mask_z0[gid])
            Er[i + gp.Nr * (j + gp.Nphi * 0)] = 0.0;
        if (!mask_zL || !mask_zL[gid])
            Er[i + gp.Nr * (j + gp.Nphi * gp.Nz)] = 0.0;
    }
}

__global__ void pec_Ephi_conductors_masked_kernel(
    double* Ephi, BndGridParams gp,
    const int* mask_inner, const int* mask_outer
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face_size = gp.Nphi * (gp.Nz + 1);

    if (gid < face_size) {
        int j = gid % gp.Nphi;
        int k = gid / gp.Nphi;
        int Nr1 = gp.Nr + 1;

        if (!mask_inner || !mask_inner[gid])
            Ephi[0 + Nr1 * (j + gp.Nphi * k)] = 0.0;
        if (!mask_outer || !mask_outer[gid])
            Ephi[gp.Nr + Nr1 * (j + gp.Nphi * k)] = 0.0;
    }
}

__global__ void pec_Ephi_endplates_masked_kernel(
    double* Ephi, BndGridParams gp,
    const int* mask_z0, const int* mask_zL
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face_size = (gp.Nr + 1) * gp.Nphi;

    if (gid < face_size) {
        int i = gid % (gp.Nr + 1);
        int j = gid / (gp.Nr + 1);
        int Nr1 = gp.Nr + 1;

        if (!mask_z0 || !mask_z0[gid])
            Ephi[i + Nr1 * (j + gp.Nphi * 0)] = 0.0;
        if (!mask_zL || !mask_zL[gid])
            Ephi[i + Nr1 * (j + gp.Nphi * gp.Nz)] = 0.0;
    }
}

__global__ void pec_Ez_conductors_masked_kernel(
    double* Ez, BndGridParams gp,
    const int* mask_inner, const int* mask_outer
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face_size = gp.Nphi * gp.Nz;

    if (gid < face_size) {
        int j = gid % gp.Nphi;
        int k = gid / gp.Nphi;
        int Nr1 = gp.Nr + 1;

        if (!mask_inner || !mask_inner[gid])
            Ez[0 + Nr1 * (j + gp.Nphi * k)] = 0.0;
        if (!mask_outer || !mask_outer[gid])
            Ez[gp.Nr + Nr1 * (j + gp.Nphi * k)] = 0.0;
    }
}

/*=============================================================================
 * IBC Surface Correction Kernels (REVISED)
 *
 * From the weak form:
 *   ∫(∇×E)·(∇×F) dV - (jωμ₀/Z_s) ∮ E_tan·F_tan dS = k² ∫E·F dV
 *
 * The discrete IBC operator is: y = ∇×∇×x + β·diag(w)·x
 *   β = (1-j)/(2α)
 *   w = 1/dr at cylindrical surfaces, 1/dz at endplates
 *
 * The curl-curl is computed WITHOUT PEC zeroing of E_tan.
 * The surface correction is ADDED to the result at wall points.
 *============================================================================*/

/* E_φ on cylindrical surface: result += β·w · E at (i_wall, j+½, k) */
__global__ void ibc_surface_Ephi_cyl_kernel(
    double* result_Ephi_re, double* result_Ephi_im,
    const double* E_Ephi_re, const double* E_Ephi_im,
    BndGridParams gp, double bw_re, double bw_im,
    int i_wall, const int* mask
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face_size = gp.Nphi * (gp.Nz + 1);
    if (gid >= face_size) return;
    if (mask && mask[gid]) return;

    int j = gid % gp.Nphi;
    int k = gid / gp.Nphi;
    int idx = i_wall + (gp.Nr + 1) * (j + gp.Nphi * k);

    double e_re = E_Ephi_re[idx], e_im = E_Ephi_im[idx];
    result_Ephi_re[idx] += bw_re * e_re - bw_im * e_im;
    result_Ephi_im[idx] += bw_re * e_im + bw_im * e_re;
}

/* E_z on cylindrical surface: result += β·w · E at (i_wall, j, k+½) */
__global__ void ibc_surface_Ez_cyl_kernel(
    double* result_Ez_re, double* result_Ez_im,
    const double* E_Ez_re, const double* E_Ez_im,
    BndGridParams gp, double bw_re, double bw_im,
    int i_wall, const int* mask
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face_size = gp.Nphi * gp.Nz;
    if (gid >= face_size) return;
    if (mask && mask[gid]) return;

    int j = gid % gp.Nphi;
    int k = gid / gp.Nphi;
    int idx = i_wall + (gp.Nr + 1) * (j + gp.Nphi * k);

    double e_re = E_Ez_re[idx], e_im = E_Ez_im[idx];
    result_Ez_re[idx] += bw_re * e_re - bw_im * e_im;
    result_Ez_im[idx] += bw_re * e_im + bw_im * e_re;
}

/* E_r on endplate: result += β·w · E at (i+½, j, k_wall) */
__global__ void ibc_surface_Er_end_kernel(
    double* result_Er_re, double* result_Er_im,
    const double* E_Er_re, const double* E_Er_im,
    BndGridParams gp, double bw_re, double bw_im,
    int k_wall, const int* mask
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face_size = gp.Nr * gp.Nphi;
    if (gid >= face_size) return;
    if (mask && mask[gid]) return;

    int i = gid % gp.Nr;
    int j = gid / gp.Nr;
    int idx = i + gp.Nr * (j + gp.Nphi * k_wall);

    double e_re = E_Er_re[idx], e_im = E_Er_im[idx];
    result_Er_re[idx] += bw_re * e_re - bw_im * e_im;
    result_Er_im[idx] += bw_re * e_im + bw_im * e_re;
}

/* E_φ on endplate: result += β·w · E at (i, j+½, k_wall) */
__global__ void ibc_surface_Ephi_end_kernel(
    double* result_Ephi_re, double* result_Ephi_im,
    const double* E_Ephi_re, const double* E_Ephi_im,
    BndGridParams gp, double bw_re, double bw_im,
    int k_wall, const int* mask
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int Nr1 = gp.Nr + 1;
    int face_size = Nr1 * gp.Nphi;
    if (gid >= face_size) return;
    if (mask && mask[gid]) return;

    int i = gid % Nr1;
    int j = gid / Nr1;
    int idx = i + Nr1 * (j + gp.Nphi * k_wall);

    double e_re = E_Ephi_re[idx], e_im = E_Ephi_im[idx];
    result_Ephi_re[idx] += bw_re * e_re - bw_im * e_im;
    result_Ephi_im[idx] += bw_re * e_im + bw_im * e_re;
}

/*=============================================================================
 * IBC Surface Correction Launcher
 *
 * Adds β·(dS/dV)·E_tan to result at all 4 conducting surfaces.
 * Called AFTER ∇×∇×E is computed (without PEC zeroing at IBC walls).
 *============================================================================*/
int gpu_apply_IBC_surface_correction(
    const GPU_EField* E,
    GPU_EField* result,
    double alpha,
    const GridParams* grid,
    const GPU_Operator* gpu_op
) {
    BndGridParams gp = make_bnd_params(grid);
    int has_masks = gpu_op->has_port_masks;

    double inv_2a = 1.0 / (2.0 * alpha);
    double bw_cyl_re =  inv_2a / grid->dr;
    double bw_cyl_im = -inv_2a / grid->dr;
    double bw_end_re =  inv_2a / grid->dz;
    double bw_end_im = -inv_2a / grid->dz;

    int face_Ephi_cyl = grid->Nphi * (grid->Nz + 1);
    int face_Ez_cyl   = grid->Nphi * grid->Nz;
    int face_Er_end   = grid->Nr * grid->Nphi;
    int face_Ephi_end = (grid->Nr + 1) * grid->Nphi;

    int b1 = (face_Ephi_cyl + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int b2 = (face_Ez_cyl   + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int b3 = (face_Er_end   + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int b4 = (face_Ephi_end + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Inner conductor (i=0) */
    ibc_surface_Ephi_cyl_kernel<<<b1, BLOCK_SIZE>>>(
        result->Ephi, result->Ephi_im, E->Ephi, E->Ephi_im,
        gp, bw_cyl_re, bw_cyl_im, 0,
        has_masks ? gpu_op->d_mask_Ephi_inner : NULL);
    ibc_surface_Ez_cyl_kernel<<<b2, BLOCK_SIZE>>>(
        result->Ez, result->Ez_im, E->Ez, E->Ez_im,
        gp, bw_cyl_re, bw_cyl_im, 0,
        has_masks ? gpu_op->d_mask_Ez_inner : NULL);

    /* Outer conductor (i=Nr) */
    ibc_surface_Ephi_cyl_kernel<<<b1, BLOCK_SIZE>>>(
        result->Ephi, result->Ephi_im, E->Ephi, E->Ephi_im,
        gp, bw_cyl_re, bw_cyl_im, grid->Nr,
        has_masks ? gpu_op->d_mask_Ephi_outer : NULL);
    ibc_surface_Ez_cyl_kernel<<<b2, BLOCK_SIZE>>>(
        result->Ez, result->Ez_im, E->Ez, E->Ez_im,
        gp, bw_cyl_re, bw_cyl_im, grid->Nr,
        has_masks ? gpu_op->d_mask_Ez_outer : NULL);

    /* Endplate z=0 (k=0) */
    ibc_surface_Er_end_kernel<<<b3, BLOCK_SIZE>>>(
        result->Er, result->Er_im, E->Er, E->Er_im,
        gp, bw_end_re, bw_end_im, 0,
        has_masks ? gpu_op->d_mask_Er_z0 : NULL);
    ibc_surface_Ephi_end_kernel<<<b4, BLOCK_SIZE>>>(
        result->Ephi, result->Ephi_im, E->Ephi, E->Ephi_im,
        gp, bw_end_re, bw_end_im, 0,
        has_masks ? gpu_op->d_mask_Ephi_z0 : NULL);

    /* Endplate z=L (k=Nz) */
    ibc_surface_Er_end_kernel<<<b3, BLOCK_SIZE>>>(
        result->Er, result->Er_im, E->Er, E->Er_im,
        gp, bw_end_re, bw_end_im, grid->Nz,
        has_masks ? gpu_op->d_mask_Er_zL : NULL);
    ibc_surface_Ephi_end_kernel<<<b4, BLOCK_SIZE>>>(
        result->Ephi, result->Ephi_im, E->Ephi, E->Ephi_im,
        gp, bw_end_re, bw_end_im, grid->Nz,
        has_masks ? gpu_op->d_mask_Ephi_zL : NULL);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/*=============================================================================
 * Build port masks on CPU, upload to GPU (UNCHANGED from original)
 *============================================================================*/
static int build_port_masks(GPU_Operator* gpu_op) {
    const CurlCurlOperator* op = gpu_op->cpu_op;
    const GridParams* grid = &op->grid;
    const PortConfig* ports = &op->ports;

    if (!op->has_ports || ports->num_ports == 0) {
        gpu_op->has_port_masks = 0;
        return 0;
    }

    int Nr = grid->Nr;
    int Nphi = grid->Nphi;
    int Nz = grid->Nz;

    /* Ephi on cylindrical conductors: Nphi * (Nz+1) */
    int cyl_size = Nphi * (Nz + 1);
    int* h_mask_Ephi_inner = (int*)calloc(cyl_size, sizeof(int));
    int* h_mask_Ephi_outer = (int*)calloc(cyl_size, sizeof(int));

    for (int k = 0; k <= Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            int idx = j + Nphi * k;

            if (point_in_port_cylindrical(ports, grid, SURFACE_INNER, phi, z))
                h_mask_Ephi_inner[idx] = 1;
            if (point_in_port_cylindrical(ports, grid, SURFACE_OUTER, phi, z))
                h_mask_Ephi_outer[idx] = 1;
        }
    }

    /* Ez on cylindrical conductors: Nphi * Nz */
    int ez_cyl_size = Nphi * Nz;
    int* h_mask_Ez_inner = (int*)calloc(ez_cyl_size, sizeof(int));
    int* h_mask_Ez_outer = (int*)calloc(ez_cyl_size, sizeof(int));

    for (int k = 0; k < Nz; k++) {
        double z = (k + 0.5) * grid->dz;
        for (int j = 0; j < Nphi; j++) {
            double phi = j * grid->dphi;
            int idx = j + Nphi * k;

            if (point_in_port_cylindrical(ports, grid, SURFACE_INNER, phi, z))
                h_mask_Ez_inner[idx] = 1;
            if (point_in_port_cylindrical(ports, grid, SURFACE_OUTER, phi, z))
                h_mask_Ez_outer[idx] = 1;
        }
    }

    /* Er on endplates: Nr * Nphi */
    int er_end_size = Nr * Nphi;
    int* h_mask_Er_z0 = (int*)calloc(er_end_size, sizeof(int));
    int* h_mask_Er_zL = (int*)calloc(er_end_size, sizeof(int));

    for (int j = 0; j < Nphi; j++) {
        double phi = (j + 0.5) * grid->dphi;
        for (int i = 0; i < Nr; i++) {
            double r = grid->a + (i + 0.5) * grid->dr;
            int idx = i + Nr * j;

            if (point_in_port_endplate(ports, grid, SURFACE_ENDPLATE_Z0, r, phi))
                h_mask_Er_z0[idx] = 1;
            if (point_in_port_endplate(ports, grid, SURFACE_ENDPLATE_ZL, r, phi))
                h_mask_Er_zL[idx] = 1;
        }
    }

    /* Ephi on endplates: (Nr+1) * Nphi */
    int ephi_end_size = (Nr + 1) * Nphi;
    int* h_mask_Ephi_z0 = (int*)calloc(ephi_end_size, sizeof(int));
    int* h_mask_Ephi_zL = (int*)calloc(ephi_end_size, sizeof(int));

    for (int j = 0; j < Nphi; j++) {
        double phi = (j + 0.5) * grid->dphi;
        for (int i = 0; i <= Nr; i++) {
            double r = grid->a + i * grid->dr;
            int idx = i + (Nr + 1) * j;

            if (point_in_port_endplate(ports, grid, SURFACE_ENDPLATE_Z0, r, phi))
                h_mask_Ephi_z0[idx] = 1;
            if (point_in_port_endplate(ports, grid, SURFACE_ENDPLATE_ZL, r, phi))
                h_mask_Ephi_zL[idx] = 1;
        }
    }

    /* Upload to GPU */
    CUDA_CHECK(cudaMalloc(&gpu_op->d_mask_Ephi_inner, cyl_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_op->d_mask_Ephi_inner, h_mask_Ephi_inner,
        cyl_size * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu_op->d_mask_Ephi_outer, cyl_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_op->d_mask_Ephi_outer, h_mask_Ephi_outer,
        cyl_size * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu_op->d_mask_Ez_inner, ez_cyl_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_op->d_mask_Ez_inner, h_mask_Ez_inner,
        ez_cyl_size * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu_op->d_mask_Ez_outer, ez_cyl_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_op->d_mask_Ez_outer, h_mask_Ez_outer,
        ez_cyl_size * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu_op->d_mask_Er_z0, er_end_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_op->d_mask_Er_z0, h_mask_Er_z0,
        er_end_size * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu_op->d_mask_Er_zL, er_end_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_op->d_mask_Er_zL, h_mask_Er_zL,
        er_end_size * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu_op->d_mask_Ephi_z0, ephi_end_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_op->d_mask_Ephi_z0, h_mask_Ephi_z0,
        ephi_end_size * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu_op->d_mask_Ephi_zL, ephi_end_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_op->d_mask_Ephi_zL, h_mask_Ephi_zL,
        ephi_end_size * sizeof(int), cudaMemcpyHostToDevice));

    /* Count free points */
    int total_free = 0;
    for (int i = 0; i < cyl_size; i++) total_free += h_mask_Ephi_inner[i];
    for (int i = 0; i < cyl_size; i++) total_free += h_mask_Ephi_outer[i];
    for (int i = 0; i < ez_cyl_size; i++) total_free += h_mask_Ez_inner[i];
    for (int i = 0; i < ez_cyl_size; i++) total_free += h_mask_Ez_outer[i];
    for (int i = 0; i < er_end_size; i++) total_free += h_mask_Er_z0[i];
    for (int i = 0; i < er_end_size; i++) total_free += h_mask_Er_zL[i];
    for (int i = 0; i < ephi_end_size; i++) total_free += h_mask_Ephi_z0[i];
    for (int i = 0; i < ephi_end_size; i++) total_free += h_mask_Ephi_zL[i];

    printf("  Port masks: %d free boundary points (ports)\n", total_free);

    /* Free CPU masks */
    free(h_mask_Ephi_inner);
    free(h_mask_Ephi_outer);
    free(h_mask_Ez_inner);
    free(h_mask_Ez_outer);
    free(h_mask_Er_z0);
    free(h_mask_Er_zL);
    free(h_mask_Ephi_z0);
    free(h_mask_Ephi_zL);

    gpu_op->has_port_masks = 1;
    return 0;
}

/*=============================================================================
 * GPU Operator Lifecycle (UNCHANGED)
 *============================================================================*/

int gpu_operator_init(GPU_Operator* gpu_op, const CurlCurlOperator* cpu_op) {
    const GridParams* grid = &cpu_op->grid;

    gpu_op->cpu_op = cpu_op;
    gpu_op->initialized = 0;
    gpu_op->has_port_masks = 0;
    gpu_op->use_ibc = 0;

    /* Initialize null pointers for masks */
    gpu_op->d_mask_Ephi_inner = NULL;
    gpu_op->d_mask_Ephi_outer = NULL;
    gpu_op->d_mask_Ez_inner = NULL;
    gpu_op->d_mask_Ez_outer = NULL;
    gpu_op->d_mask_Er_z0 = NULL;
    gpu_op->d_mask_Er_zL = NULL;
    gpu_op->d_mask_Ephi_z0 = NULL;
    gpu_op->d_mask_Ephi_zL = NULL;

    /* Allocate GPU working arrays — real only (PEC mode) */
    if (gpu_efield_alloc(&gpu_op->d_E_work, grid) != 0) return -1;
    if (gpu_efield_alloc(&gpu_op->d_result_work, grid) != 0) return -1;
    if (gpu_hfield_alloc(&gpu_op->d_H_temp, grid) != 0) return -1;

    /* Allocate reduction workspace */
    if (reduction_workspace_init(&gpu_op->reduction_ws, cpu_op->n_total) != 0) return -1;

    /* Build port masks if needed */
    if (cpu_op->has_ports) {
        if (build_port_masks(gpu_op) != 0) return -1;
    }

    /* Upload grid to constant memory */
    cuda_grid_init(grid);

    gpu_op->initialized = 1;
    return 0;
}

/*=============================================================================
 * Initialize GPU Operator — IBC mode (complex working arrays)
 *============================================================================*/

int gpu_operator_init_complex(GPU_Operator* gpu_op, const CurlCurlOperator* cpu_op) {
    const GridParams* grid = &cpu_op->grid;

    gpu_op->cpu_op = cpu_op;
    gpu_op->initialized = 0;
    gpu_op->has_port_masks = 0;
    gpu_op->use_ibc = 1;

    /* Initialize null pointers for masks */
    gpu_op->d_mask_Ephi_inner = NULL;
    gpu_op->d_mask_Ephi_outer = NULL;
    gpu_op->d_mask_Ez_inner = NULL;
    gpu_op->d_mask_Ez_outer = NULL;
    gpu_op->d_mask_Er_z0 = NULL;
    gpu_op->d_mask_Er_zL = NULL;
    gpu_op->d_mask_Ephi_z0 = NULL;
    gpu_op->d_mask_Ephi_zL = NULL;

    /* Allocate GPU working arrays — COMPLEX (IBC mode) */
    if (gpu_efield_alloc_complex(&gpu_op->d_E_work, grid) != 0) return -1;
    if (gpu_efield_alloc_complex(&gpu_op->d_result_work, grid) != 0) return -1;
    if (gpu_hfield_alloc_complex(&gpu_op->d_H_temp, grid) != 0) return -1;

    /* Allocate reduction workspace (sized for n_total, used per-half) */
    if (reduction_workspace_init(&gpu_op->reduction_ws, cpu_op->n_total) != 0) return -1;

    /* Build port masks if needed */
    if (cpu_op->has_ports) {
        if (build_port_masks(gpu_op) != 0) return -1;
    }

    /* Upload grid to constant memory */
    cuda_grid_init(grid);

    printf("  GPU Operator initialized in IBC (complex) mode\n");
    printf("  Working arrays: 6 E-field + 6 H-field arrays (re+im)\n");

    gpu_op->initialized = 1;
    return 0;
}

void gpu_operator_free(GPU_Operator* gpu_op) {
    if (!gpu_op->initialized) return;

    gpu_efield_free(&gpu_op->d_E_work);
    gpu_efield_free(&gpu_op->d_result_work);
    gpu_hfield_free(&gpu_op->d_H_temp);
    reduction_workspace_free(&gpu_op->reduction_ws);

    if (gpu_op->has_port_masks) {
        if (gpu_op->d_mask_Ephi_inner) cudaFree(gpu_op->d_mask_Ephi_inner);
        if (gpu_op->d_mask_Ephi_outer) cudaFree(gpu_op->d_mask_Ephi_outer);
        if (gpu_op->d_mask_Ez_inner)   cudaFree(gpu_op->d_mask_Ez_inner);
        if (gpu_op->d_mask_Ez_outer)   cudaFree(gpu_op->d_mask_Ez_outer);
        if (gpu_op->d_mask_Er_z0)      cudaFree(gpu_op->d_mask_Er_z0);
        if (gpu_op->d_mask_Er_zL)      cudaFree(gpu_op->d_mask_Er_zL);
        if (gpu_op->d_mask_Ephi_z0)    cudaFree(gpu_op->d_mask_Ephi_z0);
        if (gpu_op->d_mask_Ephi_zL)    cudaFree(gpu_op->d_mask_Ephi_zL);
    }

    gpu_op->initialized = 0;
}

/*=============================================================================
 * Apply PEC boundary (no ports) — UNCHANGED
 *============================================================================*/

int gpu_apply_PEC_boundary(GPU_EField* E, const GridParams* grid) {
    BndGridParams gp = make_bnd_params(grid);

    int face_Er = grid->Nr * grid->Nphi;
    int face_Ephi_cyl = grid->Nphi * (grid->Nz + 1);
    int face_Ephi_end = (grid->Nr + 1) * grid->Nphi;
    int face_Ez = grid->Nphi * grid->Nz;

    pec_Er_endplates_kernel<<<(face_Er + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Er, gp);
    pec_Ephi_conductors_kernel<<<(face_Ephi_cyl + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Ephi, gp);
    pec_Ephi_endplates_kernel<<<(face_Ephi_end + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Ephi, gp);
    pec_Ez_conductors_kernel<<<(face_Ez + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Ez, gp);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/*=============================================================================
 * Apply PEC boundary with port masks — UNCHANGED
 *============================================================================*/

int gpu_apply_PEC_boundary_with_masks(
    GPU_EField* E,
    const GridParams* grid,
    const GPU_Operator* gpu_op
) {
    BndGridParams gp = make_bnd_params(grid);

    int face_Er_end = grid->Nr * grid->Nphi;
    int face_Ephi_cyl = grid->Nphi * (grid->Nz + 1);
    int face_Ephi_end = (grid->Nr + 1) * grid->Nphi;
    int face_Ez_cyl = grid->Nphi * grid->Nz;

    pec_Er_endplates_masked_kernel<<<(face_Er_end + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Er, gp, gpu_op->d_mask_Er_z0, gpu_op->d_mask_Er_zL);

    pec_Ephi_conductors_masked_kernel<<<(face_Ephi_cyl + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Ephi, gp, gpu_op->d_mask_Ephi_inner, gpu_op->d_mask_Ephi_outer);

    pec_Ephi_endplates_masked_kernel<<<(face_Ephi_end + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Ephi, gp, gpu_op->d_mask_Ephi_z0, gpu_op->d_mask_Ephi_zL);

    pec_Ez_conductors_masked_kernel<<<(face_Ez_cyl + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Ez, gp, gpu_op->d_mask_Ez_inner, gpu_op->d_mask_Ez_outer);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/*=============================================================================
 * Apply boundary (auto-select PEC) — UNCHANGED
 *============================================================================*/

static int apply_boundary(GPU_EField* E, const GridParams* grid,
    const GPU_Operator* gpu_op) {
    if (gpu_op->has_port_masks) {
        return gpu_apply_PEC_boundary_with_masks(E, grid, gpu_op);
    }
    else {
        return gpu_apply_PEC_boundary(E, grid);
    }
}

/*=============================================================================
 * Core: y = A * x (GPU matvec) — PEC version (UNCHANGED)
 *============================================================================*/

int gpu_curlcurl_matvec(
    const GPU_Operator* gpu_op,
    const double* d_x,
    double* d_y
) {
    GPU_Operator* op = (GPU_Operator*)gpu_op;
    const CurlCurlOperator* cpu_op = gpu_op->cpu_op;
    const GridParams* grid = &cpu_op->grid;

    /* Step 1: Unpack */
    gpu_unpack_field(d_x, &op->d_E_work,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez);

    /* Step 2: Apply BC to input */
    apply_boundary(&op->d_E_work, grid, gpu_op);

    /* Step 3: Curl-curl */
    gpu_compute_curl_curl_E(&op->d_E_work, &op->d_result_work,
        &op->d_H_temp, grid);

    /* Step 4: Apply BC to result */
    apply_boundary(&op->d_result_work, grid, gpu_op);

    /* Step 5: Pack */
    gpu_pack_field(&op->d_result_work, d_y,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez);

    return 0;
}

/*=============================================================================
 * IBC Complex Matvec: y = (∇×∇× + β·S) · x
 *
 * Surface correction approach:
 *   1. Unpack x → E (re + im), do NOT zero E_tan at walls
 *   2. ∇×∇×E → result (complex curl-curl, 1 curl-E + 1 curl-H)
 *   3. result += β·diag(w)·E at wall boundaries (surface correction)
 *   4. Pack result → y
 *
 * Cost: 1 curl-E + 1 curl-H + 8 surface correction kernels
 * (MUCH cheaper than the old boundary-value approach: 3 curl-E + 1 curl-H)
 *============================================================================*/

int gpu_curlcurl_matvec_complex(
    const GPU_Operator* gpu_op,
    const double* d_x,
    double* d_y,
    double alpha
) {
    GPU_Operator* op = (GPU_Operator*)gpu_op;
    const CurlCurlOperator* cpu_op = gpu_op->cpu_op;
    const GridParams* grid = &cpu_op->grid;
    int n_real = cpu_op->n_total;

    /* Step 1: Unpack x → E_work (real + imaginary)
     * NO PEC boundary — E_tan at walls is nonzero for IBC */
    gpu_unpack_field_complex(d_x, &op->d_E_work,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez,
        n_real);

    /* Step 2: ∇×∇×E (complex) */
    gpu_compute_curl_curl_E_complex(&op->d_E_work, &op->d_result_work,
        &op->d_H_temp, grid);

    /* Step 3: Add IBC surface correction: result += β·w·E at walls */
    gpu_apply_IBC_surface_correction(&op->d_E_work, &op->d_result_work,
        alpha, grid, gpu_op);

    /* Step 4: Pack result → y */
    gpu_pack_field_complex(&op->d_result_work, d_y,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez,
        n_real);

    return 0;
}

/*=============================================================================
 * GPU Rayleigh quotient — UNCHANGED
 *============================================================================*/

int gpu_rayleigh_quotient(
    const GPU_Operator* gpu_op,
    const double* d_x,
    double* result
) {
    const CurlCurlOperator* cpu_op = gpu_op->cpu_op;
    int n = cpu_op->n_total;

    double* d_Ax;
    CUDA_CHECK(cudaMalloc(&d_Ax, n * sizeof(double)));

    gpu_curlcurl_matvec(gpu_op, d_x, d_Ax);

    double xAx;
    gpu_vec_dot_weighted_ws(d_x, d_Ax, &xAx, cpu_op,
        &((GPU_Operator*)gpu_op)->reduction_ws);

    double xx;
    gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op,
        &((GPU_Operator*)gpu_op)->reduction_ws);

    *result = xAx / xx;

    cudaFree(d_Ax);
    return 0;
}
