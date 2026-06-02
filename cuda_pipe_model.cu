#include "cuda_pipe_model.h"
#include "cuda_eigensolver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

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
 * Forward declarations for endplate PEC kernels
 *============================================================================*/

__global__ void pec_Er_at_k_kernel(double* Er, int Nr, int Nphi, int k);
__global__ void pec_Ephi_at_k_kernel(double* Ephi, int Nr1, int Nphi, int k);
__global__ void pec_Er_at_k_masked_kernel(
    double* Er, int Nr, int Nphi, int k, const int* mask);
__global__ void pec_Ephi_at_k_masked_kernel(
    double* Ephi, int Nr1, int Nphi, int k, const int* mask);

/*=============================================================================
 * Mask kernel: zero out E where mask = 0 (PEC)
 *============================================================================*/

__global__ void apply_mask_kernel(double* field, const int* mask, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (!mask[idx]) {
            field[idx] = 0.0;
        }
    }
}

/*=============================================================================
 * GPU Material Mask (unchanged)
 *============================================================================*/

int gpu_material_mask_init(GPU_MaterialMask* gpu_mask, const MaterialMask* cpu_mask) {
    gpu_mask->size_Er = cpu_mask->size_Er;
    gpu_mask->size_Ephi = cpu_mask->size_Ephi;
    gpu_mask->size_Ez = cpu_mask->size_Ez;

    CUDA_CHECK(cudaMalloc(&gpu_mask->d_mask_Er,
        gpu_mask->size_Er * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_mask->d_mask_Ephi,
        gpu_mask->size_Ephi * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_mask->d_mask_Ez,
        gpu_mask->size_Ez * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(gpu_mask->d_mask_Er, cpu_mask->mask_Er,
        gpu_mask->size_Er * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_mask->d_mask_Ephi, cpu_mask->mask_Ephi,
        gpu_mask->size_Ephi * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_mask->d_mask_Ez, cpu_mask->mask_Ez,
        gpu_mask->size_Ez * sizeof(int), cudaMemcpyHostToDevice));

    return 0;
}

void gpu_material_mask_free(GPU_MaterialMask* gpu_mask) {
    if (gpu_mask->d_mask_Er)   { cudaFree(gpu_mask->d_mask_Er);   gpu_mask->d_mask_Er = NULL; }
    if (gpu_mask->d_mask_Ephi) { cudaFree(gpu_mask->d_mask_Ephi); gpu_mask->d_mask_Ephi = NULL; }
    if (gpu_mask->d_mask_Ez)   { cudaFree(gpu_mask->d_mask_Ez);   gpu_mask->d_mask_Ez = NULL; }
}

/*=============================================================================
 * Apply material mask (real only, unchanged)
 *============================================================================*/

int gpu_apply_material_mask(GPU_EField* E, const GPU_MaterialMask* mask) {
    int blocks_Er   = (mask->size_Er   + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_Ephi = (mask->size_Ephi + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_Ez   = (mask->size_Ez   + BLOCK_SIZE - 1) / BLOCK_SIZE;

    apply_mask_kernel<<<blocks_Er, BLOCK_SIZE>>>(E->Er, mask->d_mask_Er, mask->size_Er);
    apply_mask_kernel<<<blocks_Ephi, BLOCK_SIZE>>>(E->Ephi, mask->d_mask_Ephi, mask->size_Ephi);
    apply_mask_kernel<<<blocks_Ez, BLOCK_SIZE>>>(E->Ez, mask->d_mask_Ez, mask->size_Ez);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/*=============================================================================
 * Apply material mask — complex (both re+im)
 *============================================================================*/

int gpu_apply_material_mask_complex(GPU_EField* E, const GPU_MaterialMask* mask) {
    int b1 = (mask->size_Er   + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int b2 = (mask->size_Ephi + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int b3 = (mask->size_Ez   + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Real */
    apply_mask_kernel<<<b1, BLOCK_SIZE>>>(E->Er,   mask->d_mask_Er,   mask->size_Er);
    apply_mask_kernel<<<b2, BLOCK_SIZE>>>(E->Ephi, mask->d_mask_Ephi, mask->size_Ephi);
    apply_mask_kernel<<<b3, BLOCK_SIZE>>>(E->Ez,   mask->d_mask_Ez,   mask->size_Ez);
    /* Imaginary */
    apply_mask_kernel<<<b1, BLOCK_SIZE>>>(E->Er_im,   mask->d_mask_Er,   mask->size_Er);
    apply_mask_kernel<<<b2, BLOCK_SIZE>>>(E->Ephi_im, mask->d_mask_Ephi, mask->size_Ephi);
    apply_mask_kernel<<<b3, BLOCK_SIZE>>>(E->Ez_im,   mask->d_mask_Ez,   mask->size_Ez);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/*=============================================================================
 * IBC Boundary Weight Precomputation
 *
 * For each vacuum E cell with a PEC neighbor in a direction where E is
 * tangential to the implied wall:
 *
 *   Er (tangential to φ̂ and ẑ walls):
 *     PEC neighbor in j±1 → weight += 1/(r·dφ)
 *     PEC neighbor in k±1 → weight += 1/dz
 *
 *   Ephi (tangential to r̂ and ẑ walls):
 *     PEC neighbor in i±1 → weight += 1/dr
 *     PEC neighbor in k±1 → weight += 1/dz
 *
 *   Ez (tangential to r̂ and φ̂ walls):
 *     PEC neighbor in i±1 → weight += 1/dr
 *     PEC neighbor in j±1 → weight += 1/(r·dφ)
 *============================================================================*/

int gpu_ibc_weights_build(
    GPU_IBCWeights* w,
    const MaterialMask* pec_mask,
    const MaterialMask* ibc_mask,
    const GridParams* grid
) {
    int Nr = grid->Nr, Nphi = grid->Nphi, Nz = grid->Nz;
    int Nr1 = Nr + 1;
    double dr = grid->dr, dphi = grid->dphi, dz = grid->dz;

    w->size_Er = ibc_mask->size_Er;
    w->size_Ephi = ibc_mask->size_Ephi;
    w->size_Ez = ibc_mask->size_Ez;

    double* h_w_Er = (double*)calloc(w->size_Er, sizeof(double));
    double* h_w_Ephi = (double*)calloc(w->size_Ephi, sizeof(double));
    double* h_w_Ez = (double*)calloc(w->size_Ez, sizeof(double));

    w->num_boundary_Er = 0;
    w->num_boundary_Ephi = 0;
    w->num_boundary_Ez = 0;

    /* Er(i+½, j, k): surface cell if pec=0, ibc=1.
     * Weight from vacuum neighbors in φ and z directions. */
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr; i++) {
                int idx = i + Nr * (j + Nphi * k);
                /* Only act on flipped cells: was PEC, now IBC surface */
                if (pec_mask->mask_Er[idx] != 0) continue;  /* was vacuum, skip */
                if (ibc_mask->mask_Er[idx] != 1) continue;  /* not flipped, skip */

                double wt = 0.0;
                double r = grid->a + (i + 0.5) * dr;
                double inv_rdphi = 1.0 / (r * dphi);
                double inv_dz = 1.0 / dz;

                int jm1 = (j - 1 + Nphi) % Nphi;
                int jp1 = (j + 1) % Nphi;
                if (pec_mask->mask_Er[i + Nr * (jm1 + Nphi * k)]) wt += inv_rdphi;
                if (pec_mask->mask_Er[i + Nr * (jp1 + Nphi * k)]) wt += inv_rdphi;
                if (k > 0 && pec_mask->mask_Er[i + Nr * (j + Nphi * (k - 1))]) wt += inv_dz;
                if (k < Nz && pec_mask->mask_Er[i + Nr * (j + Nphi * (k + 1))]) wt += inv_dz;

                if (wt > 0.0) { h_w_Er[idx] = wt; w->num_boundary_Er++; }
            }
        }
    }

    /* Ephi(i, j+½, k): surface cell if pec=0, ibc=1.
     * Weight from vacuum neighbors in r and z directions. */
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i <= Nr; i++) {
                int idx = i + Nr1 * (j + Nphi * k);
                if (pec_mask->mask_Ephi[idx] != 0) continue;
                if (ibc_mask->mask_Ephi[idx] != 1) continue;

                double wt = 0.0;
                double inv_dr = 1.0 / dr;
                double inv_dz = 1.0 / dz;

                if (i > 0 && pec_mask->mask_Ephi[(i - 1) + Nr1 * (j + Nphi * k)]) wt += inv_dr;
                if (i < Nr && pec_mask->mask_Ephi[(i + 1) + Nr1 * (j + Nphi * k)]) wt += inv_dr;
                if (k > 0 && pec_mask->mask_Ephi[i + Nr1 * (j + Nphi * (k - 1))]) wt += inv_dz;
                if (k < Nz && pec_mask->mask_Ephi[i + Nr1 * (j + Nphi * (k + 1))]) wt += inv_dz;

                if (wt > 0.0) { h_w_Ephi[idx] = wt; w->num_boundary_Ephi++; }
            }
        }
    }

    /* Ez(i, j, k+½): surface cell if pec=0, ibc=1.
     * Weight from vacuum neighbors in r and φ directions. */
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i <= Nr; i++) {
                int idx = i + Nr1 * (j + Nphi * k);
                if (pec_mask->mask_Ez[idx] != 0) continue;
                if (ibc_mask->mask_Ez[idx] != 1) continue;

                double wt = 0.0;
                double r = grid->a + i * dr;
                double inv_dr = 1.0 / dr;
                double inv_rdphi = (r > 1e-14) ? 1.0 / (r * dphi) : 0.0;

                if (i > 0 && pec_mask->mask_Ez[(i - 1) + Nr1 * (j + Nphi * k)]) wt += inv_dr;
                if (i < Nr && pec_mask->mask_Ez[(i + 1) + Nr1 * (j + Nphi * k)]) wt += inv_dr;
                int jm1 = (j - 1 + Nphi) % Nphi;
                int jp1 = (j + 1) % Nphi;
                if (pec_mask->mask_Ez[i + Nr1 * (jm1 + Nphi * k)]) wt += inv_rdphi;
                if (pec_mask->mask_Ez[i + Nr1 * (jp1 + Nphi * k)]) wt += inv_rdphi;

                if (wt > 0.0) { h_w_Ez[idx] = wt; w->num_boundary_Ez++; }
            }
        }
    }

    printf("  IBC boundary weights (on surface cells):\n");
    printf("    Er:   %d boundary cells\n", w->num_boundary_Er);
    printf("    Ephi: %d boundary cells\n", w->num_boundary_Ephi);
    printf("    Ez:   %d boundary cells\n", w->num_boundary_Ez);
    printf("    Total: %d\n",
        w->num_boundary_Er + w->num_boundary_Ephi + w->num_boundary_Ez);

    /* Upload to GPU — same as before */
    CUDA_CHECK(cudaMalloc(&w->d_weight_Er, w->size_Er * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(w->d_weight_Er, h_w_Er,
        w->size_Er * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&w->d_weight_Ephi, w->size_Ephi * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(w->d_weight_Ephi, h_w_Ephi,
        w->size_Ephi * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&w->d_weight_Ez, w->size_Ez * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(w->d_weight_Ez, h_w_Ez,
        w->size_Ez * sizeof(double), cudaMemcpyHostToDevice));

    free(h_w_Er); free(h_w_Ephi); free(h_w_Ez);
    return 0;
}

void gpu_ibc_weights_free(GPU_IBCWeights* w) {
    if (w->d_weight_Er)   { cudaFree(w->d_weight_Er);   w->d_weight_Er = NULL; }
    if (w->d_weight_Ephi) { cudaFree(w->d_weight_Ephi); w->d_weight_Ephi = NULL; }
    if (w->d_weight_Ez)   { cudaFree(w->d_weight_Ez);   w->d_weight_Ez = NULL; }
}

/*=============================================================================
 * IBC Boundary Weight Kernel
 *
 * For each cell: result += (β_re + j·β_im) · weight · (E_re + j·E_im)
 * Only acts where weight > 0 (boundary cells).
 *============================================================================*/

__global__ void ibc_weight_kernel(
    double* result_re, double* result_im,
    const double* E_re, const double* E_im,
    const double* weight,
    double beta_re, double beta_im,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double w = weight[idx];
        if (w > 0.0) {
            double bw_re = beta_re * w;
            double bw_im = beta_im * w;
            double e_re = E_re[idx];
            double e_im = E_im[idx];
            result_re[idx] += bw_re * e_re - bw_im * e_im;
            result_im[idx] += bw_re * e_im + bw_im * e_re;
        }
    }
}

/*=============================================================================
 * Pipe Cap PEC Kernels (IBC mode)
 *
 * In IBC mode, we only zero E at pipe terminations (grid edges), NOT at
 * the inner conductor (i=0). So we need kernels that zero only at i=Nr
 * (not i=0) for Ephi and Ez.
 *============================================================================*/

/* Ephi = 0 at i=Nr only (outer pipe cap), both re+im */
__global__ void pec_Ephi_outer_cap_kernel(
    double* Ephi_re, double* Ephi_im,
    int Nr, int Nphi, int Nz
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face = Nphi * (Nz + 1);
    if (gid < face) {
        int j = gid % Nphi;
        int k = gid / Nphi;
        int Nr1 = Nr + 1;
        int idx = Nr + Nr1 * (j + Nphi * k);  /* i=Nr */
        Ephi_re[idx] = 0.0;
        Ephi_im[idx] = 0.0;
    }
}

/* Ez = 0 at i=Nr only (outer pipe cap), both re+im */
__global__ void pec_Ez_outer_cap_kernel(
    double* Ez_re, double* Ez_im,
    int Nr, int Nphi, int Nz
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face = Nphi * Nz;
    if (gid < face) {
        int j = gid % Nphi;
        int k = gid / Nphi;
        int Nr1 = Nr + 1;
        int idx = Nr + Nr1 * (j + Nphi * k);  /* i=Nr */
        Ez_re[idx] = 0.0;
        Ez_im[idx] = 0.0;
    }
}

/* Er = 0 at k=0 and k=Nz (endplate pipe caps), both re+im */
__global__ void pec_Er_endcaps_complex_kernel(
    double* Er_re, double* Er_im,
    int Nr, int Nphi, int Nz
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face = Nr * Nphi;
    if (gid < face) {
        int i = gid % Nr;
        int j = gid / Nr;
        int idx0  = i + Nr * (j + Nphi * 0);
        int idxNz = i + Nr * (j + Nphi * Nz);
        Er_re[idx0]  = 0.0;  Er_im[idx0]  = 0.0;
        Er_re[idxNz] = 0.0;  Er_im[idxNz] = 0.0;
    }
}

/* Ephi = 0 at k=0 and k=Nz (endplate pipe caps), both re+im */
__global__ void pec_Ephi_endcaps_complex_kernel(
    double* Ephi_re, double* Ephi_im,
    int Nr1, int Nphi, int Nz
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face = Nr1 * Nphi;
    if (gid < face) {
        int i = gid % Nr1;
        int j = gid / Nr1;
        int idx0  = i + Nr1 * (j + Nphi * 0);
        int idxNz = i + Nr1 * (j + Nphi * Nz);
        Ephi_re[idx0]  = 0.0;  Ephi_im[idx0]  = 0.0;
        Ephi_re[idxNz] = 0.0;  Ephi_im[idxNz] = 0.0;
    }
}

/*=============================================================================
 * Inner Conductor IBC Kernel (i=0)
 *
 * The inner conductor is at the grid edge (i=0) and is NOT detected by
 * the boundary weight computation (no i=-1 cell). So we handle it
 * explicitly with the flat-wall surface correction: result += β/dr · E.
 *
 * Applies to both Ephi(i=0) and Ez(i=0), with port mask support.
 *============================================================================*/

/* Ephi IBC at i=0: face_size = Nphi*(Nz+1) */
__global__ void ibc_inner_Ephi_kernel(
    double* result_re, double* result_im,
    const double* E_re, const double* E_im,
    int Nr1, int Nphi, int Nz,
    double bw_re, double bw_im,
    const int* mask   /* port mask for Ephi_inner, NULL if no ports */
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face = Nphi * (Nz + 1);
    if (gid >= face) return;
    if (mask && mask[gid]) return;  /* skip port aperture */

    int j = gid % Nphi;
    int k = gid / Nphi;
    int idx = 0 + Nr1 * (j + Nphi * k);  /* i=0 */

    double e_re = E_re[idx], e_im = E_im[idx];
    result_re[idx] += bw_re * e_re - bw_im * e_im;
    result_im[idx] += bw_re * e_im + bw_im * e_re;
}

/* Ez IBC at i=0: face_size = Nphi*Nz */
__global__ void ibc_inner_Ez_kernel(
    double* result_re, double* result_im,
    const double* E_re, const double* E_im,
    int Nr1, int Nphi, int Nz,
    double bw_re, double bw_im,
    const int* mask   /* port mask for Ez_inner, NULL if no ports */
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int face = Nphi * Nz;
    if (gid >= face) return;
    if (mask && mask[gid]) return;

    int j = gid % Nphi;
    int k = gid / Nphi;
    int idx = 0 + Nr1 * (j + Nphi * k);  /* i=0 */

    double e_re = E_re[idx], e_im = E_im[idx];
    result_re[idx] += bw_re * e_re - bw_im * e_im;
    result_im[idx] += bw_re * e_im + bw_im * e_re;
}

/*=============================================================================
 * GPU Pipe Operator — PEC mode init (unchanged)
 *============================================================================*/

int gpu_pipe_operator_init(
    GPU_PipeOperator* pipe_op,
    const CurlCurlOperator* cpu_op,
    const MaterialMask* mask
) {
    if (gpu_operator_init(&pipe_op->base, cpu_op) != 0) return -1;

    if (mask) {
        if (gpu_material_mask_init(&pipe_op->mask, mask) != 0) return -1;
        pipe_op->has_mask = 1;
    } else {
        pipe_op->has_mask = 0;
    }

    pipe_op->k_endplate_z0 = -1;
    pipe_op->k_endplate_zL = -1;
    pipe_op->d_mask_Er_endplate_z0 = NULL;
    pipe_op->d_mask_Er_endplate_zL = NULL;
    pipe_op->d_mask_Ephi_endplate_z0 = NULL;
    pipe_op->d_mask_Ephi_endplate_zL = NULL;
    pipe_op->has_interior_endplates = 0;

    pipe_op->has_ibc_weights = 0;
    pipe_op->ibc_weights.d_weight_Er = NULL;
    pipe_op->ibc_weights.d_weight_Ephi = NULL;
    pipe_op->ibc_weights.d_weight_Ez = NULL;

    return 0;
}

/*=============================================================================
 * GPU Pipe Operator — IBC complex mode init
 *============================================================================*/

int gpu_pipe_operator_init_complex(
    GPU_PipeOperator* pipe_op,
    const CurlCurlOperator* cpu_op,
    const MaterialMask* mask
) {
    /* Initialize base with complex working arrays */
    if (gpu_operator_init_complex(&pipe_op->base, cpu_op) != 0) return -1;

    if (mask) {
        if (gpu_material_mask_init(&pipe_op->mask, mask) != 0) return -1;
        pipe_op->has_mask = 1;

        /* Build IBC mask: unmask surface PEC cells */
        MaterialMask ibc_mask_cpu;
        material_mask_build_ibc(&ibc_mask_cpu, mask, &cpu_op->grid);

        /* Upload IBC mask to GPU */
        if (gpu_material_mask_init(&pipe_op->ibc_mask, &ibc_mask_cpu) != 0) return -1;
        pipe_op->has_ibc_mask = 1;

        /* Build IBC weights on the SURFACE cells */
        if (gpu_ibc_weights_build(&pipe_op->ibc_weights, mask, &ibc_mask_cpu,
            &cpu_op->grid) != 0) return -1;
        pipe_op->has_ibc_weights = 1;

        material_mask_free(&ibc_mask_cpu);
    } else {
        pipe_op->has_mask = 0;
        pipe_op->has_ibc_mask = 0;
        pipe_op->has_ibc_weights = 0;
        pipe_op->ibc_weights.d_weight_Er = NULL;
        pipe_op->ibc_weights.d_weight_Ephi = NULL;
        pipe_op->ibc_weights.d_weight_Ez = NULL;
    }

    pipe_op->k_endplate_z0 = -1;
    pipe_op->k_endplate_zL = -1;
    pipe_op->d_mask_Er_endplate_z0 = NULL;
    pipe_op->d_mask_Er_endplate_zL = NULL;
    pipe_op->d_mask_Ephi_endplate_z0 = NULL;
    pipe_op->d_mask_Ephi_endplate_zL = NULL;
    pipe_op->has_interior_endplates = 0;

    return 0;
}

/*=============================================================================
 * Set endplates (unchanged — used for PEC mode)
 *============================================================================*/

int gpu_pipe_operator_set_endplates(
    GPU_PipeOperator* pipe_op,
    int k_z0, int k_zL,
    const EndcapPipeConfig* endcap_pipes,
    const GridParams* grid,
    double z0_extension
) {
    int Nr = grid->Nr;
    int Nphi = grid->Nphi;

    pipe_op->k_endplate_z0 = k_z0;
    pipe_op->k_endplate_zL = k_zL;
    pipe_op->d_mask_Er_endplate_z0 = NULL;
    pipe_op->d_mask_Er_endplate_zL = NULL;
    pipe_op->d_mask_Ephi_endplate_z0 = NULL;
    pipe_op->d_mask_Ephi_endplate_zL = NULL;
    pipe_op->has_interior_endplates = (k_z0 >= 0 || k_zL >= 0);

    if (!endcap_pipes || endcap_pipes->num_pipes == 0) {
        printf("  Interior endplates: k_z0=%d, k_zL=%d (full PEC)\n", k_z0, k_zL);
        return 0;
    }

    /* Build hole masks — same code as original, abbreviated for space */
    /* Er mask at z0 */
    if (k_z0 >= 0) {
        int er_size = Nr * Nphi;
        int* h_mask = (int*)calloc(er_size, sizeof(int));
        for (int j = 0; j < Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            for (int i = 0; i < Nr; i++) {
                double r = grid->a + (i + 0.5) * grid->dr;
                if (point_is_vacuum_endcap(endcap_pipes, grid->L, r, phi, -0.001))
                    h_mask[i + Nr * j] = 1;
            }
        }
        int free_count = 0;
        for (int i = 0; i < er_size; i++) free_count += h_mask[i];
        if (free_count > 0) {
            CUDA_CHECK(cudaMalloc(&pipe_op->d_mask_Er_endplate_z0, er_size * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(pipe_op->d_mask_Er_endplate_z0, h_mask,
                er_size * sizeof(int), cudaMemcpyHostToDevice));
        }
        free(h_mask);
    }

    /* Ephi mask at z0 */
    if (k_z0 >= 0) {
        size_t ephi_size = (size_t)(Nr + 1) * (size_t)Nphi;
        int* h_mask = (int*)calloc(ephi_size, sizeof(int));
        for (int j = 0; j < Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            for (int i = 0; i <= Nr; i++) {
                double r = grid->a + i * grid->dr;
                if (point_is_vacuum_endcap(endcap_pipes, grid->L, r, phi, -0.001))
                    h_mask[i + (Nr + 1) * j] = 1;
            }
        }
        size_t free_count = 0;
        for (size_t i = 0; i < ephi_size; i++) free_count += h_mask[i];
        if (free_count > 0) {
            CUDA_CHECK(cudaMalloc(&pipe_op->d_mask_Ephi_endplate_z0, ephi_size * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(pipe_op->d_mask_Ephi_endplate_z0, h_mask,
                ephi_size * sizeof(int), cudaMemcpyHostToDevice));
        }
        free(h_mask);
    }

    /* Er mask at zL */
    if (k_zL >= 0) {
        int er_size = Nr * Nphi;
        int* h_mask = (int*)calloc(er_size, sizeof(int));
        double L_cav = grid->L - z0_extension;  /* approximate */
        for (int j = 0; j < Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            for (int i = 0; i < Nr; i++) {
                double r = grid->a + (i + 0.5) * grid->dr;
                if (point_is_vacuum_endcap(endcap_pipes, L_cav, r, phi, L_cav + 0.001))
                    h_mask[i + Nr * j] = 1;
            }
        }
        int free_count = 0;
        for (int i = 0; i < er_size; i++) free_count += h_mask[i];
        if (free_count > 0) {
            CUDA_CHECK(cudaMalloc(&pipe_op->d_mask_Er_endplate_zL, er_size * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(pipe_op->d_mask_Er_endplate_zL, h_mask,
                er_size * sizeof(int), cudaMemcpyHostToDevice));
        }
        free(h_mask);
    }

    /* Ephi mask at zL */
    if (k_zL >= 0) {
        size_t ephi_size = (size_t)(Nr + 1) * (size_t)Nphi;
        int* h_mask = (int*)calloc(ephi_size, sizeof(int));
        double L_cav = grid->L - z0_extension;
        for (int j = 0; j < Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            for (int i = 0; i <= Nr; i++) {
                double r = grid->a + i * grid->dr;
                if (point_is_vacuum_endcap(endcap_pipes, L_cav, r, phi, L_cav + 0.001))
                    h_mask[i + (Nr + 1) * j] = 1;
            }
        }
        size_t free_count = 0;
        for (size_t i = 0; i < ephi_size; i++) free_count += h_mask[i];
        if (free_count > 0) {
            CUDA_CHECK(cudaMalloc(&pipe_op->d_mask_Ephi_endplate_zL, ephi_size * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(pipe_op->d_mask_Ephi_endplate_zL, h_mask,
                ephi_size * sizeof(int), cudaMemcpyHostToDevice));
        }
        free(h_mask);
    }

    return 0;
}

/*=============================================================================
 * Free (updated to include IBC weights)
 *============================================================================*/

void gpu_pipe_operator_free(GPU_PipeOperator* pipe_op) {
    if (pipe_op->has_mask)
        gpu_material_mask_free(&pipe_op->mask);
    pipe_op->has_mask = 0;

    if (pipe_op->has_ibc_weights)
        gpu_ibc_weights_free(&pipe_op->ibc_weights);
    pipe_op->has_ibc_weights = 0;

    if (pipe_op->d_mask_Er_endplate_z0)   cudaFree(pipe_op->d_mask_Er_endplate_z0);
    if (pipe_op->d_mask_Er_endplate_zL)   cudaFree(pipe_op->d_mask_Er_endplate_zL);
    if (pipe_op->d_mask_Ephi_endplate_z0) cudaFree(pipe_op->d_mask_Ephi_endplate_z0);
    if (pipe_op->d_mask_Ephi_endplate_zL) cudaFree(pipe_op->d_mask_Ephi_endplate_zL);
    pipe_op->has_interior_endplates = 0;

    gpu_operator_free(&pipe_op->base);

    if (pipe_op->has_ibc_mask)
        gpu_material_mask_free(&pipe_op->ibc_mask);
    pipe_op->has_ibc_mask = 0;
}

/*=============================================================================
 * PEC Matvec (unchanged)
 *============================================================================*/

static void apply_interior_endplate_pec(
    GPU_EField* E,
    const GPU_PipeOperator* pipe_op
) {
    if (!pipe_op->has_interior_endplates) return;

    const GridParams* grid = &pipe_op->base.cpu_op->grid;
    int Nr = grid->Nr, Nr1 = Nr + 1, Nphi = grid->Nphi;
    int face_Er = Nr * Nphi, face_Ephi = Nr1 * Nphi;
    int blocks_Er   = (face_Er + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_Ephi = (face_Ephi + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (pipe_op->k_endplate_z0 >= 0) {
        int k = pipe_op->k_endplate_z0;
        if (pipe_op->d_mask_Er_endplate_z0)
            pec_Er_at_k_masked_kernel<<<blocks_Er, BLOCK_SIZE>>>(E->Er, Nr, Nphi, k, pipe_op->d_mask_Er_endplate_z0);
        else
            pec_Er_at_k_kernel<<<blocks_Er, BLOCK_SIZE>>>(E->Er, Nr, Nphi, k);
        if (pipe_op->d_mask_Ephi_endplate_z0)
            pec_Ephi_at_k_masked_kernel<<<blocks_Ephi, BLOCK_SIZE>>>(E->Ephi, Nr1, Nphi, k, pipe_op->d_mask_Ephi_endplate_z0);
        else
            pec_Ephi_at_k_kernel<<<blocks_Ephi, BLOCK_SIZE>>>(E->Ephi, Nr1, Nphi, k);
    }
    if (pipe_op->k_endplate_zL >= 0) {
        int k = pipe_op->k_endplate_zL;
        if (pipe_op->d_mask_Er_endplate_zL)
            pec_Er_at_k_masked_kernel<<<blocks_Er, BLOCK_SIZE>>>(E->Er, Nr, Nphi, k, pipe_op->d_mask_Er_endplate_zL);
        else
            pec_Er_at_k_kernel<<<blocks_Er, BLOCK_SIZE>>>(E->Er, Nr, Nphi, k);
        if (pipe_op->d_mask_Ephi_endplate_zL)
            pec_Ephi_at_k_masked_kernel<<<blocks_Ephi, BLOCK_SIZE>>>(E->Ephi, Nr1, Nphi, k, pipe_op->d_mask_Ephi_endplate_zL);
        else
            pec_Ephi_at_k_kernel<<<blocks_Ephi, BLOCK_SIZE>>>(E->Ephi, Nr1, Nphi, k);
    }
}

int gpu_pipe_matvec(
    const GPU_PipeOperator* pipe_op,
    const double* d_x, double* d_y
) {
    GPU_PipeOperator* op = (GPU_PipeOperator*)pipe_op;
    GPU_Operator* base = &op->base;
    const CurlCurlOperator* cpu_op = base->cpu_op;
    const GridParams* grid = &cpu_op->grid;

    gpu_unpack_field(d_x, &base->d_E_work,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez);
    if (base->has_port_masks)
        gpu_apply_PEC_boundary_with_masks(&base->d_E_work, grid, base);
    else
        gpu_apply_PEC_boundary(&base->d_E_work, grid);
    if (op->has_mask)
        gpu_apply_material_mask(&base->d_E_work, &op->mask);
    apply_interior_endplate_pec(&base->d_E_work, pipe_op);

    gpu_compute_curl_curl_E(&base->d_E_work, &base->d_result_work,
        &base->d_H_temp, grid);

    if (base->has_port_masks)
        gpu_apply_PEC_boundary_with_masks(&base->d_result_work, grid, base);
    else
        gpu_apply_PEC_boundary(&base->d_result_work, grid);
    if (op->has_mask)
        gpu_apply_material_mask(&base->d_result_work, &op->mask);
    apply_interior_endplate_pec(&base->d_result_work, pipe_op);

    gpu_pack_field(&base->d_result_work, d_y,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez);
    return 0;
}

/*=============================================================================
 * IBC Complex Pipe Matvec: y = (∇×∇× + β·S) · x
 *
 * The IBC operator on the extended grid. Interior walls (outer conductor,
 * endplates, pipe walls) are handled by boundary weights computed from the
 * material mask. The inner conductor (i=0, grid edge) is handled explicitly.
 *
 * NO interior endplate PEC is applied — the boundary weights handle endplates
 * through mask transitions at k=k_z0 and k=k_zL. This gives O(dr) accuracy
 * at interior surfaces (E is sampled half a cell from the wall). Grid
 * refinement improves the accuracy proportionally.
 *
 * Flow:
 *   1. Unpack x → E (complex)
 *   2. PEC at pipe caps (i=Nr, k=0, k=Nz) — grid edges only
 *   3. Material mask (both re+im) — zeros PEC cells, keeps wall cells alive
 *   4. ∇×∇×E (complex)
 *   5. Inner conductor IBC: result += β/dr · E at i=0
 *   6. Boundary weight IBC: result += β · w · E (outer conductor, endplates, pipe walls)
 *   7. PEC at pipe caps (on result)
 *   8. Material mask on result
 *   9. Pack → y
 *============================================================================*/

/* Helper: apply pipe cap PEC to complex E field */
static void apply_pipe_cap_pec_complex(
    GPU_EField* E, int Nr, int Nr1, int Nphi, int Nz
) {
    int face_Ephi_cyl = Nphi * (Nz + 1);
    int face_Ez_cyl   = Nphi * Nz;
    int face_Er_end   = Nr * Nphi;
    int face_Ephi_end = Nr1 * Nphi;

    pec_Ephi_outer_cap_kernel<<<(face_Ephi_cyl + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Ephi, E->Ephi_im, Nr, Nphi, Nz);
    pec_Ez_outer_cap_kernel<<<(face_Ez_cyl + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Ez, E->Ez_im, Nr, Nphi, Nz);
    pec_Er_endcaps_complex_kernel<<<(face_Er_end + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Er, E->Er_im, Nr, Nphi, Nz);
    pec_Ephi_endcaps_complex_kernel<<<(face_Ephi_end + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Ephi, E->Ephi_im, Nr1, Nphi, Nz);
}

int gpu_pipe_matvec_complex(
    const GPU_PipeOperator* pipe_op,
    const double* d_x, double* d_y,
    double alpha
) {
    GPU_PipeOperator* op = (GPU_PipeOperator*)pipe_op;
    GPU_Operator* base = &op->base;
    const CurlCurlOperator* cpu_op = base->cpu_op;
    const GridParams* grid = &cpu_op->grid;
    int n_real = cpu_op->n_total;
    int Nr = grid->Nr, Nr1 = Nr + 1, Nphi = grid->Nphi, Nz = grid->Nz;

    /* β = (1-j)/(2α) */
    double inv_2a = 1.0 / (2.0 * alpha);
    double beta_re =  inv_2a;
    double beta_im = -inv_2a;
    double bw_inner_re = beta_re / grid->dr;
    double bw_inner_im = beta_im / grid->dr;

    /* Step 1: Unpack */
    gpu_unpack_field_complex(d_x, &base->d_E_work,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez, n_real);

    /* Step 2: PEC at pipe caps (grid edges only, NOT inner conductor) */
    apply_pipe_cap_pec_complex(&base->d_E_work, Nr, Nr1, Nphi, Nz);

    /* Step 3: Material mask — use IBC mask (surface cells stay alive) */
    if (op->has_ibc_mask)
        gpu_apply_material_mask_complex(&base->d_E_work, &op->ibc_mask);
    else if (op->has_mask)
        gpu_apply_material_mask_complex(&base->d_E_work, &op->mask);

    /* Step 4: curl-curl */
    gpu_compute_curl_curl_E_complex(&base->d_E_work, &base->d_result_work,
        &base->d_H_temp, grid);

    /* Step 5: Inner conductor IBC (i=0) */
    {
        int face_Ephi = Nphi * (Nz + 1);
        int face_Ez   = Nphi * Nz;

        ibc_inner_Ephi_kernel<<<(face_Ephi + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            base->d_result_work.Ephi, base->d_result_work.Ephi_im,
            base->d_E_work.Ephi, base->d_E_work.Ephi_im,
            Nr1, Nphi, Nz, bw_inner_re, bw_inner_im,
            base->has_port_masks ? base->d_mask_Ephi_inner : NULL);

        ibc_inner_Ez_kernel<<<(face_Ez + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            base->d_result_work.Ez, base->d_result_work.Ez_im,
            base->d_E_work.Ez, base->d_E_work.Ez_im,
            Nr1, Nphi, Nz, bw_inner_re, bw_inner_im,
            base->has_port_masks ? base->d_mask_Ez_inner : NULL);
    }

    /* Step 6: Boundary weight IBC (outer conductor + endplates + pipe walls) */
    if (op->has_ibc_weights) {
        GPU_IBCWeights* w = &op->ibc_weights;
        int b1 = (w->size_Er   + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int b2 = (w->size_Ephi + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int b3 = (w->size_Ez   + BLOCK_SIZE - 1) / BLOCK_SIZE;

        ibc_weight_kernel<<<b1, BLOCK_SIZE>>>(
            base->d_result_work.Er, base->d_result_work.Er_im,
            base->d_E_work.Er, base->d_E_work.Er_im,
            w->d_weight_Er, beta_re, beta_im, w->size_Er);

        ibc_weight_kernel<<<b2, BLOCK_SIZE>>>(
            base->d_result_work.Ephi, base->d_result_work.Ephi_im,
            base->d_E_work.Ephi, base->d_E_work.Ephi_im,
            w->d_weight_Ephi, beta_re, beta_im, w->size_Ephi);

        ibc_weight_kernel<<<b3, BLOCK_SIZE>>>(
            base->d_result_work.Ez, base->d_result_work.Ez_im,
            base->d_E_work.Ez, base->d_E_work.Ez_im,
            w->d_weight_Ez, beta_re, beta_im, w->size_Ez);
    }

    /* Step 7: PEC at pipe caps (on result) */
    apply_pipe_cap_pec_complex(&base->d_result_work, Nr, Nr1, Nphi, Nz);

    /* Step 8: Material mask on result — use IBC mask */
    if (op->has_ibc_mask)
        gpu_apply_material_mask_complex(&base->d_result_work, &op->ibc_mask);
    else if (op->has_mask)
        gpu_apply_material_mask_complex(&base->d_result_work, &op->mask);

    /* Step 9: Pack */
    gpu_pack_field_complex(&base->d_result_work, d_y,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez, n_real);

    return 0;
}

/*=============================================================================
 * Pipe matvec wrapper for ComplexMatvecFn interface
 *============================================================================*/

static int pipe_matvec_complex_wrapper(
    const void* op, const double* d_x, double* d_y, double alpha
) {
    return gpu_pipe_matvec_complex((const GPU_PipeOperator*)op, d_x, d_y, alpha);
}

GPU_ComplexEigenResult gpu_rqi_complex_pipe(
    const GPU_PipeOperator* pipe_op,
    double* d_x,
    double sigma_init,
    double conductivity,
    int max_iter,
    double tol,
    int gmres_restart
) {
    const CurlCurlOperator* cpu_op = pipe_op->base.cpu_op;
    int n_real = cpu_op->n_total;
    ComplexEigensolverWorkspace ws;
    complex_eigensolver_workspace_init(&ws, n_real, gmres_restart);

    GPU_ComplexEigenResult result = gpu_rqi_complex_ws(
        pipe_op,
        pipe_matvec_complex_wrapper,
        cpu_op,
        &((GPU_PipeOperator*)pipe_op)->base.reduction_ws,
        d_x, sigma_init, conductivity,
        max_iter, tol, &ws);

    complex_eigensolver_workspace_free(&ws);
    return result;
}

/*=============================================================================
 * Perturbative IBC Q computation (NO iteration, NO curl-curl)
 *
 * First-order eigenvalue perturbation:
 *
 *   k²_IBC = k²_PEC + ⟨x_PEC, β·S·x_PEC⟩ / ⟨x_PEC, x_PEC⟩
 *
 * Since x_PEC is an eigenvector of A_PEC, and A_IBC = A_PEC + β·S,
 * we only need to apply the surface correction S to x_PEC and compute
 * one inner product. No curl-curl, no GMRES, no iteration.
 *
 * Cost: 3 weight-multiply kernels + 1 inner conductor kernel + 3 dot products.
 * Time: milliseconds.
 *
 * The result is complex: Re gives a small frequency shift, Im gives Q.
 *============================================================================*/

GPU_ComplexEigenResult gpu_ibc_perturbative_pipe(
    const GPU_PipeOperator* pipe_op,
    const double* d_x_pec,     /* PEC eigenvector on GPU, size n_real */
    double k2_pec,             /* PEC eigenvalue */
    double conductivity
) {
    GPU_ComplexEigenResult result;
    result.iterations = 0;
    result.residual = 0.0;
    result.converged = 1;

    const CurlCurlOperator* cpu_op = pipe_op->base.cpu_op;
    const GPU_PipeOperator* pop = pipe_op;
    ReductionWorkspace* red_ws =
        &((GPU_PipeOperator*)pipe_op)->base.reduction_ws;
    const GridParams* grid = &cpu_op->grid;
    int n_real = cpu_op->n_total;
    int Nr = grid->Nr, Nr1 = Nr + 1, Nphi = grid->Nphi, Nz = grid->Nz;
    double c0 = 299792458.0;

    /* Compute alpha and beta from PEC eigenvalue */
    double alpha = ibc_compute_alpha(k2_pec, conductivity);
    double inv_2a = 1.0 / (2.0 * alpha);
    double beta_re =  inv_2a;    /* Re(β) =  1/(2α) */
    double beta_im = -inv_2a;    /* Im(β) = -1/(2α) */
    double bw_inner_re = beta_re / grid->dr;
    double bw_inner_im = beta_im / grid->dr;

    /* Unpack PEC eigenvector into E_work (real only, imag = 0) */
    GPU_EField E_input;
    gpu_efield_alloc_complex(&E_input, grid);
    /* Real part from PEC eigenvector */
    cudaMemcpy(E_input.Er, d_x_pec + cpu_op->offset_Er,
        cpu_op->size_Er * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(E_input.Ephi, d_x_pec + cpu_op->offset_Ephi,
        cpu_op->size_Ephi * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(E_input.Ez, d_x_pec + cpu_op->offset_Ez,
        cpu_op->size_Ez * sizeof(double), cudaMemcpyDeviceToDevice);
    /* Imaginary part = 0 (already from alloc_complex) */

    /* Allocate result array for β·S·x (starts at zero) */
    GPU_EField E_correction;
    gpu_efield_alloc_complex(&E_correction, grid);
    cudaMemset(E_correction.Er, 0, cpu_op->size_Er * sizeof(double));
    cudaMemset(E_correction.Ephi, 0, cpu_op->size_Ephi * sizeof(double));
    cudaMemset(E_correction.Ez, 0, cpu_op->size_Ez * sizeof(double));
    cudaMemset(E_correction.Er_im, 0, cpu_op->size_Er * sizeof(double));
    cudaMemset(E_correction.Ephi_im, 0, cpu_op->size_Ephi * sizeof(double));
    cudaMemset(E_correction.Ez_im, 0, cpu_op->size_Ez * sizeof(double));

    /* Apply inner conductor IBC: correction += β/dr · E at i=0 */
    {
        int face_Ephi = Nphi * (Nz + 1);
        int face_Ez   = Nphi * Nz;
        const GPU_Operator* base = &pop->base;

        ibc_inner_Ephi_kernel<<<(face_Ephi + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            E_correction.Ephi, E_correction.Ephi_im,
            E_input.Ephi, E_input.Ephi_im,
            Nr1, Nphi, Nz, bw_inner_re, bw_inner_im,
            base->has_port_masks ? base->d_mask_Ephi_inner : NULL);

        ibc_inner_Ez_kernel<<<(face_Ez + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            E_correction.Ez, E_correction.Ez_im,
            E_input.Ez, E_input.Ez_im,
            Nr1, Nphi, Nz, bw_inner_re, bw_inner_im,
            base->has_port_masks ? base->d_mask_Ez_inner : NULL);
    }

    /* Apply boundary weight IBC: correction += β · w · E */
    if (pop->has_ibc_weights) {
        const GPU_IBCWeights* w = &pop->ibc_weights;
        int b1 = (w->size_Er   + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int b2 = (w->size_Ephi + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int b3 = (w->size_Ez   + BLOCK_SIZE - 1) / BLOCK_SIZE;

        ibc_weight_kernel<<<b1, BLOCK_SIZE>>>(
            E_correction.Er, E_correction.Er_im,
            E_input.Er, E_input.Er_im,
            w->d_weight_Er, beta_re, beta_im, w->size_Er);

        ibc_weight_kernel<<<b2, BLOCK_SIZE>>>(
            E_correction.Ephi, E_correction.Ephi_im,
            E_input.Ephi, E_input.Ephi_im,
            w->d_weight_Ephi, beta_re, beta_im, w->size_Ephi);

        ibc_weight_kernel<<<b3, BLOCK_SIZE>>>(
            E_correction.Ez, E_correction.Ez_im,
            E_input.Ez, E_input.Ez_im,
            w->d_weight_Ez, beta_re, beta_im, w->size_Ez);
    }

    /* Pack correction into a complex vector for dot product */
    int n2 = 2 * n_real;
    double* d_Sx;   /* β·S·x packed as [re | im] */
    double* d_x_cx; /* x packed as [re | 0] */
    cudaMalloc(&d_Sx, n2 * sizeof(double));
    cudaMalloc(&d_x_cx, n2 * sizeof(double));

    gpu_pack_field_complex(&E_correction, d_Sx,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez, n_real);
    gpu_pack_field_complex(&E_input, d_x_cx,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez, n_real);

    /* Compute complex shift: Δk² = ⟨x, β·S·x⟩ / ⟨x, x⟩ */
    double xSx_re, xSx_im, xx;
    gpu_vec_dot_weighted_complex_re_ws(d_x_cx, d_Sx, &xSx_re,
        cpu_op, n_real, red_ws);
    gpu_vec_dot_weighted_complex_im_ws(d_x_cx, d_Sx, &xSx_im,
        cpu_op, n_real, red_ws);
    gpu_vec_dot_weighted_complex_re_ws(d_x_cx, d_x_cx, &xx,
        cpu_op, n_real, red_ws);

    double dk2_re = xSx_re / xx;
    double dk2_im = xSx_im / xx;

    double k2_re = k2_pec + dk2_re;
    double k2_im = dk2_im;   /* PEC has k2_im = 0 */

    double freq = (k2_re > 0) ? c0 * sqrt(k2_re) / (2.0 * M_PI) : 0.0;
    double Q = (fabs(k2_im) > 1e-30) ? -k2_re / k2_im : 0.0;

    result.k2_re = k2_re;
    result.k2_im = k2_im;
    result.frequency_Hz = freq;
    result.Q_factor = Q;

    printf("  Perturbative IBC:\n");
    printf("    k2_PEC       = %.10f\n", k2_pec);
    printf("    Δk2_re       = %+.6e  (frequency shift)\n", dk2_re);
    printf("    Δk2_im       = %+.6e  (loss term)\n", dk2_im);
    printf("    k2_IBC_re    = %.10f\n", k2_re);
    printf("    k2_IBC_im    = %.6e\n", k2_im);
    printf("    f            = %.6f MHz\n", freq / 1e6);
    printf("    Q            = %.1f\n", Q);

    /* Cleanup */
    cudaFree(d_Sx);
    cudaFree(d_x_cx);
    gpu_efield_free(&E_input);
    gpu_efield_free(&E_correction);

    return result;
}

/*=============================================================================
 * Endplate PEC kernels (definitions — unchanged from original)
 *============================================================================*/

__global__ void pec_Er_at_k_kernel(double* Er, int Nr, int Nphi, int k) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < Nr * Nphi) {
        int i = gid % Nr;
        int j = gid / Nr;
        Er[i + Nr * (j + Nphi * k)] = 0.0;
    }
}

__global__ void pec_Ephi_at_k_kernel(double* Ephi, int Nr1, int Nphi, int k) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < Nr1 * Nphi) {
        int i = gid % Nr1;
        int j = gid / Nr1;
        Ephi[i + Nr1 * (j + Nphi * k)] = 0.0;
    }
}

__global__ void pec_Er_at_k_masked_kernel(
    double* Er, int Nr, int Nphi, int k, const int* mask
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < Nr * Nphi) {
        if (!mask || !mask[gid]) {
            int i = gid % Nr;
            int j = gid / Nr;
            Er[i + Nr * (j + Nphi * k)] = 0.0;
        }
    }
}

__global__ void pec_Ephi_at_k_masked_kernel(
    double* Ephi, int Nr1, int Nphi, int k, const int* mask
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < Nr1 * Nphi) {
        if (!mask || !mask[gid]) {
            int i = gid % Nr1;
            int j = gid / Nr1;
            Ephi[i + Nr1 * (j + Nphi * k)] = 0.0;
        }
    }
}
