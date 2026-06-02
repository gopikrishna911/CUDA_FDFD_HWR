/*=============================================================================
 * cuda_conformal_pipe_cu.cpp
 *
 * IBC mask for outer conductor/pipe walls.
 * Grid-plane IBC for endplates (no cavity extension).
 * Conformal Dey-Mittra for pipe walls.
 *============================================================================*/

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "cuda_conformal_pipe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

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
 * Extern kernel declarations
 *============================================================================*/

/* Grid-edge PEC (complex) */
extern __global__ void pec_Ephi_outer_cap_kernel(
    double* Ephi_re, double* Ephi_im, int Nr, int Nphi, int Nz);
extern __global__ void pec_Ez_outer_cap_kernel(
    double* Ez_re, double* Ez_im, int Nr, int Nphi, int Nz);
extern __global__ void pec_Er_endcaps_complex_kernel(
    double* Er_re, double* Er_im, int Nr, int Nphi, int Nz);
extern __global__ void pec_Ephi_endcaps_complex_kernel(
    double* Ephi_re, double* Ephi_im, int Nr1, int Nphi, int Nz);

/* Single k-plane PEC (real only — call twice for complex) */
extern __global__ void pec_Er_at_k_masked_kernel(
    double* Er, int Nr, int Nphi, int k, const int* mask);
extern __global__ void pec_Ephi_at_k_masked_kernel(
    double* Ephi, int Nr1, int Nphi, int k, const int* mask);
extern __global__ void pec_Er_at_k_kernel(double* Er, int Nr, int Nphi, int k);
extern __global__ void pec_Ephi_at_k_kernel(double* Ephi, int Nr1, int Nphi, int k);

/* Inner conductor IBC */
extern __global__ void ibc_inner_Ephi_kernel(
    double* result_re, double* result_im,
    const double* E_re, const double* E_im,
    int Nr1, int Nphi, int Nz,
    double bw_re, double bw_im, const int* port_mask);
extern __global__ void ibc_inner_Ez_kernel(
    double* result_re, double* result_im,
    const double* E_re, const double* E_im,
    int Nr1, int Nphi, int Nz,
    double bw_re, double bw_im, const int* port_mask);

/* Staircase IBC weight kernel */
extern __global__ void ibc_weight_kernel(
    double* result_re, double* result_im,
    const double* E_re, const double* E_im,
    const double* weight, double beta_re, double beta_im, int n);

/* Grid-plane endplate IBC kernels (from cuda_operator_cu.cpp) */
typedef struct { double a, dr, dphi, dz; int Nr, Nphi, Nz; } BndGridParams;

extern __global__ void ibc_surface_Er_end_kernel(
    double* result_re, double* result_im,
    const double* E_re, const double* E_im,
    BndGridParams gp, double bw_re, double bw_im,
    int k_wall, const int* mask);
extern __global__ void ibc_surface_Ephi_end_kernel(
    double* result_re, double* result_im,
    const double* E_re, const double* E_im,
    BndGridParams gp, double bw_re, double bw_im,
    int k_wall, const int* mask);

/* Pack/unpack */
extern int gpu_unpack_field_complex(const double* d_x, GPU_EField* E,
    int offset_Er, int offset_Ephi, int offset_Ez, int n_real);
extern int gpu_pack_field_complex(const GPU_EField* E, double* d_x,
    int offset_Er, int offset_Ephi, int offset_Ez, int n_real);

/*=============================================================================
 * IBC weight kernel excluding pipe-wall cells
 *============================================================================*/
__global__ void ibc_weight_exclude_kernel(
    double* result_re, double* result_im,
    const double* E_re, const double* E_im,
    const double* staircase_weight,
    const double* conformal_weight,
    double beta_re, double beta_im, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double w = staircase_weight[idx];
        if (w > 0.0 && conformal_weight[idx] == 0.0) {
            double bw_re = beta_re * w;
            double bw_im = beta_im * w;
            double e_re = E_re[idx], e_im = E_im[idx];
            result_re[idx] += bw_re * e_re - bw_im * e_im;
            result_im[idx] += bw_re * e_im + bw_im * e_re;
        }
    }
}

/*=============================================================================
 * Helpers
 *============================================================================*/

static BndGridParams make_bnd_params(const GridParams* grid) {
    BndGridParams p;
    p.a = grid->a; p.dr = grid->dr; p.dphi = grid->dphi; p.dz = grid->dz;
    p.Nr = grid->Nr; p.Nphi = grid->Nphi; p.Nz = grid->Nz;
    return p;
}

static void apply_pipe_cap_pec_complex(
    GPU_EField* E, int Nr, int Nr1, int Nphi, int Nz
) {
    pec_Ephi_outer_cap_kernel<<<(Nphi*(Nz+1)+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Ephi, E->Ephi_im, Nr, Nphi, Nz);
    pec_Ez_outer_cap_kernel<<<(Nphi*Nz+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Ez, E->Ez_im, Nr, Nphi, Nz);
    pec_Er_endcaps_complex_kernel<<<(Nr*Nphi+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Er, E->Er_im, Nr, Nphi, Nz);
    pec_Ephi_endcaps_complex_kernel<<<((Nr+1)*Nphi+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        E->Ephi, E->Ephi_im, Nr+1, Nphi, Nz);
}

/* Re-zero endplate conductor-side surface cells (complex) */
static void rezero_endplate_surface(
    GPU_EField* E, int Nr, int Nr1, int Nphi, int Nz,
    int k_z0, int k_zL,
    const int* d_mask_Er_z0, const int* d_mask_Ephi_z0,
    const int* d_mask_Er_zL, const int* d_mask_Ephi_zL
) {
    int b_Er   = (Nr * Nphi + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int b_Ephi = (Nr1 * Nphi + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Conductor-side surface layer is k_z0-1 (z0) and k_zL+1 (zL). Guard
     * against running off the Er/Ephi arrays (valid k range is 0..Nz). The
     * z0 branch already requires k_z0>=1 so k_z0-1>=0; for zL we additionally
     * require k_zL+1<=Nz. (Bug fix: previously k_zL==Nz wrote to k=Nz+1,
     * one plane past the array end, corrupting adjacent device memory.) */
    if (k_z0 >= 1) {
        int ks = k_z0 - 1;  /* conductor-side surface layer */
        /* Re-zero Er (re+im) */
        if (d_mask_Er_z0)
            pec_Er_at_k_masked_kernel<<<b_Er, BLOCK_SIZE>>>(E->Er, Nr, Nphi, ks, d_mask_Er_z0);
        else
            pec_Er_at_k_kernel<<<b_Er, BLOCK_SIZE>>>(E->Er, Nr, Nphi, ks);
        if (d_mask_Er_z0)
            pec_Er_at_k_masked_kernel<<<b_Er, BLOCK_SIZE>>>(E->Er_im, Nr, Nphi, ks, d_mask_Er_z0);
        else
            pec_Er_at_k_kernel<<<b_Er, BLOCK_SIZE>>>(E->Er_im, Nr, Nphi, ks);
        /* Re-zero Ephi (re+im) */
        if (d_mask_Ephi_z0)
            pec_Ephi_at_k_masked_kernel<<<b_Ephi, BLOCK_SIZE>>>(E->Ephi, Nr1, Nphi, ks, d_mask_Ephi_z0);
        else
            pec_Ephi_at_k_kernel<<<b_Ephi, BLOCK_SIZE>>>(E->Ephi, Nr1, Nphi, ks);
        if (d_mask_Ephi_z0)
            pec_Ephi_at_k_masked_kernel<<<b_Ephi, BLOCK_SIZE>>>(E->Ephi_im, Nr1, Nphi, ks, d_mask_Ephi_z0);
        else
            pec_Ephi_at_k_kernel<<<b_Ephi, BLOCK_SIZE>>>(E->Ephi_im, Nr1, Nphi, ks);
    }

    if (k_zL >= 0 && k_zL + 1 <= Nz) {
        int ks = k_zL + 1;  /* conductor-side surface layer */
        if (d_mask_Er_zL)
            pec_Er_at_k_masked_kernel<<<b_Er, BLOCK_SIZE>>>(E->Er, Nr, Nphi, ks, d_mask_Er_zL);
        else
            pec_Er_at_k_kernel<<<b_Er, BLOCK_SIZE>>>(E->Er, Nr, Nphi, ks);
        if (d_mask_Er_zL)
            pec_Er_at_k_masked_kernel<<<b_Er, BLOCK_SIZE>>>(E->Er_im, Nr, Nphi, ks, d_mask_Er_zL);
        else
            pec_Er_at_k_kernel<<<b_Er, BLOCK_SIZE>>>(E->Er_im, Nr, Nphi, ks);
        if (d_mask_Ephi_zL)
            pec_Ephi_at_k_masked_kernel<<<b_Ephi, BLOCK_SIZE>>>(E->Ephi, Nr1, Nphi, ks, d_mask_Ephi_zL);
        else
            pec_Ephi_at_k_kernel<<<b_Ephi, BLOCK_SIZE>>>(E->Ephi, Nr1, Nphi, ks);
        if (d_mask_Ephi_zL)
            pec_Ephi_at_k_masked_kernel<<<b_Ephi, BLOCK_SIZE>>>(E->Ephi_im, Nr1, Nphi, ks, d_mask_Ephi_zL);
        else
            pec_Ephi_at_k_kernel<<<b_Ephi, BLOCK_SIZE>>>(E->Ephi_im, Nr1, Nphi, ks);
    }
}

/*=============================================================================
 * Init / Free
 *============================================================================*/

int gpu_conformal_pipe_operator_init(
    GPU_ConformalPipeOperator* cop,
    const CurlCurlOperator* cpu_op,
    const MaterialMask* mask,
    const ConformalData* cd,
    int k_z0, int k_zL,
    const EndcapPipeConfig* endcap_pipes,
    const GridParams* grid,
    double z0_extension
) {
    memset(cop, 0, sizeof(GPU_ConformalPipeOperator));
    int Nr = grid->Nr, Nr1 = Nr + 1, Nphi = grid->Nphi;

    /* Base pipe operator: complex arrays, IBC mask, staircase weights */
    if (gpu_pipe_operator_init_complex(&cop->pipe_op, cpu_op, mask) != 0)
        return -1;

    /* Conformal data */
    if (gpu_conformal_data_init(&cop->conformal, cd) != 0) {
        gpu_pipe_operator_free(&cop->pipe_op);
        return -1;
    }
    cop->has_conformal = 1;

    /* Endplate IBC setup */
    cop->k_endplate_z0 = k_z0;
    cop->k_endplate_zL = k_zL;
    cop->d_mask_Er_z0 = NULL;
    cop->d_mask_Ephi_z0 = NULL;
    cop->d_mask_Er_zL = NULL;
    cop->d_mask_Ephi_zL = NULL;
    cop->has_endplate_ibc = (k_z0 >= 0 || k_zL >= 0);

    if (endcap_pipes && endcap_pipes->num_pipes > 0) {
        double L_phys = (k_zL - k_z0) * grid->dz;

        /* Build endcap pipe hole masks at k=k_z0 (used for both k_z0 IBC and k_z0-1 re-zeroing) */
        if (k_z0 >= 0) {
            int er_size = Nr * Nphi;
            int* h_mask = (int*)calloc(er_size, sizeof(int));
            for (int j = 0; j < Nphi; j++) {
                double phi = (j + 0.5) * grid->dphi;
                for (int i = 0; i < Nr; i++) {
                    double r = grid->a + (i + 0.5) * grid->dr;
                    if (point_is_vacuum_endcap(endcap_pipes, L_phys, r, phi, -0.001))
                        h_mask[i + Nr * j] = 1;
                }
            }
            int cnt = 0; for (int i = 0; i < er_size; i++) cnt += h_mask[i];
            if (cnt > 0) {
                CUDA_CHECK(cudaMalloc(&cop->d_mask_Er_z0, er_size * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(cop->d_mask_Er_z0, h_mask,
                    er_size * sizeof(int), cudaMemcpyHostToDevice));
            }
            printf("  Endplate z0 Er mask: %d pipe-hole cells\n", cnt);
            free(h_mask);

            int ephi_size = Nr1 * Nphi;
            h_mask = (int*)calloc(ephi_size, sizeof(int));
            for (int j = 0; j < Nphi; j++) {
                double phi = (j + 0.5) * grid->dphi;
                for (int i = 0; i <= Nr; i++) {
                    double r = grid->a + i * grid->dr;
                    if (point_is_vacuum_endcap(endcap_pipes, L_phys, r, phi, -0.001))
                        h_mask[i + Nr1 * j] = 1;
                }
            }
            cnt = 0; for (int i = 0; i < ephi_size; i++) cnt += h_mask[i];
            if (cnt > 0) {
                CUDA_CHECK(cudaMalloc(&cop->d_mask_Ephi_z0, ephi_size * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(cop->d_mask_Ephi_z0, h_mask,
                    ephi_size * sizeof(int), cudaMemcpyHostToDevice));
            }
            printf("  Endplate z0 Ephi mask: %d pipe-hole cells\n", cnt);
            free(h_mask);
        }

        if (k_zL >= 0) {
            int er_size = Nr * Nphi;
            int* h_mask = (int*)calloc(er_size, sizeof(int));
            for (int j = 0; j < Nphi; j++) {
                double phi = (j + 0.5) * grid->dphi;
                for (int i = 0; i < Nr; i++) {
                    double r = grid->a + (i + 0.5) * grid->dr;
                    if (point_is_vacuum_endcap(endcap_pipes, L_phys, r, phi, L_phys + 0.001))
                        h_mask[i + Nr * j] = 1;
                }
            }
            int cnt = 0; for (int i = 0; i < er_size; i++) cnt += h_mask[i];
            if (cnt > 0) {
                CUDA_CHECK(cudaMalloc(&cop->d_mask_Er_zL, er_size * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(cop->d_mask_Er_zL, h_mask,
                    er_size * sizeof(int), cudaMemcpyHostToDevice));
            }
            printf("  Endplate zL Er mask: %d pipe-hole cells\n", cnt);
            free(h_mask);

            int ephi_size = Nr1 * Nphi;
            h_mask = (int*)calloc(ephi_size, sizeof(int));
            for (int j = 0; j < Nphi; j++) {
                double phi = (j + 0.5) * grid->dphi;
                for (int i = 0; i <= Nr; i++) {
                    double r = grid->a + i * grid->dr;
                    if (point_is_vacuum_endcap(endcap_pipes, L_phys, r, phi, L_phys + 0.001))
                        h_mask[i + Nr1 * j] = 1;
                }
            }
            cnt = 0; for (int i = 0; i < ephi_size; i++) cnt += h_mask[i];
            if (cnt > 0) {
                CUDA_CHECK(cudaMalloc(&cop->d_mask_Ephi_zL, ephi_size * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(cop->d_mask_Ephi_zL, h_mask,
                    ephi_size * sizeof(int), cudaMemcpyHostToDevice));
            }
            printf("  Endplate zL Ephi mask: %d pipe-hole cells\n", cnt);
            free(h_mask);
        }
    }

    printf("  Conformal pipe operator initialized (IBC mask + endplate grid-plane IBC).\n");
    return 0;
}

void gpu_conformal_pipe_operator_free(GPU_ConformalPipeOperator* cop) {
    if (cop->has_conformal) gpu_conformal_data_free(&cop->conformal);
    if (cop->d_mask_Er_z0) cudaFree(cop->d_mask_Er_z0);
    if (cop->d_mask_Ephi_z0) cudaFree(cop->d_mask_Ephi_z0);
    if (cop->d_mask_Er_zL) cudaFree(cop->d_mask_Er_zL);
    if (cop->d_mask_Ephi_zL) cudaFree(cop->d_mask_Ephi_zL);
    gpu_pipe_operator_free(&cop->pipe_op);
    memset(cop, 0, sizeof(GPU_ConformalPipeOperator));
}

/*=============================================================================
 * Conformal IBC Complex Matvec
 *============================================================================*/

int gpu_conformal_pipe_matvec_complex(
    const GPU_ConformalPipeOperator* cop,
    const double* d_x, double* d_y, double alpha
) {
    const GPU_PipeOperator* pop = &cop->pipe_op;
    GPU_Operator* base = (GPU_Operator*)&pop->base;
    const CurlCurlOperator* cpu_op = base->cpu_op;
    const GridParams* grid = &cpu_op->grid;
    int n_real = cpu_op->n_total;
    int Nr = grid->Nr, Nr1 = Nr + 1, Nphi = grid->Nphi, Nz = grid->Nz;

    BndGridParams gp = make_bnd_params(grid);

    double inv_2a = 1.0 / (2.0 * alpha);
    double beta_re =  inv_2a;
    double beta_im = -inv_2a;
    double bw_inner_re = beta_re / grid->dr;
    double bw_inner_im = beta_im / grid->dr;
    double bw_end_re = beta_re / grid->dz;
    double bw_end_im = beta_im / grid->dz;

    /* Step 1: Unpack */
    gpu_unpack_field_complex(d_x, &base->d_E_work,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez, n_real);

    /* Step 2: PEC at grid edges */
    apply_pipe_cap_pec_complex(&base->d_E_work, Nr, Nr1, Nphi, Nz);

    /* Step 3: IBC mask */
    if (pop->has_ibc_mask)
        gpu_apply_material_mask_complex(&base->d_E_work, &pop->ibc_mask);

    /* Step 3b: Re-zero endplate conductor-side surface cells.
     * The IBC mask unmasked these at k=k_z0-1 and k=k_zL+1, which would
     * extend the cavity by dz per endplate. Re-zeroing prevents this. */
    rezero_endplate_surface(&base->d_E_work, Nr, Nr1, Nphi, Nz,
        cop->k_endplate_z0, cop->k_endplate_zL,
        cop->d_mask_Er_z0, cop->d_mask_Ephi_z0,
        cop->d_mask_Er_zL, cop->d_mask_Ephi_zL);

    /* Step 4: Conformal curl-curl */
    gpu_compute_curl_curl_E_conformal_complex(
        &base->d_E_work, &base->d_result_work,
        &base->d_H_temp, grid, &cop->conformal);

    /* Step 5: Inner conductor IBC at i=0 */
    {
        int face_Ephi = Nphi * (Nz + 1);
        int face_Ez   = Nphi * Nz;
        ibc_inner_Ephi_kernel<<<(face_Ephi+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            base->d_result_work.Ephi, base->d_result_work.Ephi_im,
            base->d_E_work.Ephi, base->d_E_work.Ephi_im,
            Nr1, Nphi, Nz, bw_inner_re, bw_inner_im,
            base->has_port_masks ? base->d_mask_Ephi_inner : NULL);
        ibc_inner_Ez_kernel<<<(face_Ez+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            base->d_result_work.Ez, base->d_result_work.Ez_im,
            base->d_E_work.Ez, base->d_E_work.Ez_im,
            Nr1, Nphi, Nz, bw_inner_re, bw_inner_im,
            base->has_port_masks ? base->d_mask_Ez_inner : NULL);
    }

    /* Step 6a: Staircase IBC at flat walls (excl pipe cells) */
    if (pop->has_ibc_weights) {
        const GPU_IBCWeights* w = &pop->ibc_weights;
        ibc_weight_exclude_kernel<<<(w->size_Er+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            base->d_result_work.Er, base->d_result_work.Er_im,
            base->d_E_work.Er, base->d_E_work.Er_im,
            w->d_weight_Er, cop->conformal.d_ibc_weight_Er,
            beta_re, beta_im, w->size_Er);
        ibc_weight_exclude_kernel<<<(w->size_Ephi+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            base->d_result_work.Ephi, base->d_result_work.Ephi_im,
            base->d_E_work.Ephi, base->d_E_work.Ephi_im,
            w->d_weight_Ephi, cop->conformal.d_ibc_weight_Ephi,
            beta_re, beta_im, w->size_Ephi);
        ibc_weight_exclude_kernel<<<(w->size_Ez+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            base->d_result_work.Ez, base->d_result_work.Ez_im,
            base->d_E_work.Ez, base->d_E_work.Ez_im,
            w->d_weight_Ez, cop->conformal.d_ibc_weight_Ez,
            beta_re, beta_im, w->size_Ez);
    }

    /* Step 6b: Endplate IBC at k=k_z0 and k=k_zL (grid-plane) */
    {
        int b_Er_end   = (Nr * Nphi + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int b_Ephi_end = (Nr1 * Nphi + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if (cop->k_endplate_z0 >= 0) {
            ibc_surface_Er_end_kernel<<<b_Er_end, BLOCK_SIZE>>>(
                base->d_result_work.Er, base->d_result_work.Er_im,
                base->d_E_work.Er, base->d_E_work.Er_im,
                gp, bw_end_re, bw_end_im, cop->k_endplate_z0,
                cop->d_mask_Er_z0);
            ibc_surface_Ephi_end_kernel<<<b_Ephi_end, BLOCK_SIZE>>>(
                base->d_result_work.Ephi, base->d_result_work.Ephi_im,
                base->d_E_work.Ephi, base->d_E_work.Ephi_im,
                gp, bw_end_re, bw_end_im, cop->k_endplate_z0,
                cop->d_mask_Ephi_z0);
        }
        if (cop->k_endplate_zL >= 0) {
            ibc_surface_Er_end_kernel<<<b_Er_end, BLOCK_SIZE>>>(
                base->d_result_work.Er, base->d_result_work.Er_im,
                base->d_E_work.Er, base->d_E_work.Er_im,
                gp, bw_end_re, bw_end_im, cop->k_endplate_zL,
                cop->d_mask_Er_zL);
            ibc_surface_Ephi_end_kernel<<<b_Ephi_end, BLOCK_SIZE>>>(
                base->d_result_work.Ephi, base->d_result_work.Ephi_im,
                base->d_E_work.Ephi, base->d_E_work.Ephi_im,
                gp, bw_end_re, bw_end_im, cop->k_endplate_zL,
                cop->d_mask_Ephi_zL);
        }
    }

    /* Step 6c: Conformal IBC at pipe walls */
    gpu_apply_conformal_ibc(&base->d_E_work, &base->d_result_work,
                            alpha, &cop->conformal);

    /* Step 7: PEC at grid edges on result */
    apply_pipe_cap_pec_complex(&base->d_result_work, Nr, Nr1, Nphi, Nz);

    /* Step 8: IBC mask on result */
    if (pop->has_ibc_mask)
        gpu_apply_material_mask_complex(&base->d_result_work, &pop->ibc_mask);

    /* Step 8b: Re-zero endplate surface cells on result */
    rezero_endplate_surface(&base->d_result_work, Nr, Nr1, Nphi, Nz,
        cop->k_endplate_z0, cop->k_endplate_zL,
        cop->d_mask_Er_z0, cop->d_mask_Ephi_z0,
        cop->d_mask_Er_zL, cop->d_mask_Ephi_zL);

    /* Step 9: Pack */
    gpu_pack_field_complex(&base->d_result_work, d_y,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez, n_real);

    return 0;
}

/*=============================================================================
 * Matvec wrapper
 *============================================================================*/
static int conformal_pipe_matvec_wrapper(
    const void* op, const double* d_x, double* d_y, double alpha
) {
    return gpu_conformal_pipe_matvec_complex(
        (const GPU_ConformalPipeOperator*)op, d_x, d_y, alpha);
}

/*=============================================================================
 * RQI Complex Eigensolver
 *============================================================================*/
GPU_ComplexEigenResult gpu_rqi_complex_conformal_pipe(
    const GPU_ConformalPipeOperator* cop,
    double* d_x, double sigma_init, double conductivity,
    int max_iter, double tol, int gmres_restart
) {
    const CurlCurlOperator* cpu_op = cop->pipe_op.base.cpu_op;
    ComplexEigensolverWorkspace ws;
    complex_eigensolver_workspace_init(&ws, cpu_op->n_total, gmres_restart);
    GPU_ComplexEigenResult result = gpu_rqi_complex_ws(
        cop, conformal_pipe_matvec_wrapper, cpu_op,
        &((GPU_ConformalPipeOperator*)cop)->pipe_op.base.reduction_ws,
        d_x, sigma_init, conductivity, max_iter, tol, &ws);
    complex_eigensolver_workspace_free(&ws);
    return result;
}

/*=============================================================================
 * Diagnostic: per-surface IBC Rayleigh-shift breakdown.
 *============================================================================*/
static void zero_efield_complex(GPU_EField* E) {
    cudaMemset(E->Er,      0, E->size_Er   * sizeof(double));
    cudaMemset(E->Ephi,    0, E->size_Ephi * sizeof(double));
    cudaMemset(E->Ez,      0, E->size_Ez   * sizeof(double));
    cudaMemset(E->Er_im,   0, E->size_Er   * sizeof(double));
    cudaMemset(E->Ephi_im, 0, E->size_Ephi * sizeof(double));
    cudaMemset(E->Ez_im,   0, E->size_Ez   * sizeof(double));
}

static void finalize_correction_like_matvec(
    const GPU_ConformalPipeOperator* cop,
    GPU_EField* E_corr,
    int Nr, int Nr1, int Nphi, int Nz
) {
    const GPU_PipeOperator* pop = &cop->pipe_op;
    apply_pipe_cap_pec_complex(E_corr, Nr, Nr1, Nphi, Nz);
    if (pop->has_ibc_mask)
        gpu_apply_material_mask_complex(E_corr, &pop->ibc_mask);
    rezero_endplate_surface(E_corr, Nr, Nr1, Nphi, Nz,
        cop->k_endplate_z0, cop->k_endplate_zL,
        cop->d_mask_Er_z0, cop->d_mask_Ephi_z0,
        cop->d_mask_Er_zL, cop->d_mask_Ephi_zL);
}

static void bucket_rayleigh_shift(
    const char* label,
    const CurlCurlOperator* cpu_op,
    ReductionWorkspace* red_ws,
    const double* d_x,
    GPU_EField* E_corr,
    double* d_Sx,
    int n_real,
    double xx,
    double k2_re,
    double* out_re,
    double* out_im
) {
    gpu_pack_field_complex(E_corr, d_Sx,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez, n_real);

    double xSx_re = 0.0, xSx_im = 0.0;
    gpu_vec_dot_weighted_complex_re_ws(d_x, d_Sx, &xSx_re, cpu_op, n_real, red_ws);
    gpu_vec_dot_weighted_complex_im_ws(d_x, d_Sx, &xSx_im, cpu_op, n_real, red_ws);

    *out_re = xSx_re / xx;
    *out_im = xSx_im / xx;
    double q_equiv = (fabs(*out_im) > 1e-30) ? -k2_re / *out_im : 0.0;
    printf("    %-20s  dk2_re=%+.6e  dk2_im=%+.6e  Q_if_only=%10.1f\n",
        label, *out_re, *out_im, q_equiv);
}

int gpu_conformal_pipe_print_ibc_breakdown(
    const GPU_ConformalPipeOperator* cop,
    const double* d_x,
    double k2_re,
    double conductivity
) {
    const GPU_PipeOperator* pop = &cop->pipe_op;
    const CurlCurlOperator* cpu_op = pop->base.cpu_op;
    ReductionWorkspace* red_ws =
        &((GPU_ConformalPipeOperator*)cop)->pipe_op.base.reduction_ws;
    const GridParams* grid = &cpu_op->grid;
    int n_real = cpu_op->n_total;
    int Nr = grid->Nr, Nr1 = Nr + 1, Nphi = grid->Nphi, Nz = grid->Nz;
    int n2 = 2 * n_real;

    BndGridParams gp = make_bnd_params(grid);
    double alpha = ibc_compute_alpha(k2_re, conductivity);
    double inv_2a = 1.0 / (2.0 * alpha);
    double beta_re = inv_2a, beta_im = -inv_2a;
    double bw_inner_re = beta_re / grid->dr, bw_inner_im = beta_im / grid->dr;
    double bw_end_re = beta_re / grid->dz, bw_end_im = beta_im / grid->dz;

    GPU_EField E_input;
    GPU_EField E_corr;
    if (gpu_efield_alloc_complex(&E_input, grid) != 0) return -1;
    if (gpu_efield_alloc_complex(&E_corr, grid) != 0) {
        gpu_efield_free(&E_input);
        return -1;
    }

    double* d_Sx = NULL;
    CUDA_CHECK(cudaMalloc(&d_Sx, n2 * sizeof(double)));

    gpu_unpack_field_complex(d_x, &E_input,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez, n_real);

    apply_pipe_cap_pec_complex(&E_input, Nr, Nr1, Nphi, Nz);
    if (pop->has_ibc_mask)
        gpu_apply_material_mask_complex(&E_input, &pop->ibc_mask);
    rezero_endplate_surface(&E_input, Nr, Nr1, Nphi, Nz,
        cop->k_endplate_z0, cop->k_endplate_zL,
        cop->d_mask_Er_z0, cop->d_mask_Ephi_z0,
        cop->d_mask_Er_zL, cop->d_mask_Ephi_zL);

    double xx = 0.0;
    gpu_vec_dot_weighted_complex_re_ws(d_x, d_x, &xx, cpu_op, n_real, red_ws);

    printf("\n  IBC loss breakdown using final iterative eigenvector:\n");
    printf("    alpha = %.6e,  <x,x> = %.6e\n", alpha, xx);

    double re[7] = {0,0,0,0,0,0,0};
    double im[7] = {0,0,0,0,0,0,0};

    zero_efield_complex(&E_corr);
    ibc_inner_Ephi_kernel<<<(Nphi*(Nz+1)+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        E_corr.Ephi, E_corr.Ephi_im, E_input.Ephi, E_input.Ephi_im,
        Nr1, Nphi, Nz, bw_inner_re, bw_inner_im,
        pop->base.has_port_masks ? pop->base.d_mask_Ephi_inner : NULL);
    ibc_inner_Ez_kernel<<<(Nphi*Nz+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        E_corr.Ez, E_corr.Ez_im, E_input.Ez, E_input.Ez_im,
        Nr1, Nphi, Nz, bw_inner_re, bw_inner_im,
        pop->base.has_port_masks ? pop->base.d_mask_Ez_inner : NULL);
    finalize_correction_like_matvec(cop, &E_corr, Nr, Nr1, Nphi, Nz);
    bucket_rayleigh_shift("inner conductor", cpu_op, red_ws, d_x, &E_corr,
        d_Sx, n_real, xx, k2_re, &re[0], &im[0]);

    if (pop->has_ibc_weights) {
        const GPU_IBCWeights* w = &pop->ibc_weights;
        zero_efield_complex(&E_corr);
        ibc_weight_exclude_kernel<<<(w->size_Er+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            E_corr.Er, E_corr.Er_im, E_input.Er, E_input.Er_im,
            w->d_weight_Er, cop->conformal.d_ibc_weight_Er,
            beta_re, beta_im, w->size_Er);
        finalize_correction_like_matvec(cop, &E_corr, Nr, Nr1, Nphi, Nz);
        bucket_rayleigh_shift("staircase Er", cpu_op, red_ws, d_x, &E_corr,
            d_Sx, n_real, xx, k2_re, &re[1], &im[1]);

        zero_efield_complex(&E_corr);
        ibc_weight_exclude_kernel<<<(w->size_Ephi+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            E_corr.Ephi, E_corr.Ephi_im, E_input.Ephi, E_input.Ephi_im,
            w->d_weight_Ephi, cop->conformal.d_ibc_weight_Ephi,
            beta_re, beta_im, w->size_Ephi);
        finalize_correction_like_matvec(cop, &E_corr, Nr, Nr1, Nphi, Nz);
        bucket_rayleigh_shift("staircase Ephi", cpu_op, red_ws, d_x, &E_corr,
            d_Sx, n_real, xx, k2_re, &re[2], &im[2]);

        zero_efield_complex(&E_corr);
        ibc_weight_exclude_kernel<<<(w->size_Ez+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            E_corr.Ez, E_corr.Ez_im, E_input.Ez, E_input.Ez_im,
            w->d_weight_Ez, cop->conformal.d_ibc_weight_Ez,
            beta_re, beta_im, w->size_Ez);
        finalize_correction_like_matvec(cop, &E_corr, Nr, Nr1, Nphi, Nz);
        bucket_rayleigh_shift("staircase Ez", cpu_op, red_ws, d_x, &E_corr,
            d_Sx, n_real, xx, k2_re, &re[3], &im[3]);
    }

    zero_efield_complex(&E_corr);
    {
        int b_Er = (Nr * Nphi + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int b_Ephi = (Nr1 * Nphi + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (cop->k_endplate_z0 >= 0) {
            ibc_surface_Er_end_kernel<<<b_Er, BLOCK_SIZE>>>(
                E_corr.Er, E_corr.Er_im, E_input.Er, E_input.Er_im,
                gp, bw_end_re, bw_end_im, cop->k_endplate_z0, cop->d_mask_Er_z0);
            ibc_surface_Ephi_end_kernel<<<b_Ephi, BLOCK_SIZE>>>(
                E_corr.Ephi, E_corr.Ephi_im, E_input.Ephi, E_input.Ephi_im,
                gp, bw_end_re, bw_end_im, cop->k_endplate_z0, cop->d_mask_Ephi_z0);
        }
    }
    finalize_correction_like_matvec(cop, &E_corr, Nr, Nr1, Nphi, Nz);
    bucket_rayleigh_shift("endplate z0", cpu_op, red_ws, d_x, &E_corr,
        d_Sx, n_real, xx, k2_re, &re[4], &im[4]);

    zero_efield_complex(&E_corr);
    {
        int b_Er = (Nr * Nphi + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int b_Ephi = (Nr1 * Nphi + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (cop->k_endplate_zL >= 0) {
            ibc_surface_Er_end_kernel<<<b_Er, BLOCK_SIZE>>>(
                E_corr.Er, E_corr.Er_im, E_input.Er, E_input.Er_im,
                gp, bw_end_re, bw_end_im, cop->k_endplate_zL, cop->d_mask_Er_zL);
            ibc_surface_Ephi_end_kernel<<<b_Ephi, BLOCK_SIZE>>>(
                E_corr.Ephi, E_corr.Ephi_im, E_input.Ephi, E_input.Ephi_im,
                gp, bw_end_re, bw_end_im, cop->k_endplate_zL, cop->d_mask_Ephi_zL);
        }
    }
    finalize_correction_like_matvec(cop, &E_corr, Nr, Nr1, Nphi, Nz);
    bucket_rayleigh_shift("endplate zL", cpu_op, red_ws, d_x, &E_corr,
        d_Sx, n_real, xx, k2_re, &re[5], &im[5]);

    zero_efield_complex(&E_corr);
    gpu_apply_conformal_ibc(&E_input, &E_corr, alpha, &cop->conformal);
    finalize_correction_like_matvec(cop, &E_corr, Nr, Nr1, Nphi, Nz);
    bucket_rayleigh_shift("conformal pipes", cpu_op, red_ws, d_x, &E_corr,
        d_Sx, n_real, xx, k2_re, &re[6], &im[6]);

    double total_re = 0.0, total_im = 0.0;
    for (int i = 0; i < 7; i++) {
        total_re += re[i];
        total_im += im[i];
    }
    double q_total = (fabs(total_im) > 1e-30) ? -k2_re / total_im : 0.0;
    printf("    %-20s  dk2_re=%+.6e  dk2_im=%+.6e  Q_if_only=%10.1f\n",
        "sum", total_re, total_im, q_total);
    if (fabs(total_im) > 1e-30) {
        double stair_im = im[1] + im[2] + im[3];
        double end_im = im[4] + im[5];
        printf("    loss fractions: inner=%.1f%% staircase=%.1f%% endplate=%.1f%% conformal=%.1f%%\n",
            100.0 * im[0] / total_im,
            100.0 * stair_im / total_im,
            100.0 * end_im / total_im,
            100.0 * im[6] / total_im);
        printf("    staircase split: Er=%.1f%% Ephi=%.1f%% Ez=%.1f%% of staircase\n",
            (fabs(stair_im) > 1e-30) ? 100.0 * im[1] / stair_im : 0.0,
            (fabs(stair_im) > 1e-30) ? 100.0 * im[2] / stair_im : 0.0,
            (fabs(stair_im) > 1e-30) ? 100.0 * im[3] / stair_im : 0.0);
        printf("    endplate split: z0=%.1f%% zL=%.1f%% of endplate\n",
            (fabs(end_im) > 1e-30) ? 100.0 * im[4] / end_im : 0.0,
            (fabs(end_im) > 1e-30) ? 100.0 * im[5] / end_im : 0.0);

        printf("    endplate scale sweep (diagnostic only):\n");
        const double scales[] = {1.25, 1.50, 1.60, 1.75, 2.00};
        for (int si = 0; si < 5; si++) {
            double scaled_im = total_im + (scales[si] - 1.0) * end_im;
            double scaled_q = (fabs(scaled_im) > 1e-30) ? -k2_re / scaled_im : 0.0;
            printf("      endplate x%.2f -> dk2_im=%+.6e  Q=%.1f\n",
                scales[si], scaled_im, scaled_q);
        }
    }

    cudaFree(d_Sx);
    gpu_efield_free(&E_corr);
    gpu_efield_free(&E_input);
    return 0;
}

/*=============================================================================
 * Perturbative IBC Q
 *============================================================================*/
GPU_ComplexEigenResult gpu_ibc_perturbative_conformal_pipe(
    const GPU_ConformalPipeOperator* cop,
    const double* d_x_pec, double k2_pec, double conductivity
) {
    GPU_ComplexEigenResult result;
    result.iterations = 0; result.residual = 0.0; result.converged = 1;

    const GPU_PipeOperator* pop = &cop->pipe_op;
    const CurlCurlOperator* cpu_op = pop->base.cpu_op;
    ReductionWorkspace* red_ws =
        &((GPU_ConformalPipeOperator*)cop)->pipe_op.base.reduction_ws;
    const GridParams* grid = &cpu_op->grid;
    int n_real = cpu_op->n_total;
    int Nr = grid->Nr, Nr1 = Nr + 1, Nphi = grid->Nphi, Nz = grid->Nz;
    double c0 = 299792458.0;

    BndGridParams gp = make_bnd_params(grid);

    double alpha = ibc_compute_alpha(k2_pec, conductivity);
    double inv_2a = 1.0 / (2.0 * alpha);
    double beta_re =  inv_2a, beta_im = -inv_2a;
    double bw_inner_re = beta_re / grid->dr, bw_inner_im = beta_im / grid->dr;
    double bw_end_re = beta_re / grid->dz, bw_end_im = beta_im / grid->dz;

    GPU_EField E_input;
    gpu_efield_alloc_complex(&E_input, grid);
    cudaMemcpy(E_input.Er, d_x_pec + cpu_op->offset_Er,
        cpu_op->size_Er * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(E_input.Ephi, d_x_pec + cpu_op->offset_Ephi,
        cpu_op->size_Ephi * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(E_input.Ez, d_x_pec + cpu_op->offset_Ez,
        cpu_op->size_Ez * sizeof(double), cudaMemcpyDeviceToDevice);

    GPU_EField E_corr;
    gpu_efield_alloc_complex(&E_corr, grid);
    cudaMemset(E_corr.Er,      0, cpu_op->size_Er   * sizeof(double));
    cudaMemset(E_corr.Ephi,    0, cpu_op->size_Ephi * sizeof(double));
    cudaMemset(E_corr.Ez,      0, cpu_op->size_Ez   * sizeof(double));
    cudaMemset(E_corr.Er_im,   0, cpu_op->size_Er   * sizeof(double));
    cudaMemset(E_corr.Ephi_im, 0, cpu_op->size_Ephi * sizeof(double));
    cudaMemset(E_corr.Ez_im,   0, cpu_op->size_Ez   * sizeof(double));

    /* Inner conductor IBC at i=0 */
    ibc_inner_Ephi_kernel<<<(Nphi*(Nz+1)+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        E_corr.Ephi, E_corr.Ephi_im, E_input.Ephi, E_input.Ephi_im,
        Nr1, Nphi, Nz, bw_inner_re, bw_inner_im,
        pop->base.has_port_masks ? pop->base.d_mask_Ephi_inner : NULL);
    ibc_inner_Ez_kernel<<<(Nphi*Nz+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        E_corr.Ez, E_corr.Ez_im, E_input.Ez, E_input.Ez_im,
        Nr1, Nphi, Nz, bw_inner_re, bw_inner_im,
        pop->base.has_port_masks ? pop->base.d_mask_Ez_inner : NULL);

    /* Staircase IBC at flat walls (excl pipe cells) */
    if (pop->has_ibc_weights) {
        const GPU_IBCWeights* w = &pop->ibc_weights;
        ibc_weight_exclude_kernel<<<(w->size_Er+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            E_corr.Er, E_corr.Er_im, E_input.Er, E_input.Er_im,
            w->d_weight_Er, cop->conformal.d_ibc_weight_Er,
            beta_re, beta_im, w->size_Er);
        ibc_weight_exclude_kernel<<<(w->size_Ephi+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            E_corr.Ephi, E_corr.Ephi_im, E_input.Ephi, E_input.Ephi_im,
            w->d_weight_Ephi, cop->conformal.d_ibc_weight_Ephi,
            beta_re, beta_im, w->size_Ephi);
        ibc_weight_exclude_kernel<<<(w->size_Ez+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            E_corr.Ez, E_corr.Ez_im, E_input.Ez, E_input.Ez_im,
            w->d_weight_Ez, cop->conformal.d_ibc_weight_Ez,
            beta_re, beta_im, w->size_Ez);
    }

    /* Endplate IBC at k=k_z0, k=k_zL */
    {
        int b_Er   = (Nr * Nphi + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int b_Ephi = (Nr1 * Nphi + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if (cop->k_endplate_z0 >= 0) {
            ibc_surface_Er_end_kernel<<<b_Er, BLOCK_SIZE>>>(
                E_corr.Er, E_corr.Er_im, E_input.Er, E_input.Er_im,
                gp, bw_end_re, bw_end_im, cop->k_endplate_z0, cop->d_mask_Er_z0);
            ibc_surface_Ephi_end_kernel<<<b_Ephi, BLOCK_SIZE>>>(
                E_corr.Ephi, E_corr.Ephi_im, E_input.Ephi, E_input.Ephi_im,
                gp, bw_end_re, bw_end_im, cop->k_endplate_z0, cop->d_mask_Ephi_z0);
        }
        if (cop->k_endplate_zL >= 0) {
            ibc_surface_Er_end_kernel<<<b_Er, BLOCK_SIZE>>>(
                E_corr.Er, E_corr.Er_im, E_input.Er, E_input.Er_im,
                gp, bw_end_re, bw_end_im, cop->k_endplate_zL, cop->d_mask_Er_zL);
            ibc_surface_Ephi_end_kernel<<<b_Ephi, BLOCK_SIZE>>>(
                E_corr.Ephi, E_corr.Ephi_im, E_input.Ephi, E_input.Ephi_im,
                gp, bw_end_re, bw_end_im, cop->k_endplate_zL, cop->d_mask_Ephi_zL);
        }
    }

    /* Conformal IBC at pipe walls */
    gpu_apply_conformal_ibc(&E_input, &E_corr, alpha, &cop->conformal);

    /* Dot products */
    int n2 = 2 * n_real;
    double* d_Sx; cudaMalloc(&d_Sx, n2 * sizeof(double));
    double* d_x_cx; cudaMalloc(&d_x_cx, n2 * sizeof(double));
    gpu_pack_field_complex(&E_corr, d_Sx,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez, n_real);
    gpu_pack_field_complex(&E_input, d_x_cx,
        cpu_op->offset_Er, cpu_op->offset_Ephi, cpu_op->offset_Ez, n_real);

    double xSx_re, xSx_im, xx;
    gpu_vec_dot_weighted_complex_re_ws(d_x_cx, d_Sx, &xSx_re, cpu_op, n_real, red_ws);
    gpu_vec_dot_weighted_complex_im_ws(d_x_cx, d_Sx, &xSx_im, cpu_op, n_real, red_ws);
    gpu_vec_dot_weighted_complex_re_ws(d_x_cx, d_x_cx, &xx, cpu_op, n_real, red_ws);

    double dk2_re = xSx_re / xx, dk2_im = xSx_im / xx;
    result.k2_re = k2_pec + dk2_re;
    result.k2_im = dk2_im;
    result.frequency_Hz = (result.k2_re > 0) ? c0 * sqrt(result.k2_re) / (2.0 * M_PI) : 0.0;
    result.Q_factor = (fabs(dk2_im) > 1e-30) ? -result.k2_re / result.k2_im : 0.0;

    printf("  Conformal Perturbative IBC:\n");
    printf("    k2_PEC       = %.10f\n", k2_pec);
    printf("    Δk2_re       = %+.6e\n", dk2_re);
    printf("    Δk2_im       = %+.6e\n", dk2_im);
    printf("    f            = %.6f MHz\n", result.frequency_Hz / 1e6);
    printf("    Q            = %.1f\n", result.Q_factor);

    cudaFree(d_Sx); cudaFree(d_x_cx);
    gpu_efield_free(&E_input); gpu_efield_free(&E_corr);
    return result;
}
