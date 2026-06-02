#ifndef CUDA_CONFORMAL_H
#define CUDA_CONFORMAL_H

#include "conformal_geometry.h"
#include "cuda_fields.h"
#include "cuda_curls.h"

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * GPU Conformal Data
 *
 * Device-side copies of the ConformalData arrays. All pointers are device
 * pointers allocated with cudaMalloc.
 *
 * The conformal curl kernels read from these arrays at every grid point.
 * For non-cut cells (frac=1.0, area=standard), the conformal stencil
 * reduces exactly to the standard stencil.
 *============================================================================*/

typedef struct {
    /* Edge vacuum fractions (device pointers) */
    double* d_edge_frac_Er;     /* Nr * Nphi * (Nz+1) */
    double* d_edge_frac_Ephi;   /* (Nr+1) * Nphi * (Nz+1) */
    double* d_edge_frac_Ez;     /* (Nr+1) * Nphi * Nz */

    /* Primal face vacuum areas [m²] for curl-E normalization */
    double* d_face_area_Hr;     /* (Nr+1) * Nphi * Nz */
    double* d_face_area_Hphi;   /* Nr * Nphi * Nz */
    double* d_face_area_Hz;     /* Nr * Nphi * (Nz+1) */

    /* Dual face vacuum areas [m²] for curl-H normalization */
    double* d_dual_area_Er;     /* Nr * Nphi * (Nz+1) */
    double* d_dual_area_Ephi;   /* (Nr+1) * Nphi * (Nz+1) */
    double* d_dual_area_Ez;     /* (Nr+1) * Nphi * Nz */

    /* IBC surface weights (device pointers) */
    double* d_ibc_weight_Er;
    double* d_ibc_weight_Ephi;
    double* d_ibc_weight_Ez;

    /* Sizes (copied from CPU ConformalData for convenience) */
    int size_Er, size_Ephi, size_Ez;
    int size_Hr, size_Hphi, size_Hz;

    int initialized;
} GPU_ConformalData;

/*=============================================================================
 * Lifecycle
 *============================================================================*/

/* Upload CPU ConformalData to GPU. Allocates all device arrays. */
int gpu_conformal_data_init(
    GPU_ConformalData* gpu_cd,
    const ConformalData* cpu_cd
);

/* Free all device arrays */
void gpu_conformal_data_free(GPU_ConformalData* gpu_cd);

/*=============================================================================
 * Conformal Curl of E: E-grid → H-grid
 *
 * Same as gpu_compute_curl_E but using Faraday's integral form:
 *   H_face = (1/A_vac) · Σ (f_edge · E_edge · l_edge)
 *
 * For non-cut cells this reduces exactly to the standard stencil.
 *============================================================================*/

/* Real part only */
int gpu_compute_curl_E_conformal(
    const GPU_EField* E,
    GPU_HField* curlE,
    const GridParams* grid,
    const GPU_ConformalData* cd
);

/* Complex (both real and imaginary, no cross-coupling) */
int gpu_compute_curl_E_conformal_complex(
    const GPU_EField* E,
    GPU_HField* curlE,
    const GridParams* grid,
    const GPU_ConformalData* cd
);

/*=============================================================================
 * Conformal Curl of H: H-grid → E-grid
 *
 * Same as gpu_compute_curl_H but normalizing by dual face vacuum area:
 *   E_dual_face = (1/A_dual_vac) · Σ (H_edge · l_edge)
 *
 * H-edges are at cell centers and are typically not cut by pipe walls,
 * so no H-edge fractions are needed. Only the dual-face area changes.
 *============================================================================*/

/* Real part only */
int gpu_compute_curl_H_conformal(
    const GPU_HField* H,
    GPU_EField* curlH,
    const GridParams* grid,
    const GPU_ConformalData* cd
);

/* Complex */
int gpu_compute_curl_H_conformal_complex(
    const GPU_HField* H,
    GPU_EField* curlH,
    const GridParams* grid,
    const GPU_ConformalData* cd
);

/*=============================================================================
 * Conformal Curl-Curl: ∇ × ∇ × E
 *
 * Combines conformal curl-E and conformal curl-H.
 *============================================================================*/

/* Real part only */
int gpu_compute_curl_curl_E_conformal(
    const GPU_EField* E,
    GPU_EField* result,
    GPU_HField* temp,
    const GridParams* grid,
    const GPU_ConformalData* cd
);

/* Complex */
int gpu_compute_curl_curl_E_conformal_complex(
    const GPU_EField* E,
    GPU_EField* result,
    GPU_HField* temp,
    const GridParams* grid,
    const GPU_ConformalData* cd
);

/*=============================================================================
 * Conformal mask: zero E where edge_frac == 0 (replaces material mask)
 *============================================================================*/

int gpu_apply_conformal_mask(
    GPU_EField* E,
    const GPU_ConformalData* cd
);

int gpu_apply_conformal_mask_complex(
    GPU_EField* E,
    const GPU_ConformalData* cd
);

/*=============================================================================
 * Conformal IBC surface correction: result += β · w · E
 *
 * Uses geometrically exact conformal weights instead of staircase 1/dr.
 *============================================================================*/

int gpu_apply_conformal_ibc(
    const GPU_EField* E,
    GPU_EField* result,
    double alpha,           /* R_s / (ω μ₀) */
    const GPU_ConformalData* cd
);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_CONFORMAL_H */
