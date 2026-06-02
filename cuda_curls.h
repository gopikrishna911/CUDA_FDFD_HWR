#ifndef CUDA_CURLS_H
#define CUDA_CURLS_H

#include "cuda_fields.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*=============================================================================
     * GPU Curl of E: E-grid → H-grid
     *
     * Computes G = ∇ × E where:
     *   Input:  E-field at E-grid locations (Er, Ephi, Ez)
     *   Output: G-field at H-grid locations (stored in GPU_HField)
     *
     * Components:
     *   Gr   = (1/r) dEz/dφ - dEphi/dz     at (i, j+½, k+½)
     *   Gphi = dEr/dz - dEz/dr              at (i+½, j, k+½)
     *   Gz   = (1/r) d(rEphi)/dr - (1/r) dEr/dφ  at (i+½, j+½, k)
     *============================================================================*/

    /* Real part only (backward compatible) */
    int gpu_compute_curl_E(
        const GPU_EField* E,
        GPU_HField* curlE,
        const GridParams* grid
    );

    int gpu_compute_curl_E_r(
        const GPU_EField* E,
        GPU_HField* curlE,
        const GridParams* grid
    );

    int gpu_compute_curl_E_phi(
        const GPU_EField* E,
        GPU_HField* curlE,
        const GridParams* grid
    );

    int gpu_compute_curl_E_z(
        const GPU_EField* E,
        GPU_HField* curlE,
        const GridParams* grid
    );

    /*=============================================================================
     * GPU Curl of E — Complex (IBC extension)
     *
     * Runs the same stencils on BOTH real and imaginary parts.
     * E must have has_imag == 1; curlE must have has_imag == 1.
     *
     * Result:
     *   curlE.Hr    = curl_r(E.Er,    E.Ephi,    E.Ez)       (real)
     *   curlE.Hr_im = curl_r(E.Er_im, E.Ephi_im, E.Ez_im)   (imag)
     *   (same for phi, z components)
     *
     * NOTE: No cross-coupling here. The curl is linear, so curl(E_re + j*E_im)
     *       = curl(E_re) + j*curl(E_im). Cross-coupling enters only through
     *       the IBC boundary conditions (Phase 4).
     *============================================================================*/

    int gpu_compute_curl_E_complex(
        const GPU_EField* E,
        GPU_HField* curlE,
        const GridParams* grid
    );

    int gpu_compute_curl_E_r_complex(
        const GPU_EField* E,
        GPU_HField* curlE,
        const GridParams* grid
    );

    int gpu_compute_curl_E_phi_complex(
        const GPU_EField* E,
        GPU_HField* curlE,
        const GridParams* grid
    );

    int gpu_compute_curl_E_z_complex(
        const GPU_EField* E,
        GPU_HField* curlE,
        const GridParams* grid
    );

    /*=============================================================================
     * GPU Curl of H: H-grid → E-grid
     *
     * Computes F = ∇ × H where:
     *   Input:  H-field at H-grid locations (Hr, Hphi, Hz)
     *   Output: F-field at E-grid locations (stored in GPU_EField)
     *
     * Components:
     *   Fr   = (1/r) dHz/dφ - dHphi/dz       at (i+½, j, k)
     *   Fphi = dHr/dz - dHz/dr               at (i, j+½, k)
     *   Fz   = (1/r) d(rHphi)/dr - (1/r) dHr/dφ  at (i, j, k+½)
     *============================================================================*/

    /* Real part only (backward compatible) */
    int gpu_compute_curl_H(
        const GPU_HField* H,
        GPU_EField* curlH,
        const GridParams* grid
    );

    int gpu_compute_curl_H_r(
        const GPU_HField* H,
        GPU_EField* curlH,
        const GridParams* grid
    );

    int gpu_compute_curl_H_phi(
        const GPU_HField* H,
        GPU_EField* curlH,
        const GridParams* grid
    );

    int gpu_compute_curl_H_z(
        const GPU_HField* H,
        GPU_EField* curlH,
        const GridParams* grid
    );

    /*=============================================================================
     * GPU Curl of H — Complex (IBC extension)
     *============================================================================*/

    int gpu_compute_curl_H_complex(
        const GPU_HField* H,
        GPU_EField* curlH,
        const GridParams* grid
    );

    int gpu_compute_curl_H_r_complex(
        const GPU_HField* H,
        GPU_EField* curlH,
        const GridParams* grid
    );

    int gpu_compute_curl_H_phi_complex(
        const GPU_HField* H,
        GPU_EField* curlH,
        const GridParams* grid
    );

    int gpu_compute_curl_H_z_complex(
        const GPU_HField* H,
        GPU_EField* curlH,
        const GridParams* grid
    );

    /*=============================================================================
     * GPU Curl-Curl: combines both steps
     *
     * Computes ∇ × ∇ × E = ∇ × (∇ × E)
     *   Step 1: G = ∇ × E  (E-grid → H-grid)
     *   Step 2: result = ∇ × G  (H-grid → E-grid)
     *============================================================================*/

    /* Real part only (backward compatible) */
    int gpu_compute_curl_curl_E(
        const GPU_EField* E,
        GPU_EField* result,
        GPU_HField* temp,       /* Pre-allocated temporary H-field */
        const GridParams* grid
    );

    /* Complex: runs curl-curl on both real and imaginary parts */
    int gpu_compute_curl_curl_E_complex(
        const GPU_EField* E,
        GPU_EField* result,
        GPU_HField* temp,       /* Pre-allocated temporary H-field (must have has_imag=1) */
        const GridParams* grid
    );

#ifdef __cplusplus
}
#endif

#endif /* CUDA_CURLS_H */
