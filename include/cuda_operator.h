#ifndef CUDA_OPERATOR_H
#define CUDA_OPERATOR_H

#include "cuda_fields.h"
#include "cuda_curls.h"
#include "cuda_vector_ops.h"
#include "curlcurl_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*=============================================================================
     * GPU Operator: wraps all GPU resources for the curl-curl eigenproblem
     *
     * Owns all GPU memory needed for matvec:
     *   - Working E-field (unpacked from vector)
     *   - Result E-field (curl-curl output)
     *   - Temporary H-field (intermediate curl)
     *   - Packed vectors stay on GPU
     *============================================================================*/

    typedef struct {
        /* GPU working arrays
         * When use_ibc == 0 (PEC): real-only (has_imag = 0)
         * When use_ibc == 1 (IBC): complex  (has_imag = 1) */
        GPU_EField d_E_work;
        GPU_EField d_result_work;
        GPU_HField d_H_temp;

        /* Reduction workspace for dot products */
        ReductionWorkspace reduction_ws;

        /* Reference to CPU operator (for grid params, sizes, offsets, ports) */
        const CurlCurlOperator* cpu_op;

        /* Port boundary mask on GPU (NULL if no ports) */
        /* For each boundary point: 1 = free (port), 0 = PEC/IBC */
        int* d_mask_Ephi_inner;     /* size: Nphi * (Nz+1) */
        int* d_mask_Ephi_outer;     /* size: Nphi * (Nz+1) */
        int* d_mask_Ez_inner;       /* size: Nphi * Nz */
        int* d_mask_Ez_outer;       /* size: Nphi * Nz */
        int* d_mask_Er_z0;          /* size: Nr * Nphi */
        int* d_mask_Er_zL;          /* size: Nr * Nphi */
        int* d_mask_Ephi_z0;        /* size: (Nr+1) * Nphi */
        int* d_mask_Ephi_zL;        /* size: (Nr+1) * Nphi */
        int has_port_masks;

        /* IBC mode flag: 0 = PEC (real-only), 1 = IBC (complex) */
        int use_ibc;

        int initialized;
    } GPU_Operator;

    /*=============================================================================
     * Lifecycle
     *============================================================================*/

     /* Initialize GPU operator from CPU operator — PEC mode (real-only) */
    int gpu_operator_init(GPU_Operator* gpu_op, const CurlCurlOperator* cpu_op);

    /* Initialize GPU operator — IBC mode (complex working arrays)
     * Allocates _im arrays for d_E_work, d_result_work, d_H_temp.
     * Sets use_ibc = 1. */
    int gpu_operator_init_complex(GPU_Operator* gpu_op, const CurlCurlOperator* cpu_op);

    /* Free all GPU resources */
    void gpu_operator_free(GPU_Operator* gpu_op);

    /*=============================================================================
     * Core operation: y = A * x  (everything on GPU)
     *
     * x and y are packed vectors on GPU (device pointers)
     *============================================================================*/

    /* PEC matvec: real-only, size n_total (backward compatible) */
    int gpu_curlcurl_matvec(
        const GPU_Operator* gpu_op,
        const double* d_x,
        double* d_y
    );

    /* IBC complex matvec: x,y are size 2*n_total = [real | imag]
     *
     * Operator: A_IBC = ∇×∇× + β·diag(w)
     *
     * Flow:
     *   1. Unpack x → E (re + im), DO NOT apply PEC
     *   2. ∇×∇×E → result (complex)
     *   3. result += β·w·E at wall boundaries (surface correction)
     *   4. Pack result → y
     *
     * Cost: 1 curl-E + 1 curl-H + 8 surface kernels (much cheaper than
     * boundary-value approach).
     *
     * alpha = R_s / (omega * mu_0) from caller (changes each RQI iteration). */
    int gpu_curlcurl_matvec_complex(
        const GPU_Operator* gpu_op,
        const double* d_x,
        double* d_y,
        double alpha
    );

    /*=============================================================================
     * PEC Boundary conditions on GPU (original, backward compatible)
     *============================================================================*/

     /* Apply PEC boundary conditions (no ports) */
    int gpu_apply_PEC_boundary(
        GPU_EField* E,
        const GridParams* grid
    );

    /* Apply PEC boundary conditions with port masks */
    int gpu_apply_PEC_boundary_with_masks(
        GPU_EField* E,
        const GridParams* grid,
        const GPU_Operator* gpu_op
    );

    /*=============================================================================
     * IBC Surface Correction (IBC extension)
     *
     * Impedance Boundary Condition via weak-form surface correction.
     *
     * The continuous weak form with IBC gives:
     *   ∫(∇×E)·(∇×F) dV − (jωμ₀/Z_s) ∮ E_tan·F_tan dS = k² ∫E·F dV
     *
     * The discrete IBC operator is:
     *   A_IBC = ∇×∇× + β·diag(w)
     *
     * where β = (1−j)/(2α), α = R_s/(ωμ₀), and w is the surface-to-volume
     * ratio at each boundary E-field location:
     *   w = 1/dr  for E_φ, E_z on cylindrical surfaces (r=a, r=b)
     *   w = 1/dz  for E_r, E_φ on endplates (z=0, z=L)
     *
     * The curl-curl is computed WITHOUT zeroing E_tan at walls.
     * The surface correction is ADDED to the result.
     * At port apertures (mask = 1): correction is skipped.
     *============================================================================*/

    /* Apply IBC surface correction to curl-curl result.
     * Adds β·diag(w)·E_tan to result at all 4 wall surfaces.
     * E is the INPUT (with nonzero E_tan), result is the curl-curl output.
     * alpha = R_s / (omega * mu_0) = skin_depth / 2. */
    int gpu_apply_IBC_surface_correction(
        const GPU_EField* E,
        GPU_EField* result,
        double alpha,
        const GridParams* grid,
        const GPU_Operator* gpu_op
    );

    /*=============================================================================
     * Convenience: GPU Rayleigh quotient (PEC, real)
     * x is a packed vector on GPU
     *============================================================================*/

    int gpu_rayleigh_quotient(
        const GPU_Operator* gpu_op,
        const double* d_x,
        double* result
    );

    /*=============================================================================
     * IBC Helper: compute alpha from current eigenvalue estimate
     *
     * alpha = R_s / (omega * mu_0) = 1 / sqrt(2 * sigma * omega * mu_0)
     *
     * where omega = c * sqrt(k2_re), sigma = conductivity (e.g. 5.8e7 for Cu)
     *============================================================================*/
#ifndef __CUDACC__
#include <math.h>
#endif

    static inline double ibc_compute_alpha(double k2_re, double sigma) {
        double c0  = 299792458.0;
        double mu0 = 4.0e-7 * M_PI;
        double omega = c0 * sqrt(fabs(k2_re));
        if (omega < 1.0) omega = 1.0;  /* guard against zero */
        return 1.0 / sqrt(2.0 * sigma * omega * mu0);
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_OPERATOR_H */
