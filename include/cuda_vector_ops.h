#ifndef CUDA_VECTOR_OPS_H
#define CUDA_VECTOR_OPS_H

#include "cuda_fields.h"
#include "curlcurl_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*=============================================================================
     * Basic Vector Operations (operate on packed vectors on GPU)
     *
     * These work on ANY length.  For complex packed vectors of size 2*n_real,
     * simply pass n = 2*n_real — no special handling needed.
     *============================================================================*/

     /* x = alpha * x */
    int gpu_vec_scale(double* d_x, double alpha, int n);

    /* y = alpha * x + y */
    int gpu_vec_axpy(double alpha, const double* d_x, double* d_y, int n);

    /* y = x (copy) */
    int gpu_vec_copy(const double* d_x, double* d_y, int n);

    /* x = 0 */
    int gpu_vec_zero(double* d_x, int n);

    /* y = alpha * x + beta * y */
    int gpu_vec_axpby(double alpha, const double* d_x,
        double beta, double* d_y, int n);

    /*=============================================================================
     * Weighted Inner Products — Real (backward compatible)
     *
     * <x, y>_r = Σ r_i * x_i * y_i * dV
     *
     * Each component weighted by r at its Yee grid location:
     *   Er   at r_{i+1/2} = a + (i+0.5)*dr
     *   Ephi at r_i       = a + i*dr
     *   Ez   at r_i       = a + i*dr
     *
     * These are the most performance-critical operations since they're
     * called every linear-solver iteration.
     *============================================================================*/

     /* Weighted dot product: result = <x, y>_r */
    int gpu_vec_dot_weighted(
        const double* d_x,
        const double* d_y,
        double* result,
        const CurlCurlOperator* op
    );

    /* Weighted norm: result = sqrt(<x, x>_r) */
    int gpu_vec_norm_weighted(
        const double* d_x,
        double* result,
        const CurlCurlOperator* op
    );

    /* Weighted normalize: x = x / ||x||_r */
    int gpu_vec_normalize_weighted(
        double* d_x,
        const CurlCurlOperator* op
    );

    /* Compute dot product and norm simultaneously */
    int gpu_vec_dot_and_norm(
        const double* d_x,
        const double* d_y,
        double* dot_result,
        double* norm_x_result,
        const CurlCurlOperator* op
    );

    /*=============================================================================
     * Reduction workspace (pre-allocated for performance)
     *============================================================================*/

    typedef struct {
        double* d_partial_sums;     /* Partial sums from block reductions */
        double* d_result;           /* Final scalar result on device */
        int num_blocks;             /* Number of reduction blocks */
        int initialized;
    } ReductionWorkspace;

    int reduction_workspace_init(ReductionWorkspace* ws, int n);
    void reduction_workspace_free(ReductionWorkspace* ws);

    /* Version using pre-allocated workspace (avoids malloc per call) */
    int gpu_vec_dot_weighted_ws(
        const double* d_x,
        const double* d_y,
        double* result,
        const CurlCurlOperator* op,
        ReductionWorkspace* ws
    );

    /*=============================================================================
     * Weighted Inner Products — Complex (IBC extension)
     *
     * For complex packed vectors with layout [real_part | imag_part]:
     *   x = x_re (at offset 0) + j * x_im (at offset n_real)
     *
     * Complex inner product:  <x, y> = <x, y>_re + j * <x, y>_im
     *   Re(<x,y>) = <x_re, y_re>_r + <x_im, y_im>_r
     *   Im(<x,y>) = <x_re, y_im>_r - <x_im, y_re>_r
     *
     * Implementation trick: because the imaginary sub-layout mirrors the
     * real sub-layout (same offsets, shifted by n_real), we reuse the
     * existing dot product with pointer arithmetic:
     *   <x_im, y_im>_r = gpu_vec_dot_weighted(d_x + n_real, d_y + n_real, ...)
     *
     * No new GPU kernels are needed.
     *============================================================================*/

    /* Re(<x, y>): real part of complex weighted dot product */
    int gpu_vec_dot_weighted_complex_re(
        const double* d_x,
        const double* d_y,
        double* result,
        const CurlCurlOperator* op,
        int n_real
    );

    /* Im(<x, y>): imaginary part of complex weighted dot product */
    int gpu_vec_dot_weighted_complex_im(
        const double* d_x,
        const double* d_y,
        double* result,
        const CurlCurlOperator* op,
        int n_real
    );

    /* Workspace versions (avoid per-call malloc) */
    int gpu_vec_dot_weighted_complex_re_ws(
        const double* d_x,
        const double* d_y,
        double* result,
        const CurlCurlOperator* op,
        int n_real,
        ReductionWorkspace* ws
    );

    int gpu_vec_dot_weighted_complex_im_ws(
        const double* d_x,
        const double* d_y,
        double* result,
        const CurlCurlOperator* op,
        int n_real,
        ReductionWorkspace* ws
    );

    /* Complex weighted norm: ||x|| = sqrt( <x_re,x_re>_r + <x_im,x_im>_r ) */
    int gpu_vec_norm_weighted_complex(
        const double* d_x,
        double* result,
        const CurlCurlOperator* op,
        int n_real
    );

    int gpu_vec_norm_weighted_complex_ws(
        const double* d_x,
        double* result,
        const CurlCurlOperator* op,
        int n_real,
        ReductionWorkspace* ws
    );

    /* Complex weighted normalize: x = x / ||x||  (scales both re and im) */
    int gpu_vec_normalize_weighted_complex(
        double* d_x,
        const CurlCurlOperator* op,
        int n_real
    );

    int gpu_vec_normalize_weighted_complex_ws(
        double* d_x,
        const CurlCurlOperator* op,
        int n_real,
        ReductionWorkspace* ws
    );

    /*=============================================================================
     * Flat (Unweighted) Dot Products for GMRES
     *
     * Standard L2 inner product — no r-weighting, no component splitting.
     * 1 kernel launch + 1 reduction + 1 cudaMemcpy per call, vs 4+1+1
     * for the weighted version. ~4× faster per inner product.
     *
     * GMRES convergence is unaffected: the inner product only determines
     * which norm the residual is minimized in. The converged solution and
     * the Rayleigh quotient (which uses weighted products) are identical.
     *============================================================================*/

    /* Simple dot product: result = Σ x[i]*y[i] for i=0..n-1 */
    int gpu_vec_dot_flat(
        const double* d_x, const double* d_y,
        double* result, int n
    );

    /* Workspace version (no malloc per call — use this in hot loops) */
    int gpu_vec_dot_flat_ws(
        const double* d_x, const double* d_y,
        double* result, int n,
        double* d_partial,      /* pre-allocated, size >= min(ceil(n/256), 1024) */
        double* d_result_dev    /* pre-allocated, size 1 */
    );

    int gpu_vec_dot_flat_complex_re(
        const double* d_x, const double* d_y,
        double* result, int n_real
    );

    int gpu_vec_dot_flat_complex_im(
        const double* d_x, const double* d_y,
        double* result, int n_real
    );

    int gpu_vec_norm_flat_complex(
        const double* d_x, double* result, int n_real
    );

    /* Workspace versions for GMRES hot loop */
    int gpu_vec_dot_flat_complex_re_ws(
        const double* d_x, const double* d_y,
        double* result, int n_real,
        double* d_partial, double* d_result_dev
    );

    int gpu_vec_dot_flat_complex_im_ws(
        const double* d_x, const double* d_y,
        double* result, int n_real,
        double* d_partial, double* d_result_dev
    );

    int gpu_vec_norm_flat_complex_ws(
        const double* d_x, double* result, int n_real,
        double* d_partial, double* d_result_dev
    );

#ifdef __cplusplus
}
#endif

#endif /* CUDA_VECTOR_OPS_H */
