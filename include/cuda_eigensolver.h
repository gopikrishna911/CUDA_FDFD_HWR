#ifndef CUDA_EIGENSOLVER_H
#define CUDA_EIGENSOLVER_H

#include "cuda_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*=============================================================================
     * Common result types
     *============================================================================*/

    typedef struct {
        int iterations;
        double residual;
        int converged;
    } GPU_LinearSolverResult;

    typedef struct {
        double eigenvalue;
        int iterations;
        double residual;
        int converged;
    } GPU_EigenResult;

    /*=============================================================================
     * GPU MINRES solver for (A - σI)z = b  (PEC, real, symmetric)
     *
     * All vectors stay on GPU. Only scalar results copied to host.
     *============================================================================*/

    GPU_LinearSolverResult gpu_minres_solve_shifted(
        const GPU_Operator* gpu_op,
        double sigma,
        const double* d_b,
        double* d_x,
        int max_iter,
        double tol
    );

    /*=============================================================================
     * MINRES workspace (pre-allocated for performance)
     *============================================================================*/

    typedef struct {
        double* d_v_old;
        double* d_v_cur;
        double* d_v_new;
        double* d_Av;
        double* d_w_old;
        double* d_w_cur;
        double* d_w_new;
        int n;
        int initialized;
    } MINRES_Workspace;

    int minres_workspace_init(MINRES_Workspace* ws, int n);
    void minres_workspace_free(MINRES_Workspace* ws);

    GPU_LinearSolverResult gpu_minres_solve_shifted_ws(
        const GPU_Operator* gpu_op,
        double sigma,
        const double* d_b,
        double* d_x,
        int max_iter,
        double tol,
        MINRES_Workspace* ws
    );

    /*=============================================================================
     * GPU GMRES(m) solver for (A_IBC - σI)z = b  (IBC, complex, non-Hermitian)
     *
     * Restarted GMRES with complex arithmetic.
     * Vectors are packed as [real | imag], size 2*n_real.
     * σ is complex: σ_re + j*σ_im.
     * alpha is the IBC parameter passed to the complex matvec.
     *
     * Uses weighted complex inner product for Arnoldi orthogonalization.
     *============================================================================*/

    /*=============================================================================
     * Complex matvec function pointer
     *
     * Signature: int matvec(operator, input, output, alpha)
     * Allows GMRES and RQI to work with different operator types:
     *   - gpu_curlcurl_matvec_complex  (reference cavity, no pipes)
     *   - gpu_pipe_matvec_complex      (full model with pipes)
     *   - any future operator variant
     *
     * The void* opaque parameter carries the operator struct (GPU_Operator*
     * or GPU_PipeOperator*). The caller is responsible for casting.
     *============================================================================*/

    typedef int (*ComplexMatvecFn)(
        const void* op,             /* opaque operator pointer */
        const double* d_x,          /* input, size 2*n_real */
        double* d_y,                /* output, size 2*n_real */
        double alpha                /* IBC parameter */
    );

    /*--- Workspace ---*/

    typedef struct {
        /* GPU: Arnoldi basis vectors V[0..m], each size 2*n_real */
        double** d_V;       /* Array of (m+1) device pointers */
        double* d_w;        /* Temporary vector, size 2*n_real */

        /* GPU: Flat reduction workspace (for fast unweighted dot products) */
        double* d_flat_partial;     /* size 1024 */
        double* d_flat_result;      /* size 1 */

        /* Host: Hessenberg matrix H, (m+1) rows × m cols, complex */
        double* H_re;       /* (m+1)*m, row-major */
        double* H_im;       /* (m+1)*m, row-major */

        /* Host: Givens rotation parameters, complex */
        double* cs_re;      /* m cosines (real part) */
        double* cs_im;      /* m cosines (imag part) */
        double* sn_re;      /* m sines (real part) */
        double* sn_im;      /* m sines (imag part) */

        /* Host: RHS vector g and solution y, complex */
        double* g_re;       /* m+1 */
        double* g_im;       /* m+1 */
        double* y_re;       /* m */
        double* y_im;       /* m */

        int m;              /* Restart parameter */
        int n_real;         /* = n_total (size of real-only packed vector) */
        int initialized;
    } GMRES_Workspace;

    int gmres_workspace_init(GMRES_Workspace* ws, int n_real, int m);
    void gmres_workspace_free(GMRES_Workspace* ws);

    /*--- Solver ---*/

    GPU_LinearSolverResult gpu_gmres_solve_shifted_complex_ws(
        const void* op,             /* opaque operator (GPU_Operator* or GPU_PipeOperator*) */
        ComplexMatvecFn matvec,     /* function pointer for A*x */
        const CurlCurlOperator* cpu_op, /* for grid params, offsets, inner products */
        ReductionWorkspace* red_ws, /* for weighted dot products */
        double sigma_re,
        double sigma_im,
        const double* d_b,          /* RHS on GPU, size 2*n_real */
        double* d_x,                /* Solution on GPU, size 2*n_real (output) */
        double alpha,               /* IBC parameter for complex matvec */
        int max_iter,
        double tol,
        GMRES_Workspace* ws
    );

    /*=============================================================================
     * PEC Eigensolver (existing, unchanged)
     *============================================================================*/

    GPU_EigenResult gpu_rayleigh_quotient_iteration(
        const GPU_Operator* gpu_op,
        double* d_x,
        double sigma_init,
        int max_iter,
        double tol
    );

    GPU_EigenResult gpu_power_iteration(
        const GPU_Operator* gpu_op,
        double* d_x,
        int max_iter,
        double tol
    );

    GPU_EigenResult gpu_inverse_iteration(
        const GPU_Operator* gpu_op,
        double* d_x,
        double sigma,
        int max_iter,
        double tol
    );

    /*=============================================================================
     * PEC Eigensolver workspace (existing, unchanged)
     *============================================================================*/

    typedef struct {
        MINRES_Workspace minres_ws;
        double* d_y;
        double* d_Ax;
        int n;
        int initialized;
    } EigensolverWorkspace;

    int eigensolver_workspace_init(EigensolverWorkspace* ws, int n);
    void eigensolver_workspace_free(EigensolverWorkspace* ws);

    GPU_EigenResult gpu_rqi_ws(
        const GPU_Operator* gpu_op,
        double* d_x,
        double sigma_init,
        int max_iter,
        double tol,
        EigensolverWorkspace* ws
    );

    /*=============================================================================
     * IBC Complex Eigenresult
     *
     * k² = k2_re + j·k2_im
     * With e^{-jωt} convention: k2_im < 0 for lossy cavity.
     * f = c·sqrt(k2_re) / (2π)
     * Q = -k2_re / k2_im   (positive because k2_im < 0)
     *============================================================================*/

    typedef struct {
        double k2_re;               /* Re(k²) — gives frequency */
        double k2_im;               /* Im(k²) — negative for lossy cavity */
        double frequency_Hz;        /* f = c·sqrt(k2_re) / (2π) */
        double Q_factor;            /* Q = -k2_re / k2_im */
        int iterations;
        double residual;            /* ||Ax − σx|| / ||x|| */
        int converged;
    } GPU_ComplexEigenResult;

    /*=============================================================================
     * IBC Complex Eigensolver Workspace
     *============================================================================*/

    typedef struct {
        GMRES_Workspace gmres_ws;
        double* d_y;                /* GMRES solution, size 2*n_real */
        double* d_Ax;               /* Matvec result, size 2*n_real */
        int n_real;
        int initialized;
    } ComplexEigensolverWorkspace;

    int complex_eigensolver_workspace_init(
        ComplexEigensolverWorkspace* ws,
        int n_real,
        int gmres_restart     /* GMRES restart parameter m (e.g. 30) */
    );
    void complex_eigensolver_workspace_free(ComplexEigensolverWorkspace* ws);

    /*=============================================================================
     * IBC Complex Rayleigh Quotient Iteration
     *
     * Solves ∇×∇×E = k²E with impedance boundary conditions.
     * k² is complex: Re(k²) → frequency, Im(k²) → Q factor.
     *
     * Parameters:
     *   gpu_op     : initialized with gpu_operator_init_complex()
     *   d_x        : initial guess, size 2*n_real [real | imag], overwritten
     *   sigma_init : initial real shift estimate for k²
     *   conductivity: wall conductivity [S/m], e.g. 5.8e7 for copper
     *   max_iter   : max RQI outer iterations
     *   tol        : convergence tolerance on |Δk²|/|k²|
     *   ws         : pre-allocated workspace
     *============================================================================*/

    GPU_ComplexEigenResult gpu_rqi_complex_ws(
        const void* op,             /* opaque operator */
        ComplexMatvecFn matvec,     /* function pointer for A*x */
        const CurlCurlOperator* cpu_op,
        ReductionWorkspace* red_ws,
        double* d_x,
        double sigma_init,
        double conductivity,
        int max_iter,
        double tol,
        ComplexEigensolverWorkspace* ws
    );

    /* Convenience version for GPU_Operator (reference cavity, no pipes) */
    GPU_ComplexEigenResult gpu_rqi_complex(
        const GPU_Operator* gpu_op,
        double* d_x,
        double sigma_init,
        double conductivity,
        int max_iter,
        double tol,
        int gmres_restart
    );

#ifdef __cplusplus
}
#endif

#endif /* CUDA_EIGENSOLVER_H */
