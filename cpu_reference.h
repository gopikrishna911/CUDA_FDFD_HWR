/*=============================================================================
 * cpu_reference.h
 *
 * Single-threaded / OpenMP CPU reference implementation of the rhodotron
 * FDFD eigensolver. Algorithmically mirrors the GPU production code in
 *   cuda_operator.cu
 *   cuda_curls.cu
 *   cuda_vector_ops.cu
 *   cuda_eigensolver.cu
 *
 * Build modes:
 *   - With -fopenmp: parallel reduction + parallel kernel loops
 *   - Without  -fopenmp: same source compiles to a strict single-thread
 *     reference (#pragma omp lines become no-ops)
 *
 * The CPU reference reuses the existing CurlCurlOperator struct, GridParams,
 * EField, HField and idx_* index functions from the project. It does NOT
 * touch the existing curlcurl_matvec / power_iteration / RQI functions —
 * those remain the unmodified single-threaded baseline for cross-checking.
 *
 * Naming convention: every new function is prefixed with cpu_ to avoid
 * collision with existing symbols.
 *============================================================================*/
#ifndef CPU_REFERENCE_H
#define CPU_REFERENCE_H

#include "curl_E.h"
#include "curl_H.h"
#include "curlcurl_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * OMP utility — number of threads in use (1 if compiled without OpenMP)
 *============================================================================*/
int  cpu_omp_num_threads(void);
void cpu_omp_set_num_threads(int n);

/*=============================================================================
 * Real (PEC) building blocks — OpenMP parallel
 *
 * These produce numerically equivalent results to the existing single-thread
 * compute_curl_E / compute_curl_H / compute_curl_curl_E in curl_E.cpp and
 * curl_H.cpp, but with parallelized outer loops.
 *============================================================================*/
void cpu_compute_curl_E_omp (const EField* E, HField* G,        const GridParams* grid);
void cpu_compute_curl_H_omp (const HField* H, EField* curlH,    const GridParams* grid);
void cpu_compute_curl_curl_E_omp(const EField* E, EField* result,
                                 HField* G_temp, const GridParams* grid);

void cpu_apply_PEC_boundary_omp(EField* E, const GridParams* grid);

void cpu_pack_field_omp  (const EField* E, double* x, const CurlCurlOperator* op);
void cpu_unpack_field_omp(const double* x, EField* E, const CurlCurlOperator* op);

/*=============================================================================
 * Real PEC matvec — y = A·x
 *
 * Identical algorithm to the existing curlcurl_matvec() but every step is
 * OpenMP-parallel. Uses op->E_work, op->result_work, op->H_temp scratch
 * (one matvec at a time — caller must serialize, exactly like GPU code).
 *============================================================================*/
void cpu_curlcurl_matvec_omp(const CurlCurlOperator* op,
                             const double* x, double* y);

/*=============================================================================
 * Real vector ops (weighted by r at each component's grid location)
 *============================================================================*/
double cpu_vec_dot_weighted_omp (const double* x, const double* y,
                                 const CurlCurlOperator* op);
double cpu_vec_norm_weighted_omp(const double* x, const CurlCurlOperator* op);
void   cpu_vec_normalize_weighted_omp(double* x, const CurlCurlOperator* op);

void cpu_vec_axpy_omp  (double alpha, const double* x, double* y, int n);
void cpu_vec_scale_omp (double* x, double alpha, int n);
void cpu_vec_copy_omp  (const double* x, double* y, int n);
void cpu_vec_zero_omp  (double* x, int n);

/*=============================================================================
 * Real RQI (Rayleigh-Quotient Iteration) using MINRES inner solver.
 *
 * Mirrors gpu_rqi_ws / gpu_rayleigh_quotient_iteration. Returns the result
 * in the same EigenResult struct the existing code uses.
 *============================================================================*/
EigenResult cpu_rqi_minres_omp(const CurlCurlOperator* op,
                               double* x,            /* initial guess, overwritten */
                               double sigma_init,
                               int max_iter,
                               double tol);

/*=============================================================================
 * ============================================================================
 * Complex-valued IBC path
 * ============================================================================
 *
 * Memory layout for complex packed vector (size 2*n_total):
 *    x[0 .. n_total-1]      = real part of all DOFs
 *    x[n_total .. 2n-1]     = imaginary part of all DOFs
 *
 * Within each half the ordering is the same as the real-only PEC code:
 *    [Er block | Ephi block | Ez block]
 *
 * This matches the GPU layout in gpu_unpack_field_complex / pack_complex.
 *============================================================================*/

/*--- Complex E-field & H-field (re + im stored side by side) --------------*/
typedef struct {
    EField re;
    EField im;
} ComplexEField;

typedef struct {
    HField re;
    HField im;
} ComplexHField;

void cpu_cefield_alloc(ComplexEField* E, const GridParams* grid);
void cpu_cefield_free (ComplexEField* E);
void cpu_chfield_alloc(ComplexHField* H, const GridParams* grid);
void cpu_chfield_free (ComplexHField* H);

void cpu_pack_field_complex_omp  (const ComplexEField* E, double* x,
                                  const CurlCurlOperator* op);
void cpu_unpack_field_complex_omp(const double* x, ComplexEField* E,
                                  const CurlCurlOperator* op);

/*--- Complex IBC operator (the matvec's container) -----------------------*/
typedef struct {
    const CurlCurlOperator* op;     /* references existing real op for grid info */
    ComplexEField E_work;
    ComplexEField result_work;
    ComplexHField H_temp;
} CpuComplexOperator;

int  cpu_complex_op_init(CpuComplexOperator* cop, const CurlCurlOperator* op);
void cpu_complex_op_free(CpuComplexOperator* cop);

/*--- Complex IBC matvec  y = (∇×∇× + β·diag(w))·x , β = (1-j)/(2α) -------*/
void cpu_curlcurl_matvec_complex_omp(const CpuComplexOperator* cop,
                                     const double* x, double* y, double alpha);

/*--- Complex weighted inner products (weighted by r) ---------------------*/
void   cpu_vec_dot_weighted_complex_omp(const double* x, const double* y,
                                        double* dot_re, double* dot_im,
                                        const CurlCurlOperator* op);
double cpu_vec_norm_weighted_complex_omp(const double* x,
                                         const CurlCurlOperator* op);
void   cpu_vec_normalize_weighted_complex_omp(double* x,
                                              const CurlCurlOperator* op);

/*--- Complex flat ops (used inside GMRES Arnoldi orthogonalization) ------*/
void   cpu_vec_dot_flat_complex_omp(const double* x, const double* y,
                                    double* dot_re, double* dot_im, int n_real);
double cpu_vec_norm_flat_complex_omp(const double* x, int n_real);

/*--- Complex AXPY: w += (cr + j·ci) * v  on packed [re|im] vector -------*/
void   cpu_vec_axpy_complex_omp(double cr, double ci,
                                const double* v, double* w, int n_real);

/*--- Complex shift: Av -= (σ_re + j·σ_im)·v ------------------------------*/
void   cpu_vec_shift_complex_omp(double* Av, const double* v,
                                 double sigma_re, double sigma_im, int n_real);

/*=============================================================================
 * GMRES(m) — complex, restarted, modified Gram–Schmidt, Givens rotations
 *
 * Solves (A(α) − σI)·x = b where:
 *   A(α) is the complex IBC operator,  σ = σ_re + j σ_im is the shift,
 *   vectors are packed as [real | imag], size 2*n_real.
 *
 * Mirrors gpu_gmres_solve_shifted_complex_ws line-for-line.
 *============================================================================*/
typedef struct {
    int iterations;
    double residual;        /* ||r||/||b|| achieved */
    int converged;
} CpuLinSolveResult;

typedef struct {
    /* Arnoldi basis V[0..m], each size 2*n_real (heap-allocated) */
    double** V;
    double*  w;             /* temporary, size 2*n_real */

    /* Hessenberg matrix (m+1) × m, complex, row-major */
    double*  H_re;
    double*  H_im;

    /* Givens rotation parameters */
    double*  cs_re; double*  cs_im;
    double*  sn_re; double*  sn_im;

    /* RHS and solution */
    double*  g_re;  double*  g_im;
    double*  y_re;  double*  y_im;

    int m;
    int n_real;
    int initialized;
} CpuGmresWorkspace;

int  cpu_gmres_workspace_init(CpuGmresWorkspace* ws, int n_real, int m);
void cpu_gmres_workspace_free(CpuGmresWorkspace* ws);

CpuLinSolveResult cpu_gmres_solve_complex_omp(
    const CpuComplexOperator* cop,
    double sigma_re, double sigma_im,
    const double* b,         /* size 2*n_real */
    double*       x,         /* size 2*n_real (output, written from 0) */
    double alpha,
    int max_iter,
    double tol,
    CpuGmresWorkspace* ws);

/*=============================================================================
 * Complex Rayleigh-Quotient Iteration
 *
 * Mirrors gpu_rqi_complex_ws.  Solves ∇×∇×E + β·diag(w)·E = k² E with the
 * IBC surface correction, where k² is complex (Re→frequency, Im→Q).
 *============================================================================*/
typedef struct {
    double k2_re;
    double k2_im;
    double frequency_Hz;
    double Q_factor;
    int    iterations;
    double residual;
    int    converged;
} CpuComplexEigenResult;

typedef struct {
    CpuGmresWorkspace gmres_ws;
    double* y;
    double* Ax;
    int n_real;
    int initialized;
} CpuComplexEigenWorkspace;

int  cpu_complex_eigensolver_workspace_init (CpuComplexEigenWorkspace* ws,
                                             int n_real, int gmres_restart);
void cpu_complex_eigensolver_workspace_free (CpuComplexEigenWorkspace* ws);

CpuComplexEigenResult cpu_rqi_complex_omp(
    const CpuComplexOperator* cop,
    double* x,                      /* initial guess, size 2*n_real, overwritten */
    double sigma_init,
    double conductivity,            /* S/m  (e.g. 5.8e7 for Cu) */
    int    max_iter,
    double tol,
    int    gmres_restart,           /* m (e.g. 30) */
    CpuComplexEigenWorkspace* ws);

#ifdef __cplusplus
}
#endif

#endif /* CPU_REFERENCE_H */
