#include "cuda_eigensolver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

/* Wall-clock helper for per-RQI-iter timing (matches CPU side's
 * conf_wall_seconds; needed so the paper benchmark can compute per-GMRES-iter
 * time on the GPU as iter_s/GMRES_count). */
static double rqi_wall_seconds() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
        steady_clock::now().time_since_epoch()).count();
}

/*=============================================================================
 * Error checking
 *============================================================================*/
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
    }                                                                   \
} while(0)

#define CUDA_CHECK_RET(call) do {                                       \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        return -1;                                                      \
    }                                                                   \
} while(0)

#define BLOCK_SIZE 256

/*=============================================================================
 * Shifted matvec kernel (PEC, real): y = Ax - σx (UNCHANGED)
 *============================================================================*/
__global__ void shift_kernel(double* Av, const double* v, double sigma, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Av[idx] -= sigma * v[idx];
    }
}

/*=============================================================================
 * Complex shifted matvec kernel (IBC): Av -= σ*v
 *
 * σ = σ_re + j*σ_im,  v = [v_re | v_im]
 *
 * Av[0:n]   -= σ_re*v[0:n]   - σ_im*v[n:2n]
 * Av[n:2n]  -= σ_re*v[n:2n]  + σ_im*v[0:n]
 *============================================================================*/
__global__ void complex_shift_kernel(
    double* Av,
    const double* v,
    double sigma_re,
    double sigma_im,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double v_re = v[idx];
        double v_im = v[idx + n];
        Av[idx]     -= sigma_re * v_re - sigma_im * v_im;
        Av[idx + n] -= sigma_re * v_im + sigma_im * v_re;
    }
}

/*=============================================================================
 * Complex scalar × vector addition: w += (cr + j*ci) * v
 *
 * v, w are packed as [real | imag], each half of size n.
 *
 * w[0:n]   += cr*v[0:n]   - ci*v[n:2n]
 * w[n:2n]  += cr*v[n:2n]  + ci*v[0:n]
 *
 * To subtract, caller passes (-cr, -ci).
 *============================================================================*/
__global__ void complex_axpy_kernel(
    double cr,
    double ci,
    const double* v,
    double* w,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double v_re = v[idx];
        double v_im = v[idx + n];
        w[idx]     += cr * v_re - ci * v_im;
        w[idx + n] += cr * v_im + ci * v_re;
    }
}

/*=============================================================================
 * MINRES Workspace (UNCHANGED)
 *============================================================================*/

int minres_workspace_init(MINRES_Workspace* ws, int n) {
    ws->n = n;
    ws->initialized = 0;

    CUDA_CHECK(cudaMalloc(&ws->d_v_old, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws->d_v_cur, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws->d_v_new, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws->d_Av, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws->d_w_old, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws->d_w_cur, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws->d_w_new, n * sizeof(double)));

    ws->initialized = 1;
    return 0;
}

void minres_workspace_free(MINRES_Workspace* ws) {
    if (!ws->initialized) return;

    cudaFree(ws->d_v_old);
    cudaFree(ws->d_v_cur);
    cudaFree(ws->d_v_new);
    cudaFree(ws->d_Av);
    cudaFree(ws->d_w_old);
    cudaFree(ws->d_w_cur);
    cudaFree(ws->d_w_new);

    ws->initialized = 0;
}

/*=============================================================================
 * Eigensolver Workspace (UNCHANGED)
 *============================================================================*/

int eigensolver_workspace_init(EigensolverWorkspace* ws, int n) {
    ws->n = n;
    ws->initialized = 0;

    minres_workspace_init(&ws->minres_ws, n);

    CUDA_CHECK(cudaMalloc(&ws->d_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws->d_Ax, n * sizeof(double)));

    ws->initialized = 1;
    return 0;
}

void eigensolver_workspace_free(EigensolverWorkspace* ws) {
    if (!ws->initialized) return;

    minres_workspace_free(&ws->minres_ws);
    cudaFree(ws->d_y);
    cudaFree(ws->d_Ax);

    ws->initialized = 0;
}

/*=============================================================================
 * GPU MINRES solver for (A - σI)z = b  (UNCHANGED)
 *============================================================================*/

GPU_LinearSolverResult gpu_minres_solve_shifted_ws(
    const GPU_Operator* gpu_op,
    double sigma,
    const double* d_b,
    double* d_x,
    int max_iter,
    double tol,
    MINRES_Workspace* ws
) {
    GPU_LinearSolverResult result;
    result.iterations = 0;
    result.residual = 1.0;
    result.converged = 0;

    const CurlCurlOperator* cpu_op = gpu_op->cpu_op;
    int n = cpu_op->n_total;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gpu_vec_zero(d_x, n);
    gpu_vec_zero(ws->d_v_old, n);
    gpu_vec_zero(ws->d_w_old, n);
    gpu_vec_zero(ws->d_w_cur, n);

    double b_norm;
    gpu_vec_dot_weighted_ws(d_b, d_b, &b_norm, cpu_op,
        &((GPU_Operator*)gpu_op)->reduction_ws);
    b_norm = sqrt(b_norm);

    if (b_norm < 1e-14) {
        result.converged = 1;
        result.residual = 0.0;
        return result;
    }

    gpu_vec_copy(d_b, ws->d_v_cur, n);
    gpu_vec_scale(ws->d_v_cur, 1.0 / b_norm, n);

    double beta_cur = b_norm;
    double eta = b_norm;
    double c_old = 1.0, c_cur = 1.0;
    double s_old = 0.0, s_cur = 0.0;

    for (int iter = 0; iter < max_iter; iter++) {
        gpu_curlcurl_matvec(gpu_op, ws->d_v_cur, ws->d_Av);
        shift_kernel<<<blocks, BLOCK_SIZE>>>(ws->d_Av, ws->d_v_cur, sigma, n);

        double alpha;
        gpu_vec_dot_weighted_ws(ws->d_v_cur, ws->d_Av, &alpha, cpu_op,
            &((GPU_Operator*)gpu_op)->reduction_ws);

        gpu_vec_copy(ws->d_Av, ws->d_v_new, n);
        gpu_vec_axpy(-alpha, ws->d_v_cur, ws->d_v_new, n);
        gpu_vec_axpy(-beta_cur, ws->d_v_old, ws->d_v_new, n);

        double beta_new;
        gpu_vec_dot_weighted_ws(ws->d_v_new, ws->d_v_new, &beta_new, cpu_op,
            &((GPU_Operator*)gpu_op)->reduction_ws);
        beta_new = sqrt(beta_new);

        if (beta_new > 1e-14) {
            gpu_vec_scale(ws->d_v_new, 1.0 / beta_new, n);
        }

        double rho1 = s_old * beta_cur;
        double rho2 = c_old * c_cur * beta_cur + s_cur * alpha;
        double rho3_bar = c_cur * alpha - c_old * s_cur * beta_cur;
        double gamma = sqrt(rho3_bar * rho3_bar + beta_new * beta_new);

        double c_new, s_new;
        if (gamma > 1e-14) {
            c_new = rho3_bar / gamma;
            s_new = beta_new / gamma;
        } else {
            c_new = 1.0;
            s_new = 0.0;
        }

        gpu_vec_copy(ws->d_v_cur, ws->d_w_new, n);
        gpu_vec_axpy(-rho2, ws->d_w_cur, ws->d_w_new, n);
        gpu_vec_axpy(-rho1, ws->d_w_old, ws->d_w_new, n);
        if (fabs(gamma) > 1e-14) {
            gpu_vec_scale(ws->d_w_new, 1.0 / gamma, n);
        }

        double update_coef = c_new * eta;
        gpu_vec_axpy(update_coef, ws->d_w_new, d_x, n);
        eta = -s_new * eta;

        result.residual = fabs(eta) / b_norm;
        result.iterations = iter + 1;

        if (result.residual < tol) {
            result.converged = 1;
            break;
        }

        double* temp;
        temp = ws->d_v_old;
        ws->d_v_old = ws->d_v_cur;
        ws->d_v_cur = ws->d_v_new;
        ws->d_v_new = temp;

        temp = ws->d_w_old;
        ws->d_w_old = ws->d_w_cur;
        ws->d_w_cur = ws->d_w_new;
        ws->d_w_new = temp;

        beta_cur = beta_new;
        c_old = c_cur; c_cur = c_new;
        s_old = s_cur; s_cur = s_new;
    }

    return result;
}

GPU_LinearSolverResult gpu_minres_solve_shifted(
    const GPU_Operator* gpu_op,
    double sigma,
    const double* d_b,
    double* d_x,
    int max_iter,
    double tol
) {
    int n = gpu_op->cpu_op->n_total;
    MINRES_Workspace ws;
    minres_workspace_init(&ws, n);

    GPU_LinearSolverResult result =
        gpu_minres_solve_shifted_ws(gpu_op, sigma, d_b, d_x, max_iter, tol, &ws);

    minres_workspace_free(&ws);
    return result;
}

/*=============================================================================
 * GMRES(m) Workspace
 *============================================================================*/

int gmres_workspace_init(GMRES_Workspace* ws, int n_real, int m) {
    ws->m = m;
    ws->n_real = n_real;
    ws->initialized = 0;

    int n = 2 * n_real;  /* complex packed vector size */

    /* Allocate m+1 basis vectors on GPU */
    ws->d_V = (double**)malloc((m + 1) * sizeof(double*));
    if (!ws->d_V) return -1;

    for (int j = 0; j <= m; j++) {
        ws->d_V[j] = NULL;
    }
    for (int j = 0; j <= m; j++) {
        CUDA_CHECK_RET(cudaMalloc(&ws->d_V[j], n * sizeof(double)));
        CUDA_CHECK_RET(cudaMemset(ws->d_V[j], 0, n * sizeof(double)));
    }

    /* Temporary vector for matvec result */
    CUDA_CHECK_RET(cudaMalloc(&ws->d_w, n * sizeof(double)));

    /* Flat reduction workspace for fast unweighted dot products */
    CUDA_CHECK_RET(cudaMalloc(&ws->d_flat_partial, 1024 * sizeof(double)));
    CUDA_CHECK_RET(cudaMalloc(&ws->d_flat_result, sizeof(double)));

    /* Host arrays for Hessenberg matrix: (m+1) × m complex */
    ws->H_re = (double*)calloc((size_t)(m + 1) * m, sizeof(double));
    ws->H_im = (double*)calloc((size_t)(m + 1) * m, sizeof(double));

    /* Givens rotation parameters: m complex pairs */
    ws->cs_re = (double*)calloc(m, sizeof(double));
    ws->cs_im = (double*)calloc(m, sizeof(double));
    ws->sn_re = (double*)calloc(m, sizeof(double));
    ws->sn_im = (double*)calloc(m, sizeof(double));

    /* RHS and solution vectors */
    ws->g_re = (double*)calloc(m + 1, sizeof(double));
    ws->g_im = (double*)calloc(m + 1, sizeof(double));
    ws->y_re = (double*)calloc(m, sizeof(double));
    ws->y_im = (double*)calloc(m, sizeof(double));

    ws->initialized = 1;
    return 0;
}

void gmres_workspace_free(GMRES_Workspace* ws) {
    if (!ws->initialized) return;

    if (ws->d_V) {
        for (int j = 0; j <= ws->m; j++) {
            if (ws->d_V[j]) cudaFree(ws->d_V[j]);
        }
        free(ws->d_V);
        ws->d_V = NULL;
    }

    if (ws->d_w) { cudaFree(ws->d_w); ws->d_w = NULL; }

    if (ws->d_flat_partial) { cudaFree(ws->d_flat_partial); ws->d_flat_partial = NULL; }
    if (ws->d_flat_result)  { cudaFree(ws->d_flat_result);  ws->d_flat_result = NULL; }

    free(ws->H_re);  free(ws->H_im);
    free(ws->cs_re); free(ws->cs_im);
    free(ws->sn_re); free(ws->sn_im);
    free(ws->g_re);  free(ws->g_im);
    free(ws->y_re);  free(ws->y_im);

    ws->initialized = 0;
}

/*=============================================================================
 * GMRES(m) Complex Solver
 *
 * Solves (A_IBC - σI)x = b where:
 *   - A_IBC is the complex curl-curl operator with IBC
 *   - σ = σ_re + j·σ_im is a complex shift
 *   - All vectors are packed as [real | imag], size 2*n_real
 *
 * Uses restarted Arnoldi with complex Givens rotations for the
 * least-squares problem.
 *
 * The weighted complex inner product is used for orthogonalization:
 *   <x,y> = <x_re,y_re>_r + <x_im,y_im>_r
 *           + j·(<x_re,y_im>_r - <x_im,y_re>_r)
 *============================================================================*/

GPU_LinearSolverResult gpu_gmres_solve_shifted_complex_ws(
    const void* op,
    ComplexMatvecFn matvec,
    const CurlCurlOperator* cpu_op,
    ReductionWorkspace* red_ws,
    double sigma_re,
    double sigma_im,
    const double* d_b,
    double* d_x,
    double alpha,
    int max_iter,
    double tol,
    GMRES_Workspace* ws
) {
    GPU_LinearSolverResult result;
    result.iterations = 0;
    result.residual = 1.0;
    result.converged = 0;

    int n_real = cpu_op->n_total;
    int n = 2 * n_real;
    int m = ws->m;
    int blocks = (n_real + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* x = 0 */
    gpu_vec_zero(d_x, n);

    /* Compute ||b|| (flat L2 norm — faster than weighted, same convergence) */
    double b_norm;
    gpu_vec_norm_flat_complex_ws(d_b, &b_norm, n_real,
        ws->d_flat_partial, ws->d_flat_result);

    if (b_norm < 1e-14) {
        result.converged = 1;
        result.residual = 0.0;
        return result;
    }

    int total_iters = 0;
    double residual_norm = b_norm;

    /* ================================================================
     * Outer restart loop
     * ================================================================ */
    while (total_iters < max_iter) {

        /* ---- Compute residual r = b - (A-σI)*x ---- */
        if (total_iters == 0) {
            /* First pass: x=0, so r = b */
            gpu_vec_copy(d_b, ws->d_V[0], n);
            residual_norm = b_norm;
        } else {
            /* Restart: compute r = b - (A-σI)*x */
            matvec(op, d_x, ws->d_w, alpha);
            complex_shift_kernel<<<blocks, BLOCK_SIZE>>>(
                ws->d_w, d_x, sigma_re, sigma_im, n_real);

            gpu_vec_copy(d_b, ws->d_V[0], n);
            gpu_vec_axpy(-1.0, ws->d_w, ws->d_V[0], n);

            gpu_vec_norm_flat_complex_ws(
                ws->d_V[0], &residual_norm, n_real,
                ws->d_flat_partial, ws->d_flat_result);

            if (residual_norm / b_norm < tol) {
                result.converged = 1;
                result.residual = residual_norm / b_norm;
                result.iterations = total_iters;
                return result;
            }
        }

        /* v[0] = r / ||r|| */
        gpu_vec_scale(ws->d_V[0], 1.0 / residual_norm, n);

        /* g = [β, 0, 0, ...], β = ||r|| */
        ws->g_re[0] = residual_norm;
        ws->g_im[0] = 0.0;
        for (int i = 1; i <= m; i++) {
            ws->g_re[i] = 0.0;
            ws->g_im[i] = 0.0;
        }

        /* Zero Hessenberg matrix */
        memset(ws->H_re, 0, (size_t)(m + 1) * m * sizeof(double));
        memset(ws->H_im, 0, (size_t)(m + 1) * m * sizeof(double));

        int j_final = 0;

        /* ================================================================
         * Inner Arnoldi loop
         * ================================================================ */
        for (int j = 0; j < m && total_iters < max_iter; j++) {
            total_iters++;

            /* ---- w = (A - σI) * v[j] ---- */
            matvec(op, ws->d_V[j], ws->d_w, alpha);
            complex_shift_kernel<<<blocks, BLOCK_SIZE>>>(
                ws->d_w, ws->d_V[j], sigma_re, sigma_im, n_real);

            /* ---- Modified Gram-Schmidt orthogonalization ---- */
            for (int i = 0; i <= j; i++) {
                /* h[i][j] = <v[i], w>  (flat complex inner product) */
                double h_re, h_im;
                gpu_vec_dot_flat_complex_re_ws(
                    ws->d_V[i], ws->d_w, &h_re, n_real,
                    ws->d_flat_partial, ws->d_flat_result);
                gpu_vec_dot_flat_complex_im_ws(
                    ws->d_V[i], ws->d_w, &h_im, n_real,
                    ws->d_flat_partial, ws->d_flat_result);

                ws->H_re[i * m + j] = h_re;
                ws->H_im[i * m + j] = h_im;

                /* w -= h[i][j] * v[i]  (complex scalar subtraction) */
                complex_axpy_kernel<<<blocks, BLOCK_SIZE>>>(
                    -h_re, -h_im, ws->d_V[i], ws->d_w, n_real);
            }

            /* ---- h[j+1][j] = ||w||  (always real positive) ---- */
            double w_norm;
            gpu_vec_norm_flat_complex_ws(ws->d_w, &w_norm, n_real,
                ws->d_flat_partial, ws->d_flat_result);

            ws->H_re[(j + 1) * m + j] = w_norm;
            ws->H_im[(j + 1) * m + j] = 0.0;

            /* ---- v[j+1] = w / ||w|| ---- */
            if (w_norm > 1e-14) {
                gpu_vec_copy(ws->d_w, ws->d_V[j + 1], n);
                gpu_vec_scale(ws->d_V[j + 1], 1.0 / w_norm, n);
            } else {
                /* Lucky breakdown */
                gpu_vec_zero(ws->d_V[j + 1], n);
            }

            /* ---- Apply previous Givens rotations to column j ---- */
            for (int i = 0; i < j; i++) {
                double a_re = ws->H_re[i * m + j];
                double a_im = ws->H_im[i * m + j];
                double b_re = ws->H_re[(i + 1) * m + j];
                double b_im = ws->H_im[(i + 1) * m + j];

                double c_re = ws->cs_re[i], c_im = ws->cs_im[i];
                double s_re = ws->sn_re[i], s_im = ws->sn_im[i];

                /* new_a = conj(c)*a + conj(s)*b */
                double na_re = (c_re * a_re + c_im * a_im)
                             + (s_re * b_re + s_im * b_im);
                double na_im = (c_re * a_im - c_im * a_re)
                             + (s_re * b_im - s_im * b_re);

                /* new_b = -s*a + c*b */
                double nb_re = -(s_re * a_re - s_im * a_im)
                             +  (c_re * b_re - c_im * b_im);
                double nb_im = -(s_re * a_im + s_im * a_re)
                             +  (c_re * b_im + c_im * b_re);

                ws->H_re[i * m + j]       = na_re;
                ws->H_im[i * m + j]       = na_im;
                ws->H_re[(i + 1) * m + j] = nb_re;
                ws->H_im[(i + 1) * m + j] = nb_im;
            }

            /* ---- Compute new Givens rotation for row j ---- */
            {
                double a_re = ws->H_re[j * m + j];
                double a_im = ws->H_im[j * m + j];
                double b_re = ws->H_re[(j + 1) * m + j];
                double b_im = ws->H_im[(j + 1) * m + j];

                double r = sqrt(a_re * a_re + a_im * a_im
                              + b_re * b_re + b_im * b_im);

                if (r > 1e-14) {
                    ws->cs_re[j] = a_re / r;
                    ws->cs_im[j] = a_im / r;
                    ws->sn_re[j] = b_re / r;
                    ws->sn_im[j] = b_im / r;
                } else {
                    ws->cs_re[j] = 1.0;  ws->cs_im[j] = 0.0;
                    ws->sn_re[j] = 0.0;  ws->sn_im[j] = 0.0;
                }

                /* Apply to H: diagonal becomes r, sub-diagonal becomes 0 */
                ws->H_re[j * m + j]       = r;
                ws->H_im[j * m + j]       = 0.0;
                ws->H_re[(j + 1) * m + j] = 0.0;
                ws->H_im[(j + 1) * m + j] = 0.0;

                /* Apply to g vector */
                double gj_re  = ws->g_re[j],     gj_im  = ws->g_im[j];
                double gj1_re = ws->g_re[j + 1],  gj1_im = ws->g_im[j + 1];

                double c_re = ws->cs_re[j], c_im = ws->cs_im[j];
                double s_re = ws->sn_re[j], s_im = ws->sn_im[j];

                /* g_new[j] = conj(c)*g[j] + conj(s)*g[j+1] */
                ws->g_re[j] = (c_re * gj_re + c_im * gj_im)
                             + (s_re * gj1_re + s_im * gj1_im);
                ws->g_im[j] = (c_re * gj_im - c_im * gj_re)
                             + (s_re * gj1_im - s_im * gj1_re);

                /* g_new[j+1] = -s*g[j] + c*g[j+1] */
                ws->g_re[j + 1] = -(s_re * gj_re - s_im * gj_im)
                                 +  (c_re * gj1_re - c_im * gj1_im);
                ws->g_im[j + 1] = -(s_re * gj_im + s_im * gj_re)
                                 +  (c_re * gj1_im + c_im * gj1_re);
            }

            /* ---- Residual estimate ---- */
            double res_est = sqrt(ws->g_re[j + 1] * ws->g_re[j + 1]
                                + ws->g_im[j + 1] * ws->g_im[j + 1]);
            result.residual = res_est / b_norm;
            result.iterations = total_iters;

            j_final = j + 1;

            if (result.residual < tol) {
                result.converged = 1;
                break;
            }

            /* Lucky breakdown */
            if (w_norm < 1e-14) {
                break;
            }
        } /* end Arnoldi loop */

        /* ================================================================
         * Back-substitution: solve upper triangular H*y = g (complex)
         * H is j_final × j_final upper triangular after Givens rotations.
         * ================================================================ */
        for (int i = j_final - 1; i >= 0; i--) {
            double yi_re = ws->g_re[i];
            double yi_im = ws->g_im[i];

            for (int k = i + 1; k < j_final; k++) {
                double h_re = ws->H_re[i * m + k];
                double h_im = ws->H_im[i * m + k];
                double yk_re = ws->y_re[k];
                double yk_im = ws->y_im[k];

                /* yi -= H[i][k] * y[k]  (complex multiply) */
                yi_re -= h_re * yk_re - h_im * yk_im;
                yi_im -= h_re * yk_im + h_im * yk_re;
            }

            /* yi /= H[i][i]  (should be real after Givens, but handle general) */
            double h_re = ws->H_re[i * m + i];
            double h_im = ws->H_im[i * m + i];
            double denom = h_re * h_re + h_im * h_im;

            if (denom > 1e-28) {
                ws->y_re[i] = (yi_re * h_re + yi_im * h_im) / denom;
                ws->y_im[i] = (yi_im * h_re - yi_re * h_im) / denom;
            } else {
                ws->y_re[i] = 0.0;
                ws->y_im[i] = 0.0;
            }
        }

        /* ================================================================
         * Update solution: x += V * y = Σ y[j] * v[j]
         * ================================================================ */
        for (int j = 0; j < j_final; j++) {
            /* x += (y_re + j*y_im) * v[j] */
            complex_axpy_kernel<<<blocks, BLOCK_SIZE>>>(
                ws->y_re[j], ws->y_im[j], ws->d_V[j], d_x, n_real);
        }
        /* Kernels execute in order on default stream — no sync needed */

        /* Check if converged */
        if (result.converged) break;

    } /* end restart loop */

    return result;
}

/*=============================================================================
 * GPU Weighted Residual (PEC, real): ||Ax - λx||_r  (UNCHANGED)
 *============================================================================*/
static double gpu_compute_residual(
    const GPU_Operator* gpu_op,
    const double* d_Ax,
    const double* d_x,
    double lambda
) {
    const CurlCurlOperator* cpu_op = gpu_op->cpu_op;

    double AxAx, Axx, xx;
    gpu_vec_dot_weighted_ws(d_Ax, d_Ax, &AxAx, cpu_op,
        &((GPU_Operator*)gpu_op)->reduction_ws);
    gpu_vec_dot_weighted_ws(d_Ax, d_x, &Axx, cpu_op,
        &((GPU_Operator*)gpu_op)->reduction_ws);
    gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op,
        &((GPU_Operator*)gpu_op)->reduction_ws);

    double residual_sq = AxAx - 2.0 * lambda * Axx + lambda * lambda * xx;
    if (residual_sq < 0.0) residual_sq = 0.0;

    return sqrt(residual_sq);
}

/*=============================================================================
 * GPU Rayleigh Quotient Iteration (PEC, real) (UNCHANGED)
 *============================================================================*/

GPU_EigenResult gpu_rqi_ws(
    const GPU_Operator* gpu_op,
    double* d_x,
    double sigma_init,
    int max_iter,
    double tol,
    EigensolverWorkspace* ws
) {
    GPU_EigenResult result;
    result.eigenvalue = 0.0;
    result.iterations = 0;
    result.residual = 1.0;
    result.converged = 0;

    const CurlCurlOperator* cpu_op = gpu_op->cpu_op;
    int n = cpu_op->n_total;

    gpu_vec_normalize_weighted(d_x, cpu_op);

    gpu_curlcurl_matvec(gpu_op, d_x, ws->d_Ax);

    double sigma;
    double xAx, xx;
    gpu_vec_dot_weighted_ws(d_x, ws->d_Ax, &xAx, cpu_op,
        &((GPU_Operator*)gpu_op)->reduction_ws);
    gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op,
        &((GPU_Operator*)gpu_op)->reduction_ws);
    sigma = xAx / xx;

    if (fabs(sigma) < 1e-10 || sigma < 0) {
        sigma = sigma_init;
    }

    printf("  Rayleigh Quotient Iteration (GPU):\n");
    printf("  %5s %15s %15s %8s\n", "Iter", "Eigenvalue", "Residual", "LS its");
    printf("  -------------------------------------------------\n");

    for (int iter = 0; iter < max_iter; iter++) {
        GPU_LinearSolverResult ls = gpu_minres_solve_shifted_ws(
            gpu_op, sigma, d_x, ws->d_y, 1000, 1e-6, &ws->minres_ws);

        gpu_vec_normalize_weighted(ws->d_y, cpu_op);
        gpu_vec_copy(ws->d_y, d_x, n);

        gpu_curlcurl_matvec(gpu_op, d_x, ws->d_Ax);

        gpu_vec_dot_weighted_ws(d_x, ws->d_Ax, &xAx, cpu_op,
            &((GPU_Operator*)gpu_op)->reduction_ws);
        gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op,
            &((GPU_Operator*)gpu_op)->reduction_ws);
        double sigma_new = xAx / xx;

        double residual = gpu_compute_residual(gpu_op, ws->d_Ax, d_x, sigma_new);

        result.eigenvalue = sigma_new;
        result.residual = residual;
        result.iterations = iter + 1;

        printf("  %5d %15.10f %15.8e %8d\n",
            iter, sigma_new, residual, ls.iterations);

        if (residual < tol) {
            result.converged = 1;
            printf("  Converged!\n");
            break;
        }

        sigma = sigma_new;
    }

    return result;
}

GPU_EigenResult gpu_rayleigh_quotient_iteration(
    const GPU_Operator* gpu_op,
    double* d_x,
    double sigma_init,
    int max_iter,
    double tol
) {
    int n = gpu_op->cpu_op->n_total;
    EigensolverWorkspace ws;
    eigensolver_workspace_init(&ws, n);

    GPU_EigenResult result = gpu_rqi_ws(gpu_op, d_x, sigma_init, max_iter, tol, &ws);

    eigensolver_workspace_free(&ws);
    return result;
}

/*=============================================================================
 * GPU Power Iteration (PEC, real) (UNCHANGED)
 *============================================================================*/

GPU_EigenResult gpu_power_iteration(
    const GPU_Operator* gpu_op,
    double* d_x,
    int max_iter,
    double tol
) {
    GPU_EigenResult result;
    result.eigenvalue = 0.0;
    result.iterations = 0;
    result.residual = 1.0;
    result.converged = 0;

    const CurlCurlOperator* cpu_op = gpu_op->cpu_op;
    int n = cpu_op->n_total;

    double* d_y;
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));

    gpu_vec_normalize_weighted(d_x, cpu_op);

    double lambda_old = 0.0;

    printf("  Power Iteration (GPU):\n");
    printf("  %5s %18s %15s\n", "Iter", "Eigenvalue", "Change");
    printf("  ------------------------------------------\n");

    for (int iter = 0; iter < max_iter; iter++) {
        gpu_curlcurl_matvec(gpu_op, d_x, d_y);

        double lambda;
        gpu_vec_dot_weighted_ws(d_x, d_y, &lambda, cpu_op,
            &((GPU_Operator*)gpu_op)->reduction_ws);

        gpu_vec_normalize_weighted(d_y, cpu_op);

        result.eigenvalue = lambda;
        result.iterations = iter + 1;

        double change = fabs(lambda - lambda_old);

        if (iter % 10 == 0 || change < tol) {
            printf("  %5d %18.12f %15.8e\n", iter, lambda, change);
        }

        if (change < tol && iter > 0) {
            result.converged = 1;
            double residual = gpu_compute_residual(gpu_op, d_y, d_x, lambda);
            result.residual = residual;
            gpu_vec_copy(d_y, d_x, n);
            printf("  Converged!\n");
            break;
        }

        gpu_vec_copy(d_y, d_x, n);
        lambda_old = lambda;
    }

    cudaFree(d_y);
    return result;
}

/*=============================================================================
 * GPU Inverse Iteration (PEC, real) (UNCHANGED)
 *============================================================================*/

GPU_EigenResult gpu_inverse_iteration(
    const GPU_Operator* gpu_op,
    double* d_x,
    double sigma,
    int max_iter,
    double tol
) {
    GPU_EigenResult result;
    result.eigenvalue = sigma;
    result.iterations = 0;
    result.residual = 1.0;
    result.converged = 0;

    const CurlCurlOperator* cpu_op = gpu_op->cpu_op;
    int n = cpu_op->n_total;

    double* d_y, *d_Ax;
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ax, n * sizeof(double)));

    MINRES_Workspace ws;
    minres_workspace_init(&ws, n);

    gpu_vec_normalize_weighted(d_x, cpu_op);

    printf("  Inverse Iteration (GPU, sigma = %g):\n", sigma);
    printf("  %5s %15s %15s %8s\n", "Iter", "Eigenvalue", "Residual", "LS its");
    printf("  -------------------------------------------------\n");

    for (int iter = 0; iter < max_iter; iter++) {
        GPU_LinearSolverResult ls = gpu_minres_solve_shifted_ws(
            gpu_op, sigma, d_x, d_y, 500, 1e-10, &ws);

        gpu_vec_normalize_weighted(d_y, cpu_op);

        gpu_curlcurl_matvec(gpu_op, d_y, d_Ax);

        double xAx, xx_val;
        gpu_vec_dot_weighted_ws(d_y, d_Ax, &xAx, cpu_op,
            &((GPU_Operator*)gpu_op)->reduction_ws);
        gpu_vec_dot_weighted_ws(d_y, d_y, &xx_val, cpu_op,
            &((GPU_Operator*)gpu_op)->reduction_ws);
        double lambda = xAx / xx_val;

        double residual = gpu_compute_residual(gpu_op, d_Ax, d_y, lambda);

        result.eigenvalue = lambda;
        result.residual = residual;
        result.iterations = iter + 1;

        printf("  %5d %15.10f %15.8e %8d\n",
            iter, lambda, residual, ls.iterations);

        if (residual < tol) {
            result.converged = 1;
            gpu_vec_copy(d_y, d_x, n);
            printf("  Converged!\n");
            break;
        }

        gpu_vec_copy(d_y, d_x, n);
    }

    minres_workspace_free(&ws);
    cudaFree(d_y);
    cudaFree(d_Ax);

    return result;
}

/*=============================================================================
 * IBC Complex Eigensolver Workspace
 *============================================================================*/

int complex_eigensolver_workspace_init(
    ComplexEigensolverWorkspace* ws,
    int n_real,
    int gmres_restart
) {
    ws->n_real = n_real;
    ws->initialized = 0;

    int n = 2 * n_real;

    if (gmres_workspace_init(&ws->gmres_ws, n_real, gmres_restart) != 0)
        return -1;

    CUDA_CHECK_RET(cudaMalloc(&ws->d_y, n * sizeof(double)));
    CUDA_CHECK_RET(cudaMalloc(&ws->d_Ax, n * sizeof(double)));

    ws->initialized = 1;
    return 0;
}

void complex_eigensolver_workspace_free(ComplexEigensolverWorkspace* ws) {
    if (!ws->initialized) return;

    gmres_workspace_free(&ws->gmres_ws);
    if (ws->d_y)  { cudaFree(ws->d_y);  ws->d_y = NULL; }
    if (ws->d_Ax) { cudaFree(ws->d_Ax); ws->d_Ax = NULL; }

    ws->initialized = 0;
}

/*=============================================================================
 * Complex Rayleigh Quotient
 *
 * σ = <x, Ax> / <x, x>  where both inner products are complex.
 *
 *   <x, Ax> = Re(<x,Ax>) + j·Im(<x,Ax>)
 *   <x, x>  = Re(<x,x>)  (Im part is zero by definition: <x,x> = ||x||²)
 *
 * So σ_re = Re(<x,Ax>) / ||x||²
 *    σ_im = Im(<x,Ax>) / ||x||²
 *============================================================================*/
static void compute_complex_rayleigh_quotient(
    const CurlCurlOperator* cpu_op,
    ReductionWorkspace* red_ws,
    const double* d_x,
    const double* d_Ax,
    int n_real,
    double* sigma_re,
    double* sigma_im
) {
    double xAx_re, xAx_im, xx;

    gpu_vec_dot_weighted_complex_re_ws(d_x, d_Ax, &xAx_re, cpu_op, n_real, red_ws);
    gpu_vec_dot_weighted_complex_im_ws(d_x, d_Ax, &xAx_im, cpu_op, n_real, red_ws);
    gpu_vec_dot_weighted_complex_re_ws(d_x, d_x,  &xx,     cpu_op, n_real, red_ws);

    if (xx > 1e-28) {
        *sigma_re = xAx_re / xx;
        *sigma_im = xAx_im / xx;
    } else {
        *sigma_re = 0.0;
        *sigma_im = 0.0;
    }
}

/*=============================================================================
 * Complex Residual: || Ax - σx ||_r  where σ is complex
 *
 * r = Ax - σx, computed component-wise:
 *   r_re = Ax_re - σ_re*x_re + σ_im*x_im
 *   r_im = Ax_im - σ_re*x_im - σ_im*x_re
 *
 * We compute ||r||² = <r_re,r_re>_r + <r_im,r_im>_r without forming r
 * explicitly, by expanding:
 *   ||r||² = ||Ax||² - 2·Re(σ̄·<Ax,x>) + |σ|²·||x||²
 *
 * But it's simpler and more robust to just compute r explicitly in a
 * temp vector, then take its norm.
 *============================================================================*/
static double compute_complex_residual(
    const CurlCurlOperator* cpu_op,
    ReductionWorkspace* red_ws,
    const double* d_Ax,
    const double* d_x,
    double sigma_re,
    double sigma_im,
    int n_real,
    double* d_temp       /* scratch vector, size 2*n_real */
) {
    int n = 2 * n_real;
    int blocks = (n_real + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* r = Ax */
    gpu_vec_copy(d_Ax, d_temp, n);

    /* r -= σ*x  (complex subtraction) */
    complex_shift_kernel<<<blocks, BLOCK_SIZE>>>(
        d_temp, d_x, sigma_re, sigma_im, n_real);

    /* ||r||_r */
    double r_norm;
    gpu_vec_norm_weighted_complex_ws(d_temp, &r_norm, cpu_op, n_real, red_ws);

    /* Normalize by ||x||_r */
    double x_norm;
    gpu_vec_norm_weighted_complex_ws(d_x, &x_norm, cpu_op, n_real, red_ws);

    if (x_norm > 1e-28) {
        return r_norm / x_norm;
    }
    return r_norm;
}

/*=============================================================================
 * IBC Complex Rayleigh Quotient Iteration
 *
 * Algorithm:
 *   1. Start with initial guess x, compute σ = Rayleigh quotient
 *   2. Compute α = R_s/(ωμ₀) from current σ_re
 *   3. Solve (A_IBC(α) − σI)y = x  via GMRES(m)
 *   4. Normalize y, set x = y
 *   5. Compute new σ, check convergence on |Δσ|/|σ|
 *   6. Repeat
 *
 * Note: α changes each iteration because it depends on ω = c·√(σ_re).
 * This makes the operator itself iteration-dependent, which is why α is
 * passed to the matvec rather than baked into the operator.
 *============================================================================*/

GPU_ComplexEigenResult gpu_rqi_complex_ws(
    const void* op,
    ComplexMatvecFn matvec,
    const CurlCurlOperator* cpu_op,
    ReductionWorkspace* red_ws,
    double* d_x,
    double sigma_init,
    double conductivity,
    int max_iter,
    double tol,
    ComplexEigensolverWorkspace* ws
) {
    GPU_ComplexEigenResult result;
    result.k2_re = 0.0;
    result.k2_im = 0.0;
    result.frequency_Hz = 0.0;
    result.Q_factor = 0.0;
    result.iterations = 0;
    result.residual = 1.0;
    result.converged = 0;

    int n_real = cpu_op->n_total;
    int n = 2 * n_real;

    double c0 = 299792458.0;

    /* Normalize initial guess */
    gpu_vec_normalize_weighted_complex(d_x, cpu_op, n_real);

    /* Compute initial alpha from sigma_init */
    double alpha = ibc_compute_alpha(sigma_init, conductivity);

    /* Compute initial Rayleigh quotient */
    matvec(op, d_x, ws->d_Ax, alpha);

    double sigma_re, sigma_im;
    compute_complex_rayleigh_quotient(cpu_op, red_ws, d_x, ws->d_Ax, n_real,
                                      &sigma_re, &sigma_im);

    /* If initial estimate seems bad, use hint */
    /* Always use sigma_init (k2_pec) as the GMRES shift.
     * This prevents the inflated Rayleigh quotient from interior
     * boundary weights from corrupting the shift. */
    sigma_re = sigma_init;
    sigma_im = 0.0;

    printf("\n  Complex Rayleigh Quotient Iteration (IBC, GPU):\n");
    printf("  %-5s  %-15s  %-12s  %-12s  %-12s  %-6s  %-8s\n",
        "Iter", "k2_re", "k2_im", "Q", "Residual", "GMRES", "iter_s");
    printf("  ---------------------------------------------------------------"
           "----------------------\n");

    double sigma_re_old = sigma_re;
    double sigma_im_old = sigma_im;

    for (int iter = 0; iter < max_iter; iter++) {
        double iter_t0 = rqi_wall_seconds();
        /* Update alpha from current eigenvalue estimate */
        alpha = ibc_compute_alpha(sigma_re, conductivity);

        int gmres_maxiter = (iter == 0) ? 300 : 2000;
        double gmres_tol = (iter == 0) ? 1e-1 : 1e-4;  /* tightened to match CPU:
                              Q depends on k2_im (~1e-4); a loose 1e-1 inner solve
                              leaves it imprecise and makes Q grid/path-dependent. */

        /* Solve (A_IBC(α) - σI)y = x  via GMRES */
        GPU_LinearSolverResult ls = gpu_gmres_solve_shifted_complex_ws(
            op, matvec, cpu_op, red_ws,
            sigma_re, sigma_im,
            d_x, ws->d_y, alpha,
            gmres_maxiter, gmres_tol,
            &ws->gmres_ws);

        /* Normalize y */
        gpu_vec_normalize_weighted_complex(ws->d_y, cpu_op, n_real);

        /* Update x = y */
        gpu_vec_copy(ws->d_y, d_x, n);

        /* Update alpha with latest sigma for the matvec */
        alpha = ibc_compute_alpha(sigma_re, conductivity);

        /* Compute Ax for Rayleigh quotient */
        matvec(op, d_x, ws->d_Ax, alpha);

        /* New complex Rayleigh quotient */
        double sigma_re_new, sigma_im_new;
        compute_complex_rayleigh_quotient(cpu_op, red_ws, d_x, ws->d_Ax, n_real,
                                          &sigma_re_new, &sigma_im_new);

        /* Compute residual ||Ax - σx||/||x|| using d_y as scratch */
        double residual = compute_complex_residual(
            cpu_op, red_ws, ws->d_Ax, d_x,
            sigma_re_new, sigma_im_new,
            n_real, ws->d_y);

        /* Physical quantities
         * With e^{-jωt}: k2_im < 0 for lossy cavity, Q = -k2_re/k2_im */
        double freq = (sigma_re_new > 0)
            ? c0 * sqrt(sigma_re_new) / (2.0 * M_PI)
            : 0.0;
        double Q = (fabs(sigma_im_new) > 1e-30)
            ? -sigma_re_new / sigma_im_new
            : 0.0;

        result.k2_re = sigma_re_new;
        result.k2_im = sigma_im_new;
        result.frequency_Hz = freq;
        result.Q_factor = Q;
        result.residual = residual;
        result.iterations = iter + 1;

        double iter_s = rqi_wall_seconds() - iter_t0;
        printf("  %-5d  %15.10f  %12.5e  %12.1f  %12.5e  %6d  %7.2fs\n",
            iter, sigma_re_new, sigma_im_new, Q, residual, ls.iterations,
            iter_s);
        fflush(stdout);

        /* Check convergence: relative change in complex eigenvalue */
        double sigma_mag = sqrt(sigma_re_new * sigma_re_new
                              + sigma_im_new * sigma_im_new);
        double delta_re = sigma_re_new - sigma_re_old;
        double delta_im = sigma_im_new - sigma_im_old;
        double delta_mag = sqrt(delta_re * delta_re + delta_im * delta_im);

        double rel_change = (sigma_mag > 1e-28)
            ? delta_mag / sigma_mag : delta_mag;

        if (residual < tol || (rel_change < tol && iter > 0)) {
            result.converged = 1;
            printf("  Converged!\n");
            printf("  f = %.6f MHz,  Q = %.1f\n", freq / 1.0e6, Q);
            break;
        }

        sigma_re_old = sigma_re_new;
        sigma_im_old = sigma_im_new;
        /* Keep sigma fixed at initial value (≈ k2_pec).
         * The IBC eigenvalue is close enough that GMRES stays
         * well-conditioned. Rayleigh quotient still tracks the
         * true complex eigenvalue for convergence checking. */
    }

    return result;
}

/*=============================================================================
 * Convenience wrapper for GPU_Operator (reference cavity, no pipes)
 *
 * Wraps gpu_curlcurl_matvec_complex into the function pointer interface.
 *============================================================================*/

static int curlcurl_matvec_complex_wrapper(
    const void* op, const double* d_x, double* d_y, double alpha
) {
    return gpu_curlcurl_matvec_complex((const GPU_Operator*)op, d_x, d_y, alpha);
}

GPU_ComplexEigenResult gpu_rqi_complex(
    const GPU_Operator* gpu_op,
    double* d_x,
    double sigma_init,
    double conductivity,
    int max_iter,
    double tol,
    int gmres_restart
) {
    int n_real = gpu_op->cpu_op->n_total;
    ComplexEigensolverWorkspace ws;
    complex_eigensolver_workspace_init(&ws, n_real, gmres_restart);

    GPU_ComplexEigenResult result = gpu_rqi_complex_ws(
        gpu_op,
        curlcurl_matvec_complex_wrapper,
        gpu_op->cpu_op,
        &((GPU_Operator*)gpu_op)->reduction_ws,
        d_x, sigma_init, conductivity,
        max_iter, tol, &ws);

    complex_eigensolver_workspace_free(&ws);
    return result;
}
