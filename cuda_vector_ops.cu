#include "cuda_vector_ops.h"
#include <stdio.h>
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

 /*=============================================================================
  * Kernel parameters
  *============================================================================*/
#define BLOCK_SIZE 256
#define WARP_SIZE 32

  /*=============================================================================
   * Grid parameters struct for passing to kernels
   * (Avoids __constant__ memory cross-file issues)
   *============================================================================*/
typedef struct {
    double a, dr, dphi, dz;
    int Nr, Nphi, Nz;
} KernelGridParams;

static KernelGridParams make_kernel_params(const CurlCurlOperator* op) {
    KernelGridParams p;
    p.a = op->grid.a;
    p.dr = op->grid.dr;
    p.dphi = op->grid.dphi;
    p.dz = op->grid.dz;
    p.Nr = op->grid.Nr;
    p.Nphi = op->grid.Nphi;
    p.Nz = op->grid.Nz;
    return p;
}

/*=============================================================================
 * Basic Vector Kernels (UNCHANGED)
 *============================================================================*/

__global__ void scale_kernel(double* x, double alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= alpha;
    }
}

__global__ void axpy_kernel(double alpha, const double* x, double* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

__global__ void axpby_kernel(double alpha, const double* x,
    double beta, double* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + beta * y[idx];
    }
}

__global__ void copy_kernel(const double* x, double* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx];
    }
}

/*=============================================================================
 * Basic Vector Operations (UNCHANGED)
 *============================================================================*/

int gpu_vec_scale(double* d_x, double alpha, int n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scale_kernel<<<grid, BLOCK_SIZE>>>(d_x, alpha, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int gpu_vec_axpy(double alpha, const double* d_x, double* d_y, int n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    axpy_kernel<<<grid, BLOCK_SIZE>>>(alpha, d_x, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int gpu_vec_axpby(double alpha, const double* d_x,
    double beta, double* d_y, int n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    axpby_kernel<<<grid, BLOCK_SIZE>>>(alpha, d_x, beta, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int gpu_vec_copy(const double* d_x, double* d_y, int n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    copy_kernel<<<grid, BLOCK_SIZE>>>(d_x, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int gpu_vec_zero(double* d_x, int n) {
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(double)));
    return 0;
}

/*=============================================================================
 * Warp-level reduction helper (shuffle-based) (UNCHANGED)
 *============================================================================*/
__device__ double warp_reduce_sum(double val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/*=============================================================================
 * Block-level reduction helper (UNCHANGED)
 *============================================================================*/
__device__ void block_reduce_sum(double val, double* sdata, double* block_result) {
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    /* Reduce down to 64 elements */
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Final warp reduction: must first combine the 64→32 step */
    if (tid < 32) {
        double wval = sdata[tid] + sdata[tid + 32];
        wval = warp_reduce_sum(wval);
        if (tid == 0) {
            *block_result = wval;
        }
    }
}

/*=============================================================================
 * Weighted dot product kernels (UNCHANGED)
 *============================================================================*/

__global__ void dot_weighted_Er_kernel(
    const double* x,
    const double* y,
    double* partial_sums,
    int offset_Er,
    int size_Er,
    KernelGridParams gp
) {
    __shared__ double sdata[BLOCK_SIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;

    if (gid < size_Er) {
        int i = gid % gp.Nr;

        double r = gp.a + (i + 0.5) * gp.dr;
        double dV = gp.dr * gp.dphi * gp.dz;

        int global_idx = offset_Er + gid;
        sum = r * x[global_idx] * y[global_idx] * dV;
    }

    block_reduce_sum(sum, sdata, &partial_sums[blockIdx.x]);
}

__global__ void dot_weighted_Ephi_kernel(
    const double* x,
    const double* y,
    double* partial_sums,
    int offset_Ephi,
    int size_Ephi,
    KernelGridParams gp
) {
    __shared__ double sdata[BLOCK_SIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;

    if (gid < size_Ephi) {
        int Nr1 = gp.Nr + 1;
        int i = gid % Nr1;

        double r = gp.a + i * gp.dr;
        double dV = gp.dr * gp.dphi * gp.dz;

        int global_idx = offset_Ephi + gid;
        sum = r * x[global_idx] * y[global_idx] * dV;
    }

    block_reduce_sum(sum, sdata, &partial_sums[blockIdx.x]);
}

__global__ void dot_weighted_Ez_kernel(
    const double* x,
    const double* y,
    double* partial_sums,
    int offset_Ez,
    int size_Ez,
    KernelGridParams gp
) {
    __shared__ double sdata[BLOCK_SIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;

    if (gid < size_Ez) {
        int Nr1 = gp.Nr + 1;
        int i = gid % Nr1;

        double r = gp.a + i * gp.dr;
        double dV = gp.dr * gp.dphi * gp.dz;

        int global_idx = offset_Ez + gid;
        sum = r * x[global_idx] * y[global_idx] * dV;
    }

    block_reduce_sum(sum, sdata, &partial_sums[blockIdx.x]);
}

/*=============================================================================
 * Final reduction kernel (UNCHANGED)
 *============================================================================*/
__global__ void final_reduce_kernel(
    const double* partial_sums,
    double* result,
    int n
) {
    __shared__ double sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    double sum = 0.0;

    /* Each thread sums multiple elements if n > BLOCK_SIZE */
    for (int i = tid; i < n; i += blockDim.x) {
        sum += partial_sums[i];
    }

    /* Use the fixed block reduction */
    block_reduce_sum(sum, sdata, &result[0]);
}

/*=============================================================================
 * Reduction Workspace (UNCHANGED)
 *============================================================================*/

int reduction_workspace_init(ReductionWorkspace* ws, int n) {
    ws->num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ws->initialized = 0;

    CUDA_CHECK(cudaMalloc(&ws->d_partial_sums,
        ws->num_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws->d_result, sizeof(double)));

    ws->initialized = 1;
    return 0;
}

void reduction_workspace_free(ReductionWorkspace* ws) {
    if (ws->d_partial_sums) {
        cudaFree(ws->d_partial_sums);
        ws->d_partial_sums = NULL;
    }
    if (ws->d_result) {
        cudaFree(ws->d_result);
        ws->d_result = NULL;
    }
    ws->initialized = 0;
}

/*=============================================================================
 * Weighted Dot Product — Real (UNCHANGED behavior)
 *============================================================================*/

int gpu_vec_dot_weighted(
    const double* d_x,
    const double* d_y,
    double* result,
    const CurlCurlOperator* op
) {
    KernelGridParams gp = make_kernel_params(op);

    int size_Er = op->size_Er;
    int size_Ephi = op->size_Ephi;
    int size_Ez = op->size_Ez;

    int blocks_Er = (size_Er + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_Ephi = (size_Ephi + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_Ez = (size_Ez + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int total_blocks = blocks_Er + blocks_Ephi + blocks_Ez;

    double* d_partial;
    double* d_result_dev;
    CUDA_CHECK(cudaMalloc(&d_partial, total_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result_dev, sizeof(double)));

    dot_weighted_Er_kernel<<<blocks_Er, BLOCK_SIZE>>>(
        d_x, d_y, d_partial,
        op->offset_Er, size_Er, gp
        );

    dot_weighted_Ephi_kernel<<<blocks_Ephi, BLOCK_SIZE>>>(
        d_x, d_y, d_partial + blocks_Er,
        op->offset_Ephi, size_Ephi, gp
        );

    dot_weighted_Ez_kernel<<<blocks_Ez, BLOCK_SIZE>>>(
        d_x, d_y, d_partial + blocks_Er + blocks_Ephi,
        op->offset_Ez, size_Ez, gp
        );

    final_reduce_kernel<<<1, BLOCK_SIZE>>>(d_partial, d_result_dev, total_blocks);

    CUDA_CHECK(cudaMemcpy(result, d_result_dev, sizeof(double),
        cudaMemcpyDeviceToHost));

    cudaFree(d_partial);
    cudaFree(d_result_dev);

    return 0;
}

/*=============================================================================
 * Workspace version (UNCHANGED behavior)
 *============================================================================*/

int gpu_vec_dot_weighted_ws(
    const double* d_x,
    const double* d_y,
    double* result,
    const CurlCurlOperator* op,
    ReductionWorkspace* ws
) {
    KernelGridParams gp = make_kernel_params(op);

    int size_Er = op->size_Er;
    int size_Ephi = op->size_Ephi;
    int size_Ez = op->size_Ez;

    int blocks_Er = (size_Er + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_Ephi = (size_Ephi + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_Ez = (size_Ez + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int total_blocks = blocks_Er + blocks_Ephi + blocks_Ez;

    dot_weighted_Er_kernel<<<blocks_Er, BLOCK_SIZE>>>(
        d_x, d_y, ws->d_partial_sums,
        op->offset_Er, size_Er, gp
        );

    dot_weighted_Ephi_kernel<<<blocks_Ephi, BLOCK_SIZE>>>(
        d_x, d_y, ws->d_partial_sums + blocks_Er,
        op->offset_Ephi, size_Ephi, gp
        );

    dot_weighted_Ez_kernel<<<blocks_Ez, BLOCK_SIZE>>>(
        d_x, d_y, ws->d_partial_sums + blocks_Er + blocks_Ephi,
        op->offset_Ez, size_Ez, gp
        );

    final_reduce_kernel<<<1, BLOCK_SIZE>>>(
        ws->d_partial_sums, ws->d_result, total_blocks
        );

    CUDA_CHECK(cudaMemcpy(result, ws->d_result, sizeof(double),
        cudaMemcpyDeviceToHost));

    return 0;
}

/*=============================================================================
 * Weighted Norm / Normalize — Real (UNCHANGED)
 *============================================================================*/

int gpu_vec_norm_weighted(
    const double* d_x,
    double* result,
    const CurlCurlOperator* op
) {
    double dot;
    int status = gpu_vec_dot_weighted(d_x, d_x, &dot, op);
    if (status != 0) return status;
    *result = sqrt(dot);
    return 0;
}

int gpu_vec_normalize_weighted(
    double* d_x,
    const CurlCurlOperator* op
) {
    double norm;
    int status = gpu_vec_norm_weighted(d_x, &norm, op);
    if (status != 0) return status;

    if (norm > 1e-14) {
        gpu_vec_scale(d_x, 1.0 / norm, op->n_total);
    }
    return 0;
}

/*=============================================================================
 * Combined dot + norm (UNCHANGED)
 *============================================================================*/

int gpu_vec_dot_and_norm(
    const double* d_x,
    const double* d_y,
    double* dot_result,
    double* norm_x_result,
    const CurlCurlOperator* op
) {
    int status = gpu_vec_dot_weighted(d_x, d_y, dot_result, op);
    if (status != 0) return status;

    double dot_xx;
    status = gpu_vec_dot_weighted(d_x, d_x, &dot_xx, op);
    if (status != 0) return status;

    *norm_x_result = sqrt(dot_xx);
    return 0;
}

/*=============================================================================
 * Complex Weighted Dot Product (IBC extension)
 *
 * Exploits the mirrored packed-vector layout:
 *   d_x + 0       → real part    (offsets: offset_Er, offset_Ephi, offset_Ez)
 *   d_x + n_real   → imaginary part (same offsets work because layout mirrors)
 *
 * So we reuse the existing dot product function with pointer arithmetic.
 * Two calls to the real dot product per complex result — the overhead of
 * two host-side cudaMemcpy's for scalars is negligible vs kernel time.
 *============================================================================*/

/* Re(<x, y>) = <x_re, y_re>_r + <x_im, y_im>_r */
int gpu_vec_dot_weighted_complex_re(
    const double* d_x,
    const double* d_y,
    double* result,
    const CurlCurlOperator* op,
    int n_real
) {
    double dot_rr, dot_ii;

    int status = gpu_vec_dot_weighted(d_x, d_y, &dot_rr, op);
    if (status != 0) return status;

    status = gpu_vec_dot_weighted(d_x + n_real, d_y + n_real, &dot_ii, op);
    if (status != 0) return status;

    *result = dot_rr + dot_ii;
    return 0;
}

/* Im(<x, y>) = <x_re, y_im>_r - <x_im, y_re>_r */
int gpu_vec_dot_weighted_complex_im(
    const double* d_x,
    const double* d_y,
    double* result,
    const CurlCurlOperator* op,
    int n_real
) {
    double dot_ri, dot_ir;

    int status = gpu_vec_dot_weighted(d_x, d_y + n_real, &dot_ri, op);
    if (status != 0) return status;

    status = gpu_vec_dot_weighted(d_x + n_real, d_y, &dot_ir, op);
    if (status != 0) return status;

    *result = dot_ri - dot_ir;
    return 0;
}

/* Workspace versions */

int gpu_vec_dot_weighted_complex_re_ws(
    const double* d_x,
    const double* d_y,
    double* result,
    const CurlCurlOperator* op,
    int n_real,
    ReductionWorkspace* ws
) {
    double dot_rr, dot_ii;

    int status = gpu_vec_dot_weighted_ws(d_x, d_y, &dot_rr, op, ws);
    if (status != 0) return status;

    status = gpu_vec_dot_weighted_ws(d_x + n_real, d_y + n_real, &dot_ii, op, ws);
    if (status != 0) return status;

    *result = dot_rr + dot_ii;
    return 0;
}

int gpu_vec_dot_weighted_complex_im_ws(
    const double* d_x,
    const double* d_y,
    double* result,
    const CurlCurlOperator* op,
    int n_real,
    ReductionWorkspace* ws
) {
    double dot_ri, dot_ir;

    int status = gpu_vec_dot_weighted_ws(d_x, d_y + n_real, &dot_ri, op, ws);
    if (status != 0) return status;

    status = gpu_vec_dot_weighted_ws(d_x + n_real, d_y, &dot_ir, op, ws);
    if (status != 0) return status;

    *result = dot_ri - dot_ir;
    return 0;
}

/*=============================================================================
 * Complex Weighted Norm (IBC extension)
 *
 * ||x||² = <x_re, x_re>_r + <x_im, x_im>_r
 *============================================================================*/

int gpu_vec_norm_weighted_complex(
    const double* d_x,
    double* result,
    const CurlCurlOperator* op,
    int n_real
) {
    double dot_rr, dot_ii;

    int status = gpu_vec_dot_weighted(d_x, d_x, &dot_rr, op);
    if (status != 0) return status;

    status = gpu_vec_dot_weighted(d_x + n_real, d_x + n_real, &dot_ii, op);
    if (status != 0) return status;

    *result = sqrt(dot_rr + dot_ii);
    return 0;
}

int gpu_vec_norm_weighted_complex_ws(
    const double* d_x,
    double* result,
    const CurlCurlOperator* op,
    int n_real,
    ReductionWorkspace* ws
) {
    double dot_rr, dot_ii;

    int status = gpu_vec_dot_weighted_ws(d_x, d_x, &dot_rr, op, ws);
    if (status != 0) return status;

    status = gpu_vec_dot_weighted_ws(d_x + n_real, d_x + n_real, &dot_ii, op, ws);
    if (status != 0) return status;

    *result = sqrt(dot_rr + dot_ii);
    return 0;
}

/*=============================================================================
 * Complex Weighted Normalize (IBC extension)
 *
 * Scales the ENTIRE 2*n_real vector by 1/||x||, keeping re/im ratio intact.
 *============================================================================*/

int gpu_vec_normalize_weighted_complex(
    double* d_x,
    const CurlCurlOperator* op,
    int n_real
) {
    double norm;
    int status = gpu_vec_norm_weighted_complex(d_x, &norm, op, n_real);
    if (status != 0) return status;

    if (norm > 1e-14) {
        /* Scale both real and imaginary parts in one call */
        gpu_vec_scale(d_x, 1.0 / norm, 2 * n_real);
    }
    return 0;
}

int gpu_vec_normalize_weighted_complex_ws(
    double* d_x,
    const CurlCurlOperator* op,
    int n_real,
    ReductionWorkspace* ws
) {
    double norm;
    int status = gpu_vec_norm_weighted_complex_ws(d_x, &norm, op, n_real, ws);
    if (status != 0) return status;

    if (norm > 1e-14) {
        gpu_vec_scale(d_x, 1.0 / norm, 2 * n_real);
    }
    return 0;
}

/*=============================================================================
 * Flat (Unweighted) Dot Product — single kernel, no component splitting
 *
 * ~4× fewer kernel launches than weighted version.
 *============================================================================*/

__global__ void dot_flat_kernel(
    const double* x, const double* y,
    double* partial_sums, int n
) {
    __shared__ double sdata[BLOCK_SIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    /* Grid-stride loop for large arrays */
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        sum += x[i] * y[i];
    }

    block_reduce_sum(sum, sdata, &partial_sums[blockIdx.x]);
}

int gpu_vec_dot_flat(
    const double* d_x, const double* d_y,
    double* result, int n
) {
    /* Cap grid size to avoid excessive partial sums */
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    double* d_partial;
    double* d_result_dev;
    cudaMalloc(&d_partial, num_blocks * sizeof(double));
    cudaMalloc(&d_result_dev, sizeof(double));

    dot_flat_kernel<<<num_blocks, BLOCK_SIZE>>>(d_x, d_y, d_partial, n);
    final_reduce_kernel<<<1, BLOCK_SIZE>>>(d_partial, d_result_dev, num_blocks);

    cudaMemcpy(result, d_result_dev, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_partial);
    cudaFree(d_result_dev);
    return 0;
}

/*=============================================================================
 * Complex flat inner product (packed [re|im] layout)
 *
 * Re(<x,y>) = x_re·y_re + x_im·y_im = dot(x, y, 2*n_real)
 *   Because layout is [x_re | x_im], [y_re | y_im], a flat dot of
 *   the full 2*n_real vector naturally gives Σ(x_re*y_re + x_im*y_im).
 *
 * Im(<x,y>) = x_re·y_im - x_im·y_re
 *   = dot(x[0:n], y[n:2n]) - dot(x[n:2n], y[0:n])
 *============================================================================*/

int gpu_vec_dot_flat_complex_re(
    const double* d_x, const double* d_y,
    double* result, int n_real
) {
    return gpu_vec_dot_flat(d_x, d_y, result, 2 * n_real);
}

int gpu_vec_dot_flat_complex_im(
    const double* d_x, const double* d_y,
    double* result, int n_real
) {
    double dot_ri, dot_ir;

    int status = gpu_vec_dot_flat(d_x, d_y + n_real, &dot_ri, n_real);
    if (status != 0) return status;

    status = gpu_vec_dot_flat(d_x + n_real, d_y, &dot_ir, n_real);
    if (status != 0) return status;

    *result = dot_ri - dot_ir;
    return 0;
}

int gpu_vec_norm_flat_complex(
    const double* d_x, double* result, int n_real
) {
    double dot;
    int status = gpu_vec_dot_flat(d_x, d_x, &dot, 2 * n_real);
    if (status != 0) return status;
    *result = sqrt(dot);
    return 0;
}

/*=============================================================================
 * Workspace versions — no malloc per call
 *============================================================================*/

int gpu_vec_dot_flat_ws(
    const double* d_x, const double* d_y,
    double* result, int n,
    double* d_partial, double* d_result_dev
) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    dot_flat_kernel<<<num_blocks, BLOCK_SIZE>>>(d_x, d_y, d_partial, n);
    final_reduce_kernel<<<1, BLOCK_SIZE>>>(d_partial, d_result_dev, num_blocks);
    cudaMemcpy(result, d_result_dev, sizeof(double), cudaMemcpyDeviceToHost);
    return 0;
}

int gpu_vec_dot_flat_complex_re_ws(
    const double* d_x, const double* d_y,
    double* result, int n_real,
    double* d_partial, double* d_result_dev
) {
    return gpu_vec_dot_flat_ws(d_x, d_y, result, 2 * n_real,
        d_partial, d_result_dev);
}

int gpu_vec_dot_flat_complex_im_ws(
    const double* d_x, const double* d_y,
    double* result, int n_real,
    double* d_partial, double* d_result_dev
) {
    double dot_ri, dot_ir;
    int status = gpu_vec_dot_flat_ws(d_x, d_y + n_real, &dot_ri, n_real,
        d_partial, d_result_dev);
    if (status != 0) return status;
    status = gpu_vec_dot_flat_ws(d_x + n_real, d_y, &dot_ir, n_real,
        d_partial, d_result_dev);
    if (status != 0) return status;
    *result = dot_ri - dot_ir;
    return 0;
}

int gpu_vec_norm_flat_complex_ws(
    const double* d_x, double* result, int n_real,
    double* d_partial, double* d_result_dev
) {
    double dot;
    int status = gpu_vec_dot_flat_ws(d_x, d_x, &dot, 2 * n_real,
        d_partial, d_result_dev);
    if (status != 0) return status;
    *result = sqrt(dot);
    return 0;
}
