#include "cuda_fields.h"
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

#define CUDA_CHECK_VOID(call) do {                                      \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        return;                                                         \
    }                                                                   \
} while(0)

 /*=============================================================================
  * Device constant memory for grid parameters
  * Accessible by ALL kernels without passing as argument
  *============================================================================*/
__constant__ GridParams d_grid;

int cuda_grid_init(const GridParams* grid) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_grid, grid, sizeof(GridParams)));
    return 0;
}

/*=============================================================================
 * Device Info
 *============================================================================*/
void cuda_print_device_info(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("No CUDA devices found!\n");
        return;
    }

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("\n");
    printf("============================================\n");
    printf("  CUDA Device: %s\n", prop.name);
    printf("============================================\n");
    printf("  Compute capability : %d.%d\n", prop.major, prop.minor);
    printf("  Multiprocessors    : %d\n", prop.multiProcessorCount);
    printf("  Global memory      : %.1f GB\n", prop.totalGlobalMem / 1.0e9);
    printf("  Shared mem / block : %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("  Max threads / block: %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size          : %d\n", prop.warpSize);
    printf("  Clock rate         : %.1f GHz\n", prop.clockRate / 1.0e6);
    printf("  Memory clock       : %.1f GHz\n", prop.memoryClockRate / 1.0e6);
    printf("  Memory bus width   : %d bits\n", prop.memoryBusWidth);
    printf("  L2 cache size      : %.0f KB\n", prop.l2CacheSize / 1024.0);
    printf("  Max grid dims      : [%d, %d, %d]\n",
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("============================================\n\n");
}

void cuda_print_memory_info(const char* label) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used = total_mem - free_mem;
    printf("  GPU Memory [%s]: %.1f MB used / %.1f MB total (%.1f MB free)\n",
        label, used / 1.0e6, total_mem / 1.0e6, free_mem / 1.0e6);
}

/*=============================================================================
 * GPU E-Field Memory Management
 *============================================================================*/

/* Allocate real part only (backward compatible, has_imag = 0) */
int gpu_efield_alloc(GPU_EField* E, const GridParams* grid) {
    E->size_Er = grid->Nr * grid->Nphi * (grid->Nz + 1);
    E->size_Ephi = (grid->Nr + 1) * grid->Nphi * (grid->Nz + 1);
    E->size_Ez = (grid->Nr + 1) * grid->Nphi * grid->Nz;

    size_t bytes_Er = E->size_Er * sizeof(double);
    size_t bytes_Ephi = E->size_Ephi * sizeof(double);
    size_t bytes_Ez = E->size_Ez * sizeof(double);

    CUDA_CHECK(cudaMalloc(&E->Er, bytes_Er));
    CUDA_CHECK(cudaMalloc(&E->Ephi, bytes_Ephi));
    CUDA_CHECK(cudaMalloc(&E->Ez, bytes_Ez));

    CUDA_CHECK(cudaMemset(E->Er, 0, bytes_Er));
    CUDA_CHECK(cudaMemset(E->Ephi, 0, bytes_Ephi));
    CUDA_CHECK(cudaMemset(E->Ez, 0, bytes_Ez));

    /* Imaginary pointers NULL for real-only mode */
    E->Er_im = NULL;
    E->Ephi_im = NULL;
    E->Ez_im = NULL;
    E->has_imag = 0;

    return 0;
}

/* Allocate real + imaginary parts for IBC (has_imag = 1) */
int gpu_efield_alloc_complex(GPU_EField* E, const GridParams* grid) {
    /* Allocate real part first */
    if (gpu_efield_alloc(E, grid) != 0) return -1;

    /* Now allocate imaginary part */
    size_t bytes_Er = E->size_Er * sizeof(double);
    size_t bytes_Ephi = E->size_Ephi * sizeof(double);
    size_t bytes_Ez = E->size_Ez * sizeof(double);

    CUDA_CHECK(cudaMalloc(&E->Er_im, bytes_Er));
    CUDA_CHECK(cudaMalloc(&E->Ephi_im, bytes_Ephi));
    CUDA_CHECK(cudaMalloc(&E->Ez_im, bytes_Ez));

    CUDA_CHECK(cudaMemset(E->Er_im, 0, bytes_Er));
    CUDA_CHECK(cudaMemset(E->Ephi_im, 0, bytes_Ephi));
    CUDA_CHECK(cudaMemset(E->Ez_im, 0, bytes_Ez));

    E->has_imag = 1;

    return 0;
}

void gpu_efield_free(GPU_EField* E) {
    if (E->Er)   { cudaFree(E->Er);   E->Er = NULL; }
    if (E->Ephi) { cudaFree(E->Ephi); E->Ephi = NULL; }
    if (E->Ez)   { cudaFree(E->Ez);   E->Ez = NULL; }

    /* Free imaginary arrays if allocated */
    if (E->Er_im)   { cudaFree(E->Er_im);   E->Er_im = NULL; }
    if (E->Ephi_im) { cudaFree(E->Ephi_im); E->Ephi_im = NULL; }
    if (E->Ez_im)   { cudaFree(E->Ez_im);   E->Ez_im = NULL; }

    E->size_Er = E->size_Ephi = E->size_Ez = 0;
    E->has_imag = 0;
}

int gpu_efield_zero(GPU_EField* E) {
    CUDA_CHECK(cudaMemset(E->Er, 0, E->size_Er * sizeof(double)));
    CUDA_CHECK(cudaMemset(E->Ephi, 0, E->size_Ephi * sizeof(double)));
    CUDA_CHECK(cudaMemset(E->Ez, 0, E->size_Ez * sizeof(double)));

    /* Zero imaginary part if present */
    if (E->has_imag) {
        CUDA_CHECK(cudaMemset(E->Er_im, 0, E->size_Er * sizeof(double)));
        CUDA_CHECK(cudaMemset(E->Ephi_im, 0, E->size_Ephi * sizeof(double)));
        CUDA_CHECK(cudaMemset(E->Ez_im, 0, E->size_Ez * sizeof(double)));
    }

    return 0;
}

/* CPU → GPU: real part */
int gpu_efield_to_device(GPU_EField* dst, const EField* src) {
    CUDA_CHECK(cudaMemcpy(dst->Er, src->Er,
        dst->size_Er * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst->Ephi, src->Ephi,
        dst->size_Ephi * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst->Ez, src->Ez,
        dst->size_Ez * sizeof(double), cudaMemcpyHostToDevice));
    return 0;
}

/* GPU → CPU: real part */
int gpu_efield_to_host(EField* dst, const GPU_EField* src) {
    CUDA_CHECK(cudaMemcpy(dst->Er, src->Er,
        src->size_Er * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst->Ephi, src->Ephi,
        src->size_Ephi * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst->Ez, src->Ez,
        src->size_Ez * sizeof(double), cudaMemcpyDeviceToHost));
    return 0;
}

/* CPU → GPU: imaginary part (src_im is a CPU EField holding imaginary data) */
int gpu_efield_im_to_device(GPU_EField* dst, const EField* src_im) {
    if (!dst->has_imag) {
        fprintf(stderr, "gpu_efield_im_to_device: imaginary arrays not allocated\n");
        return -1;
    }
    CUDA_CHECK(cudaMemcpy(dst->Er_im, src_im->Er,
        dst->size_Er * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst->Ephi_im, src_im->Ephi,
        dst->size_Ephi * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst->Ez_im, src_im->Ez,
        dst->size_Ez * sizeof(double), cudaMemcpyHostToDevice));
    return 0;
}

/* GPU → CPU: imaginary part */
int gpu_efield_im_to_host(EField* dst_im, const GPU_EField* src) {
    if (!src->has_imag) {
        fprintf(stderr, "gpu_efield_im_to_host: imaginary arrays not allocated\n");
        return -1;
    }
    CUDA_CHECK(cudaMemcpy(dst_im->Er, src->Er_im,
        src->size_Er * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst_im->Ephi, src->Ephi_im,
        src->size_Ephi * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst_im->Ez, src->Ez_im,
        src->size_Ez * sizeof(double), cudaMemcpyDeviceToHost));
    return 0;
}

/*=============================================================================
 * GPU H-Field Memory Management
 *============================================================================*/

/* Allocate real part only */
int gpu_hfield_alloc(GPU_HField* H, const GridParams* grid) {
    H->size_Hr = (grid->Nr + 1) * grid->Nphi * grid->Nz;
    H->size_Hphi = grid->Nr * grid->Nphi * grid->Nz;
    H->size_Hz = grid->Nr * grid->Nphi * (grid->Nz + 1);

    size_t bytes_Hr = H->size_Hr * sizeof(double);
    size_t bytes_Hphi = H->size_Hphi * sizeof(double);
    size_t bytes_Hz = H->size_Hz * sizeof(double);

    CUDA_CHECK(cudaMalloc(&H->Hr, bytes_Hr));
    CUDA_CHECK(cudaMalloc(&H->Hphi, bytes_Hphi));
    CUDA_CHECK(cudaMalloc(&H->Hz, bytes_Hz));

    CUDA_CHECK(cudaMemset(H->Hr, 0, bytes_Hr));
    CUDA_CHECK(cudaMemset(H->Hphi, 0, bytes_Hphi));
    CUDA_CHECK(cudaMemset(H->Hz, 0, bytes_Hz));

    H->Hr_im = NULL;
    H->Hphi_im = NULL;
    H->Hz_im = NULL;
    H->has_imag = 0;

    return 0;
}

/* Allocate real + imaginary parts for IBC */
int gpu_hfield_alloc_complex(GPU_HField* H, const GridParams* grid) {
    if (gpu_hfield_alloc(H, grid) != 0) return -1;

    size_t bytes_Hr = H->size_Hr * sizeof(double);
    size_t bytes_Hphi = H->size_Hphi * sizeof(double);
    size_t bytes_Hz = H->size_Hz * sizeof(double);

    CUDA_CHECK(cudaMalloc(&H->Hr_im, bytes_Hr));
    CUDA_CHECK(cudaMalloc(&H->Hphi_im, bytes_Hphi));
    CUDA_CHECK(cudaMalloc(&H->Hz_im, bytes_Hz));

    CUDA_CHECK(cudaMemset(H->Hr_im, 0, bytes_Hr));
    CUDA_CHECK(cudaMemset(H->Hphi_im, 0, bytes_Hphi));
    CUDA_CHECK(cudaMemset(H->Hz_im, 0, bytes_Hz));

    H->has_imag = 1;

    return 0;
}

void gpu_hfield_free(GPU_HField* H) {
    if (H->Hr)   { cudaFree(H->Hr);   H->Hr = NULL; }
    if (H->Hphi) { cudaFree(H->Hphi); H->Hphi = NULL; }
    if (H->Hz)   { cudaFree(H->Hz);   H->Hz = NULL; }

    if (H->Hr_im)   { cudaFree(H->Hr_im);   H->Hr_im = NULL; }
    if (H->Hphi_im) { cudaFree(H->Hphi_im); H->Hphi_im = NULL; }
    if (H->Hz_im)   { cudaFree(H->Hz_im);   H->Hz_im = NULL; }

    H->size_Hr = H->size_Hphi = H->size_Hz = 0;
    H->has_imag = 0;
}

int gpu_hfield_zero(GPU_HField* H) {
    CUDA_CHECK(cudaMemset(H->Hr, 0, H->size_Hr * sizeof(double)));
    CUDA_CHECK(cudaMemset(H->Hphi, 0, H->size_Hphi * sizeof(double)));
    CUDA_CHECK(cudaMemset(H->Hz, 0, H->size_Hz * sizeof(double)));

    if (H->has_imag) {
        CUDA_CHECK(cudaMemset(H->Hr_im, 0, H->size_Hr * sizeof(double)));
        CUDA_CHECK(cudaMemset(H->Hphi_im, 0, H->size_Hphi * sizeof(double)));
        CUDA_CHECK(cudaMemset(H->Hz_im, 0, H->size_Hz * sizeof(double)));
    }

    return 0;
}

int gpu_hfield_to_device(GPU_HField* dst, const HField* src) {
    CUDA_CHECK(cudaMemcpy(dst->Hr, src->Hr,
        dst->size_Hr * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst->Hphi, src->Hphi,
        dst->size_Hphi * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst->Hz, src->Hz,
        dst->size_Hz * sizeof(double), cudaMemcpyHostToDevice));
    return 0;
}

int gpu_hfield_to_host(HField* dst, const GPU_HField* src) {
    CUDA_CHECK(cudaMemcpy(dst->Hr, src->Hr,
        src->size_Hr * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst->Hphi, src->Hphi,
        src->size_Hphi * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst->Hz, src->Hz,
        src->size_Hz * sizeof(double), cudaMemcpyDeviceToHost));
    return 0;
}

/*=============================================================================
 * GPU Packed Vector Management
 *============================================================================*/

int gpu_vector_alloc(double** d_vec, int n) {
    CUDA_CHECK(cudaMalloc(d_vec, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(*d_vec, 0, n * sizeof(double)));
    return 0;
}

void gpu_vector_free(double* d_vec) {
    if (d_vec) cudaFree(d_vec);
}

int gpu_vector_zero(double* d_vec, int n) {
    CUDA_CHECK(cudaMemset(d_vec, 0, n * sizeof(double)));
    return 0;
}

int gpu_vector_to_device(double* d_dst, const double* h_src, int n) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, n * sizeof(double),
        cudaMemcpyHostToDevice));
    return 0;
}

int gpu_vector_to_host(double* h_dst, const double* d_src, int n) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, n * sizeof(double),
        cudaMemcpyDeviceToHost));
    return 0;
}

/*=============================================================================
 * GPU Pack/Unpack Kernels
 *
 * These run on GPU to avoid copying fields back to CPU just to repack.
 * Simply copy between separate component arrays and a single flat vector.
 *============================================================================*/

__global__ void pack_component_kernel(
    const double* component,
    double* packed,
    int offset,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        packed[offset + idx] = component[idx];
    }
}

__global__ void unpack_component_kernel(
    const double* packed,
    double* component,
    int offset,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        component[idx] = packed[offset + idx];
    }
}

/* Block size for simple copy kernels */
#define BLOCK_COPY 256

/*=============================================================================
 * Pack/Unpack — Real part only (backward compatible, unchanged)
 *============================================================================*/

int gpu_pack_field(const GPU_EField* E, double* d_x,
    int offset_Er, int offset_Ephi, int offset_Ez)
{
    int grid_Er = (E->size_Er + BLOCK_COPY - 1) / BLOCK_COPY;
    int grid_Ephi = (E->size_Ephi + BLOCK_COPY - 1) / BLOCK_COPY;
    int grid_Ez = (E->size_Ez + BLOCK_COPY - 1) / BLOCK_COPY;

    pack_component_kernel<<<grid_Er, BLOCK_COPY>>>(E->Er, d_x, offset_Er, E->size_Er);
    pack_component_kernel<<<grid_Ephi, BLOCK_COPY>>>(E->Ephi, d_x, offset_Ephi, E->size_Ephi);
    pack_component_kernel<<<grid_Ez, BLOCK_COPY>>>(E->Ez, d_x, offset_Ez, E->size_Ez);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int gpu_unpack_field(const double* d_x, GPU_EField* E,
    int offset_Er, int offset_Ephi, int offset_Ez)
{
    int grid_Er = (E->size_Er + BLOCK_COPY - 1) / BLOCK_COPY;
    int grid_Ephi = (E->size_Ephi + BLOCK_COPY - 1) / BLOCK_COPY;
    int grid_Ez = (E->size_Ez + BLOCK_COPY - 1) / BLOCK_COPY;

    unpack_component_kernel<<<grid_Er, BLOCK_COPY>>>(d_x, E->Er, offset_Er, E->size_Er);
    unpack_component_kernel<<<grid_Ephi, BLOCK_COPY>>>(d_x, E->Ephi, offset_Ephi, E->size_Ephi);
    unpack_component_kernel<<<grid_Ez, BLOCK_COPY>>>(d_x, E->Ez, offset_Ez, E->size_Ez);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/*=============================================================================
 * Pack/Unpack — Complex (real + imaginary)
 *
 * Packed vector layout:
 *   [0       .. n_real-1]  = real part  [Er_re | Ephi_re | Ez_re]
 *   [n_real  .. 2*n_real-1] = imag part [Er_im | Ephi_im | Ez_im]
 *
 * The imaginary sub-offsets mirror the real ones, shifted by n_real.
 *============================================================================*/

int gpu_pack_field_complex(const GPU_EField* E, double* d_x,
    int offset_Er, int offset_Ephi, int offset_Ez,
    int n_real)
{
    /* Pack real part (same as original) */
    int status = gpu_pack_field(E, d_x, offset_Er, offset_Ephi, offset_Ez);
    if (status != 0) return status;

    /* Pack imaginary part at offset + n_real */
    if (E->has_imag && E->Er_im) {
        int grid_Er = (E->size_Er + BLOCK_COPY - 1) / BLOCK_COPY;
        int grid_Ephi = (E->size_Ephi + BLOCK_COPY - 1) / BLOCK_COPY;
        int grid_Ez = (E->size_Ez + BLOCK_COPY - 1) / BLOCK_COPY;

        pack_component_kernel<<<grid_Er, BLOCK_COPY>>>(
            E->Er_im, d_x, offset_Er + n_real, E->size_Er);
        pack_component_kernel<<<grid_Ephi, BLOCK_COPY>>>(
            E->Ephi_im, d_x, offset_Ephi + n_real, E->size_Ephi);
        pack_component_kernel<<<grid_Ez, BLOCK_COPY>>>(
            E->Ez_im, d_x, offset_Ez + n_real, E->size_Ez);

        CUDA_CHECK(cudaGetLastError());
    }

    return 0;
}

int gpu_unpack_field_complex(const double* d_x, GPU_EField* E,
    int offset_Er, int offset_Ephi, int offset_Ez,
    int n_real)
{
    /* Unpack real part (same as original) */
    int status = gpu_unpack_field(d_x, E, offset_Er, offset_Ephi, offset_Ez);
    if (status != 0) return status;

    /* Unpack imaginary part from offset + n_real */
    if (E->has_imag && E->Er_im) {
        int grid_Er = (E->size_Er + BLOCK_COPY - 1) / BLOCK_COPY;
        int grid_Ephi = (E->size_Ephi + BLOCK_COPY - 1) / BLOCK_COPY;
        int grid_Ez = (E->size_Ez + BLOCK_COPY - 1) / BLOCK_COPY;

        unpack_component_kernel<<<grid_Er, BLOCK_COPY>>>(
            d_x, E->Er_im, offset_Er + n_real, E->size_Er);
        unpack_component_kernel<<<grid_Ephi, BLOCK_COPY>>>(
            d_x, E->Ephi_im, offset_Ephi + n_real, E->size_Ephi);
        unpack_component_kernel<<<grid_Ez, BLOCK_COPY>>>(
            d_x, E->Ez_im, offset_Ez + n_real, E->size_Ez);

        CUDA_CHECK(cudaGetLastError());
    }

    return 0;
}
