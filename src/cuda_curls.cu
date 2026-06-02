#include "cuda_curls.h"
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

#define BLOCK_SIZE 256

 /*=============================================================================
  * Grid parameters passed to kernels (same approach as vector ops)
  *============================================================================*/
typedef struct {
    double a, dr, dphi, dz, L;
    int Nr, Nphi, Nz;
} CurlGridParams;

static CurlGridParams make_curl_params(const GridParams* grid) {
    CurlGridParams p;
    p.a = grid->a;
    p.dr = grid->dr;
    p.dphi = grid->dphi;
    p.dz = grid->dz;
    p.L = grid->L;
    p.Nr = grid->Nr;
    p.Nphi = grid->Nphi;
    p.Nz = grid->Nz;
    return p;
}

/*=============================================================================
 * Device index functions
 * Mirror the CPU idx_* functions exactly
 *============================================================================*/

 /* Er at (i+1/2, j, k): i=0..Nr-1, j=0..Nphi-1, k=0..Nz */
__device__ __forceinline__
int d_idx_Er(int Nr, int Nphi, int i, int j, int k) {
    return i + Nr * (j + Nphi * k);
}

/* Ephi at (i, j+1/2, k): i=0..Nr, j=0..Nphi-1, k=0..Nz */
__device__ __forceinline__
int d_idx_Ephi(int Nr, int Nphi, int i, int j, int k) {
    return i + (Nr + 1) * (j + Nphi * k);
}

/* Ez at (i, j, k+1/2): i=0..Nr, j=0..Nphi-1, k=0..Nz-1 */
__device__ __forceinline__
int d_idx_Ez(int Nr, int Nphi, int i, int j, int k) {
    return i + (Nr + 1) * (j + Nphi * k);
}

/* Hr at (i, j+1/2, k+1/2): i=0..Nr, j=0..Nphi-1, k=0..Nz-1 */
__device__ __forceinline__
int d_idx_Hr(int Nr, int Nphi, int i, int j, int k) {
    return i + (Nr + 1) * (j + Nphi * k);
}

/* Hphi at (i+1/2, j, k+1/2): i=0..Nr-1, j=0..Nphi-1, k=0..Nz-1 */
__device__ __forceinline__
int d_idx_Hphi(int Nr, int Nphi, int i, int j, int k) {
    return i + Nr * (j + Nphi * k);
}

/* Hz at (i+1/2, j+1/2, k): i=0..Nr-1, j=0..Nphi-1, k=0..Nz */
__device__ __forceinline__
int d_idx_Hz(int Nr, int Nphi, int i, int j, int k) {
    return i + Nr * (j + Nphi * k);
}

/* Periodic phi index */
__device__ __forceinline__
int d_periodic_j(int j, int Nphi) {
    return ((j % Nphi) + Nphi) % Nphi;
}

/*=============================================================================
 * Curl of E kernels (UNCHANGED from original)
 *============================================================================*/

 /*-----------------------------------------------------------------------------
  * (curl E)_r = (1/r) * dEz/dphi - dEphi/dz
  * Location: (i, j+1/2, k+1/2) for i=0..Nr, j=0..Nphi-1, k=0..Nz-1
  * Size: (Nr+1) * Nphi * Nz
  *---------------------------------------------------------------------------*/
__global__ void curl_E_r_kernel(
    const double* Ez,
    const double* Ephi,
    double* Hr,             /* output: (curl E)_r stored in Hr */
    CurlGridParams gp
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (gp.Nr + 1) * gp.Nphi * gp.Nz;

    if (gid >= total) return;

    /* Recover (i, j, k) from flat index */
    int Nr1 = gp.Nr + 1;
    int tmp = gid;
    int i = tmp % Nr1;    tmp /= Nr1;
    int j = tmp % gp.Nphi;
    int k = tmp / gp.Nphi;

    double r_i = gp.a + i * gp.dr;
    double inv_dphi = 1.0 / gp.dphi;
    double inv_dz = 1.0 / gp.dz;

    int jp1 = d_periodic_j(j + 1, gp.Nphi);

    /* dEz/dphi: Ez(i, j+1, k) - Ez(i, j, k) */
    double Ez_jp1 = Ez[d_idx_Ez(gp.Nr, gp.Nphi, i, jp1, k)];
    double Ez_j = Ez[d_idx_Ez(gp.Nr, gp.Nphi, i, j, k)];
    double dEz_dphi = (Ez_jp1 - Ez_j) * inv_dphi;

    /* dEphi/dz: Ephi(i, j, k+1) - Ephi(i, j, k) */
    double Ephi_kp1 = Ephi[d_idx_Ephi(gp.Nr, gp.Nphi, i, j, k + 1)];
    double Ephi_k = Ephi[d_idx_Ephi(gp.Nr, gp.Nphi, i, j, k)];
    double dEphi_dz = (Ephi_kp1 - Ephi_k) * inv_dz;

    double result;
    if (r_i < 1e-14) {
        result = -dEphi_dz;
    }
    else {
        result = dEz_dphi / r_i - dEphi_dz;
    }

    Hr[gid] = result;
}

/*-----------------------------------------------------------------------------
 * (curl E)_phi = dEr/dz - dEz/dr
 * Location: (i+1/2, j, k+1/2) for i=0..Nr-1, j=0..Nphi-1, k=0..Nz-1
 * Size: Nr * Nphi * Nz
 *---------------------------------------------------------------------------*/
__global__ void curl_E_phi_kernel(
    const double* Er,
    const double* Ez,
    double* Hphi,           /* output: (curl E)_phi stored in Hphi */
    CurlGridParams gp
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gp.Nr * gp.Nphi * gp.Nz;

    if (gid >= total) return;

    int tmp = gid;
    int i = tmp % gp.Nr;     tmp /= gp.Nr;
    int j = tmp % gp.Nphi;
    int k = tmp / gp.Nphi;

    double inv_dz = 1.0 / gp.dz;
    double inv_dr = 1.0 / gp.dr;

    /* dEr/dz: Er(i, j, k+1) - Er(i, j, k) */
    double Er_kp1 = Er[d_idx_Er(gp.Nr, gp.Nphi, i, j, k + 1)];
    double Er_k = Er[d_idx_Er(gp.Nr, gp.Nphi, i, j, k)];
    double dEr_dz = (Er_kp1 - Er_k) * inv_dz;

    /* dEz/dr: Ez(i+1, j, k) - Ez(i, j, k) */
    double Ez_ip1 = Ez[d_idx_Ez(gp.Nr, gp.Nphi, i + 1, j, k)];
    double Ez_i = Ez[d_idx_Ez(gp.Nr, gp.Nphi, i, j, k)];
    double dEz_dr = (Ez_ip1 - Ez_i) * inv_dr;

    Hphi[gid] = dEr_dz - dEz_dr;
}

/*-----------------------------------------------------------------------------
 * (curl E)_z = (1/r) * d(r*Ephi)/dr - (1/r) * dEr/dphi
 * Location: (i+1/2, j+1/2, k) for i=0..Nr-1, j=0..Nphi-1, k=0..Nz
 * Size: Nr * Nphi * (Nz+1)
 *---------------------------------------------------------------------------*/
__global__ void curl_E_z_kernel(
    const double* Er,
    const double* Ephi,
    double* Hz,             /* output: (curl E)_z stored in Hz */
    CurlGridParams gp
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gp.Nr * gp.Nphi * (gp.Nz + 1);

    if (gid >= total) return;

    int tmp = gid;
    int i = tmp % gp.Nr;     tmp /= gp.Nr;
    int j = tmp % gp.Nphi;
    int k = tmp / gp.Nphi;

    double r_i = gp.a + i * gp.dr;
    double r_ip1 = gp.a + (i + 1) * gp.dr;
    double r_iph = gp.a + (i + 0.5) * gp.dr;

    double inv_dr = 1.0 / gp.dr;
    double inv_dphi = 1.0 / gp.dphi;

    int jp1 = d_periodic_j(j + 1, gp.Nphi);

    /* d(r*Ephi)/dr: r_{i+1}*Ephi(i+1,j,k) - r_i*Ephi(i,j,k) */
    double Ephi_ip1 = Ephi[d_idx_Ephi(gp.Nr, gp.Nphi, i + 1, j, k)];
    double Ephi_i = Ephi[d_idx_Ephi(gp.Nr, gp.Nphi, i, j, k)];
    double d_rEphi_dr = (r_ip1 * Ephi_ip1 - r_i * Ephi_i) * inv_dr;

    /* dEr/dphi: Er(i, j+1, k) - Er(i, j, k) */
    double Er_jp1 = Er[d_idx_Er(gp.Nr, gp.Nphi, i, jp1, k)];
    double Er_j = Er[d_idx_Er(gp.Nr, gp.Nphi, i, j, k)];
    double dEr_dphi = (Er_jp1 - Er_j) * inv_dphi;

    Hz[gid] = d_rEphi_dr / r_iph - dEr_dphi / r_iph;
}

/*=============================================================================
 * Curl of H kernels (UNCHANGED from original)
 *============================================================================*/

 /*-----------------------------------------------------------------------------
  * (curl H)_r = (1/r) * dHz/dphi - dHphi/dz
  * Output: (i+1/2, j, k) for i=0..Nr-1, j=0..Nphi-1, k=0..Nz
  * Size: Nr * Nphi * (Nz+1)
  *---------------------------------------------------------------------------*/
__global__ void curl_H_r_kernel(
    const double* Hz,
    const double* Hphi,
    double* Er_out,         /* output: (curl H)_r stored in Er */
    CurlGridParams gp
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gp.Nr * gp.Nphi * (gp.Nz + 1);

    if (gid >= total) return;

    int tmp = gid;
    int i = tmp % gp.Nr;     tmp /= gp.Nr;
    int j = tmp % gp.Nphi;
    int k = tmp / gp.Nphi;

    double r_iph = gp.a + (i + 0.5) * gp.dr;
    double inv_dphi = 1.0 / gp.dphi;
    double inv_dz = 1.0 / gp.dz;

    int jm1 = d_periodic_j(j - 1, gp.Nphi);

    /* dHz/dphi: Hz(i, j, k) - Hz(i, j-1, k) */
    double Hz_j = Hz[d_idx_Hz(gp.Nr, gp.Nphi, i, j, k)];
    double Hz_jm1 = Hz[d_idx_Hz(gp.Nr, gp.Nphi, i, jm1, k)];
    double dHz_dphi = (Hz_j - Hz_jm1) * inv_dphi;

    /* dHphi/dz: Hphi(i, j, k) - Hphi(i, j, k-1) */
    double Hphi_kph = 0.0;
    double Hphi_kmh = 0.0;

    if (k < gp.Nz) {
        Hphi_kph = Hphi[d_idx_Hphi(gp.Nr, gp.Nphi, i, j, k)];
    }
    if (k > 0) {
        Hphi_kmh = Hphi[d_idx_Hphi(gp.Nr, gp.Nphi, i, j, k - 1)];
    }
    double dHphi_dz = (Hphi_kph - Hphi_kmh) * inv_dz;

    Er_out[gid] = dHz_dphi / r_iph - dHphi_dz;
}

/*-----------------------------------------------------------------------------
 * (curl H)_phi = dHr/dz - dHz/dr
 * Output: (i, j+1/2, k) for i=0..Nr, j=0..Nphi-1, k=0..Nz
 * Size: (Nr+1) * Nphi * (Nz+1)
 *---------------------------------------------------------------------------*/
__global__ void curl_H_phi_kernel(
    const double* Hr,
    const double* Hz,
    double* Ephi_out,       /* output: (curl H)_phi stored in Ephi */
    CurlGridParams gp
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int Nr1 = gp.Nr + 1;
    int total = Nr1 * gp.Nphi * (gp.Nz + 1);

    if (gid >= total) return;

    int tmp = gid;
    int i = tmp % Nr1;        tmp /= Nr1;
    int j = tmp % gp.Nphi;
    int k = tmp / gp.Nphi;

    double inv_dz = 1.0 / gp.dz;
    double inv_dr = 1.0 / gp.dr;

    /* dHr/dz: Hr(i, j, k) - Hr(i, j, k-1) */
    double Hr_kph = 0.0;
    double Hr_kmh = 0.0;

    if (k < gp.Nz) {
        Hr_kph = Hr[d_idx_Hr(gp.Nr, gp.Nphi, i, j, k)];
    }
    if (k > 0) {
        Hr_kmh = Hr[d_idx_Hr(gp.Nr, gp.Nphi, i, j, k - 1)];
    }
    double dHr_dz = (Hr_kph - Hr_kmh) * inv_dz;

    /* dHz/dr: Hz(i, j, k) - Hz(i-1, j, k) */
    double Hz_iph = 0.0;
    double Hz_imh = 0.0;

    if (i < gp.Nr) {
        Hz_iph = Hz[d_idx_Hz(gp.Nr, gp.Nphi, i, j, k)];
    }
    if (i > 0) {
        Hz_imh = Hz[d_idx_Hz(gp.Nr, gp.Nphi, i - 1, j, k)];
    }
    double dHz_dr = (Hz_iph - Hz_imh) * inv_dr;

    Ephi_out[gid] = dHr_dz - dHz_dr;
}

/*-----------------------------------------------------------------------------
 * (curl H)_z = (1/r) * d(r*Hphi)/dr - (1/r) * dHr/dphi
 * Output: (i, j, k+1/2) for i=0..Nr, j=0..Nphi-1, k=0..Nz-1
 * Size: (Nr+1) * Nphi * Nz
 *---------------------------------------------------------------------------*/
__global__ void curl_H_z_kernel(
    const double* Hr,
    const double* Hphi,
    double* Ez_out,         /* output: (curl H)_z stored in Ez */
    CurlGridParams gp
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int Nr1 = gp.Nr + 1;
    int total = Nr1 * gp.Nphi * gp.Nz;

    if (gid >= total) return;

    int tmp = gid;
    int i = tmp % Nr1;        tmp /= Nr1;
    int j = tmp % gp.Nphi;
    int k = tmp / gp.Nphi;

    double r_i = gp.a + i * gp.dr;
    double r_iph = gp.a + (i + 0.5) * gp.dr;
    double r_imh = (i > 0) ? gp.a + (i - 0.5) * gp.dr : 0.0;

    double inv_dr = 1.0 / gp.dr;
    double inv_dphi = 1.0 / gp.dphi;

    int jm1 = d_periodic_j(j - 1, gp.Nphi);

    /* d(r*Hphi)/dr: r_{i+1/2}*Hphi(i,j,k) - r_{i-1/2}*Hphi(i-1,j,k) */
    double rHphi_iph = 0.0;
    double rHphi_imh = 0.0;

    if (i < gp.Nr) {
        rHphi_iph = r_iph * Hphi[d_idx_Hphi(gp.Nr, gp.Nphi, i, j, k)];
    }
    if (i > 0) {
        rHphi_imh = r_imh * Hphi[d_idx_Hphi(gp.Nr, gp.Nphi, i - 1, j, k)];
    }
    double d_rHphi_dr = (rHphi_iph - rHphi_imh) * inv_dr;

    /* dHr/dphi: Hr(i, j, k) - Hr(i, j-1, k) */
    double Hr_jph = Hr[d_idx_Hr(gp.Nr, gp.Nphi, i, j, k)];
    double Hr_jmh = Hr[d_idx_Hr(gp.Nr, gp.Nphi, i, jm1, k)];
    double dHr_dphi = (Hr_jph - Hr_jmh) * inv_dphi;

    double result;
    if (r_i < 1e-14) {
        result = d_rHphi_dr * 2.0;
    }
    else {
        result = d_rHphi_dr / r_i - dHr_dphi / r_i;
    }

    Ez_out[gid] = result;
}

/*=============================================================================
 * Internal raw-pointer launch helpers
 *
 * These factor out the grid-params + block-sizing logic so that both the
 * original struct-based API and the new _complex variants can share them
 * without duplicating code.
 *============================================================================*/

static int launch_curl_E_r(
    const double* Ez, const double* Ephi, double* Hr,
    const GridParams* grid
) {
    CurlGridParams gp = make_curl_params(grid);
    int total = (grid->Nr + 1) * grid->Nphi * grid->Nz;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    curl_E_r_kernel<<<blocks, BLOCK_SIZE>>>(Ez, Ephi, Hr, gp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int launch_curl_E_phi(
    const double* Er, const double* Ez, double* Hphi,
    const GridParams* grid
) {
    CurlGridParams gp = make_curl_params(grid);
    int total = grid->Nr * grid->Nphi * grid->Nz;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    curl_E_phi_kernel<<<blocks, BLOCK_SIZE>>>(Er, Ez, Hphi, gp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int launch_curl_E_z(
    const double* Er, const double* Ephi, double* Hz,
    const GridParams* grid
) {
    CurlGridParams gp = make_curl_params(grid);
    int total = grid->Nr * grid->Nphi * (grid->Nz + 1);
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    curl_E_z_kernel<<<blocks, BLOCK_SIZE>>>(Er, Ephi, Hz, gp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int launch_curl_H_r(
    const double* Hz, const double* Hphi, double* Er_out,
    const GridParams* grid
) {
    CurlGridParams gp = make_curl_params(grid);
    int total = grid->Nr * grid->Nphi * (grid->Nz + 1);
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    curl_H_r_kernel<<<blocks, BLOCK_SIZE>>>(Hz, Hphi, Er_out, gp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int launch_curl_H_phi(
    const double* Hr, const double* Hz, double* Ephi_out,
    const GridParams* grid
) {
    CurlGridParams gp = make_curl_params(grid);
    int total = (grid->Nr + 1) * grid->Nphi * (grid->Nz + 1);
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    curl_H_phi_kernel<<<blocks, BLOCK_SIZE>>>(Hr, Hz, Ephi_out, gp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int launch_curl_H_z(
    const double* Hr, const double* Hphi, double* Ez_out,
    const GridParams* grid
) {
    CurlGridParams gp = make_curl_params(grid);
    int total = (grid->Nr + 1) * grid->Nphi * grid->Nz;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    curl_H_z_kernel<<<blocks, BLOCK_SIZE>>>(Hr, Hphi, Ez_out, gp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/*=============================================================================
 * Public API — Real part only (backward compatible)
 *
 * These now delegate to the internal raw-pointer helpers.
 * Behavior is IDENTICAL to the original code.
 *============================================================================*/

int gpu_compute_curl_E_r(const GPU_EField* E, GPU_HField* curlE,
    const GridParams* grid) {
    return launch_curl_E_r(E->Ez, E->Ephi, curlE->Hr, grid);
}

int gpu_compute_curl_E_phi(const GPU_EField* E, GPU_HField* curlE,
    const GridParams* grid) {
    return launch_curl_E_phi(E->Er, E->Ez, curlE->Hphi, grid);
}

int gpu_compute_curl_E_z(const GPU_EField* E, GPU_HField* curlE,
    const GridParams* grid) {
    return launch_curl_E_z(E->Er, E->Ephi, curlE->Hz, grid);
}

int gpu_compute_curl_E(const GPU_EField* E, GPU_HField* curlE,
    const GridParams* grid) {
    int status;
    status = gpu_compute_curl_E_r(E, curlE, grid);
    if (status != 0) return status;
    status = gpu_compute_curl_E_phi(E, curlE, grid);
    if (status != 0) return status;
    status = gpu_compute_curl_E_z(E, curlE, grid);
    if (status != 0) return status;
    return 0;
}

int gpu_compute_curl_H_r(const GPU_HField* H, GPU_EField* curlH,
    const GridParams* grid) {
    return launch_curl_H_r(H->Hz, H->Hphi, curlH->Er, grid);
}

int gpu_compute_curl_H_phi(const GPU_HField* H, GPU_EField* curlH,
    const GridParams* grid) {
    return launch_curl_H_phi(H->Hr, H->Hz, curlH->Ephi, grid);
}

int gpu_compute_curl_H_z(const GPU_HField* H, GPU_EField* curlH,
    const GridParams* grid) {
    return launch_curl_H_z(H->Hr, H->Hphi, curlH->Ez, grid);
}

int gpu_compute_curl_H(const GPU_HField* H, GPU_EField* curlH,
    const GridParams* grid) {
    int status;
    status = gpu_compute_curl_H_r(H, curlH, grid);
    if (status != 0) return status;
    status = gpu_compute_curl_H_phi(H, curlH, grid);
    if (status != 0) return status;
    status = gpu_compute_curl_H_z(H, curlH, grid);
    if (status != 0) return status;
    return 0;
}

/*=============================================================================
 * Curl-Curl: ∇ × ∇ × E  (real part only, backward compatible)
 *============================================================================*/
int gpu_compute_curl_curl_E(
    const GPU_EField* E,
    GPU_EField* result,
    GPU_HField* temp,
    const GridParams* grid
) {
    int status;

    /* Step 1: temp = ∇ × E */
    status = gpu_compute_curl_E(E, temp, grid);
    if (status != 0) return status;

    /* Step 2: result = ∇ × temp */
    status = gpu_compute_curl_H(temp, result, grid);
    if (status != 0) return status;

    return 0;
}

/*=============================================================================
 * Public API — Complex (IBC extension)
 *
 * Run the same stencils on both real and imaginary component arrays.
 * No cross-coupling: curl is linear.
 *============================================================================*/

int gpu_compute_curl_E_r_complex(const GPU_EField* E, GPU_HField* curlE,
    const GridParams* grid) {
    int status;
    /* Real part */
    status = launch_curl_E_r(E->Ez, E->Ephi, curlE->Hr, grid);
    if (status != 0) return status;
    /* Imaginary part */
    status = launch_curl_E_r(E->Ez_im, E->Ephi_im, curlE->Hr_im, grid);
    return status;
}

int gpu_compute_curl_E_phi_complex(const GPU_EField* E, GPU_HField* curlE,
    const GridParams* grid) {
    int status;
    status = launch_curl_E_phi(E->Er, E->Ez, curlE->Hphi, grid);
    if (status != 0) return status;
    status = launch_curl_E_phi(E->Er_im, E->Ez_im, curlE->Hphi_im, grid);
    return status;
}

int gpu_compute_curl_E_z_complex(const GPU_EField* E, GPU_HField* curlE,
    const GridParams* grid) {
    int status;
    status = launch_curl_E_z(E->Er, E->Ephi, curlE->Hz, grid);
    if (status != 0) return status;
    status = launch_curl_E_z(E->Er_im, E->Ephi_im, curlE->Hz_im, grid);
    return status;
}

int gpu_compute_curl_E_complex(const GPU_EField* E, GPU_HField* curlE,
    const GridParams* grid) {
    int status;
    status = gpu_compute_curl_E_r_complex(E, curlE, grid);
    if (status != 0) return status;
    status = gpu_compute_curl_E_phi_complex(E, curlE, grid);
    if (status != 0) return status;
    status = gpu_compute_curl_E_z_complex(E, curlE, grid);
    if (status != 0) return status;
    return 0;
}

int gpu_compute_curl_H_r_complex(const GPU_HField* H, GPU_EField* curlH,
    const GridParams* grid) {
    int status;
    status = launch_curl_H_r(H->Hz, H->Hphi, curlH->Er, grid);
    if (status != 0) return status;
    status = launch_curl_H_r(H->Hz_im, H->Hphi_im, curlH->Er_im, grid);
    return status;
}

int gpu_compute_curl_H_phi_complex(const GPU_HField* H, GPU_EField* curlH,
    const GridParams* grid) {
    int status;
    status = launch_curl_H_phi(H->Hr, H->Hz, curlH->Ephi, grid);
    if (status != 0) return status;
    status = launch_curl_H_phi(H->Hr_im, H->Hz_im, curlH->Ephi_im, grid);
    return status;
}

int gpu_compute_curl_H_z_complex(const GPU_HField* H, GPU_EField* curlH,
    const GridParams* grid) {
    int status;
    status = launch_curl_H_z(H->Hr, H->Hphi, curlH->Ez, grid);
    if (status != 0) return status;
    status = launch_curl_H_z(H->Hr_im, H->Hphi_im, curlH->Ez_im, grid);
    return status;
}

int gpu_compute_curl_H_complex(const GPU_HField* H, GPU_EField* curlH,
    const GridParams* grid) {
    int status;
    status = gpu_compute_curl_H_r_complex(H, curlH, grid);
    if (status != 0) return status;
    status = gpu_compute_curl_H_phi_complex(H, curlH, grid);
    if (status != 0) return status;
    status = gpu_compute_curl_H_z_complex(H, curlH, grid);
    if (status != 0) return status;
    return 0;
}

/*=============================================================================
 * Curl-Curl complex: ∇ × ∇ × E on both real and imaginary parts
 *============================================================================*/
int gpu_compute_curl_curl_E_complex(
    const GPU_EField* E,
    GPU_EField* result,
    GPU_HField* temp,
    const GridParams* grid
) {
    int status;

    /* Step 1: temp = ∇ × E  (both real and imaginary) */
    status = gpu_compute_curl_E_complex(E, temp, grid);
    if (status != 0) return status;

    /* Step 2: result = ∇ × temp  (both real and imaginary) */
    status = gpu_compute_curl_H_complex(temp, result, grid);
    if (status != 0) return status;

    return 0;
}
