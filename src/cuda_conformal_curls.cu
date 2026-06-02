/*=============================================================================
 * cuda_conformal_curls_cu.cpp
 *
 * Dey-Mittra conformal curl kernels for the rhodotron FDFD eigensolver.
 *
 * Each kernel uses the integral form of Faraday's / Ampere's law:
 *
 *   Curl-E (Faraday): H_face = (1/A_vac) · ∮ f·E·dl
 *   Curl-H (Ampere):  E_dual = (1/A_dual_vac) · ∮ H·dl
 *
 * where f is the edge vacuum fraction and A_vac / A_dual_vac are the
 * precomputed vacuum face areas from the conformal geometry engine.
 *
 * When all fractions = 1.0 and areas = standard values, these kernels
 * produce IDENTICAL results to the standard curl kernels.
 *
 * GPU memory: reads 3 edge_frac arrays + 3 face_area arrays (curl-E)
 * or 3 dual_area arrays (curl-H). Total ~72 bytes/DOF extra reads.
 *============================================================================*/

#include "cuda_conformal.h"
#include <stdio.h>
#include <cuda_runtime.h>

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
 * Grid params struct (same as cuda_curls_cu.cpp)
 *============================================================================*/
typedef struct {
    double a, dr, dphi, dz, L;
    int Nr, Nphi, Nz;
} CurlGridParams;

static CurlGridParams make_curl_params(const GridParams* grid) {
    CurlGridParams p;
    p.a = grid->a; p.dr = grid->dr; p.dphi = grid->dphi;
    p.dz = grid->dz; p.L = grid->L;
    p.Nr = grid->Nr; p.Nphi = grid->Nphi; p.Nz = grid->Nz;
    return p;
}

/*=============================================================================
 * Device index functions (same as cuda_curls_cu.cpp)
 *============================================================================*/

__device__ __forceinline__
int d_idx_Er(int Nr, int Nphi, int i, int j, int k) {
    return i + Nr * (j + Nphi * k);
}

__device__ __forceinline__
int d_idx_Ephi(int Nr, int Nphi, int i, int j, int k) {
    return i + (Nr + 1) * (j + Nphi * k);
}

__device__ __forceinline__
int d_idx_Ez(int Nr, int Nphi, int i, int j, int k) {
    return i + (Nr + 1) * (j + Nphi * k);
}

__device__ __forceinline__
int d_idx_Hr(int Nr, int Nphi, int i, int j, int k) {
    return i + (Nr + 1) * (j + Nphi * k);
}

__device__ __forceinline__
int d_idx_Hphi(int Nr, int Nphi, int i, int j, int k) {
    return i + Nr * (j + Nphi * k);
}

__device__ __forceinline__
int d_idx_Hz(int Nr, int Nphi, int i, int j, int k) {
    return i + Nr * (j + Nphi * k);
}

__device__ __forceinline__
int d_periodic_j(int j, int Nphi) {
    return ((j % Nphi) + Nphi) % Nphi;
}

/*=============================================================================
 * GPU Conformal Data: upload & free
 *============================================================================*/

static int upload_array(double** d_ptr, const double* h_ptr, int n) {
    CUDA_CHECK(cudaMalloc(d_ptr, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(*d_ptr, h_ptr, n * sizeof(double),
                           cudaMemcpyHostToDevice));
    return 0;
}

int gpu_conformal_data_init(
    GPU_ConformalData* gpu_cd,
    const ConformalData* cpu_cd
) {
    memset(gpu_cd, 0, sizeof(GPU_ConformalData));

    gpu_cd->size_Er   = cpu_cd->size_Er;
    gpu_cd->size_Ephi = cpu_cd->size_Ephi;
    gpu_cd->size_Ez   = cpu_cd->size_Ez;
    gpu_cd->size_Hr   = cpu_cd->size_Hr;
    gpu_cd->size_Hphi = cpu_cd->size_Hphi;
    gpu_cd->size_Hz   = cpu_cd->size_Hz;

    /* Edge fractions */
    if (upload_array(&gpu_cd->d_edge_frac_Er,   cpu_cd->edge_frac_Er,   cpu_cd->size_Er))   return -1;
    if (upload_array(&gpu_cd->d_edge_frac_Ephi, cpu_cd->edge_frac_Ephi, cpu_cd->size_Ephi)) return -1;
    if (upload_array(&gpu_cd->d_edge_frac_Ez,   cpu_cd->edge_frac_Ez,   cpu_cd->size_Ez))   return -1;

    /* Primal face areas */
    if (upload_array(&gpu_cd->d_face_area_Hr,   cpu_cd->face_area_Hr,   cpu_cd->size_Hr))   return -1;
    if (upload_array(&gpu_cd->d_face_area_Hphi, cpu_cd->face_area_Hphi, cpu_cd->size_Hphi)) return -1;
    if (upload_array(&gpu_cd->d_face_area_Hz,   cpu_cd->face_area_Hz,   cpu_cd->size_Hz))   return -1;

    /* Dual face areas */
    if (upload_array(&gpu_cd->d_dual_area_Er,   cpu_cd->dual_area_Er,   cpu_cd->size_Er))   return -1;
    if (upload_array(&gpu_cd->d_dual_area_Ephi, cpu_cd->dual_area_Ephi, cpu_cd->size_Ephi)) return -1;
    if (upload_array(&gpu_cd->d_dual_area_Ez,   cpu_cd->dual_area_Ez,   cpu_cd->size_Ez))   return -1;

    /* IBC weights */
    if (upload_array(&gpu_cd->d_ibc_weight_Er,   cpu_cd->ibc_weight_Er,   cpu_cd->size_Er))   return -1;
    if (upload_array(&gpu_cd->d_ibc_weight_Ephi, cpu_cd->ibc_weight_Ephi, cpu_cd->size_Ephi)) return -1;
    if (upload_array(&gpu_cd->d_ibc_weight_Ez,   cpu_cd->ibc_weight_Ez,   cpu_cd->size_Ez))   return -1;

    gpu_cd->initialized = 1;

    /* Memory report */
    size_t total = (cpu_cd->size_Er + cpu_cd->size_Ephi + cpu_cd->size_Ez) * 4
                 + (cpu_cd->size_Hr + cpu_cd->size_Hphi + cpu_cd->size_Hz);
    total += (cpu_cd->size_Er + cpu_cd->size_Ephi + cpu_cd->size_Ez); /* dual areas */
    printf("  GPU conformal data: %.1f MB uploaded\n",
           total * sizeof(double) / (1024.0 * 1024.0));

    return 0;
}

static void safe_free(double** p) {
    if (*p) { cudaFree(*p); *p = NULL; }
}

void gpu_conformal_data_free(GPU_ConformalData* gpu_cd) {
    safe_free(&gpu_cd->d_edge_frac_Er);
    safe_free(&gpu_cd->d_edge_frac_Ephi);
    safe_free(&gpu_cd->d_edge_frac_Ez);
    safe_free(&gpu_cd->d_face_area_Hr);
    safe_free(&gpu_cd->d_face_area_Hphi);
    safe_free(&gpu_cd->d_face_area_Hz);
    safe_free(&gpu_cd->d_dual_area_Er);
    safe_free(&gpu_cd->d_dual_area_Ephi);
    safe_free(&gpu_cd->d_dual_area_Ez);
    safe_free(&gpu_cd->d_ibc_weight_Er);
    safe_free(&gpu_cd->d_ibc_weight_Ephi);
    safe_free(&gpu_cd->d_ibc_weight_Ez);
    gpu_cd->initialized = 0;
}

/*=============================================================================
 * Conformal Curl-E Kernels
 *
 * Faraday's law in integral form on each primal face:
 *   ∮ E·dl = -∂/∂t ∫ B·dS
 *
 * Discretized:
 *   (∇×E)_face = (1 / A_vac) · Σ_edges (f_edge · E_edge · l_edge)
 *
 * where f_edge is the vacuum fraction and l_edge is the edge length
 * (with appropriate sign from the circulation direction).
 *============================================================================*/

/*-----------------------------------------------------------------------------
 * Conformal (curl E)_r at (i, j+½, k+½)
 *
 * Face in (φ,z) plane at r = r_i.
 * Standard: (1/r) dEz/dφ - dEphi/dz
 *
 * Integral form:
 *   + f_Ez(i,j+1,k) · Ez(i,j+1,k) · dz    (top edge, +φ direction)
 *   - f_Ez(i,j,k)   · Ez(i,j,k)   · dz    (bottom edge)
 *   - f_Ephi(i,j,k+1)· Ephi(i,j,k+1) · r·dφ  (right edge, +z direction)
 *   + f_Ephi(i,j,k)  · Ephi(i,j,k)  · r·dφ  (left edge)
 *   all divided by A_vac_Hr(i,j,k)
 *
 * Note: edge indexing follows the circulation around the face.
 * Ez edges are at j and j+1 (φ direction), Eφ edges at k and k+1 (z direction).
 *---------------------------------------------------------------------------*/
__global__ void conformal_curl_E_r_kernel(
    const double* Ez, const double* Ephi,
    double* Hr,
    const double* frac_Ez, const double* frac_Ephi,
    const double* face_area_Hr,
    CurlGridParams gp
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int Nr1 = gp.Nr + 1;
    int total = Nr1 * gp.Nphi * gp.Nz;
    if (gid >= total) return;

    int tmp = gid;
    int i = tmp % Nr1;     tmp /= Nr1;
    int j = tmp % gp.Nphi;
    int k = tmp / gp.Nphi;

    double A_vac = face_area_Hr[gid];
    if (A_vac < 1e-30) {
        Hr[gid] = 0.0;
        return;
    }

    double r_i = gp.a + i * gp.dr;
    int jp1 = d_periodic_j(j + 1, gp.Nphi);

    /* Ez edges in φ-direction (at k, varying j) */
    int idx_Ez_jp1 = d_idx_Ez(gp.Nr, gp.Nphi, i, jp1, k);
    int idx_Ez_j   = d_idx_Ez(gp.Nr, gp.Nphi, i, j,   k);

    /* Eφ edges in z-direction (at j, varying k) */
    int idx_Ephi_kp1 = d_idx_Ephi(gp.Nr, gp.Nphi, i, j, k + 1);
    int idx_Ephi_k   = d_idx_Ephi(gp.Nr, gp.Nphi, i, j, k);

    double line_integral =
        + frac_Ez[idx_Ez_jp1]   * Ez[idx_Ez_jp1]   * gp.dz
        - frac_Ez[idx_Ez_j]     * Ez[idx_Ez_j]     * gp.dz
        - frac_Ephi[idx_Ephi_kp1] * Ephi[idx_Ephi_kp1] * r_i * gp.dphi
        + frac_Ephi[idx_Ephi_k]   * Ephi[idx_Ephi_k]   * r_i * gp.dphi;

    Hr[gid] = line_integral / A_vac;
}

/*-----------------------------------------------------------------------------
 * Conformal (curl E)_φ at (i+½, j, k+½)
 *
 * Face in (r,z) plane at φ = φ_j.
 * Standard: dEr/dz - dEz/dr
 *
 * Integral form:
 *   + f_Er(i,j,k+1)  · Er(i,j,k+1)  · dr    (top, +r direction)
 *   - f_Er(i,j,k)    · Er(i,j,k)    · dr    (bottom)
 *   - f_Ez(i+1,j,k)  · Ez(i+1,j,k)  · dz    (right, +z direction)
 *   + f_Ez(i,j,k)    · Ez(i,j,k)    · dz    (left)
 *   / A_vac_Hphi(i,j,k)
 *---------------------------------------------------------------------------*/
__global__ void conformal_curl_E_phi_kernel(
    const double* Er, const double* Ez,
    double* Hphi,
    const double* frac_Er, const double* frac_Ez,
    const double* face_area_Hphi,
    CurlGridParams gp
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gp.Nr * gp.Nphi * gp.Nz;
    if (gid >= total) return;

    int tmp = gid;
    int i = tmp % gp.Nr;     tmp /= gp.Nr;
    int j = tmp % gp.Nphi;
    int k = tmp / gp.Nphi;

    double A_vac = face_area_Hphi[gid];
    if (A_vac < 1e-30) {
        Hphi[gid] = 0.0;
        return;
    }

    int idx_Er_kp1 = d_idx_Er(gp.Nr, gp.Nphi, i, j, k + 1);
    int idx_Er_k   = d_idx_Er(gp.Nr, gp.Nphi, i, j, k);
    int idx_Ez_ip1 = d_idx_Ez(gp.Nr, gp.Nphi, i + 1, j, k);
    int idx_Ez_i   = d_idx_Ez(gp.Nr, gp.Nphi, i, j, k);

    double line_integral =
        + frac_Er[idx_Er_kp1] * Er[idx_Er_kp1] * gp.dr
        - frac_Er[idx_Er_k]   * Er[idx_Er_k]   * gp.dr
        - frac_Ez[idx_Ez_ip1] * Ez[idx_Ez_ip1] * gp.dz
        + frac_Ez[idx_Ez_i]   * Ez[idx_Ez_i]   * gp.dz;

    Hphi[gid] = line_integral / A_vac;
}

/*-----------------------------------------------------------------------------
 * Conformal (curl E)_z at (i+½, j+½, k)
 *
 * Face in (r,φ) plane at z = z_k.
 * Standard: (1/r) d(r·Ephi)/dr - (1/r) dEr/dφ
 *
 * Integral form:
 *   + f_Ephi(i+1,j,k) · Ephi(i+1,j,k) · r_{i+1} · dphi   (outer, +φ)
 *   - f_Ephi(i,j,k)   · Ephi(i,j,k)   · r_i     · dphi   (inner)
 *   - f_Er(i,j+1,k)   · Er(i,j+1,k)   · dr                (+r direction)
 *   + f_Er(i,j,k)     · Er(i,j,k)     · dr
 *   / A_vac_Hz(i,j,k)
 *
 * Note: the Ephi edge lengths are r·dφ which varies with r.
 *---------------------------------------------------------------------------*/
__global__ void conformal_curl_E_z_kernel(
    const double* Er, const double* Ephi,
    double* Hz,
    const double* frac_Er, const double* frac_Ephi,
    const double* face_area_Hz,
    CurlGridParams gp
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gp.Nr * gp.Nphi * (gp.Nz + 1);
    if (gid >= total) return;

    int tmp = gid;
    int i = tmp % gp.Nr;     tmp /= gp.Nr;
    int j = tmp % gp.Nphi;
    int k = tmp / gp.Nphi;

    double A_vac = face_area_Hz[gid];
    if (A_vac < 1e-30) {
        Hz[gid] = 0.0;
        return;
    }

    double r_i   = gp.a + i * gp.dr;
    double r_ip1 = gp.a + (i + 1) * gp.dr;

    int jp1 = d_periodic_j(j + 1, gp.Nphi);

    int idx_Ephi_ip1 = d_idx_Ephi(gp.Nr, gp.Nphi, i + 1, j, k);
    int idx_Ephi_i   = d_idx_Ephi(gp.Nr, gp.Nphi, i, j, k);
    int idx_Er_jp1   = d_idx_Er(gp.Nr, gp.Nphi, i, jp1, k);
    int idx_Er_j     = d_idx_Er(gp.Nr, gp.Nphi, i, j, k);

    double line_integral =
        + frac_Ephi[idx_Ephi_ip1] * Ephi[idx_Ephi_ip1] * r_ip1 * gp.dphi
        - frac_Ephi[idx_Ephi_i]   * Ephi[idx_Ephi_i]   * r_i   * gp.dphi
        - frac_Er[idx_Er_jp1]     * Er[idx_Er_jp1]     * gp.dr
        + frac_Er[idx_Er_j]       * Er[idx_Er_j]       * gp.dr;

    Hz[gid] = line_integral / A_vac;
}

/*=============================================================================
 * Conformal Curl-H Kernels
 *
 * Ampere's law in integral form on each dual face:
 *   ∮ H·dl = ∂/∂t ∫ D·dS
 *
 * H-edges are at cell centers and are typically not cut by pipe walls.
 * The main Dey-Mittra modification here is the dual-face area normalization.
 *
 * The H-line-integral is computed identically to the standard stencil;
 * only the denominator changes from the standard area to A_dual_vac.
 *============================================================================*/

/*-----------------------------------------------------------------------------
 * Conformal (curl H)_r at (i+½, j, k)
 *
 * Dual face in (φ,z) plane at r = r_{i+½}.
 * Standard: (1/r) dHz/dφ - dHphi/dz
 *
 * H-line-integral (standard stencil, unchanged):
 *   [Hz(i,j,k) - Hz(i,j-1,k)] · dz
 *   - [Hphi(i,j,k) - Hphi(i,j,k-1)] · r_{i+½}·dφ
 *
 * Divided by A_dual_Er (instead of standard r_{i+½}·dφ·dz).
 *---------------------------------------------------------------------------*/
__global__ void conformal_curl_H_r_kernel(
    const double* Hz, const double* Hphi,
    double* Er_out,
    const double* dual_area_Er,
    CurlGridParams gp
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gp.Nr * gp.Nphi * (gp.Nz + 1);
    if (gid >= total) return;

    int tmp = gid;
    int i = tmp % gp.Nr;     tmp /= gp.Nr;
    int j = tmp % gp.Nphi;
    int k = tmp / gp.Nphi;

    double A_dual = dual_area_Er[gid];
    if (A_dual < 1e-30) {
        Er_out[gid] = 0.0;
        return;
    }

    double r_iph = gp.a + (i + 0.5) * gp.dr;
    int jm1 = d_periodic_j(j - 1, gp.Nphi);

    /* Hz edges in φ-direction */
    double Hz_j   = Hz[d_idx_Hz(gp.Nr, gp.Nphi, i, j, k)];
    double Hz_jm1 = Hz[d_idx_Hz(gp.Nr, gp.Nphi, i, jm1, k)];

    /* Hφ edges in z-direction (boundary: zero at k=0 and k=Nz) */
    double Hphi_k   = (k < gp.Nz) ? Hphi[d_idx_Hphi(gp.Nr, gp.Nphi, i, j, k)]     : 0.0;
    double Hphi_km1 = (k > 0)     ? Hphi[d_idx_Hphi(gp.Nr, gp.Nphi, i, j, k - 1)] : 0.0;

    double line_integral =
        + (Hz_j - Hz_jm1) * gp.dz
        - (Hphi_k - Hphi_km1) * r_iph * gp.dphi;

    Er_out[gid] = line_integral / A_dual;
}

/*-----------------------------------------------------------------------------
 * Conformal (curl H)_φ at (i, j+½, k)
 *
 * Dual face in (r,z) plane at φ = (j+½)·dφ.
 * Standard: dHr/dz - dHz/dr
 *---------------------------------------------------------------------------*/
__global__ void conformal_curl_H_phi_kernel(
    const double* Hr, const double* Hz,
    double* Ephi_out,
    const double* dual_area_Ephi,
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

    double A_dual = dual_area_Ephi[gid];
    if (A_dual < 1e-30) {
        Ephi_out[gid] = 0.0;
        return;
    }

    /* Hr edges in z-direction (boundary: zero at k boundaries) */
    double Hr_k   = (k < gp.Nz) ? Hr[d_idx_Hr(gp.Nr, gp.Nphi, i, j, k)]     : 0.0;
    double Hr_km1 = (k > 0)     ? Hr[d_idx_Hr(gp.Nr, gp.Nphi, i, j, k - 1)] : 0.0;

    /* Hz edges in r-direction (boundary: zero at i boundaries) */
    double Hz_i   = (i < gp.Nr) ? Hz[d_idx_Hz(gp.Nr, gp.Nphi, i, j, k)]     : 0.0;
    double Hz_im1 = (i > 0)     ? Hz[d_idx_Hz(gp.Nr, gp.Nphi, i - 1, j, k)] : 0.0;

    double line_integral =
        + (Hr_k - Hr_km1) * gp.dr
        - (Hz_i - Hz_im1) * gp.dz;

    Ephi_out[gid] = line_integral / A_dual;
}

/*-----------------------------------------------------------------------------
 * Conformal (curl H)_z at (i, j, k+½)
 *
 * Dual face in (r,φ) plane at z = (k+½)·dz.
 * Standard: (1/r) d(r·Hphi)/dr - (1/r) dHr/dφ
 *---------------------------------------------------------------------------*/
__global__ void conformal_curl_H_z_kernel(
    const double* Hr, const double* Hphi,
    double* Ez_out,
    const double* dual_area_Ez,
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

    double A_dual = dual_area_Ez[gid];
    if (A_dual < 1e-30) {
        Ez_out[gid] = 0.0;
        return;
    }

    double r_iph = gp.a + (i + 0.5) * gp.dr;
    double r_imh = (i > 0) ? gp.a + (i - 0.5) * gp.dr : 0.0;

    int jm1 = d_periodic_j(j - 1, gp.Nphi);

    /* r·Hφ edges in r-direction */
    double rHphi_iph = (i < gp.Nr) ? r_iph * Hphi[d_idx_Hphi(gp.Nr, gp.Nphi, i, j, k)]     : 0.0;
    double rHphi_imh = (i > 0)     ? r_imh * Hphi[d_idx_Hphi(gp.Nr, gp.Nphi, i - 1, j, k)] : 0.0;

    /* Hr edges in φ-direction */
    double Hr_j   = Hr[d_idx_Hr(gp.Nr, gp.Nphi, i, j, k)];
    double Hr_jm1 = Hr[d_idx_Hr(gp.Nr, gp.Nphi, i, jm1, k)];

    double line_integral =
        + (rHphi_iph - rHphi_imh) * gp.dphi
        - (Hr_j - Hr_jm1) * gp.dr;

    Ez_out[gid] = line_integral / A_dual;
}

/*=============================================================================
 * Launch helpers (raw pointer, one component at a time)
 *============================================================================*/

static int launch_conformal_curl_E_r(
    const double* Ez, const double* Ephi, double* Hr,
    const double* frac_Ez, const double* frac_Ephi,
    const double* face_area_Hr,
    const GridParams* grid
) {
    CurlGridParams gp = make_curl_params(grid);
    int total = (grid->Nr + 1) * grid->Nphi * grid->Nz;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conformal_curl_E_r_kernel<<<blocks, BLOCK_SIZE>>>(
        Ez, Ephi, Hr, frac_Ez, frac_Ephi, face_area_Hr, gp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int launch_conformal_curl_E_phi(
    const double* Er, const double* Ez, double* Hphi,
    const double* frac_Er, const double* frac_Ez,
    const double* face_area_Hphi,
    const GridParams* grid
) {
    CurlGridParams gp = make_curl_params(grid);
    int total = grid->Nr * grid->Nphi * grid->Nz;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conformal_curl_E_phi_kernel<<<blocks, BLOCK_SIZE>>>(
        Er, Ez, Hphi, frac_Er, frac_Ez, face_area_Hphi, gp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int launch_conformal_curl_E_z(
    const double* Er, const double* Ephi, double* Hz,
    const double* frac_Er, const double* frac_Ephi,
    const double* face_area_Hz,
    const GridParams* grid
) {
    CurlGridParams gp = make_curl_params(grid);
    int total = grid->Nr * grid->Nphi * (grid->Nz + 1);
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conformal_curl_E_z_kernel<<<blocks, BLOCK_SIZE>>>(
        Er, Ephi, Hz, frac_Er, frac_Ephi, face_area_Hz, gp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int launch_conformal_curl_H_r(
    const double* Hz, const double* Hphi, double* Er_out,
    const double* dual_area_Er,
    const GridParams* grid
) {
    CurlGridParams gp = make_curl_params(grid);
    int total = grid->Nr * grid->Nphi * (grid->Nz + 1);
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conformal_curl_H_r_kernel<<<blocks, BLOCK_SIZE>>>(
        Hz, Hphi, Er_out, dual_area_Er, gp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int launch_conformal_curl_H_phi(
    const double* Hr, const double* Hz, double* Ephi_out,
    const double* dual_area_Ephi,
    const GridParams* grid
) {
    CurlGridParams gp = make_curl_params(grid);
    int total = (grid->Nr + 1) * grid->Nphi * (grid->Nz + 1);
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conformal_curl_H_phi_kernel<<<blocks, BLOCK_SIZE>>>(
        Hr, Hz, Ephi_out, dual_area_Ephi, gp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static int launch_conformal_curl_H_z(
    const double* Hr, const double* Hphi, double* Ez_out,
    const double* dual_area_Ez,
    const GridParams* grid
) {
    CurlGridParams gp = make_curl_params(grid);
    int total = (grid->Nr + 1) * grid->Nphi * grid->Nz;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conformal_curl_H_z_kernel<<<blocks, BLOCK_SIZE>>>(
        Hr, Hphi, Ez_out, dual_area_Ez, gp);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/*=============================================================================
 * Public API — Conformal Curl-E (real)
 *============================================================================*/

int gpu_compute_curl_E_conformal(
    const GPU_EField* E,
    GPU_HField* curlE,
    const GridParams* grid,
    const GPU_ConformalData* cd
) {
    int status;
    status = launch_conformal_curl_E_r(
        E->Ez, E->Ephi, curlE->Hr,
        cd->d_edge_frac_Ez, cd->d_edge_frac_Ephi,
        cd->d_face_area_Hr, grid);
    if (status != 0) return status;

    status = launch_conformal_curl_E_phi(
        E->Er, E->Ez, curlE->Hphi,
        cd->d_edge_frac_Er, cd->d_edge_frac_Ez,
        cd->d_face_area_Hphi, grid);
    if (status != 0) return status;

    status = launch_conformal_curl_E_z(
        E->Er, E->Ephi, curlE->Hz,
        cd->d_edge_frac_Er, cd->d_edge_frac_Ephi,
        cd->d_face_area_Hz, grid);
    return status;
}

/*=============================================================================
 * Public API — Conformal Curl-E (complex)
 *
 * Curl is linear: curl(E_re + j·E_im) = curl(E_re) + j·curl(E_im)
 * Run the same conformal stencil on both real and imaginary parts.
 *============================================================================*/

int gpu_compute_curl_E_conformal_complex(
    const GPU_EField* E,
    GPU_HField* curlE,
    const GridParams* grid,
    const GPU_ConformalData* cd
) {
    int status;

    /* Real part */
    status = launch_conformal_curl_E_r(
        E->Ez, E->Ephi, curlE->Hr,
        cd->d_edge_frac_Ez, cd->d_edge_frac_Ephi,
        cd->d_face_area_Hr, grid);
    if (status != 0) return status;
    status = launch_conformal_curl_E_phi(
        E->Er, E->Ez, curlE->Hphi,
        cd->d_edge_frac_Er, cd->d_edge_frac_Ez,
        cd->d_face_area_Hphi, grid);
    if (status != 0) return status;
    status = launch_conformal_curl_E_z(
        E->Er, E->Ephi, curlE->Hz,
        cd->d_edge_frac_Er, cd->d_edge_frac_Ephi,
        cd->d_face_area_Hz, grid);
    if (status != 0) return status;

    /* Imaginary part */
    status = launch_conformal_curl_E_r(
        E->Ez_im, E->Ephi_im, curlE->Hr_im,
        cd->d_edge_frac_Ez, cd->d_edge_frac_Ephi,
        cd->d_face_area_Hr, grid);
    if (status != 0) return status;
    status = launch_conformal_curl_E_phi(
        E->Er_im, E->Ez_im, curlE->Hphi_im,
        cd->d_edge_frac_Er, cd->d_edge_frac_Ez,
        cd->d_face_area_Hphi, grid);
    if (status != 0) return status;
    status = launch_conformal_curl_E_z(
        E->Er_im, E->Ephi_im, curlE->Hz_im,
        cd->d_edge_frac_Er, cd->d_edge_frac_Ephi,
        cd->d_face_area_Hz, grid);
    return status;
}

/*=============================================================================
 * Public API — Conformal Curl-H (real)
 *============================================================================*/

int gpu_compute_curl_H_conformal(
    const GPU_HField* H,
    GPU_EField* curlH,
    const GridParams* grid,
    const GPU_ConformalData* cd
) {
    int status;
    status = launch_conformal_curl_H_r(
        H->Hz, H->Hphi, curlH->Er,
        cd->d_dual_area_Er, grid);
    if (status != 0) return status;

    status = launch_conformal_curl_H_phi(
        H->Hr, H->Hz, curlH->Ephi,
        cd->d_dual_area_Ephi, grid);
    if (status != 0) return status;

    status = launch_conformal_curl_H_z(
        H->Hr, H->Hphi, curlH->Ez,
        cd->d_dual_area_Ez, grid);
    return status;
}

/*=============================================================================
 * Public API — Conformal Curl-H (complex)
 *============================================================================*/

int gpu_compute_curl_H_conformal_complex(
    const GPU_HField* H,
    GPU_EField* curlH,
    const GridParams* grid,
    const GPU_ConformalData* cd
) {
    int status;

    /* Real part */
    status = launch_conformal_curl_H_r(
        H->Hz, H->Hphi, curlH->Er,
        cd->d_dual_area_Er, grid);
    if (status != 0) return status;
    status = launch_conformal_curl_H_phi(
        H->Hr, H->Hz, curlH->Ephi,
        cd->d_dual_area_Ephi, grid);
    if (status != 0) return status;
    status = launch_conformal_curl_H_z(
        H->Hr, H->Hphi, curlH->Ez,
        cd->d_dual_area_Ez, grid);
    if (status != 0) return status;

    /* Imaginary part */
    status = launch_conformal_curl_H_r(
        H->Hz_im, H->Hphi_im, curlH->Er_im,
        cd->d_dual_area_Er, grid);
    if (status != 0) return status;
    status = launch_conformal_curl_H_phi(
        H->Hr_im, H->Hz_im, curlH->Ephi_im,
        cd->d_dual_area_Ephi, grid);
    if (status != 0) return status;
    status = launch_conformal_curl_H_z(
        H->Hr_im, H->Hphi_im, curlH->Ez_im,
        cd->d_dual_area_Ez, grid);
    return status;
}

/*=============================================================================
 * Public API — Conformal Curl-Curl (real + complex)
 *============================================================================*/

int gpu_compute_curl_curl_E_conformal(
    const GPU_EField* E,
    GPU_EField* result,
    GPU_HField* temp,
    const GridParams* grid,
    const GPU_ConformalData* cd
) {
    int status;
    status = gpu_compute_curl_E_conformal(E, temp, grid, cd);
    if (status != 0) return status;
    status = gpu_compute_curl_H_conformal(temp, result, grid, cd);
    return status;
}

int gpu_compute_curl_curl_E_conformal_complex(
    const GPU_EField* E,
    GPU_EField* result,
    GPU_HField* temp,
    const GridParams* grid,
    const GPU_ConformalData* cd
) {
    int status;
    status = gpu_compute_curl_E_conformal_complex(E, temp, grid, cd);
    if (status != 0) return status;
    status = gpu_compute_curl_H_conformal_complex(temp, result, grid, cd);
    return status;
}

/*=============================================================================
 * Conformal mask: zero E where edge_frac == 0
 *
 * Replaces the binary material mask. At cut edges (0 < frac < 1),
 * E is NOT zeroed — the conformal curl handles the partial contribution.
 *============================================================================*/

__global__ void conformal_mask_kernel(
    double* field, const double* edge_frac, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (edge_frac[idx] == 0.0) {
            field[idx] = 0.0;
        }
    }
}

int gpu_apply_conformal_mask(GPU_EField* E, const GPU_ConformalData* cd) {
    int b1 = (cd->size_Er   + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int b2 = (cd->size_Ephi + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int b3 = (cd->size_Ez   + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conformal_mask_kernel<<<b1, BLOCK_SIZE>>>(E->Er,   cd->d_edge_frac_Er,   cd->size_Er);
    conformal_mask_kernel<<<b2, BLOCK_SIZE>>>(E->Ephi, cd->d_edge_frac_Ephi, cd->size_Ephi);
    conformal_mask_kernel<<<b3, BLOCK_SIZE>>>(E->Ez,   cd->d_edge_frac_Ez,   cd->size_Ez);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int gpu_apply_conformal_mask_complex(GPU_EField* E, const GPU_ConformalData* cd) {
    int b1 = (cd->size_Er   + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int b2 = (cd->size_Ephi + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int b3 = (cd->size_Ez   + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conformal_mask_kernel<<<b1, BLOCK_SIZE>>>(E->Er,     cd->d_edge_frac_Er,   cd->size_Er);
    conformal_mask_kernel<<<b1, BLOCK_SIZE>>>(E->Er_im,  cd->d_edge_frac_Er,   cd->size_Er);
    conformal_mask_kernel<<<b2, BLOCK_SIZE>>>(E->Ephi,   cd->d_edge_frac_Ephi, cd->size_Ephi);
    conformal_mask_kernel<<<b2, BLOCK_SIZE>>>(E->Ephi_im,cd->d_edge_frac_Ephi, cd->size_Ephi);
    conformal_mask_kernel<<<b3, BLOCK_SIZE>>>(E->Ez,     cd->d_edge_frac_Ez,   cd->size_Ez);
    conformal_mask_kernel<<<b3, BLOCK_SIZE>>>(E->Ez_im,  cd->d_edge_frac_Ez,   cd->size_Ez);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/*=============================================================================
 * Conformal IBC surface correction: result += β · w · E
 *
 * Uses the geometrically exact conformal weights (wall_arc / dual_volume).
 * Same kernel structure as the staircase ibc_weight_kernel, but reads
 * from conformal IBC weight arrays.
 *============================================================================*/

__global__ void conformal_ibc_kernel(
    double* result_re, double* result_im,
    const double* E_re, const double* E_im,
    const double* weight,
    double beta_re, double beta_im,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double w = weight[idx];
        if (w > 0.0) {
            double bw_re = beta_re * w;
            double bw_im = beta_im * w;
            double e_re = E_re[idx];
            double e_im = E_im[idx];
            result_re[idx] += bw_re * e_re - bw_im * e_im;
            result_im[idx] += bw_re * e_im + bw_im * e_re;
        }
    }
}

int gpu_apply_conformal_ibc(
    const GPU_EField* E,
    GPU_EField* result,
    double alpha,
    const GPU_ConformalData* cd
) {
    double inv_2a = 1.0 / (2.0 * alpha);
    double beta_re =  inv_2a;
    double beta_im = -inv_2a;

    int b1 = (cd->size_Er   + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int b2 = (cd->size_Ephi + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int b3 = (cd->size_Ez   + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conformal_ibc_kernel<<<b1, BLOCK_SIZE>>>(
        result->Er, result->Er_im,
        E->Er, E->Er_im,
        cd->d_ibc_weight_Er, beta_re, beta_im, cd->size_Er);

    conformal_ibc_kernel<<<b2, BLOCK_SIZE>>>(
        result->Ephi, result->Ephi_im,
        E->Ephi, E->Ephi_im,
        cd->d_ibc_weight_Ephi, beta_re, beta_im, cd->size_Ephi);

    conformal_ibc_kernel<<<b3, BLOCK_SIZE>>>(
        result->Ez, result->Ez_im,
        E->Ez, E->Ez_im,
        cd->d_ibc_weight_Ez, beta_re, beta_im, cd->size_Ez);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}
