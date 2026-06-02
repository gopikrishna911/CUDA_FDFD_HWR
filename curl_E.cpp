#include "curl_E.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/*=============================================================================
 * Grid functions
 *============================================================================*/
void grid_init(GridParams* grid, double a, double b, double L, 
               int Nr, int Nphi, int Nz) {
    grid->a = a;
    grid->b = b;
    grid->L = L;
    grid->Nr = Nr;
    grid->Nphi = Nphi;
    grid->Nz = Nz;
    
    grid->dr = (b - a) / Nr;
    grid->dphi = 2.0 * M_PI / Nphi;
    grid->dz = L / Nz;
}

double r_at_i(const GridParams* grid, int i) {
    return grid->a + i * grid->dr;
}

double r_at_i_half(const GridParams* grid, int i) {
    return grid->a + (i + 0.5) * grid->dr;
}

void grid_print(const GridParams* grid) {
    printf("Grid Parameters:\n");
    printf("  a = %g m, b = %g m, L = %g m\n", grid->a, grid->b, grid->L);
    printf("  Nr = %d, Nphi = %d, Nz = %d\n", grid->Nr, grid->Nphi, grid->Nz);
    printf("  dr = %g m, dphi = %g rad, dz = %g m\n", 
           grid->dr, grid->dphi, grid->dz);
}

/*=============================================================================
 * Index functions for E-field
 *============================================================================*/

/* Er at (i+1/2, j, k): i=0..Nr-1, j=0..Nphi-1, k=0..Nz */
int idx_Er(const GridParams* grid, int i, int j, int k) {
    /* Size: Nr * Nphi * (Nz+1) */
    return i + grid->Nr * (j + grid->Nphi * k);
}

/* Ephi at (i, j+1/2, k): i=0..Nr, j=0..Nphi-1, k=0..Nz */
int idx_Ephi(const GridParams* grid, int i, int j, int k) {
    /* Size: (Nr+1) * Nphi * (Nz+1) */
    return i + (grid->Nr + 1) * (j + grid->Nphi * k);
}

/* Ez at (i, j, k+1/2): i=0..Nr, j=0..Nphi-1, k=0..Nz-1 */
int idx_Ez(const GridParams* grid, int i, int j, int k) {
    /* Size: (Nr+1) * Nphi * Nz */
    return i + (grid->Nr + 1) * (j + grid->Nphi * k);
}

/*=============================================================================
 * Index functions for curl components
 *============================================================================*/

/* curl_r at (i, j+1/2, k+1/2): i=0..Nr, j=0..Nphi-1, k=0..Nz-1 */
int idx_curl_r(const GridParams* grid, int i, int j, int k) {
    /* Size: (Nr+1) * Nphi * Nz */
    return i + (grid->Nr + 1) * (j + grid->Nphi * k);
}

/* curl_phi at (i+1/2, j, k+1/2): i=0..Nr-1, j=0..Nphi-1, k=0..Nz-1 */
int idx_curl_phi(const GridParams* grid, int i, int j, int k) {
    /* Size: Nr * Nphi * Nz */
    return i + grid->Nr * (j + grid->Nphi * k);
}

/* curl_z at (i+1/2, j+1/2, k): i=0..Nr-1, j=0..Nphi-1, k=0..Nz */
int idx_curl_z(const GridParams* grid, int i, int j, int k) {
    /* Size: Nr * Nphi * (Nz+1) */
    return i + grid->Nr * (j + grid->Nphi * k);
}

/*=============================================================================
 * E-field allocation/deallocation
 *============================================================================*/
void efield_alloc(EField* E, const GridParams* grid) {
    /* Er: Nr * Nphi * (Nz+1) */
    E->size_Er = grid->Nr * grid->Nphi * (grid->Nz + 1);
    E->Er = (double*)malloc(E->size_Er * sizeof(double));
    
    /* Ephi: (Nr+1) * Nphi * (Nz+1) */
    E->size_Ephi = (grid->Nr + 1) * grid->Nphi * (grid->Nz + 1);
    E->Ephi = (double*)malloc(E->size_Ephi * sizeof(double));
    
    /* Ez: (Nr+1) * Nphi * Nz */
    E->size_Ez = (grid->Nr + 1) * grid->Nphi * grid->Nz;
    E->Ez = (double*)malloc(E->size_Ez * sizeof(double));
    
    efield_zero(E);
}

void efield_free(EField* E) {
    if (E->Er) { free(E->Er); E->Er = NULL; }
    if (E->Ephi) { free(E->Ephi); E->Ephi = NULL; }
    if (E->Ez) { free(E->Ez); E->Ez = NULL; }
}

void efield_zero(EField* E) {
    memset(E->Er, 0, E->size_Er * sizeof(double));
    memset(E->Ephi, 0, E->size_Ephi * sizeof(double));
    memset(E->Ez, 0, E->size_Ez * sizeof(double));
}

/*=============================================================================
 * Curl allocation/deallocation
 *============================================================================*/
void curl_alloc(CurlE* curl, const GridParams* grid) {
    /* curl_r: (Nr+1) * Nphi * Nz */
    curl->size_curl_r = (grid->Nr + 1) * grid->Nphi * grid->Nz;
    curl->curl_r = (double*)malloc(curl->size_curl_r * sizeof(double));
    
    /* curl_phi: Nr * Nphi * Nz */
    curl->size_curl_phi = grid->Nr * grid->Nphi * grid->Nz;
    curl->curl_phi = (double*)malloc(curl->size_curl_phi * sizeof(double));
    
    /* curl_z: Nr * Nphi * (Nz+1) */
    curl->size_curl_z = grid->Nr * grid->Nphi * (grid->Nz + 1);
    curl->curl_z = (double*)malloc(curl->size_curl_z * sizeof(double));
    
    curl_zero(curl);
}

void curl_free(CurlE* curl) {
    if (curl->curl_r) { free(curl->curl_r); curl->curl_r = NULL; }
    if (curl->curl_phi) { free(curl->curl_phi); curl->curl_phi = NULL; }
    if (curl->curl_z) { free(curl->curl_z); curl->curl_z = NULL; }
}

void curl_zero(CurlE* curl) {
    memset(curl->curl_r, 0, curl->size_curl_r * sizeof(double));
    memset(curl->curl_phi, 0, curl->size_curl_phi * sizeof(double));
    memset(curl->curl_z, 0, curl->size_curl_z * sizeof(double));
}

/*=============================================================================
 * Periodic index helper for phi direction
 *============================================================================*/
static inline int periodic_j(int j, int Nphi) {
    return ((j % Nphi) + Nphi) % Nphi;
}

/*=============================================================================
 * Curl computation: r-component
 * 
 * (curl E)_r = (1/r) * dEz/dphi - dEphi/dz
 * 
 * Location: (i, j+1/2, k+1/2) for i=0..Nr, j=0..Nphi-1, k=0..Nz-1
 * 
 * Uses:
 *   Ez at (i, j, k+1/2) and (i, j+1, k+1/2)
 *   Ephi at (i, j+1/2, k) and (i, j+1/2, k+1)
 *============================================================================*/
double curl_E_r_at(const EField* E, const GridParams* grid, int i, int j, int k) {
    double r_i = r_at_i(grid, i);
    double inv_dphi = 1.0 / grid->dphi;
    double inv_dz = 1.0 / grid->dz;
    
    int jp1 = periodic_j(j + 1, grid->Nphi);
    
    /* Ez(i, j+1, k+1/2) - Ez(i, j, k+1/2) */
    double Ez_jp1 = E->Ez[idx_Ez(grid, i, jp1, k)];
    double Ez_j   = E->Ez[idx_Ez(grid, i, j, k)];
    double dEz_dphi = (Ez_jp1 - Ez_j) * inv_dphi;
    
    /* Ephi(i, j+1/2, k+1) - Ephi(i, j+1/2, k) */
    double Ephi_kp1 = E->Ephi[idx_Ephi(grid, i, j, k + 1)];
    double Ephi_k   = E->Ephi[idx_Ephi(grid, i, j, k)];
    double dEphi_dz = (Ephi_kp1 - Ephi_k) * inv_dz;
    
    /* Handle r = 0 case (shouldn't happen for coaxial, but be safe) */
    if (r_i < 1e-14) {
        return -dEphi_dz;  /* 1/r term vanishes at r=0 for regular fields */
    }
    
    return dEz_dphi / r_i - dEphi_dz;
}

void compute_curl_E_r(const EField* E, CurlE* curl, const GridParams* grid) {
    /* curl_r at (i, j+1/2, k+1/2) for i=0..Nr, j=0..Nphi-1, k=0..Nz-1 */
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++) {
                int idx = idx_curl_r(grid, i, j, k);
                curl->curl_r[idx] = curl_E_r_at(E, grid, i, j, k);
            }
        }
    }
}

/*=============================================================================
 * Curl computation: phi-component
 * 
 * (curl E)_phi = dEr/dz - dEz/dr
 * 
 * Location: (i+1/2, j, k+1/2) for i=0..Nr-1, j=0..Nphi-1, k=0..Nz-1
 * 
 * Uses:
 *   Er at (i+1/2, j, k) and (i+1/2, j, k+1)
 *   Ez at (i, j, k+1/2) and (i+1, j, k+1/2)
 *============================================================================*/
double curl_E_phi_at(const EField* E, const GridParams* grid, int i, int j, int k) {
    double inv_dz = 1.0 / grid->dz;
    double inv_dr = 1.0 / grid->dr;
    
    /* Er(i+1/2, j, k+1) - Er(i+1/2, j, k) */
    double Er_kp1 = E->Er[idx_Er(grid, i, j, k + 1)];
    double Er_k   = E->Er[idx_Er(grid, i, j, k)];
    double dEr_dz = (Er_kp1 - Er_k) * inv_dz;
    
    /* Ez(i+1, j, k+1/2) - Ez(i, j, k+1/2) */
    double Ez_ip1 = E->Ez[idx_Ez(grid, i + 1, j, k)];
    double Ez_i   = E->Ez[idx_Ez(grid, i, j, k)];
    double dEz_dr = (Ez_ip1 - Ez_i) * inv_dr;
    
    return dEr_dz - dEz_dr;
}

void compute_curl_E_phi(const EField* E, CurlE* curl, const GridParams* grid) {
    /* curl_phi at (i+1/2, j, k+1/2) for i=0..Nr-1, j=0..Nphi-1, k=0..Nz-1 */
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                int idx = idx_curl_phi(grid, i, j, k);
                curl->curl_phi[idx] = curl_E_phi_at(E, grid, i, j, k);
            }
        }
    }
}

/*=============================================================================
 * Curl computation: z-component
 * 
 * (curl E)_z = (1/r) * d(r*Ephi)/dr - (1/r) * dEr/dphi
 *            = (1/r) * [r_{i+1}*Ephi_{i+1} - r_i*Ephi_i]/dr - (1/r) * dEr/dphi
 * 
 * Location: (i+1/2, j+1/2, k) for i=0..Nr-1, j=0..Nphi-1, k=0..Nz
 * 
 * Uses:
 *   Ephi at (i, j+1/2, k) and (i+1, j+1/2, k)
 *   Er at (i+1/2, j, k) and (i+1/2, j+1, k)
 *============================================================================*/
double curl_E_z_at(const EField* E, const GridParams* grid, int i, int j, int k) {
    double r_i   = r_at_i(grid, i);
    double r_ip1 = r_at_i(grid, i + 1);
    double r_iph = r_at_i_half(grid, i);  /* r at i+1/2 */
    
    double inv_dr = 1.0 / grid->dr;
    double inv_dphi = 1.0 / grid->dphi;
    
    int jp1 = periodic_j(j + 1, grid->Nphi);
    
    /* r_{i+1} * Ephi(i+1, j+1/2, k) - r_i * Ephi(i, j+1/2, k) */
    double Ephi_ip1 = E->Ephi[idx_Ephi(grid, i + 1, j, k)];
    double Ephi_i   = E->Ephi[idx_Ephi(grid, i, j, k)];
    double d_rEphi_dr = (r_ip1 * Ephi_ip1 - r_i * Ephi_i) * inv_dr;
    
    /* Er(i+1/2, j+1, k) - Er(i+1/2, j, k) */
    double Er_jp1 = E->Er[idx_Er(grid, i, jp1, k)];
    double Er_j   = E->Er[idx_Er(grid, i, j, k)];
    double dEr_dphi = (Er_jp1 - Er_j) * inv_dphi;
    
    return d_rEphi_dr / r_iph - dEr_dphi / r_iph;
}

void compute_curl_E_z(const EField* E, CurlE* curl, const GridParams* grid) {
    /* curl_z at (i+1/2, j+1/2, k) for i=0..Nr-1, j=0..Nphi-1, k=0..Nz */
    for (int k = 0; k <= grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                int idx = idx_curl_z(grid, i, j, k);
                curl->curl_z[idx] = curl_E_z_at(E, grid, i, j, k);
            }
        }
    }
}

/*=============================================================================
 * Compute all curl components
 *============================================================================*/
void compute_curl_E(const EField* E, CurlE* curl, const GridParams* grid) {
    compute_curl_E_r(E, curl, grid);
    compute_curl_E_phi(E, curl, grid);
    compute_curl_E_z(E, curl, grid);
}

/*=============================================================================
 * Utility: Print curl values at a z-plane
 *============================================================================*/
void curl_print_zplane(const CurlE* curl, const GridParams* grid, int k) {
    printf("\nCurl values at k = %d:\n", k);
    
    printf("\ncurl_r (at k+1/2 = %d.5):\n", k);
    if (k < grid->Nz) {
        for (int j = 0; j < grid->Nphi; j++) {
            printf("  j=%d: ", j);
            for (int i = 0; i <= grid->Nr; i++) {
                printf("%10.4e ", curl->curl_r[idx_curl_r(grid, i, j, k)]);
            }
            printf("\n");
        }
    }
    
    printf("\ncurl_phi (at k+1/2 = %d.5):\n", k);
    if (k < grid->Nz) {
        for (int j = 0; j < grid->Nphi; j++) {
            printf("  j=%d: ", j);
            for (int i = 0; i < grid->Nr; i++) {
                printf("%10.4e ", curl->curl_phi[idx_curl_phi(grid, i, j, k)]);
            }
            printf("\n");
        }
    }
    
    printf("\ncurl_z (at k = %d):\n", k);
    for (int j = 0; j < grid->Nphi; j++) {
        printf("  j=%d: ", j);
        for (int i = 0; i < grid->Nr; i++) {
            printf("%10.4e ", curl->curl_z[idx_curl_z(grid, i, j, k)]);
        }
        printf("\n");
    }
}