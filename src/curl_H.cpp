#include "curl_H.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/*=============================================================================
 * Index functions for H-field components
 *============================================================================*/

/* Hr at (i, j+1/2, k+1/2): i=0..Nr, j=0..Nphi-1, k=0..Nz-1 */
int idx_Hr(const GridParams* grid, int i, int j, int k) {
    /* Size: (Nr+1) * Nphi * Nz */
    return i + (grid->Nr + 1) * (j + grid->Nphi * k);
}

/* Hphi at (i+1/2, j, k+1/2): i=0..Nr-1, j=0..Nphi-1, k=0..Nz-1 */
int idx_Hphi(const GridParams* grid, int i, int j, int k) {
    /* Size: Nr * Nphi * Nz */
    return i + grid->Nr * (j + grid->Nphi * k);
}

/* Hz at (i+1/2, j+1/2, k): i=0..Nr-1, j=0..Nphi-1, k=0..Nz */
int idx_Hz(const GridParams* grid, int i, int j, int k) {
    /* Size: Nr * Nphi * (Nz+1) */
    return i + grid->Nr * (j + grid->Nphi * k);
}

/*=============================================================================
 * H-field allocation/deallocation
 *============================================================================*/
void hfield_alloc(HField* H, const GridParams* grid) {
    /* Hr: (Nr+1) * Nphi * Nz */
    H->size_Hr = (grid->Nr + 1) * grid->Nphi * grid->Nz;
    H->Hr = (double*)malloc(H->size_Hr * sizeof(double));
    
    /* Hphi: Nr * Nphi * Nz */
    H->size_Hphi = grid->Nr * grid->Nphi * grid->Nz;
    H->Hphi = (double*)malloc(H->size_Hphi * sizeof(double));
    
    /* Hz: Nr * Nphi * (Nz+1) */
    H->size_Hz = grid->Nr * grid->Nphi * (grid->Nz + 1);
    H->Hz = (double*)malloc(H->size_Hz * sizeof(double));
    
    hfield_zero(H);
}

void hfield_free(HField* H) {
    if (H->Hr) { free(H->Hr); H->Hr = NULL; }
    if (H->Hphi) { free(H->Hphi); H->Hphi = NULL; }
    if (H->Hz) { free(H->Hz); H->Hz = NULL; }
}

void hfield_zero(HField* H) {
    memset(H->Hr, 0, H->size_Hr * sizeof(double));
    memset(H->Hphi, 0, H->size_Hphi * sizeof(double));
    memset(H->Hz, 0, H->size_Hz * sizeof(double));
}

/*=============================================================================
 * Periodic index helper for phi direction
 *============================================================================*/
static inline int periodic_j(int j, int Nphi) {
    return ((j % Nphi) + Nphi) % Nphi;
}

/*=============================================================================
 * Curl of H: r-component
 * 
 * (curl H)_r = (1/r) * dHz/dphi - dHphi/dz
 * 
 * Output location: (i+1/2, j, k) for i=0..Nr-1, j=0..Nphi-1, k=0..Nz
 * 
 * Uses:
 *   Hz at (i+1/2, j+1/2, k) and (i+1/2, j-1/2, k)
 *   Hphi at (i+1/2, j, k+1/2) and (i+1/2, j, k-1/2)
 *============================================================================*/
double curl_H_r_at(const HField* H, const GridParams* grid, int i, int j, int k) {
    double r_iph = r_at_i_half(grid, i);  /* r at i+1/2 */
    double inv_dphi = 1.0 / grid->dphi;
    double inv_dz = 1.0 / grid->dz;
    
    int jm1 = periodic_j(j - 1, grid->Nphi);
    
    /* Hz(i+1/2, j+1/2, k) - Hz(i+1/2, j-1/2, k) */
    /* Hz at (i+1/2, j+1/2, k) uses index j */
    /* Hz at (i+1/2, j-1/2, k) uses index j-1 */
    double Hz_jph = H->Hz[idx_Hz(grid, i, j, k)];
    double Hz_jmh = H->Hz[idx_Hz(grid, i, jm1, k)];
    double dHz_dphi = (Hz_jph - Hz_jmh) * inv_dphi;
    
    /* Hphi(i+1/2, j, k+1/2) - Hphi(i+1/2, j, k-1/2) */
    /* Need to handle boundaries at k=0 and k=Nz */
    double Hphi_kph = 0.0;
    double Hphi_kmh = 0.0;
    
    if (k < grid->Nz) {
        Hphi_kph = H->Hphi[idx_Hphi(grid, i, j, k)];
    }
    if (k > 0) {
        Hphi_kmh = H->Hphi[idx_Hphi(grid, i, j, k - 1)];
    }
    double dHphi_dz = (Hphi_kph - Hphi_kmh) * inv_dz;
    
    return dHz_dphi / r_iph - dHphi_dz;
}

void compute_curl_H_r(const HField* H, EField* curlH, const GridParams* grid) {
    /* Output: (curl H)_r at (i+1/2, j, k) for i=0..Nr-1, j=0..Nphi-1, k=0..Nz */
    for (int k = 0; k <= grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                int idx = idx_Er(grid, i, j, k);
                curlH->Er[idx] = curl_H_r_at(H, grid, i, j, k);
            }
        }
    }
}

/*=============================================================================
 * Curl of H: phi-component
 * 
 * (curl H)_phi = dHr/dz - dHz/dr
 * 
 * Output location: (i, j+1/2, k) for i=0..Nr, j=0..Nphi-1, k=0..Nz
 * 
 * Uses:
 *   Hr at (i, j+1/2, k+1/2) and (i, j+1/2, k-1/2)
 *   Hz at (i+1/2, j+1/2, k) and (i-1/2, j+1/2, k)
 *============================================================================*/
double curl_H_phi_at(const HField* H, const GridParams* grid, int i, int j, int k) {
    double inv_dz = 1.0 / grid->dz;
    double inv_dr = 1.0 / grid->dr;
    
    /* Hr(i, j+1/2, k+1/2) - Hr(i, j+1/2, k-1/2) */
    /* Need to handle boundaries at k=0 and k=Nz */
    double Hr_kph = 0.0;
    double Hr_kmh = 0.0;
    
    if (k < grid->Nz) {
        Hr_kph = H->Hr[idx_Hr(grid, i, j, k)];
    }
    if (k > 0) {
        Hr_kmh = H->Hr[idx_Hr(grid, i, j, k - 1)];
    }
    double dHr_dz = (Hr_kph - Hr_kmh) * inv_dz;
    
    /* Hz(i+1/2, j+1/2, k) - Hz(i-1/2, j+1/2, k) */
    /* Need to handle boundaries at i=0 and i=Nr */
    double Hz_iph = 0.0;
    double Hz_imh = 0.0;
    
    if (i < grid->Nr) {
        Hz_iph = H->Hz[idx_Hz(grid, i, j, k)];
    }
    if (i > 0) {
        Hz_imh = H->Hz[idx_Hz(grid, i - 1, j, k)];
    }
    double dHz_dr = (Hz_iph - Hz_imh) * inv_dr;
    
    return dHr_dz - dHz_dr;
}

void compute_curl_H_phi(const HField* H, EField* curlH, const GridParams* grid) {
    /* Output: (curl H)_phi at (i, j+1/2, k) for i=0..Nr, j=0..Nphi-1, k=0..Nz */
    for (int k = 0; k <= grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++) {
                int idx = idx_Ephi(grid, i, j, k);
                curlH->Ephi[idx] = curl_H_phi_at(H, grid, i, j, k);
            }
        }
    }
}

/*=============================================================================
 * Curl of H: z-component
 * 
 * (curl H)_z = (1/r) * d(r*Hphi)/dr - (1/r) * dHr/dphi
 * 
 * Output location: (i, j, k+1/2) for i=0..Nr, j=0..Nphi-1, k=0..Nz-1
 * 
 * Uses:
 *   Hphi at (i+1/2, j, k+1/2) and (i-1/2, j, k+1/2)
 *   Hr at (i, j+1/2, k+1/2) and (i, j-1/2, k+1/2)
 *============================================================================*/
double curl_H_z_at(const HField* H, const GridParams* grid, int i, int j, int k) {
    double r_i = r_at_i(grid, i);
    double r_iph = r_at_i_half(grid, i);      /* r at i+1/2 */
    double r_imh = (i > 0) ? r_at_i_half(grid, i - 1) : 0.0;  /* r at i-1/2 */
    
    double inv_dr = 1.0 / grid->dr;
    double inv_dphi = 1.0 / grid->dphi;
    
    int jm1 = periodic_j(j - 1, grid->Nphi);
    
    /* r_{i+1/2} * Hphi(i+1/2, j, k+1/2) - r_{i-1/2} * Hphi(i-1/2, j, k+1/2) */
    /* Need to handle boundaries at i=0 and i=Nr */
    double rHphi_iph = 0.0;
    double rHphi_imh = 0.0;
    
    if (i < grid->Nr) {
        rHphi_iph = r_iph * H->Hphi[idx_Hphi(grid, i, j, k)];
    }
    if (i > 0) {
        rHphi_imh = r_imh * H->Hphi[idx_Hphi(grid, i - 1, j, k)];
    }
    double d_rHphi_dr = (rHphi_iph - rHphi_imh) * inv_dr;
    
    /* Hr(i, j+1/2, k+1/2) - Hr(i, j-1/2, k+1/2) */
    double Hr_jph = H->Hr[idx_Hr(grid, i, j, k)];
    double Hr_jmh = H->Hr[idx_Hr(grid, i, jm1, k)];
    double dHr_dphi = (Hr_jph - Hr_jmh) * inv_dphi;
    
    /* Handle r = 0 case */
    if (r_i < 1e-14) {
        /* At r = 0, use L'Hopital or assume field is regular */
        return d_rHphi_dr * 2.0;  /* Approximation for (1/r)*d(r*f)/dr at r=0 */
    }
    
    return d_rHphi_dr / r_i - dHr_dphi / r_i;
}

void compute_curl_H_z(const HField* H, EField* curlH, const GridParams* grid) {
    /* Output: (curl H)_z at (i, j, k+1/2) for i=0..Nr, j=0..Nphi-1, k=0..Nz-1 */
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++) {
                int idx = idx_Ez(grid, i, j, k);
                curlH->Ez[idx] = curl_H_z_at(H, grid, i, j, k);
            }
        }
    }
}

/*=============================================================================
 * Compute all curl H components
 *============================================================================*/
void compute_curl_H(const HField* H, EField* curlH, const GridParams* grid) {
    compute_curl_H_r(H, curlH, grid);
    compute_curl_H_phi(H, curlH, grid);
    compute_curl_H_z(H, curlH, grid);
}

/*=============================================================================
 * Compute curl(curl(E))
 * 
 * Step 1: G = curl E  (E-grid → H-grid)  using curl_E
 * Step 2: result = curl G  (H-grid → E-grid)  using curl_H
 *============================================================================*/
void compute_curl_curl_E(const EField* E, EField* curlcurlE, const GridParams* grid) {
    /* Allocate temporary storage for intermediate result G = curl E */
    HField G;
    hfield_alloc(&G, grid);
    
    compute_curl_curl_E_with_temp(E, curlcurlE, &G, grid);
    
    hfield_free(&G);
}

void compute_curl_curl_E_with_temp(const EField* E, EField* curlcurlE, 
                                    HField* G, const GridParams* grid) {
    /* Step 1: G = curl E */
    /* curl_E outputs to CurlE struct, but it has same layout as HField */
    /* We need to compute curl_E and store in G (HField) */
    
    /* Compute (curl E)_r and store in G->Hr */
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++) {
                int idx = idx_Hr(grid, i, j, k);
                G->Hr[idx] = curl_E_r_at(E, grid, i, j, k);
            }
        }
    }
    
    /* Compute (curl E)_phi and store in G->Hphi */
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                int idx = idx_Hphi(grid, i, j, k);
                G->Hphi[idx] = curl_E_phi_at(E, grid, i, j, k);
            }
        }
    }
    
    /* Compute (curl E)_z and store in G->Hz */
    for (int k = 0; k <= grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                int idx = idx_Hz(grid, i, j, k);
                G->Hz[idx] = curl_E_z_at(E, grid, i, j, k);
            }
        }
    }
    
    /* Step 2: curlcurlE = curl G */
    compute_curl_H(G, curlcurlE, grid);
}