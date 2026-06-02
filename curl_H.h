#ifndef CURL_H_H
#define CURL_H_H

#include "curl_E.h"

/*=============================================================================
 * H-field (or G = curl E) arrays structure
 * Lives at H-field locations (dual grid to E-field)
 * 
 * H_r   at (i, j+1/2, k+1/2)     for i=0..Nr,   j=0..Nphi-1, k=0..Nz-1
 * H_phi at (i+1/2, j, k+1/2)     for i=0..Nr-1, j=0..Nphi-1, k=0..Nz-1
 * H_z   at (i+1/2, j+1/2, k)     for i=0..Nr-1, j=0..Nphi-1, k=0..Nz
 *============================================================================*/
typedef struct {
    double* Hr;
    double* Hphi;
    double* Hz;
    
    int size_Hr;
    int size_Hphi;
    int size_Hz;
} HField;

/*=============================================================================
 * Allocation / Deallocation
 *============================================================================*/
void hfield_alloc(HField* H, const GridParams* grid);
void hfield_free(HField* H);
void hfield_zero(HField* H);

/*=============================================================================
 * Index functions for H-field components
 *============================================================================*/

/* Hr at (i, j+1/2, k+1/2): i=0..Nr, j=0..Nphi-1, k=0..Nz-1 */
int idx_Hr(const GridParams* grid, int i, int j, int k);

/* Hphi at (i+1/2, j, k+1/2): i=0..Nr-1, j=0..Nphi-1, k=0..Nz-1 */
int idx_Hphi(const GridParams* grid, int i, int j, int k);

/* Hz at (i+1/2, j+1/2, k): i=0..Nr-1, j=0..Nphi-1, k=0..Nz */
int idx_Hz(const GridParams* grid, int i, int j, int k);

/*=============================================================================
 * Curl of H (or any vector at H-locations)
 * 
 * Input:  H-field at H-grid locations
 * Output: curl H at E-grid locations
 * 
 * This is used for:
 *   1. Maxwell's equation: dE/dt = (1/ε) * curl H
 *   2. Computing curl(curl E) = curl(G) where G = curl E
 *============================================================================*/

/* Compute all components of curl H */
void compute_curl_H(const HField* H, EField* curlH, const GridParams* grid);

/* Compute individual components */
void compute_curl_H_r(const HField* H, EField* curlH, const GridParams* grid);
void compute_curl_H_phi(const HField* H, EField* curlH, const GridParams* grid);
void compute_curl_H_z(const HField* H, EField* curlH, const GridParams* grid);

/* Get single curl value at a point (for testing) */
double curl_H_r_at(const HField* H, const GridParams* grid, int i, int j, int k);
double curl_H_phi_at(const HField* H, const GridParams* grid, int i, int j, int k);
double curl_H_z_at(const HField* H, const GridParams* grid, int i, int j, int k);

/*=============================================================================
 * Curl of Curl of E
 * 
 * Combines curl_E and curl_H to compute ∇ × ∇ × E
 *============================================================================*/

/* Compute curl(curl(E)) - allocates internal temporary storage */
void compute_curl_curl_E(const EField* E, EField* curlcurlE, const GridParams* grid);

/* Compute curl(curl(E)) - user provides temporary H-field storage */
void compute_curl_curl_E_with_temp(const EField* E, EField* curlcurlE, 
                                    HField* temp, const GridParams* grid);

#endif /* CURL_H_H */