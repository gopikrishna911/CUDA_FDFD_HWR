#ifndef CURL_E_H
#define CURL_E_H
#define _USE_MATH_DEFINES
#include <stdlib.h>

/*=============================================================================
 * Grid parameters structure
 *============================================================================*/
typedef struct {
    double a;       // Inner radius [m]
    double b;       // Outer radius [m]
    double L;       // Cavity length [m]
    
    int Nr;         // Number of radial cells
    int Nphi;       // Number of azimuthal cells
    int Nz;         // Number of axial cells
    
    double dr;      // Radial step
    double dphi;    // Azimuthal step
    double dz;      // Axial step
} GridParams;

/*=============================================================================
 * E-field arrays structure
 * 
 * Er   at (i+1/2, j, k)     for i=0..Nr-1, j=0..Nphi-1, k=0..Nz
 * Ephi at (i, j+1/2, k)     for i=0..Nr,   j=0..Nphi-1, k=0..Nz
 * Ez   at (i, j, k+1/2)     for i=0..Nr,   j=0..Nphi-1, k=0..Nz-1
 *============================================================================*/
typedef struct {
    double* Er;
    double* Ephi;
    double* Ez;
    
    int size_Er;
    int size_Ephi;
    int size_Ez;
} EField;

/*=============================================================================
 * Curl of E arrays structure (lives at H-field locations)
 * 
 * curl_r   at (i, j+1/2, k+1/2)   for i=0..Nr,   j=0..Nphi-1, k=0..Nz-1
 * curl_phi at (i+1/2, j, k+1/2)   for i=0..Nr-1, j=0..Nphi-1, k=0..Nz-1
 * curl_z   at (i+1/2, j+1/2, k)   for i=0..Nr-1, j=0..Nphi-1, k=0..Nz
 *============================================================================*/
typedef struct {
    double* curl_r;
    double* curl_phi;
    double* curl_z;
    
    int size_curl_r;
    int size_curl_phi;
    int size_curl_z;
} CurlE;

/*=============================================================================
 * Function declarations
 *============================================================================*/

/* Initialize grid parameters */
void grid_init(GridParams* grid, double a, double b, double L, 
               int Nr, int Nphi, int Nz);

/* Get r coordinate at integer index */
double r_at_i(const GridParams* grid, int i);

/* Get r coordinate at half index */
double r_at_i_half(const GridParams* grid, int i);

/* Allocate E-field arrays */
void efield_alloc(EField* E, const GridParams* grid);

/* Free E-field arrays */
void efield_free(EField* E);

/* Zero E-field arrays */
void efield_zero(EField* E);

/* Allocate curl arrays */
void curl_alloc(CurlE* curl, const GridParams* grid);

/* Free curl arrays */
void curl_free(CurlE* curl);

/* Zero curl arrays */
void curl_zero(CurlE* curl);

/*=============================================================================
 * Index functions for E-field components
 *============================================================================*/

/* Er at (i+1/2, j, k): i=0..Nr-1, j=0..Nphi-1, k=0..Nz */
int idx_Er(const GridParams* grid, int i, int j, int k);

/* Ephi at (i, j+1/2, k): i=0..Nr, j=0..Nphi-1, k=0..Nz */
int idx_Ephi(const GridParams* grid, int i, int j, int k);

/* Ez at (i, j, k+1/2): i=0..Nr, j=0..Nphi-1, k=0..Nz-1 */
int idx_Ez(const GridParams* grid, int i, int j, int k);

/*=============================================================================
 * Index functions for curl components
 *============================================================================*/

/* curl_r at (i, j+1/2, k+1/2): i=0..Nr, j=0..Nphi-1, k=0..Nz-1 */
int idx_curl_r(const GridParams* grid, int i, int j, int k);

/* curl_phi at (i+1/2, j, k+1/2): i=0..Nr-1, j=0..Nphi-1, k=0..Nz-1 */
int idx_curl_phi(const GridParams* grid, int i, int j, int k);

/* curl_z at (i+1/2, j+1/2, k): i=0..Nr-1, j=0..Nphi-1, k=0..Nz */
int idx_curl_z(const GridParams* grid, int i, int j, int k);

/*=============================================================================
 * Main curl computation functions
 *============================================================================*/

/* Compute all components of curl E */
void compute_curl_E(const EField* E, CurlE* curl, const GridParams* grid);

/* Compute individual components */
void compute_curl_E_r(const EField* E, CurlE* curl, const GridParams* grid);
void compute_curl_E_phi(const EField* E, CurlE* curl, const GridParams* grid);
void compute_curl_E_z(const EField* E, CurlE* curl, const GridParams* grid);

/* Get single curl value at a point (for testing) */
double curl_E_r_at(const EField* E, const GridParams* grid, int i, int j, int k);
double curl_E_phi_at(const EField* E, const GridParams* grid, int i, int j, int k);
double curl_E_z_at(const EField* E, const GridParams* grid, int i, int j, int k);

/*=============================================================================
 * Utility functions
 *============================================================================*/

/* Print grid info */
void grid_print(const GridParams* grid);

/* Print curl values at a specific z-plane */
void curl_print_zplane(const CurlE* curl, const GridParams* grid, int k);

#endif /* CURL_E_H */