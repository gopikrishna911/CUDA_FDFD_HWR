#ifndef FIELD_MAP_EXPORT_H
#define FIELD_MAP_EXPORT_H

#include <stdint.h>
#include "curlcurl_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * Cavity Field Map Exporter (v2: complex-phase aware, cavity-only L)
 *
 * Writes the spatial amplitudes of E=(Er,Ephi,Ez) and B=(Br,Bphi,Bz) on a
 * cylindrical *cell-centred* grid covering the cavity slab z in [0, L_cavity].
 *
 * The exported amplitudes are real-valued.  When the eigenvector is complex
 * (IBC iterative solve), RQI returns v with arbitrary global phase, so
 * Re(v) by itself does not look like the physical TEM mode.  Pass the
 * imaginary part via cfg->eigenvector_imag and the exporter will phase-rotate
 * so that the dominant Er becomes purely real before extracting Re(v).
 *
 * Array layout (per component): i fastest, k slowest:
 *   idx = i + Nr*(j + Nphi*k)     with i in [0,Nr), j in [0,Nphi), k in [0,Nz_cav)
 *
 * RF time dependence in the tracker (90 deg from Faraday's law):
 *   E_RF(r, t) = E_spatial(r) * cos(omega t + phi0)
 *   B_RF(r, t) = B_spatial(r) * sin(omega t + phi0)
 *   B_spatial  = -curl(E_spatial) / omega
 *============================================================================*/

#define FIELD_MAP_MAGIC   0xCAFE0001u
#define FIELD_MAP_VERSION 2u

typedef struct {
    uint32_t magic;
    uint32_t version;                /* now 2 */
    int32_t  Nr;
    int32_t  Nphi;
    int32_t  Nz_cav;
    int32_t  _pad0;
    double   a;                      /* inner conductor radius [m]              */
    double   b;                      /* outer extent of radial grid [m]         */
    double   L;                      /* CAVITY axial length [m] (not extended)  */
    double   dr;
    double   dphi;
    double   dz;
    double   omega;
    double   freq_Hz;
    double   peak_Er_before_norm;
    double   _reserved[3];
} FieldMapHeader;

typedef struct {
    int     k_cavity_start;
    int     k_cavity_end;
    double  z0_offset;
    double  omega;
    int     normalize_to_peak;
    int     flip_sign_for_positive_Er;

    /* If non-NULL, treat the eigenvector as complex (separate Re and Im
     * blocks).  The exporter phase-rotates so the dominant Er is purely real
     * before extracting Re.  Pass NULL for PEC (real eigenvector). */
    const double* eigenvector_imag;
} FieldMapExportConfig;

void export_cavity_field_map(
    const CurlCurlOperator*       op,
    const double*                 eigenvector,    /* real part */
    const FieldMapExportConfig*   cfg,
    const char*                   filename
);

#ifdef __cplusplus
}
#endif

#endif /* FIELD_MAP_EXPORT_H */
