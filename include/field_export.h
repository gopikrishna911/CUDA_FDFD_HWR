#ifndef FIELD_EXPORT_H
#define FIELD_EXPORT_H

#include "curlcurl_operator.h"
#include "curl_E.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*=============================================================================
     * Field Export — CSV data for publication figures
     *
     * Exports 2D slices and 1D profiles of E-field and surface loss
     * density from the eigensolver result. All coordinates are physical
     * (meters, degrees), not grid indices.
     *
     * Output files (prefix = e.g. "ibc"):
     *
     *   2D color maps:
     *     {prefix}_Er_rz.csv          Er(r,z) at phi=0
     *     {prefix}_Er_rphi.csv        Er(r,phi) at z=L/2
     *     {prefix}_Hloss_inner.csv    |H_tan|^2(phi,z) on r=a
     *     {prefix}_Hloss_outer.csv    |H_tan|^2(phi,z) on r=b
     *     {prefix}_Hloss_z0.csv       |H_tan|^2(r,phi) at z=0 endplate
     *     {prefix}_Hloss_zL.csv       |H_tan|^2(r,phi) at z=L endplate
     *
     *   1D line profiles:
     *     {prefix}_Er_vs_r.csv        Er(r) at z=L/2, phi=0
     *     {prefix}_Er_vs_z.csv        Er(z) at r_mid, phi=0
     *     {prefix}_Er_vs_phi.csv      Er(phi) at r_mid, z=L/2
     *     {prefix}_Hphi_vs_z.csv      |H_phi|(z) on inner wall, phi=0
     *
     *   All CSV files have descriptive headers and metadata comments.
     *   2D files include Nr, Nphi, Nz dimensions for easy reshaping.
     *============================================================================*/

    /*=============================================================================
     * Export configuration
     *============================================================================*/
    typedef struct {
        double b_cavity;        /* Physical outer conductor radius [m] */
        double L_cavity;        /* Physical cavity length [m] */
        double z0_offset;       /* Grid z-offset for extended grids [m] */
        int k_cavity_start;     /* k-index where cavity region starts */
        int k_cavity_end;       /* k-index where cavity region ends */
        const char* prefix;     /* Output filename prefix */
    } FieldExportConfig;

    /*=============================================================================
     * Export all data in one call
     *
     * Unpacks the eigenvector, computes curl for H-field, and writes
     * all CSV files. The eigenvector is NOT modified.
     *============================================================================*/
    void export_all_field_data(
        const CurlCurlOperator* op,
        const double* eigenvector,
        const FieldExportConfig* config
    );

    /*=============================================================================
     * Individual exports (if you only need specific files)
     *
     * These all require the E-field and/or curl to be pre-computed.
     *============================================================================*/

    /* --- 2D slices --- */

    /* Er(r,z) at phi=0 (closest grid phi index) */
    void export_Er_rz(
        const CurlCurlOperator* op,
        const EField* E,
        const FieldExportConfig* config,
        const char* filename
    );

    /* Er(r,phi) at z = L/2 (closest grid k index) */
    void export_Er_rphi(
        const CurlCurlOperator* op,
        const EField* E,
        const FieldExportConfig* config,
        const char* filename
    );

    /* |H_tan|^2(phi,z) on inner conductor r = a */
    void export_Hloss_inner(
        const CurlCurlOperator* op,
        const CurlE* curlE,
        const FieldExportConfig* config,
        const char* filename
    );

    /* |H_tan|^2(phi,z) on outer conductor r = b */
    void export_Hloss_outer(
        const CurlCurlOperator* op,
        const CurlE* curlE,
        const FieldExportConfig* config,
        const char* filename
    );

    /* |H_tan|^2(r,phi) on endplate at z = 0 */
    void export_Hloss_endplate_z0(
        const CurlCurlOperator* op,
        const CurlE* curlE,
        const FieldExportConfig* config,
        const char* filename
    );

    /* |H_tan|^2(r,phi) on endplate at z = L */
    void export_Hloss_endplate_zL(
        const CurlCurlOperator* op,
        const CurlE* curlE,
        const FieldExportConfig* config,
        const char* filename
    );

    /* --- 1D line profiles --- */

    /* Er(r) at z = L/2, phi = 0, with analytical comparison */
    void export_Er_vs_r(
        const CurlCurlOperator* op,
        const EField* E,
        const FieldExportConfig* config,
        const char* filename
    );

    /* Er(z) at r = r_mid, phi = 0, with analytical comparison */
    void export_Er_vs_z(
        const CurlCurlOperator* op,
        const EField* E,
        const FieldExportConfig* config,
        const char* filename
    );

    /* Er(phi) at r = r_mid, z = L/2 */
    void export_Er_vs_phi(
        const CurlCurlOperator* op,
        const EField* E,
        const FieldExportConfig* config,
        const char* filename
    );

    /* |H_phi|(z) on inner wall at phi = 0, with analytical comparison */
    void export_Hphi_vs_z(
        const CurlCurlOperator* op,
        const CurlE* curlE,
        const FieldExportConfig* config,
        const char* filename
    );

#ifdef __cplusplus
}
#endif

#endif /* FIELD_EXPORT_H */
