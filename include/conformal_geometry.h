#ifndef CONFORMAL_GEOMETRY_H
#define CONFORMAL_GEOMETRY_H

#include "curl_E.h"
#include "pipe_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * Dey-Mittra Conformal Subcell Geometry Data
 *
 * For each E-field edge and each H-field face in the grid, stores the
 * vacuum fraction that accounts for curved PEC pipe walls cutting
 * through Yee cells.
 *
 * Edge fractions: what fraction of each E-edge length is in vacuum.
 *   1.0 = fully vacuum (standard stencil)
 *   0.0 = fully PEC (E = 0 enforced)
 *   0 < f < 1 = partially cut by pipe wall (conformal treatment)
 *
 * Face areas: the actual vacuum area of each H-face (for curl-E)
 *   and each E-dual-face (for curl-H), in physical units [m²].
 *
 * IBC weights: wall arc length / dual cell volume at each cut edge,
 *   replacing the staircase 1/dr or 1/dz weights.
 *
 * COORDINATE CONVENTIONS (from curl stencil analysis):
 *   Er(i,j,k)  at  r = a + (i+½)dr,  φ = j·dφ,      z = k·dz
 *   Eφ(i,j,k)  at  r = a + i·dr,     φ = (j+½)·dφ,   z = k·dz
 *   Ez(i,j,k)  at  r = a + i·dr,     φ = j·dφ,       z = (k+½)·dz
 *
 *   Hr(i,j,k)  at  r = a + i·dr,     φ = (j+½)·dφ,   z = (k+½)·dz
 *   Hφ(i,j,k)  at  r = a + (i+½)·dr, φ = j·dφ,       z = (k+½)·dz
 *   Hz(i,j,k)  at  r = a + (i+½)·dr, φ = (j+½)·dφ,   z = k·dz
 *
 * NOTE: The existing material_mask_build_full() uses φ = (j+0.5)·dφ
 * for Er, which is a half-cell offset. This module uses the correct
 * positions from the curl stencils.
 *============================================================================*/

#define DM_AREA_THRESHOLD 0.1  /* Min vacuum fraction before treating as PEC */

typedef struct {
    /*-----------------------------------------------------------------
     * Edge vacuum fractions (same sizes as MaterialMask arrays)
     *-----------------------------------------------------------------*/
    double* edge_frac_Er;     /* Nr * Nphi * (Nz+1) */
    double* edge_frac_Ephi;   /* (Nr+1) * Nphi * (Nz+1) */
    double* edge_frac_Ez;     /* (Nr+1) * Nphi * Nz */

    int size_Er;
    int size_Ephi;
    int size_Ez;

    /*-----------------------------------------------------------------
     * Primal face vacuum areas [m²] for curl-E normalization
     * Each H-component lives at the center of a face.
     *
     * Hr face: (φ,z) plane at r = r_i
     *   standard area = r_i · dφ · dz
     *
     * Hφ face: (r,z) plane at φ = φ_j
     *   standard area = dr · dz
     *
     * Hz face: (r,φ) plane at z = z_k
     *   standard area = r_{i+½} · dφ · dr
     *-----------------------------------------------------------------*/
    double* face_area_Hr;     /* (Nr+1) * Nphi * Nz */
    double* face_area_Hphi;   /* Nr * Nphi * Nz */
    double* face_area_Hz;     /* Nr * Nphi * (Nz+1) */

    int size_Hr;
    int size_Hphi;
    int size_Hz;

    /*-----------------------------------------------------------------
     * Dual face vacuum areas [m²] for curl-H normalization
     *
     * Er dual face: (φ,z) plane at r = r_{i+½}
     *   spans φ ∈ [(j-½)dφ, (j+½)dφ], z ∈ [(k-½)dz, (k+½)dz]
     *   standard area = r_{i+½} · dφ · dz
     *
     * Eφ dual face: (r,z) plane at φ = (j+½)·dφ
     *   standard area = dr · dz
     *
     * Ez dual face: (r,φ) plane at z = (k+½)·dz
     *   standard area = r_i · dφ · dr
     *-----------------------------------------------------------------*/
    double* dual_area_Er;     /* Nr * Nphi * (Nz+1) */
    double* dual_area_Ephi;   /* (Nr+1) * Nphi * (Nz+1) */
    double* dual_area_Ez;     /* (Nr+1) * Nphi * Nz */

    /*-----------------------------------------------------------------
     * IBC surface weights at cut edges
     * weight = wall_arc_length_in_cell / dual_cell_vacuum_volume
     * Replaces staircase 1/dr, 1/dz weights.
     * Zero for non-boundary cells.
     *-----------------------------------------------------------------*/
    double* ibc_weight_Er;    /* Nr * Nphi * (Nz+1) */
    double* ibc_weight_Ephi;  /* (Nr+1) * Nphi * (Nz+1) */
    double* ibc_weight_Ez;    /* (Nr+1) * Nphi * Nz */

    /*-----------------------------------------------------------------
     * Diagnostics
     *-----------------------------------------------------------------*/
    int num_cut_Er;       /* edges with 0 < frac < 1 */
    int num_cut_Ephi;
    int num_cut_Ez;
    int num_ibc_Er;       /* edges with nonzero IBC weight */
    int num_ibc_Ephi;
    int num_ibc_Ez;
} ConformalData;

/*=============================================================================
 * Lifecycle
 *============================================================================*/

/* Build conformal data for radial beam pipes only */
void conformal_data_build(
    ConformalData* cd,
    const PipeConfig* radial_pipes,
    const GridParams* grid,
    double r_outer,             /* b = outer conductor radius */
    double z0_extension         /* grid z-offset (z_phys = z_grid - z0_ext) */
);

/* Build conformal data for radial + endcap pipes */
void conformal_data_build_full(
    ConformalData* cd,
    const PipeConfig* radial_pipes,
    const EndcapPipeConfig* endcap_pipes,
    const GridParams* grid,
    double r_outer,
    double L_cavity,
    double z0_extension
);

void conformal_data_free(ConformalData* cd);

void conformal_data_print_stats(const ConformalData* cd, const GridParams* grid);

/*=============================================================================
 * Core Geometry Primitives
 *============================================================================*/

/* Area of intersection of axis-aligned rectangle [x0,x1]×[y0,y1]
 * with circle centered at (cx,cy) with radius R.
 * All coordinates in physical units.
 * Returns 0 if no intersection. */
double circle_rect_intersection_area(
    double x0, double x1,
    double y0, double y1,
    double cx, double cy, double R
);

/* Length of overlap between 1D intervals [a0,a1] and [b0,b1].
 * Returns 0 if no overlap. */
double interval_overlap(double a0, double a1, double b0, double b1);

/* Pipe cross-section radius at radial position r.
 * Includes quarter-circle fillet between aperture and pipe radii.
 * Wrapper matching pipe_radius_at_r() but taking individual params. */
double conformal_pipe_radius(
    double r,
    double r_outer,             /* b */
    double aperture_radius,
    double pipe_radius
);

/*=============================================================================
 * Edge Fraction Computation (for individual edges near a radial pipe)
 *
 * Each returns the vacuum fraction of the edge for the region r > r_outer.
 * For r ≤ r_outer, edges are fully vacuum (fraction = 1).
 *============================================================================*/

/* Er edge at (i+½, j, k): radial edge at fixed (φ,z).
 * Edge extends in r from r0 = a+i·dr to r1 = a+(i+1)·dr.
 * Uses bisection for fillet region where R(r) varies. */
double edge_fraction_Er_radial_pipe(
    double r0, double r1,           /* edge radial extent */
    double phi, double z,           /* edge (φ,z) position */
    double phi_c, double z_c,       /* pipe center */
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
);

/* Eφ edge at (i, j+½, k): azimuthal edge at fixed (r, z).
 * Edge extends in φ from φ0 to φ1 = φ0 + dφ.
 * Analytical: 1D interval intersection. */
double edge_fraction_Ephi_radial_pipe(
    double r, double z,
    double phi0, double phi1,       /* edge φ extent */
    double phi_c, double z_c,       /* pipe center */
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
);

/* Ez edge at (i, j, k+½): axial edge at fixed (r, φ).
 * Edge extends in z from z0 to z1 = z0 + dz.
 * Analytical: 1D interval intersection. */
double edge_fraction_Ez_radial_pipe(
    double r, double phi,
    double z0, double z1,           /* edge z extent */
    double phi_c, double z_c,
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
);

/*=============================================================================
 * Face Area Computation (for individual faces near a radial pipe)
 *
 * Returns the vacuum area of the face [m²], accounting for the pipe
 * cross-section cutting through it.
 *============================================================================*/

/* Hr face: (φ,z) plane at fixed r.
 * Uses analytical circle-rect intersection. */
double face_area_Hr_radial_pipe(
    double r,                       /* face radial position */
    double phi0, double phi1,       /* face φ extent */
    double z0, double z1,           /* face z extent */
    double phi_c, double z_c,       /* pipe center */
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
);

/* Hφ face: (r,z) plane at fixed φ.
 * Uses Gauss-Legendre quadrature (pipe boundary varies with r). */
double face_area_Hphi_radial_pipe(
    double phi,
    double r0, double r1,
    double z0, double z1,
    double phi_c, double z_c,
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
);

/* Hz face: (r,φ) plane at fixed z.
 * Uses Gauss-Legendre quadrature. */
double face_area_Hz_radial_pipe(
    double z,
    double r0, double r1,
    double phi0, double phi1,
    double phi_c, double z_c,
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
);

/*=============================================================================
 * IBC Wall Arc Length (for individual cells near a radial pipe)
 *
 * Returns the length of the pipe wall arc intersecting the given cell face.
 *============================================================================*/

/* Wall arc length on Hr face: (φ,z) plane at fixed r.
 * Arc of circle with radius R(r), intersecting rectangle [φ0,φ1]×[z0,z1]. */
double wall_arc_length_Hr(
    double r,
    double phi0, double phi1,
    double z0, double z1,
    double phi_c, double z_c,
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
);

/*=============================================================================
 * Validation
 *============================================================================*/

/* Compute total pipe wall area from conformal data (should match 2πR·L_pipe
 * for a straight pipe, or the integrated fillet profile). */
double conformal_total_wall_area(
    const ConformalData* cd,
    const GridParams* grid,
    double r_outer
);

void conformal_data_apply_ibc_unmask(
    ConformalData* cd,
    const MaterialMask* pec_mask,
    const MaterialMask* ibc_mask,
    const GridParams* grid
);

/* Test circle-rect intersection against known analytical cases */
int conformal_test_primitives(void);

void conformal_data_zero_endplate_faces(
    ConformalData* cd,
    const GridParams* grid,
    int k_z0, int k_zL
);

#ifdef __cplusplus
}
#endif

#endif /* CONFORMAL_GEOMETRY_H */
