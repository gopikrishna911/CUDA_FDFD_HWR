/*=============================================================================
 * conformal_geometry.c
 *
 * Dey-Mittra conformal subcell geometry engine for the rhodotron FDFD
 * eigensolver. Computes exact vacuum edge fractions, face areas, and
 * IBC wall weights where curved pipe walls intersect Yee grid cells.
 *
 * Phase 1: Radial beam pipes only (endcap pipes in Phase 4).
 *
 * Key geometry:
 *   Each radial pipe is a cylinder of radius R(r) passing through
 *   the outer conductor at r = b. In the (φ,z) plane at any r > b,
 *   the pipe cross-section is a circle of radius R(r) centered at
 *   (φ_c, z_c). The cross-section uses physical arc distance:
 *       dist² = (r·Δφ)² + Δz² < R(r)²
 *   where R(r) includes a quarter-circle fillet near r = b.
 *
 * All face areas and edge fractions are computed analytically where
 * possible, with Gauss-Legendre quadrature for faces where the pipe
 * boundary is not a simple circle (Hφ and Hz faces where R varies
 * with r along the face).
 *============================================================================*/

#define _USE_MATH_DEFINES
#include "conformal_geometry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/*=============================================================================
 * Gauss-Legendre quadrature nodes and weights (8-point)
 * Nodes on [-1, 1], mapped to [a, b] via x = (b-a)/2 * t + (a+b)/2
 *============================================================================*/
#define GL_N 8
static const double gl_nodes[GL_N] = {
    -0.9602898564975362, -0.7966664774136267,
    -0.5255324099163290, -0.1834346424956498,
     0.1834346424956498,  0.5255324099163290,
     0.7966664774136267,  0.9602898564975362
};
static const double gl_weights[GL_N] = {
    0.1012285362903763, 0.2223810344533745,
    0.3137066458778873, 0.3626837833783620,
    0.3626837833783620, 0.3137066458778873,
    0.2223810344533745, 0.1012285362903763
};

/*=============================================================================
 * Low-level helpers
 *============================================================================*/

static inline double dmin(double a, double b) { return a < b ? a : b; }
static inline double dmax(double a, double b) { return a > b ? a : b; }
static inline double clamp(double v, double lo, double hi) {
    return dmax(lo, dmin(hi, v));
}

/* Wrap angle difference to [-π, π] */
static inline double wrap_dphi(double dphi) {
    while (dphi >  M_PI) dphi -= 2.0 * M_PI;
    while (dphi < -M_PI) dphi += 2.0 * M_PI;
    return dphi;
}

/*=============================================================================
 * Pipe cross-section radius at radial position r
 *
 * Quarter-circle fillet profile:
 *   r <= r_outer:            aperture_radius  (inside cavity)
 *   r_outer < r < r_outer+Rf: fillet arc
 *   r >= r_outer + Rf:       pipe_radius
 *
 * where Rf = aperture_radius - pipe_radius (fillet radius)
 *============================================================================*/

double conformal_pipe_radius(
    double r,
    double r_outer,
    double aperture_radius,
    double pipe_radius
) {
    if (r <= r_outer) return aperture_radius;

    double R_f = aperture_radius - pipe_radius;
    if (R_f < 1e-14) return pipe_radius;

    double dr = r - r_outer;
    if (dr >= R_f) return pipe_radius;

    /* Quarter-circle arc: R(r) = (pipe_radius + Rf) - sqrt(Rf² - (Rf-dr)²) */
    double arg = R_f * R_f - (R_f - dr) * (R_f - dr);
    if (arg < 0.0) arg = 0.0;
    return (pipe_radius + R_f) - sqrt(arg);
}

/* Is point (r, phi, z) inside a specific radial pipe? (vacuum test)
 * r > r_outer only. */
static int point_in_pipe(
    double r, double phi, double z,
    double phi_c, double z_c,
    double r_outer, double aperture_radius, double pipe_radius
) {
    double R = conformal_pipe_radius(r, r_outer, aperture_radius, pipe_radius);
    double dphi = wrap_dphi(phi - phi_c);
    double arc = r * dphi;
    double dz = z - z_c;
    return (arc * arc + dz * dz) < (R * R);
}

/*=============================================================================
 * Core Primitive: 1D interval overlap
 *============================================================================*/

double interval_overlap(double a0, double a1, double b0, double b1) {
    double lo = dmax(a0, b0);
    double hi = dmin(a1, b1);
    return dmax(0.0, hi - lo);
}

/*=============================================================================
 * Core Primitive: Circle-Rectangle intersection area
 *
 * Analytical computation using vertical strip integration.
 *
 * Intersection area = ∫ h(x) dx over x ∈ [x0,x1] ∩ [-R,R]
 * where h(x) = max(0, min(y1, sqrt(R²-x²)) - max(y0, -sqrt(R²-x²)))
 *
 * The integrand changes form at breakpoints where the circle crosses
 * y = y0 or y = y1. Between breakpoints, each piece integrates to
 * linear × width  or  circular_segment_area.
 *============================================================================*/

/* Antiderivative of sqrt(R² - x²) */
static double F_circle(double x, double R) {
    double s = clamp(x / R, -1.0, 1.0);
    double sq = R * R - x * x;
    if (sq < 0.0) sq = 0.0;
    return 0.5 * (x * sqrt(sq) + R * R * asin(s));
}

/* ∫_a^b sqrt(R² - x²) dx,  with a,b clamped to [-R, R] */
static double I_circle(double a, double b, double R) {
    a = dmax(a, -R);
    b = dmin(b,  R);
    if (a >= b) return 0.0;
    return F_circle(b, R) - F_circle(a, R);
}

double circle_rect_intersection_area(
    double x0, double x1,
    double y0, double y1,
    double cx, double cy, double R
) {
    /* Translate circle center to origin */
    x0 -= cx;  x1 -= cx;
    y0 -= cy;  y1 -= cy;

    /* Quick reject */
    if (x0 >= R || x1 <= -R || y0 >= R || y1 <= -R) return 0.0;
    if (R <= 0.0 || x0 >= x1 || y0 >= y1) return 0.0;

    /* Quick accept: rectangle fully inside circle */
    double corner_max_r2 = dmax(x0*x0, x1*x1) + dmax(y0*y0, y1*y1);
    if (corner_max_r2 <= R * R) {
        return (x1 - x0) * (y1 - y0);  /* whole rectangle */
    }

    /* Collect breakpoints in x where the integrand changes form.
     * Breakpoints come from:
     *   x = ±R               (circle boundary)
     *   sqrt(R²-x²) = |y0|   → x = ±sqrt(R²-y0²)
     *   sqrt(R²-x²) = |y1|   → x = ±sqrt(R²-y1²)
     */
    double bp[10];
    int nbp = 0;

    double x_lo = dmax(x0, -R);
    double x_hi = dmin(x1,  R);
    if (x_lo >= x_hi) return 0.0;

    bp[nbp++] = x_lo;
    bp[nbp++] = x_hi;

    /* Breakpoints from y0 */
    if (fabs(y0) < R) {
        double xb = sqrt(R * R - y0 * y0);
        if (-xb > x_lo && -xb < x_hi) bp[nbp++] = -xb;
        if ( xb > x_lo &&  xb < x_hi) bp[nbp++] =  xb;
    }
    /* Breakpoints from y1 */
    if (fabs(y1) < R) {
        double xb = sqrt(R * R - y1 * y1);
        if (-xb > x_lo && -xb < x_hi) bp[nbp++] = -xb;
        if ( xb > x_lo &&  xb < x_hi) bp[nbp++] =  xb;
    }

    /* Sort (insertion sort, ≤10 elements) */
    for (int i = 1; i < nbp; i++) {
        double key = bp[i];
        int j = i - 1;
        while (j >= 0 && bp[j] > key) { bp[j+1] = bp[j]; j--; }
        bp[j+1] = key;
    }

    /* Remove near-duplicates */
    double tol = 1e-14 * R;
    int nu = 1;
    for (int i = 1; i < nbp; i++) {
        if (bp[i] - bp[nu-1] > tol) bp[nu++] = bp[i];
    }
    nbp = nu;

    /* Integrate each segment analytically */
    double area = 0.0;
    for (int s = 0; s < nbp - 1; s++) {
        double xa = bp[s], xb_seg = bp[s+1];
        if (xb_seg - xa < tol) continue;

        /* Sample midpoint to determine which case applies */
        double xm = 0.5 * (xa + xb_seg);
        double Rsq_xm = R * R - xm * xm;
        if (Rsq_xm < 0.0) continue;  /* outside circle */
        double y_up = sqrt(Rsq_xm);
        double y_lo_circ = -y_up;

        double v_lo = dmax(y0, y_lo_circ);
        double v_hi = dmin(y1, y_up);
        if (v_hi <= v_lo) continue;

        /* Classify edges */
        int lo_is_y0 = (y0 >= y_lo_circ);   /* bottom clipped by rect */
        int hi_is_y1 = (y1 <= y_up);         /* top clipped by rect */

        if (lo_is_y0 && hi_is_y1) {
            /* h(x) = y1 - y0 (constant, rect fully inside circle vertically) */
            area += (y1 - y0) * (xb_seg - xa);
        }
        else if (lo_is_y0 && !hi_is_y1) {
            /* h(x) = sqrt(R²-x²) - y0 */
            area += I_circle(xa, xb_seg, R) - y0 * (xb_seg - xa);
        }
        else if (!lo_is_y0 && hi_is_y1) {
            /* h(x) = y1 + sqrt(R²-x²)   [since lower bound is -sqrt(R²-x²)] */
            area += y1 * (xb_seg - xa) + I_circle(xa, xb_seg, R);
        }
        else {
            /* h(x) = 2·sqrt(R²-x²) (full circle height within rect) */
            area += 2.0 * I_circle(xa, xb_seg, R);
        }
    }

    return area;
}

/*=============================================================================
 * Circle-rectangle arc length
 *
 * Length of the circular arc (radius R, centered at origin) that lies
 * within rectangle [x0,x1]×[y0,y1].
 *
 * Uses high-resolution angular sweep: divide the circle into N
 * sub-arcs, check if each midpoint is inside the rectangle, and
 * sum the arc lengths. Simple, robust, no angular breakpoint issues.
 *
 * Accuracy: ~R·(2π/N)/2 ≈ 0.05% for N=4096.
 * Performance: fine for precomputation (called once per build).
 *============================================================================*/

#define ARC_N 4096

static double circle_rect_arc_length(
    double x0, double x1, double y0, double y1, double R
) {
    if (R <= 0.0) return 0.0;

    /* Quick reject: circle doesn't touch rectangle */
    if (x0 >= R || x1 <= -R || y0 >= R || y1 <= -R) return 0.0;

    double dtheta = 2.0 * M_PI / ARC_N;
    double total_arc = 0.0;

    for (int i = 0; i < ARC_N; i++) {
        double theta = (i + 0.5) * dtheta;
        double xm = R * cos(theta);
        double ym = R * sin(theta);
        if (xm >= x0 && xm <= x1 && ym >= y0 && ym <= y1) {
            total_arc += R * dtheta;
        }
    }

    return total_arc;
}

/*=============================================================================
 * Edge Fraction: Eφ (azimuthal edge, 1D in φ)
 *
 * Edge at (r, j+½, k) extends from φ0 to φ1 = φ0 + dφ at fixed (r, z).
 * For r > r_outer: the pipe covers an angular range around φ_c.
 * Vacuum fraction = fraction of edge inside pipe.
 * For r ≤ r_outer: fraction = 1.0 (cavity vacuum).
 *============================================================================*/

double edge_fraction_Ephi_radial_pipe(
    double r, double z,
    double phi0, double phi1,
    double phi_c, double z_c,
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
) {
    if (r <= r_outer) return 1.0;  /* inside cavity = vacuum */
    if (r >= r_outer + pipe_length - 1e-10) return 0.0;  /* past pipe cap = PEC */

    double R = conformal_pipe_radius(r, r_outer, aperture_radius, pipe_radius);
    double dz = z - z_c;

    /* Pipe extent in physical arc at this (r, z): circle of radius R */
    double arg = R * R - dz * dz;
    if (arg <= 0.0) return 0.0;  /* z too far from pipe center, all PEC */

    double half_arc = sqrt(arg);  /* physical half-width */
    double half_phi = half_arc / r;  /* angular half-width */

    /* Pipe covers φ ∈ [φ_c - half_phi, φ_c + half_phi] */
    /* Edge covers φ ∈ [φ0, φ1] */
    /* Need overlap accounting for periodic φ */
    double dphi_edge = phi1 - phi0;

    /* Center the edge relative to pipe */
    double edge_center = 0.5 * (phi0 + phi1);
    double d_center = wrap_dphi(edge_center - phi_c);

    /* Transform to local coordinate: pipe centered at 0 */
    double local_lo = d_center - 0.5 * dphi_edge;
    double local_hi = d_center + 0.5 * dphi_edge;

    double overlap = interval_overlap(local_lo, local_hi,
                                      -half_phi, half_phi);
    return overlap / dphi_edge;  /* fraction inside pipe = vacuum */
}

/*=============================================================================
 * Edge Fraction: Ez (axial edge, 1D in z)
 *
 * Edge at (i, j, k+½) extends from z0 to z1 = z0 + dz at fixed (r, φ).
 * For r > r_outer: pipe covers a z-range centered on z_c.
 *============================================================================*/

double edge_fraction_Ez_radial_pipe(
    double r, double phi,
    double z0, double z1,
    double phi_c, double z_c,
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
) {
    if (r <= r_outer) return 1.0;
    if (r >= r_outer + pipe_length - 1e-10) return 0.0;  /* past pipe cap = PEC */

    double R = conformal_pipe_radius(r, r_outer, aperture_radius, pipe_radius);
    double dphi = wrap_dphi(phi - phi_c);
    double arc = r * dphi;

    double arg = R * R - arc * arc;
    if (arg <= 0.0) return 0.0;  /* too far in φ, all PEC */

    double half_z = sqrt(arg);

    double overlap = interval_overlap(z0, z1, z_c - half_z, z_c + half_z);
    return overlap / (z1 - z0);
}

/*=============================================================================
 * Edge Fraction: Er (radial edge, 1D in r — complex due to fillet)
 *
 * Edge at (i+½, j, k) extends from r0 to r1 = r0 + dr at fixed (φ, z).
 * For r ≤ r_outer: fully in cavity = vacuum.
 * For r > r_outer: in vacuum iff inside pipe, i.e.,
 *   (r·Δφ)² + Δz² < R(r)²
 * R(r) varies through the fillet region.
 * Use bisection to find the crossover r.
 *============================================================================*/

double edge_fraction_Er_radial_pipe(
    double r0, double r1,
    double phi, double z,
    double phi_c, double z_c,
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
) {
    double r_cap = r_outer + pipe_length;

    /* Part in cavity (r ≤ r_outer) is always vacuum */
    double r_cav_hi = dmin(r1, r_outer);
    double vac_from_cavity = 0.0;
    if (r0 < r_cav_hi) {
        vac_from_cavity = r_cav_hi - r0;
    }

    /* Part in pipe region (r_outer < r < r_cap) */
    double r_pipe_lo = dmax(r0, r_outer);
    double r_pipe_hi = dmin(r1, r_cap);  /* clamp at pipe cap */
    if (r_pipe_lo >= r_pipe_hi) {
        return vac_from_cavity / (r1 - r0);
    }

    double dphi = wrap_dphi(phi - phi_c);
    double dz = z - z_c;

    /* Test both endpoints of the pipe segment */
    double R_lo = conformal_pipe_radius(r_pipe_lo, r_outer, aperture_radius, pipe_radius);
    double dist2_lo = r_pipe_lo * r_pipe_lo * dphi * dphi + dz * dz;
    int in_lo = (dist2_lo < R_lo * R_lo);

    double R_hi = conformal_pipe_radius(r_pipe_hi, r_outer, aperture_radius, pipe_radius);
    double dist2_hi = r_pipe_hi * r_pipe_hi * dphi * dphi + dz * dz;
    int in_hi = (dist2_hi < R_hi * R_hi);

    double vac_from_pipe = 0.0;

    if (in_lo && in_hi) {
        /* Entire pipe segment is inside vacuum */
        vac_from_pipe = r_pipe_hi - r_pipe_lo;
    }
    else if (!in_lo && !in_hi) {
        /* Check if there's a pocket of vacuum in between.
         * This can happen if R(r) has a local maximum (fillet region).
         * For a monotonically decreasing R(r), if both ends are outside,
         * the entire segment is outside. */
        /* Sample a few points to check */
        int any_inside = 0;
        for (int s = 1; s <= 7; s++) {
            double rt = r_pipe_lo + s * (r_pipe_hi - r_pipe_lo) / 8.0;
            double Rt = conformal_pipe_radius(rt, r_outer, aperture_radius, pipe_radius);
            double dt2 = rt * rt * dphi * dphi + dz * dz;
            if (dt2 < Rt * Rt) { any_inside = 1; break; }
        }
        if (!any_inside) {
            vac_from_pipe = 0.0;
        } else {
            /* Fallback: Gauss-Legendre quadrature */
            double half = 0.5 * (r_pipe_hi - r_pipe_lo);
            double mid = 0.5 * (r_pipe_lo + r_pipe_hi);
            double sum = 0.0;
            for (int g = 0; g < GL_N; g++) {
                double rt = mid + half * gl_nodes[g];
                double Rt = conformal_pipe_radius(rt, r_outer, aperture_radius, pipe_radius);
                double dt2 = rt * rt * dphi * dphi + dz * dz;
                if (dt2 < Rt * Rt) sum += gl_weights[g];
            }
            vac_from_pipe = half * sum;
        }
    }
    else {
        /* One end inside, one outside: find crossover by bisection */
        double ra = in_lo ? r_pipe_lo : r_pipe_hi;
        double rb = in_lo ? r_pipe_hi : r_pipe_lo;
        /* ra is inside pipe, rb is outside */
        for (int iter = 0; iter < 50; iter++) {
            double rm = 0.5 * (ra + rb);
            double Rm = conformal_pipe_radius(rm, r_outer, aperture_radius, pipe_radius);
            double dm2 = rm * rm * dphi * dphi + dz * dz;
            if (dm2 < Rm * Rm)
                ra = rm;  /* inside */
            else
                rb = rm;  /* outside */
        }
        double r_cross = 0.5 * (ra + rb);

        if (in_lo) {
            vac_from_pipe = r_cross - r_pipe_lo;
        } else {
            vac_from_pipe = r_pipe_hi - r_cross;
        }
    }

    return (vac_from_cavity + vac_from_pipe) / (r1 - r0);
}

/*=============================================================================
 * Face Area: Hr (φ,z plane at fixed r) — analytical
 *
 * Pipe cross-section at this r is a circle of radius R(r) in physical
 * (arc, z) space. Face is a rectangle. Vacuum area = face - pipe overlap.
 * For r > r_outer, vacuum is INSIDE the pipe.
 * So: vacuum_area = pipe_circle ∩ face_rectangle.
 *
 * For r ≤ r_outer: entire face is vacuum = r · dφ · dz.
 *============================================================================*/

double face_area_Hr_radial_pipe(
    double r,
    double phi0, double phi1,
    double z0, double z1,
    double phi_c, double z_c,
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
) {
    double face_area = r * (phi1 - phi0) * (z1 - z0);
    if (r <= r_outer) return face_area;  /* cavity: all vacuum */
    if (r >= r_outer + pipe_length - 1e-10) return 0.0;  /* past cap: all PEC */

    double R = conformal_pipe_radius(r, r_outer, aperture_radius, pipe_radius);

    /* Physical coordinates centered on pipe axis */
    double u0 = r * wrap_dphi(phi0 - phi_c);
    double u1 = r * wrap_dphi(phi1 - phi_c);

    /* Ensure u0 < u1 */
    if (u0 > u1) { double tmp = u0; u0 = u1; u1 = tmp; }

    /* Vacuum area = intersection of pipe circle with face rectangle */
    double pipe_in_face = circle_rect_intersection_area(
        u0, u1, z0, z1, 0.0, z_c, R);

    return pipe_in_face;
}

/*=============================================================================
 * Face Area: Hφ (r,z plane at fixed φ) — numerical
 *
 * The pipe boundary in (r, z) at fixed φ is the curve:
 *   |z - z_c| = sqrt(R(r)² - r²·Δφ²)
 * where R(r) varies through the fillet. This is not a simple circle.
 * Use 2D Gauss-Legendre quadrature.
 *
 * For r > r_outer: vacuum = inside pipe.
 * For r ≤ r_outer: vacuum = everything.
 *============================================================================*/

double face_area_Hphi_radial_pipe(
    double phi,
    double r0, double r1,
    double z0, double z1,
    double phi_c, double z_c,
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
) {
    double face_std = (r1 - r0) * (z1 - z0);  /* standard area = dr · dz */

    /* If entirely in cavity, all vacuum */
    if (r1 <= r_outer) return face_std;

    /* If entirely past pipe cap, all PEC */
    if (r0 >= r_outer + pipe_length - 1e-10) return 0.0;

    double dphi_val = wrap_dphi(phi - phi_c);
    double r_cap = r_outer + pipe_length;

    /* 2D Gauss-Legendre quadrature over [r0,r1] × [z0,z1] */
    double half_r = 0.5 * (r1 - r0);
    double mid_r  = 0.5 * (r0 + r1);
    double half_z = 0.5 * (z1 - z0);
    double mid_z  = 0.5 * (z0 + z1);

    double sum = 0.0;
    for (int gi = 0; gi < GL_N; gi++) {
        double r = mid_r + half_r * gl_nodes[gi];
        for (int gk = 0; gk < GL_N; gk++) {
            double z = mid_z + half_z * gl_nodes[gk];
            int is_vac;
            if (r <= r_outer) {
                is_vac = 1;  /* cavity */
            } else if (r >= r_cap) {
                is_vac = 0;  /* past pipe cap = PEC */
            } else {
                double R = conformal_pipe_radius(r, r_outer, aperture_radius, pipe_radius);
                double arc = r * dphi_val;
                double dz = z - z_c;
                is_vac = (arc * arc + dz * dz < R * R);
            }
            if (is_vac) {
                sum += gl_weights[gi] * gl_weights[gk];
            }
        }
    }

    return half_r * half_z * sum;
}

/*=============================================================================
 * Face Area: Hz (r,φ plane at fixed z) — numerical
 *
 * The pipe boundary in (r, φ) at fixed z is the curve:
 *   |r·(φ-φ_c)| = sqrt(R(r)² - (z-z_c)²)
 * Use 2D Gauss-Legendre quadrature.
 * Note: face area in physical units = ∫∫ r dr dφ
 *============================================================================*/

double face_area_Hz_radial_pipe(
    double z,
    double r0, double r1,
    double phi0, double phi1,
    double phi_c, double z_c,
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
) {
    /* Standard area = r_{i+½} · dφ · dr */
    double r_mid = 0.5 * (r0 + r1);
    double face_std = r_mid * (phi1 - phi0) * (r1 - r0);

    if (r1 <= r_outer) return face_std;
    if (r0 >= r_outer + pipe_length - 1e-10) return 0.0;

    double dz = z - z_c;
    double r_cap = r_outer + pipe_length;

    /* 2D GL quadrature over [r0,r1] × [φ0,φ1], integrand = r (for area element) */
    double half_r = 0.5 * (r1 - r0);
    double mid_r  = 0.5 * (r0 + r1);
    double half_p = 0.5 * (phi1 - phi0);
    double mid_p  = 0.5 * (phi0 + phi1);

    double sum = 0.0;
    for (int gi = 0; gi < GL_N; gi++) {
        double r = mid_r + half_r * gl_nodes[gi];
        for (int gj = 0; gj < GL_N; gj++) {
            double phi = mid_p + half_p * gl_nodes[gj];
            int is_vac;
            if (r <= r_outer) {
                is_vac = 1;
            } else if (r >= r_cap) {
                is_vac = 0;  /* past pipe cap */
            } else {
                double R = conformal_pipe_radius(r, r_outer, aperture_radius, pipe_radius);
                double dphi_val = wrap_dphi(phi - phi_c);
                double arc = r * dphi_val;
                is_vac = (arc * arc + dz * dz < R * R);
            }
            if (is_vac) {
                sum += gl_weights[gi] * gl_weights[gj] * r;
            }
        }
    }

    return half_r * half_p * sum;
}

/*=============================================================================
 * Wall Arc Length on Hr face (φ,z plane at fixed r) — analytical
 *
 * Length of circular arc of radius R(r) inside the face rectangle.
 *============================================================================*/

double wall_arc_length_Hr(
    double r,
    double phi0, double phi1,
    double z0, double z1,
    double phi_c, double z_c,
    double r_outer,
    double aperture_radius,
    double pipe_radius,
    double pipe_length
) {
    if (r <= r_outer) return 0.0;
    if (r >= r_outer + pipe_length - 1e-10) return 0.0;  /* past pipe cap */

    double R = conformal_pipe_radius(r, r_outer, aperture_radius, pipe_radius);

    double u0 = r * wrap_dphi(phi0 - phi_c);
    double u1 = r * wrap_dphi(phi1 - phi_c);
    if (u0 > u1) { double tmp = u0; u0 = u1; u1 = tmp; }

    return circle_rect_arc_length(u0, u1, z0 - z_c, z1 - z_c, R);
}

/*=============================================================================
 * Main builder: conformal_data_build()
 *
 * For each E-edge and H-face in the grid, compute the vacuum fraction
 * and face area, accounting for all radial pipes.
 *
 * Strategy:
 * 1. Initialize all edge fractions and face areas to their "no pipe" values:
 *    - For r ≤ r_outer (cavity): frac=1, area=standard
 *    - For r > r_outer (PEC body): frac=0, area=0
 * 2. For each pipe, find the bounding box of affected cells.
 * 3. Overwrite fractions/areas with exact values at affected cells.
 * 4. Handle overlapping pipes (shouldn't occur for well-separated pipes).
 * 5. Apply area threshold (DM_AREA_THRESHOLD).
 *============================================================================*/

/* Helper: index of cell containing radial position r */
static int r_to_i(double r, const GridParams* grid) {
    return (int)floor((r - grid->a) / grid->dr);
}

/* Helper: index of cell containing azimuthal position φ */
static int phi_to_j(double phi, const GridParams* grid) {
    int j = (int)floor(phi / grid->dphi);
    return ((j % grid->Nphi) + grid->Nphi) % grid->Nphi;
}

/* Helper: index of cell containing axial position z (grid coordinates) */
static int z_to_k(double z_grid, const GridParams* grid) {
    return (int)floor(z_grid / grid->dz);
}

void conformal_data_build(
    ConformalData* cd,
    const PipeConfig* radial_pipes,
    const GridParams* grid,
    double r_outer,
    double z0_extension
) {
    int Nr = grid->Nr, Nphi = grid->Nphi, Nz = grid->Nz;
    int Nr1 = Nr + 1;
    double dr = grid->dr, dphi = grid->dphi, dz = grid->dz;

    /* Sizes (same as MaterialMask) */
    cd->size_Er   = Nr * Nphi * (Nz + 1);
    cd->size_Ephi = Nr1 * Nphi * (Nz + 1);
    cd->size_Ez   = Nr1 * Nphi * Nz;

    cd->size_Hr   = Nr1 * Nphi * Nz;
    cd->size_Hphi = Nr * Nphi * Nz;
    cd->size_Hz   = Nr * Nphi * (Nz + 1);

    /* Allocate */
    cd->edge_frac_Er   = (double*)malloc(cd->size_Er   * sizeof(double));
    cd->edge_frac_Ephi = (double*)malloc(cd->size_Ephi * sizeof(double));
    cd->edge_frac_Ez   = (double*)malloc(cd->size_Ez   * sizeof(double));

    cd->face_area_Hr   = (double*)malloc(cd->size_Hr   * sizeof(double));
    cd->face_area_Hphi = (double*)malloc(cd->size_Hphi * sizeof(double));
    cd->face_area_Hz   = (double*)malloc(cd->size_Hz   * sizeof(double));

    cd->dual_area_Er   = (double*)malloc(cd->size_Er   * sizeof(double));
    cd->dual_area_Ephi = (double*)malloc(cd->size_Ephi * sizeof(double));
    cd->dual_area_Ez   = (double*)malloc(cd->size_Ez   * sizeof(double));

    cd->ibc_weight_Er   = (double*)calloc(cd->size_Er,   sizeof(double));
    cd->ibc_weight_Ephi = (double*)calloc(cd->size_Ephi, sizeof(double));
    cd->ibc_weight_Ez   = (double*)calloc(cd->size_Ez,   sizeof(double));

    cd->num_cut_Er = cd->num_cut_Ephi = cd->num_cut_Ez = 0;
    cd->num_ibc_Er = cd->num_ibc_Ephi = cd->num_ibc_Ez = 0;

    /*=========================================================================
     * Pass 1: Initialize to binary values (no pipes, staircase-like)
     *========================================================================*/

    /* Er(i,j,k): r = a + (i+½)dr, φ = j·dφ, z_phys = k·dz - z0_ext */
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr; i++) {
                int idx = i + Nr * (j + Nphi * k);
                double r = grid->a + (i + 0.5) * dr;
                if (r <= r_outer) {
                    cd->edge_frac_Er[idx] = 1.0;
                } else {
                    cd->edge_frac_Er[idx] = 0.0;  /* PEC by default */
                }
            }
        }
    }

    /* Eφ(i,j,k): r = a + i·dr, φ = (j+½)·dφ, z_phys = k·dz - z0_ext */
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i <= Nr; i++) {
                int idx = i + Nr1 * (j + Nphi * k);
                double r = grid->a + i * dr;
                if (r <= r_outer) {
                    cd->edge_frac_Ephi[idx] = 1.0;
                } else {
                    cd->edge_frac_Ephi[idx] = 0.0;
                }
            }
        }
    }

    /* Ez(i,j,k): r = a + i·dr, φ = j·dφ, z_phys = (k+½)·dz - z0_ext */
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i <= Nr; i++) {
                int idx = i + Nr1 * (j + Nphi * k);
                double r = grid->a + i * dr;
                if (r <= r_outer) {
                    cd->edge_frac_Ez[idx] = 1.0;
                } else {
                    cd->edge_frac_Ez[idx] = 0.0;
                }
            }
        }
    }

    /* Initialize face areas: standard values for r ≤ r_outer, 0 for r > r_outer */

    /* Hr face at (i, j+½, k+½): r = a + i·dr */
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i <= Nr; i++) {
                int idx = i + Nr1 * (j + Nphi * k);
                double r = grid->a + i * dr;
                double std_area = r * dphi * dz;
                cd->face_area_Hr[idx] = (r <= r_outer) ? std_area : 0.0;
            }
        }
    }

    /* Hφ face at (i+½, j, k+½): standard area = dr · dz */
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr; i++) {
                int idx = i + Nr * (j + Nphi * k);
                double r = grid->a + (i + 0.5) * dr;
                double std_area = dr * dz;
                cd->face_area_Hphi[idx] = (r <= r_outer) ? std_area : 0.0;
            }
        }
    }

    /* Hz face at (i+½, j+½, k): standard area = r_{i+½} · dφ · dr */
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr; i++) {
                int idx = i + Nr * (j + Nphi * k);
                double r_mid = grid->a + (i + 0.5) * dr;
                double std_area = r_mid * dphi * dr;
                cd->face_area_Hz[idx] = (r_mid <= r_outer) ? std_area : 0.0;
            }
        }
    }

    /* Dual face areas — same pattern */

    /* Er dual face at (i+½, j, k): r = a + (i+½)dr, spans [(j-½)dφ,(j+½)dφ]×[(k-½)dz,(k+½)dz] */
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr; i++) {
                int idx = i + Nr * (j + Nphi * k);
                double r = grid->a + (i + 0.5) * dr;
                double std_area = r * dphi * dz;
                cd->dual_area_Er[idx] = (r <= r_outer) ? std_area : 0.0;
            }
        }
    }

    /* Eφ dual face at (i, j+½, k): r = a + i·dr, spans [(i-½)dr,(i+½)dr]×[(k-½)dz,(k+½)dz] */
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i <= Nr; i++) {
                int idx = i + Nr1 * (j + Nphi * k);
                double r = grid->a + i * dr;
                double std_area = dr * dz;
                cd->dual_area_Ephi[idx] = (r <= r_outer) ? std_area : 0.0;
            }
        }
    }

    /* Ez dual face at (i, j, k+½): r = a + i·dr, spans [(j-½)dφ,(j+½)dφ]×[(i-½)dr,(i+½)dr] */
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i <= Nr; i++) {
                int idx = i + Nr1 * (j + Nphi * k);
                double r = grid->a + i * dr;
                double std_area = r * dphi * dr;
                cd->dual_area_Ez[idx] = (r <= r_outer) ? std_area : 0.0;
            }
        }
    }

    /*=========================================================================
     * Pass 2: Process each radial pipe
     *
     * For each pipe, find the bounding box of cells it can affect,
     * then compute exact edge fractions and face areas.
     *========================================================================*/

    /* Index of the outer conductor wall */
    int i_wall = r_to_i(r_outer, grid);  /* r_outer should be at an integer i */

    for (int p = 0; p < radial_pipes->num_pipes; p++) {
        double phi_c = radial_pipes->pipes[p].phi_center;
        double z_c   = radial_pipes->pipes[p].z_center;
        double z_c_grid = z_c + z0_extension;  /* z in grid coordinates */

        double R_aper = radial_pipes->aperture_radius;
        double R_pipe = radial_pipes->pipe_radius;
        double R_max = R_aper;  /* maximum cross-section radius */
        double pipe_len = radial_pipes->pipe_length;
        double b = r_outer;

        /* Bounding box in grid indices */
        /* Radial: from i_wall-1 (aperture region) to i covering r=b+pipe_length */
        int i_lo = dmax(0, i_wall - 2);
        int i_hi = dmin(Nr, r_to_i(b + pipe_len, grid) + 2);

        /* Azimuthal: pipe extends ±R_max/r in φ at radius r */
        /* Maximum angular extent at r = b */
        double max_dphi = R_max / b + 2.0 * dphi;  /* safety margin */
        int dj_max = (int)ceil(max_dphi / dphi) + 1;

        int j_center = phi_to_j(phi_c, grid);

        /* Axial: pipe extends ±R_max in z from z_c */
        int dk_max = (int)ceil(R_max / dz) + 2;
        int k_center = z_to_k(z_c_grid, grid);

        printf("    Pipe %d: φ=%.4f, z=%.4f, j_cen=%d, k_cen=%d, "
               "i=[%d,%d], dj=%d, dk=%d\n",
               p, phi_c, z_c, j_center, k_center,
               i_lo, i_hi, dj_max, dk_max);

        /*-----------------------------------------------------------------
         * Edge fractions: Er
         * Er(i,j,k) at r=(i+½)dr, φ=j·dφ, z=k·dz
         * Edge extends from r=a+i·dr to r=a+(i+1)·dr at fixed (φ,z)
         *-----------------------------------------------------------------*/
        for (int i = i_lo; i < dmin(i_hi, Nr); i++) {
            double r0 = grid->a + i * dr;
            double r1 = r0 + dr;

            for (int dj = -dj_max; dj <= dj_max; dj++) {
                int j = ((j_center + dj) % Nphi + Nphi) % Nphi;
                double phi = j * dphi;

                for (int dk = -dk_max; dk <= dk_max; dk++) {
                    int k = k_center + dk;
                    if (k < 0 || k > Nz) continue;
                    double z_grid = k * dz;
                    double z_phys = z_grid - z0_extension;

                    int idx = i + Nr * (j + Nphi * k);
                    double frac = edge_fraction_Er_radial_pipe(
                        r0, r1, phi, z_phys,
                        phi_c, z_c, b, R_aper, R_pipe, pipe_len);

                    /* Take maximum with existing (handles multiple pipes) */
                    if (frac > cd->edge_frac_Er[idx])
                        cd->edge_frac_Er[idx] = frac;
                }
            }
        }

        /*-----------------------------------------------------------------
         * Edge fractions: Eφ
         * Eφ(i,j,k) at r=i·dr, φ=(j+½)·dφ, z=k·dz
         * Edge extends from φ=j·dφ to φ=(j+1)·dφ at fixed (r,z)
         *-----------------------------------------------------------------*/
        for (int i = dmax(i_lo, 0); i <= dmin(i_hi, Nr); i++) {
            double r = grid->a + i * dr;

            for (int dj = -dj_max; dj <= dj_max; dj++) {
                int j = ((j_center + dj) % Nphi + Nphi) % Nphi;
                double phi0 = j * dphi;
                double phi1 = phi0 + dphi;

                for (int dk = -dk_max; dk <= dk_max; dk++) {
                    int k = k_center + dk;
                    if (k < 0 || k > Nz) continue;
                    double z_phys = k * dz - z0_extension;

                    int idx = i + Nr1 * (j + Nphi * k);
                    double frac = edge_fraction_Ephi_radial_pipe(
                        r, z_phys, phi0, phi1,
                        phi_c, z_c, b, R_aper, R_pipe, pipe_len);

                    if (frac > cd->edge_frac_Ephi[idx])
                        cd->edge_frac_Ephi[idx] = frac;
                }
            }
        }

        /*-----------------------------------------------------------------
         * Edge fractions: Ez
         * Ez(i,j,k) at r=i·dr, φ=j·dφ, z=(k+½)·dz
         * Edge extends from z=k·dz to z=(k+1)·dz at fixed (r,φ)
         *-----------------------------------------------------------------*/
        for (int i = dmax(i_lo, 0); i <= dmin(i_hi, Nr); i++) {
            double r = grid->a + i * dr;

            for (int dj = -dj_max; dj <= dj_max; dj++) {
                int j = ((j_center + dj) % Nphi + Nphi) % Nphi;
                double phi = j * dphi;

                for (int dk = -dk_max; dk <= dk_max; dk++) {
                    int k = k_center + dk;
                    if (k < 0 || k >= Nz) continue;
                    double z0 = k * dz - z0_extension;
                    double z1 = z0 + dz;

                    int idx = i + Nr1 * (j + Nphi * k);
                    double frac = edge_fraction_Ez_radial_pipe(
                        r, phi, z0, z1,
                        phi_c, z_c, b, R_aper, R_pipe, pipe_len);

                    if (frac > cd->edge_frac_Ez[idx])
                        cd->edge_frac_Ez[idx] = frac;
                }
            }
        }

        /*-----------------------------------------------------------------
         * Primal face areas: Hr
         * Hr(i,j,k) at (i, j+½, k+½) in (φ,z) plane at r = a + i·dr
         * Face: φ ∈ [j·dφ, (j+1)·dφ], z ∈ [k·dz, (k+1)·dz]
         *-----------------------------------------------------------------*/
        for (int i = dmax(i_lo, 0); i <= dmin(i_hi, Nr); i++) {
            double r = grid->a + i * dr;

            for (int dj = -dj_max; dj <= dj_max; dj++) {
                int j = ((j_center + dj) % Nphi + Nphi) % Nphi;
                double phi0 = j * dphi;
                double phi1 = phi0 + dphi;

                for (int dk = -dk_max; dk <= dk_max; dk++) {
                    int k = k_center + dk;
                    if (k < 0 || k >= Nz) continue;
                    double z0 = k * dz - z0_extension;
                    double z1 = z0 + dz;

                    int idx = i + Nr1 * (j + Nphi * k);
                    double area = face_area_Hr_radial_pipe(
                        r, phi0, phi1, z0, z1,
                        phi_c, z_c, b, R_aper, R_pipe, pipe_len);

                    /* For multiple pipes: add vacuum areas (union) */
                    /* Since pipes don't overlap, max works too */
                    if (area > cd->face_area_Hr[idx])
                        cd->face_area_Hr[idx] = area;
                }
            }
        }

        /*-----------------------------------------------------------------
         * Primal face areas: Hφ
         * Hφ(i,j,k) at (i+½, j, k+½) in (r,z) plane at φ = j·dφ
         * Face: r ∈ [a+i·dr, a+(i+1)·dr], z ∈ [k·dz, (k+1)·dz]
         *-----------------------------------------------------------------*/
        for (int i = dmax(i_lo, 0); i < dmin(i_hi, Nr); i++) {
            double r0 = grid->a + i * dr;
            double r1 = r0 + dr;

            for (int dj = -dj_max; dj <= dj_max; dj++) {
                int j = ((j_center + dj) % Nphi + Nphi) % Nphi;
                double phi = j * dphi;

                for (int dk = -dk_max; dk <= dk_max; dk++) {
                    int k = k_center + dk;
                    if (k < 0 || k >= Nz) continue;
                    double z0 = k * dz - z0_extension;
                    double z1 = z0 + dz;

                    int idx = i + Nr * (j + Nphi * k);
                    double area = face_area_Hphi_radial_pipe(
                        phi, r0, r1, z0, z1,
                        phi_c, z_c, b, R_aper, R_pipe, pipe_len);

                    if (area > cd->face_area_Hphi[idx])
                        cd->face_area_Hphi[idx] = area;
                }
            }
        }

        /*-----------------------------------------------------------------
         * Primal face areas: Hz
         * Hz(i,j,k) at (i+½, j+½, k) in (r,φ) plane at z = k·dz
         * Face: r ∈ [a+i·dr, a+(i+1)·dr], φ ∈ [j·dφ, (j+1)·dφ]
         *-----------------------------------------------------------------*/
        for (int i = dmax(i_lo, 0); i < dmin(i_hi, Nr); i++) {
            double r0 = grid->a + i * dr;
            double r1 = r0 + dr;

            for (int dj = -dj_max; dj <= dj_max; dj++) {
                int j = ((j_center + dj) % Nphi + Nphi) % Nphi;
                double phi0 = j * dphi;
                double phi1 = phi0 + dphi;

                for (int dk = -dk_max; dk <= dk_max; dk++) {
                    int k = k_center + dk;
                    if (k < 0 || k > Nz) continue;
                    double z_phys = k * dz - z0_extension;

                    int idx = i + Nr * (j + Nphi * k);
                    double area = face_area_Hz_radial_pipe(
                        z_phys, r0, r1, phi0, phi1,
                        phi_c, z_c, b, R_aper, R_pipe, pipe_len);

                    if (area > cd->face_area_Hz[idx])
                        cd->face_area_Hz[idx] = area;
                }
            }
        }

        /*-----------------------------------------------------------------
         * Dual face areas and IBC weights
         *
         * Dual faces are offset by half a cell from primal faces.
         * Use the same geometry primitives with shifted coordinates.
         *-----------------------------------------------------------------*/

        /* Er dual face at (i+½, j, k): (φ,z) plane at r = a+(i+½)dr
         * spans φ ∈ [(j-½)dφ, (j+½)dφ], z ∈ [(k-½)dz, (k+½)dz]  */
        for (int i = dmax(i_lo, 0); i < dmin(i_hi, Nr); i++) {
            double r = grid->a + (i + 0.5) * dr;

            for (int dj = -dj_max; dj <= dj_max; dj++) {
                int j = ((j_center + dj) % Nphi + Nphi) % Nphi;
                double phi_lo = (j - 0.5) * dphi;
                double phi_hi = (j + 0.5) * dphi;

                for (int dk = -dk_max; dk <= dk_max; dk++) {
                    int k = k_center + dk;
                    if (k < 0 || k > Nz) continue;
                    double z_lo = (k - 0.5) * dz - z0_extension;
                    double z_hi = (k + 0.5) * dz - z0_extension;

                    int idx = i + Nr * (j + Nphi * k);
                    double area = face_area_Hr_radial_pipe(
                        r, phi_lo, phi_hi, z_lo, z_hi,
                        phi_c, z_c, b, R_aper, R_pipe, pipe_len);

                    if (area > cd->dual_area_Er[idx])
                        cd->dual_area_Er[idx] = area;
                }
            }
        }

        /* Eφ dual face at (i, j+½, k): (r,z) plane at φ = (j+½)dφ
         * spans r ∈ [a+(i-½)dr, a+(i+½)dr], z ∈ [(k-½)dz, (k+½)dz] */
        for (int i = dmax(i_lo, 0); i <= dmin(i_hi, Nr); i++) {
            double r0 = grid->a + (i - 0.5) * dr;
            double r1 = grid->a + (i + 0.5) * dr;
            r0 = dmax(r0, grid->a);
            r1 = dmin(r1, grid->a + Nr * dr);

            for (int dj = -dj_max; dj <= dj_max; dj++) {
                int j = ((j_center + dj) % Nphi + Nphi) % Nphi;
                double phi = (j + 0.5) * dphi;

                for (int dk = -dk_max; dk <= dk_max; dk++) {
                    int k = k_center + dk;
                    if (k < 0 || k > Nz) continue;
                    double z_lo = (k - 0.5) * dz - z0_extension;
                    double z_hi = (k + 0.5) * dz - z0_extension;

                    int idx = i + Nr1 * (j + Nphi * k);
                    double area = face_area_Hphi_radial_pipe(
                        phi, r0, r1, z_lo, z_hi,
                        phi_c, z_c, b, R_aper, R_pipe, pipe_len);

                    if (area > cd->dual_area_Ephi[idx])
                        cd->dual_area_Ephi[idx] = area;
                }
            }
        }

        /* Ez dual face at (i, j, k+½): (r,φ) plane at z = (k+½)dz
         * spans r ∈ [a+(i-½)dr, a+(i+½)dr], φ ∈ [(j-½)dφ, (j+½)dφ] */
        for (int i = dmax(i_lo, 0); i <= dmin(i_hi, Nr); i++) {
            double r0 = grid->a + (i - 0.5) * dr;
            double r1 = grid->a + (i + 0.5) * dr;
            r0 = dmax(r0, grid->a);
            r1 = dmin(r1, grid->a + Nr * dr);

            for (int dj = -dj_max; dj <= dj_max; dj++) {
                int j = ((j_center + dj) % Nphi + Nphi) % Nphi;
                double phi_lo = (j - 0.5) * dphi;
                double phi_hi = (j + 0.5) * dphi;

                for (int dk = -dk_max; dk <= dk_max; dk++) {
                    int k = k_center + dk;
                    if (k < 0 || k >= Nz) continue;
                    double z_phys = (k + 0.5) * dz - z0_extension;

                    int idx = i + Nr1 * (j + Nphi * k);
                    double area = face_area_Hz_radial_pipe(
                        z_phys, r0, r1, phi_lo, phi_hi,
                        phi_c, z_c, b, R_aper, R_pipe, pipe_len);

                    if (area > cd->dual_area_Ez[idx])
                        cd->dual_area_Ez[idx] = area;
                }
            }
        }

        /*-----------------------------------------------------------------
         * IBC weights: wall arc length / dual cell volume
         *
         * For tangential E components at the pipe wall boundary:
         * Eφ and Ez are tangential to a cylindrical pipe wall.
         * Er is normal to the wall and doesn't get IBC.
         *
         * IBC weight for Eφ at cut cell = wall_arc_in_dual / V_dual
         * IBC weight for Ez at cut cell = wall_arc_in_dual / V_dual
         *
         * For now, compute using the Hr-face arc length at the edge's r.
         *-----------------------------------------------------------------*/
        for (int i = dmax(i_lo, 0); i <= dmin(i_hi, Nr); i++) {
            double r = grid->a + i * dr;
            if (r <= r_outer) continue;

            for (int dj = -dj_max; dj <= dj_max; dj++) {
                int j = ((j_center + dj) % Nphi + Nphi) % Nphi;

                for (int dk = -dk_max; dk <= dk_max; dk++) {
                    int k = k_center + dk;
                    if (k < 0 || k > Nz) continue;

                    /* Eφ at (i, j+½, k): IBC weight */
                    {
                        int idx = i + Nr1 * (j + Nphi * k);
                        double frac = cd->edge_frac_Ephi[idx];
                        if (frac > 0.0 && frac < 1.0) {
                            /* Wall passes through this edge's dual cell */
                            double phi_lo = (j) * dphi;
                            double phi_hi = (j + 1) * dphi;
                            double z_lo = (k - 0.5) * dz - z0_extension;
                            double z_hi = (k + 0.5) * dz - z0_extension;

                            double arc_len = wall_arc_length_Hr(
                                r, phi_lo, phi_hi, z_lo, z_hi,
                                phi_c, z_c, b, R_aper, R_pipe, pipe_len);

                            /* Dual cell volume = dr · dz · r·dφ
                             * But we want weight = arc_len · dr_extent / V_dual
                             * For a cylindrical wall: IBC surface element = arc_len · dr
                             * V_dual = dr · r·dφ · dz (but cut by pipe, use dual_area) */
                            double V_dual = cd->dual_area_Ephi[idx] * dr;
                            if (V_dual > 1e-30) {
                                double w = arc_len * dr / V_dual;
                                cd->ibc_weight_Ephi[idx] = dmax(cd->ibc_weight_Ephi[idx], w);
                            }
                        }
                    }

                    /* Ez at (i, j, k+½): IBC weight — only for k < Nz */
                    if (k < Nz) {
                        int idx = i + Nr1 * (j + Nphi * k);
                        double frac = cd->edge_frac_Ez[idx];
                        if (frac > 0.0 && frac < 1.0) {
                            double phi_lo = (j - 0.5) * dphi;
                            double phi_hi = (j + 0.5) * dphi;
                            double z_lo = k * dz - z0_extension;
                            double z_hi = (k + 1) * dz - z0_extension;

                            double arc_len = wall_arc_length_Hr(
                                r, phi_lo, phi_hi, z_lo, z_hi,
                                phi_c, z_c, b, R_aper, R_pipe, pipe_len);

                            double V_dual = cd->dual_area_Ez[idx] * dr;
                            if (V_dual > 1e-30) {
                                double w = arc_len * dr / V_dual;
                                cd->ibc_weight_Ez[idx] = dmax(cd->ibc_weight_Ez[idx], w);
                            }
                        }
                    }
                }
            }
        }

    }  /* end loop over pipes */

    /*=========================================================================
     * Pass 3: Apply area threshold (stability)
     *
     * If a face's vacuum area is less than DM_AREA_THRESHOLD of the
     * standard area, treat it as fully PEC (zero area).
     *========================================================================*/

    for (int idx = 0; idx < cd->size_Hr; idx++) {
        int i = idx % Nr1;
        double r = grid->a + i * dr;
        double std = r * dphi * dz;
        if (cd->face_area_Hr[idx] > 0.0 && cd->face_area_Hr[idx] < DM_AREA_THRESHOLD * std) {
            cd->face_area_Hr[idx] = 0.0;
        }
    }
    for (int idx = 0; idx < cd->size_Hphi; idx++) {
        double std = dr * dz;
        if (cd->face_area_Hphi[idx] > 0.0 && cd->face_area_Hphi[idx] < DM_AREA_THRESHOLD * std) {
            cd->face_area_Hphi[idx] = 0.0;
        }
    }
    for (int idx = 0; idx < cd->size_Hz; idx++) {
        int i = idx % Nr;
        double r_mid = grid->a + (i + 0.5) * dr;
        double std = r_mid * dphi * dr;
        if (cd->face_area_Hz[idx] > 0.0 && cd->face_area_Hz[idx] < DM_AREA_THRESHOLD * std) {
            cd->face_area_Hz[idx] = 0.0;
        }
    }
    /* Same for dual areas */
    for (int idx = 0; idx < cd->size_Er; idx++) {
        int i = idx % Nr;
        double r = grid->a + (i + 0.5) * dr;
        double std = r * dphi * dz;
        if (cd->dual_area_Er[idx] > 0.0 && cd->dual_area_Er[idx] < DM_AREA_THRESHOLD * std)
            cd->dual_area_Er[idx] = 0.0;
    }
    for (int idx = 0; idx < cd->size_Ephi; idx++) {
        double std = dr * dz;
        if (cd->dual_area_Ephi[idx] > 0.0 && cd->dual_area_Ephi[idx] < DM_AREA_THRESHOLD * std)
            cd->dual_area_Ephi[idx] = 0.0;
    }
    for (int idx = 0; idx < cd->size_Ez; idx++) {
        int i = idx % Nr1;
        double r = grid->a + i * dr;
        double std = r * dphi * dr;
        if (cd->dual_area_Ez[idx] > 0.0 && cd->dual_area_Ez[idx] < DM_AREA_THRESHOLD * std)
            cd->dual_area_Ez[idx] = 0.0;
    }

    /*=========================================================================
     * Pass 4: Count cut edges and IBC boundary cells
     *========================================================================*/

    for (int idx = 0; idx < cd->size_Er; idx++) {
        double f = cd->edge_frac_Er[idx];
        if (f > 0.0 && f < 1.0) cd->num_cut_Er++;
        if (cd->ibc_weight_Er[idx] > 0.0) cd->num_ibc_Er++;
    }
    for (int idx = 0; idx < cd->size_Ephi; idx++) {
        double f = cd->edge_frac_Ephi[idx];
        if (f > 0.0 && f < 1.0) cd->num_cut_Ephi++;
        if (cd->ibc_weight_Ephi[idx] > 0.0) cd->num_ibc_Ephi++;
    }
    for (int idx = 0; idx < cd->size_Ez; idx++) {
        double f = cd->edge_frac_Ez[idx];
        if (f > 0.0 && f < 1.0) cd->num_cut_Ez++;
        if (cd->ibc_weight_Ez[idx] > 0.0) cd->num_ibc_Ez++;
    }
}

/*=============================================================================
 * Build with endcap pipes (Phase 4 — stub for now)
 *============================================================================*/

void conformal_data_build_full(
    ConformalData* cd,
    const PipeConfig* radial_pipes,
    const EndcapPipeConfig* endcap_pipes,
    const GridParams* grid,
    double r_outer,
    double L_cavity,
    double z0_extension
) {
    /* Start with radial pipes */
    conformal_data_build(cd, radial_pipes, grid, r_outer, z0_extension);

    /* TODO (Phase 4): process endcap pipes similarly */
    if (endcap_pipes && endcap_pipes->num_pipes > 0) {
        printf("  WARNING: Endcap pipe conformal geometry not yet implemented.\n");
        printf("           Using staircase approximation for %d endcap pipes.\n",
               endcap_pipes->num_pipes);
    }
}

/*=============================================================================
 * IBC Unmask: fix conformal data at flat-wall surface cells
 *
 * The IBC method unmarks surface PEC cells so E_tan is nonzero there.
 * The conformal curl must also "see" these cells (edge_frac=1, standard
 * face areas), otherwise the curl-curl operator has zero rows at surface
 * cells, forcing the eigenvector to E=0 there, which kills the IBC
 * contribution from flat walls.
 *
 * This function sets edge_frac=1 at IBC surface cells and restores
 * standard face/dual areas at faces bordering those cells.
 *============================================================================*/

void conformal_data_apply_ibc_unmask(
    ConformalData* cd,
    const MaterialMask* pec_mask,
    const MaterialMask* ibc_mask,
    const GridParams* grid
) {
    int Nr = grid->Nr, Nphi = grid->Nphi, Nz = grid->Nz;
    int Nr1 = Nr + 1;
    double dr = grid->dr, dphi = grid->dphi, dz = grid->dz;
    int unmasked_Er = 0, unmasked_Ephi = 0, unmasked_Ez = 0;

    /* --- Pass 1: Set edge_frac = 1 and dual_area = standard at surface cells --- */

    /* Er(i,j,k) */
    for (int idx = 0; idx < cd->size_Er; idx++) {
        if (pec_mask->mask_Er[idx] == 0 && ibc_mask->mask_Er[idx] == 1) {
            /* IBC surface cell: was PEC, now unmasked */
            if (cd->edge_frac_Er[idx] < 1.0) {
                cd->edge_frac_Er[idx] = 1.0;
                /* Dual area: (φ,z) plane at r_{i+½} */
                int i = idx % Nr;
                double r = grid->a + (i + 0.5) * dr;
                cd->dual_area_Er[idx] = r * dphi * dz;
                unmasked_Er++;
            }
        }
    }

    /* Ephi(i,j,k) */
    for (int idx = 0; idx < cd->size_Ephi; idx++) {
        if (pec_mask->mask_Ephi[idx] == 0 && ibc_mask->mask_Ephi[idx] == 1) {
            if (cd->edge_frac_Ephi[idx] < 1.0) {
                cd->edge_frac_Ephi[idx] = 1.0;
                /* Dual area: (r,z) plane */
                cd->dual_area_Ephi[idx] = dr * dz;
                unmasked_Ephi++;
            }
        }
    }

    /* Ez(i,j,k) */
    for (int idx = 0; idx < cd->size_Ez; idx++) {
        if (pec_mask->mask_Ez[idx] == 0 && ibc_mask->mask_Ez[idx] == 1) {
            if (cd->edge_frac_Ez[idx] < 1.0) {
                cd->edge_frac_Ez[idx] = 1.0;
                /* Dual area: (r,φ) plane at r_i */
                int i = idx % Nr1;
                double r = grid->a + i * dr;
                cd->dual_area_Ez[idx] = r * dphi * dr;
                unmasked_Ez++;
            }
        }
    }

    printf("  IBC unmask: set edge_frac=1 at %d Er + %d Ephi + %d Ez = %d surface cells\n",
           unmasked_Er, unmasked_Ephi, unmasked_Ez,
           unmasked_Er + unmasked_Ephi + unmasked_Ez);

    /* --- Pass 2: Restore face areas ONLY at cavity-side H-faces ---
     *
     * A face is "cavity-side" if it has at least one bordering edge that
     * was ORIGINALLY VACUUM in the PEC mask. Faces where ALL bordering
     * edges are PEC/unmasked (PEC-body side) stay at face_area=0.
     * This makes surface cells one-sided (coupled only to cavity),
     * exactly like grid-edge boundary cells in the reference cavity.
     */

    if (unmasked_Er + unmasked_Ephi + unmasked_Ez == 0) return;

    /* Build fast-lookup: was this edge just unmasked? */
    int* um_Er   = (int*)calloc(cd->size_Er,   sizeof(int));
    int* um_Ephi = (int*)calloc(cd->size_Ephi, sizeof(int));
    int* um_Ez   = (int*)calloc(cd->size_Ez,   sizeof(int));

    for (int idx = 0; idx < cd->size_Er; idx++)
        if (pec_mask->mask_Er[idx] == 0 && ibc_mask->mask_Er[idx] == 1)
            um_Er[idx] = 1;
    for (int idx = 0; idx < cd->size_Ephi; idx++)
        if (pec_mask->mask_Ephi[idx] == 0 && ibc_mask->mask_Ephi[idx] == 1)
            um_Ephi[idx] = 1;
    for (int idx = 0; idx < cd->size_Ez; idx++)
        if (pec_mask->mask_Ez[idx] == 0 && ibc_mask->mask_Ez[idx] == 1)
            um_Ez[idx] = 1;

    int fixed_Hr = 0, fixed_Hphi = 0, fixed_Hz = 0;

    /* Hr faces: Hr(i,j,k) at (i, j+½, k+½), i=0..Nr, j=0..Nphi-1, k=0..Nz-1
     * Bordering edges: Ez(i,j+1,k), Ez(i,j,k), Ephi(i,j,k+1), Ephi(i,j,k) */
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            int jp1 = (j + 1) % Nphi;
            for (int i = 0; i <= Nr; i++) {
                int fidx = i + Nr1 * (j + Nphi * k);
                if (cd->face_area_Hr[fidx] > 0.0) continue;

                int ez_j   = i + Nr1 * (j   + Nphi * k);
                int ez_jp1 = i + Nr1 * (jp1 + Nphi * k);
                int ep_k   = i + Nr1 * (j + Nphi * k);
                int ep_kp1 = i + Nr1 * (j + Nphi * (k + 1));

                int has_unmasked = (um_Ez[ez_j] || um_Ez[ez_jp1] ||
                                    um_Ephi[ep_k] || um_Ephi[ep_kp1]);
                int has_orig_vac = (pec_mask->mask_Ez[ez_j] == 1 ||
                                    pec_mask->mask_Ez[ez_jp1] == 1 ||
                                    pec_mask->mask_Ephi[ep_k] == 1 ||
                                    pec_mask->mask_Ephi[ep_kp1] == 1);

                if (has_unmasked && has_orig_vac) {
                    double r = grid->a + i * dr;
                    cd->face_area_Hr[fidx] = r * dphi * dz;
                    fixed_Hr++;
                }
            }
        }
    }

    /* Hphi faces: Hphi(i,j,k) at (i+½, j, k+½), i=0..Nr-1, j=0..Nphi-1, k=0..Nz-1
     * Bordering edges: Er(i,j,k+1), Er(i,j,k), Ez(i+1,j,k), Ez(i,j,k) */
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr; i++) {
                int fidx = i + Nr * (j + Nphi * k);
                if (cd->face_area_Hphi[fidx] > 0.0) continue;

                int er_k   = i + Nr * (j + Nphi * k);
                int er_kp1 = i + Nr * (j + Nphi * (k + 1));
                int ez_i   = i     + Nr1 * (j + Nphi * k);
                int ez_ip1 = (i+1) + Nr1 * (j + Nphi * k);

                int has_unmasked = (um_Er[er_k] || um_Er[er_kp1] ||
                                    um_Ez[ez_i] || um_Ez[ez_ip1]);
                int has_orig_vac = (pec_mask->mask_Er[er_k] == 1 ||
                                    pec_mask->mask_Er[er_kp1] == 1 ||
                                    pec_mask->mask_Ez[ez_i] == 1 ||
                                    pec_mask->mask_Ez[ez_ip1] == 1);

                if (has_unmasked && has_orig_vac) {
                    cd->face_area_Hphi[fidx] = dr * dz;
                    fixed_Hphi++;
                }
            }
        }
    }

    /* Hz faces: Hz(i,j,k) at (i+½, j+½, k), i=0..Nr-1, j=0..Nphi-1, k=0..Nz
     * Bordering edges: Ephi(i+1,j,k), Ephi(i,j,k), Er(i,j+1,k), Er(i,j,k) */
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            int jp1 = (j + 1) % Nphi;
            for (int i = 0; i < Nr; i++) {
                int fidx = i + Nr * (j + Nphi * k);
                if (cd->face_area_Hz[fidx] > 0.0) continue;

                int ep_i   = i     + Nr1 * (j + Nphi * k);
                int ep_ip1 = (i+1) + Nr1 * (j + Nphi * k);
                int er_j   = i + Nr * (j   + Nphi * k);
                int er_jp1 = i + Nr * (jp1 + Nphi * k);

                int has_unmasked = (um_Ephi[ep_i] || um_Ephi[ep_ip1] ||
                                    um_Er[er_j] || um_Er[er_jp1]);
                int has_orig_vac = (pec_mask->mask_Ephi[ep_i] == 1 ||
                                    pec_mask->mask_Ephi[ep_ip1] == 1 ||
                                    pec_mask->mask_Er[er_j] == 1 ||
                                    pec_mask->mask_Er[er_jp1] == 1);

                if (has_unmasked && has_orig_vac) {
                    double r_mid = grid->a + (i + 0.5) * dr;
                    cd->face_area_Hz[fidx] = r_mid * dphi * dr;
                    fixed_Hz++;
                }
            }
        }
    }

    printf("  Face area fixup: %d Hr + %d Hphi + %d Hz faces restored (cavity-side only)\n",
           fixed_Hr, fixed_Hphi, fixed_Hz);

    free(um_Er);
    free(um_Ephi);
    free(um_Ez);
}

/*=============================================================================
 * Free
 *============================================================================*/

void conformal_data_free(ConformalData* cd) {
    free(cd->edge_frac_Er);    cd->edge_frac_Er = NULL;
    free(cd->edge_frac_Ephi);  cd->edge_frac_Ephi = NULL;
    free(cd->edge_frac_Ez);    cd->edge_frac_Ez = NULL;

    free(cd->face_area_Hr);    cd->face_area_Hr = NULL;
    free(cd->face_area_Hphi);  cd->face_area_Hphi = NULL;
    free(cd->face_area_Hz);    cd->face_area_Hz = NULL;

    free(cd->dual_area_Er);    cd->dual_area_Er = NULL;
    free(cd->dual_area_Ephi);  cd->dual_area_Ephi = NULL;
    free(cd->dual_area_Ez);    cd->dual_area_Ez = NULL;

    free(cd->ibc_weight_Er);   cd->ibc_weight_Er = NULL;
    free(cd->ibc_weight_Ephi); cd->ibc_weight_Ephi = NULL;
    free(cd->ibc_weight_Ez);   cd->ibc_weight_Ez = NULL;
}

/*=============================================================================
 * Diagnostics
 *============================================================================*/

void conformal_data_print_stats(const ConformalData* cd, const GridParams* grid) {
    int total_edges = cd->size_Er + cd->size_Ephi + cd->size_Ez;
    int total_cut = cd->num_cut_Er + cd->num_cut_Ephi + cd->num_cut_Ez;
    int total_ibc = cd->num_ibc_Er + cd->num_ibc_Ephi + cd->num_ibc_Ez;

    /* Count fully-vacuum and fully-PEC edges */
    int vac_Er = 0, pec_Er = 0;
    for (int i = 0; i < cd->size_Er; i++) {
        if (cd->edge_frac_Er[i] >= 1.0) vac_Er++;
        else if (cd->edge_frac_Er[i] <= 0.0) pec_Er++;
    }
    int vac_Ephi = 0, pec_Ephi = 0;
    for (int i = 0; i < cd->size_Ephi; i++) {
        if (cd->edge_frac_Ephi[i] >= 1.0) vac_Ephi++;
        else if (cd->edge_frac_Ephi[i] <= 0.0) pec_Ephi++;
    }
    int vac_Ez = 0, pec_Ez = 0;
    for (int i = 0; i < cd->size_Ez; i++) {
        if (cd->edge_frac_Ez[i] >= 1.0) vac_Ez++;
        else if (cd->edge_frac_Ez[i] <= 0.0) pec_Ez++;
    }

    printf("\n  Conformal Geometry Statistics:\n");
    printf("    Er:   %d vacuum / %d cut / %d PEC  (of %d)\n",
           vac_Er, cd->num_cut_Er, pec_Er, cd->size_Er);
    printf("    Ephi: %d vacuum / %d cut / %d PEC  (of %d)\n",
           vac_Ephi, cd->num_cut_Ephi, pec_Ephi, cd->size_Ephi);
    printf("    Ez:   %d vacuum / %d cut / %d PEC  (of %d)\n",
           vac_Ez, cd->num_cut_Ez, pec_Ez, cd->size_Ez);
    printf("    Total edges: %d, cut: %d (%.3f%%)\n",
           total_edges, total_cut, 100.0 * total_cut / total_edges);
    printf("    IBC boundary: Er=%d, Ephi=%d, Ez=%d, total=%d\n",
           cd->num_ibc_Er, cd->num_ibc_Ephi, cd->num_ibc_Ez, total_ibc);

    /* Memory usage */
    size_t bytes = (cd->size_Er + cd->size_Ephi + cd->size_Ez) * 4 * sizeof(double)
                 + (cd->size_Hr + cd->size_Hphi + cd->size_Hz) * sizeof(double)
                 + (cd->size_Er + cd->size_Ephi + cd->size_Ez) * sizeof(double);
    printf("    Memory: %.1f MB (CPU)\n", bytes / (1024.0 * 1024.0));
}

/*=============================================================================
 * Self-test: circle-rectangle intersection
 *============================================================================*/

int conformal_test_primitives(void) {
    int pass = 1;
    double tol = 1e-10;

    printf("\n  Testing circle-rect intersection primitives...\n");

    /* Test 1: Circle fully inside rectangle */
    {
        double area = circle_rect_intersection_area(-5, 5, -5, 5, 0, 0, 1.0);
        double expected = M_PI;  /* π·R² */
        double err = fabs(area - expected) / expected;
        printf("    Circle in rect: area=%.10f, expected=%.10f, err=%.2e %s\n",
               area, expected, err, err < tol ? "PASS" : "FAIL");
        if (err >= tol) pass = 0;
    }

    /* Test 2: Rectangle fully inside circle */
    {
        double area = circle_rect_intersection_area(-0.1, 0.1, -0.1, 0.1, 0, 0, 10.0);
        double expected = 0.04;
        double err = fabs(area - expected) / expected;
        printf("    Rect in circle: area=%.10f, expected=%.10f, err=%.2e %s\n",
               area, expected, err, err < tol ? "PASS" : "FAIL");
        if (err >= tol) pass = 0;
    }

    /* Test 3: Half circle (clipped by y = 0) */
    {
        double area = circle_rect_intersection_area(-2, 2, 0, 2, 0, 0, 1.0);
        double expected = M_PI / 2.0;
        double err = fabs(area - expected) / expected;
        printf("    Half circle:    area=%.10f, expected=%.10f, err=%.2e %s\n",
               area, expected, err, err < tol ? "PASS" : "FAIL");
        if (err >= tol) pass = 0;
    }

    /* Test 4: Quarter circle (clipped by x=0 and y=0) */
    {
        double area = circle_rect_intersection_area(0, 2, 0, 2, 0, 0, 1.0);
        double expected = M_PI / 4.0;
        double err = fabs(area - expected) / expected;
        printf("    Quarter circle: area=%.10f, expected=%.10f, err=%.2e %s\n",
               area, expected, err, err < tol ? "PASS" : "FAIL");
        if (err >= tol) pass = 0;
    }

    /* Test 5: No overlap */
    {
        double area = circle_rect_intersection_area(5, 6, 5, 6, 0, 0, 1.0);
        printf("    No overlap:     area=%.10f, expected=0 %s\n",
               area, area < 1e-15 ? "PASS" : "FAIL");
        if (area >= 1e-15) pass = 0;
    }

    /* Test 6: Circle at non-origin */
    {
        double area = circle_rect_intersection_area(-5, 15, -5, 15, 3.0, 4.0, 2.0);
        double expected = M_PI * 4.0;  /* fully inside rect */
        double err = fabs(area - expected) / expected;
        printf("    Off-center:     area=%.10f, expected=%.10f, err=%.2e %s\n",
               area, expected, err, err < tol ? "PASS" : "FAIL");
        if (err >= tol) pass = 0;
    }

    /* Test 7: Edge fraction — Eφ fully inside pipe */
    {
        double frac = edge_fraction_Ephi_radial_pipe(
            1.05, 0.6975,    /* r=1.05 (past wall at 1.0), z = z_c */
            0.0, 0.01,       /* small φ range near pipe center */
            0.005, 0.6975,   /* pipe at φ=0.005, z=0.6975 */
            1.0, 0.0175, 0.0125, 0.050);
        printf("    Ephi inside:    frac=%.6f %s\n",
               frac, frac > 0.99 ? "PASS" : "FAIL");
        if (frac < 0.99) pass = 0;
    }

    /* Test 8: Edge fraction — Eφ fully outside pipe */
    {
        double frac = edge_fraction_Ephi_radial_pipe(
            1.05, 0.6975,
            1.0, 1.01,       /* far from pipe in φ */
            0.005, 0.6975,
            1.0, 0.0175, 0.0125, 0.050);
        printf("    Ephi outside:   frac=%.6f %s\n",
               frac, frac < 0.01 ? "PASS" : "FAIL");
        if (frac >= 0.01) pass = 0;
    }

    printf("  Primitive tests: %s\n\n", pass ? "ALL PASSED" : "SOME FAILED");
    return pass;
}

void conformal_data_zero_endplate_faces(
    ConformalData* cd,
    const GridParams* grid,
    int k_z0, int k_zL
) {
    int Nr = grid->Nr, Nr1 = Nr + 1, Nphi = grid->Nphi;
    int zeroed = 0;

    /* Hr(i,j,k) at (i, j+½, k+½): face between k and k+1 */
    /* Hphi(i,j,k) at (i+½, j, k+½): face between k and k+1 */

    /* Zero faces at k = k_z0-1 (between conductor body and endplate) */
    if (k_z0 >= 1) {
        int kf = k_z0 - 1;
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i <= Nr; i++) {
                int idx_hr = i + Nr1 * (j + Nphi * kf);
                if (cd->face_area_Hr[idx_hr] > 0.0) {
                    cd->face_area_Hr[idx_hr] = 0.0;
                    zeroed++;
                }
            }
            for (int i = 0; i < Nr; i++) {
                int idx_hp = i + Nr * (j + Nphi * kf);
                if (cd->face_area_Hphi[idx_hp] > 0.0) {
                    cd->face_area_Hphi[idx_hp] = 0.0;
                    zeroed++;
                }
            }
        }
    }

    /* Zero faces at k = k_zL (between endplate and conductor body) */
    if (k_zL >= 0) {
        int kf = k_zL;
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i <= Nr; i++) {
                int idx_hr = i + Nr1 * (j + Nphi * kf);
                if (cd->face_area_Hr[idx_hr] > 0.0) {
                    cd->face_area_Hr[idx_hr] = 0.0;
                    zeroed++;
                }
            }
            for (int i = 0; i < Nr; i++) {
                int idx_hp = i + Nr * (j + Nphi * kf);
                if (cd->face_area_Hphi[idx_hp] > 0.0) {
                    cd->face_area_Hphi[idx_hp] = 0.0;
                    zeroed++;
                }
            }
        }
    }

    printf("  Endplate face zeroing: %d faces set to 0 (making curl one-sided)\n", zeroed);
}