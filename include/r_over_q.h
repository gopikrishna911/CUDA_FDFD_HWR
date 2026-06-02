#ifndef R_OVER_Q_H
#define R_OVER_Q_H

#include "curlcurl_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*=============================================================================
     * R/Q (Shunt Impedance per unit Q) for HWR Cavity
     *
     * In this HWR the beam travels RADIALLY, not axially. The accelerating
     * voltage per gap crossing is:
     *
     *     V_gap = integral_a^b  E_r(r, phi_beam, z_mid) dr
     *
     * evaluated at the midplane z = L/2 where E_r is maximum for the
     * fundamental TEM-like half-wave mode.
     *
     * R/Q is a purely geometric quantity (independent of wall material):
     *
     *   Linac convention:    R/Q = V^2 / (omega * W)       [Ohm]
     *   Circuit convention:  R/Q = V^2 / (2 * omega * W)   [Ohm]
     *
     * where W = (eps_0 / 2) * integral |E|^2 r dr dphi dz  is the stored
     * energy and omega is the angular resonant frequency.
     *
     * For the analytical TEM coaxial HWR (no pipes):
     *
     *   R/Q_linac  = 2 * eta_0 * ln(b/a) / pi^2
     *   R/Q_circuit = eta_0 * ln(b/a) / pi^2
     *
     * where eta_0 = mu_0 * c ~ 376.73 Ohm.
     * These are PER SINGLE GAP CROSSING (a -> b or b -> a).
     *
     * Per pass (2 crossings, voltage doubles):
     *   R/Q_pass = 4 * R/Q_crossing    (since (2V)^2 = 4V^2)
     *
     * NOTES:
     *   - The voltage V here is the "peak voltage" (V_0), i.e. the integral
     *     at the instant of maximum field. The transit time factor (TTF)
     *     is NOT included — it depends on beam energy and is applied
     *     separately.
     *   - For the unperturbed TEM mode, all crossings see the same voltage.
     *     With beam pipes/apertures, there are small azimuthal variations
     *     which this code resolves per crossing.
     *============================================================================*/

    /*=============================================================================
     * Physical constants (same as q_factor.h)
     *============================================================================*/
#define ROQ_C0      299792458.0
#define ROQ_MU0     (4.0e-7 * M_PI)
#define ROQ_EPS0    (1.0 / (ROQ_MU0 * ROQ_C0 * ROQ_C0))
#define ROQ_ETA0    (ROQ_MU0 * ROQ_C0)

    /*=============================================================================
     * Per-crossing voltage result
     *============================================================================*/
    typedef struct {
        double V_gap;       /* Gap voltage [V, arb normalization] */
        double phi;         /* Azimuthal angle of beam path [rad] */
        int pass;           /* Pass number (0-indexed) */
        int is_exit;        /* 0 = entry crossing, 1 = exit crossing */
    } CrossingVoltage;

    /*=============================================================================
     * R/Q result structure
     *============================================================================*/
    typedef struct {
        /* Eigenmode properties */
        double k_squared;
        double frequency_Hz;
        double omega;

        /* Stored energy (from weighted inner product) */
        double W_stored;

        /* Per-crossing breakdown */
        int n_crossings;
        CrossingVoltage* crossings;     /* Array [n_crossings] */

        /* Per-pass breakdown (each pass = entry + exit crossing) */
        int n_passes;
        double* V_pass;                 /* Voltage per pass (entry + exit) */
        double* R_over_Q_pass_linac;    /* R/Q per pass, linac convention */

        /* Single-crossing statistics */
        double V_gap_avg;               /* Average gap voltage */
        double V_gap_min;
        double V_gap_max;
        double V_gap_spread;            /* (max - min) / avg as percentage */

        /* R/Q per single crossing (averaged over all crossings) */
        double R_over_Q_crossing_linac;     /* V_avg^2 / (omega * W) */
        double R_over_Q_crossing_circuit;   /* V_avg^2 / (2 * omega * W) */

        /* R/Q per pass (averaged) */
        double R_over_Q_per_pass_linac;     /* (2*V_avg)^2 / (omega * W) */
        double R_over_Q_per_pass_circuit;

        /* Total machine R/Q: all crossings coherently summed
         * V_total = sum of all V_gap (beam accelerated each crossing)
         * R/Q_total = V_total^2 / (omega * W)                         */
        double V_total;
        double R_over_Q_total_linac;
        double R_over_Q_total_circuit;

        /* Analytical comparison (TEM coaxial HWR, per crossing) */
        double R_over_Q_analytical_linac;
        double R_over_Q_analytical_circuit;
    } RoverQResult;

    /*=============================================================================
     * Compute gap voltage along a single radial line
     *
     * Integrates E_r from r=a to r=b_cavity at fixed (phi, z_phys).
     * Uses bilinear interpolation in (phi, z) for sub-cell accuracy.
     *
     * Parameters:
     *   op          - Operator (provides grid, offsets, field indexing)
     *   eigenvector - Packed E-field from eigensolver
     *   phi         - Azimuthal angle of beam path [rad]
     *   z_phys      - Physical axial position [m] (typically L/2)
     *   z0_offset   - Grid z-offset for extended grids (0 if not extended)
     *   b_cavity    - Physical outer conductor radius [m]
     *
     * Returns the gap voltage (trapezoidal integral of E_r * dr).
     *============================================================================*/
    double compute_gap_voltage(
        const CurlCurlOperator* op,
        const double* eigenvector,
        double phi,
        double z_phys,
        double z0_offset,
        double b_cavity
    );

    /*=============================================================================
     * Main R/Q computation for multi-pass HWR
     *
     * Computes V_gap for every beam crossing (2 per pass) and assembles
     * R/Q in both linac and circuit conventions.
     *
     * Parameters:
     *   op          - Curl-curl operator
     *   eigenvector - Packed E-field eigenvector
     *   eigenvalue  - k^2 from Rayleigh quotient
     *   b_cavity    - Physical outer conductor radius [m]
     *   L_cavity    - Physical cavity length [m]
     *   z0_offset   - Grid z-offset for extended grids
     *   num_passes  - Number of beam passes (entry/exit pair each)
     *
     * The beam geometry follows the standard multi-pass radial-port layout:
     *   Pass i: entry at phi = i * pi/num_passes
     *           exit  at phi = entry + pi
     * All crossings at z = L_cavity / 2 (midplane).
     *============================================================================*/
    RoverQResult compute_r_over_q(
        const CurlCurlOperator* op,
        const double* eigenvector,
        double eigenvalue,
        double b_cavity,
        double L_cavity,
        double z0_offset,
        int num_passes
    );

    /*=============================================================================
     * Analytical R/Q for coaxial HWR (TEM mode, per crossing)
     *
     * R/Q_linac  = 2 * eta_0 * ln(b/a) / pi^2
     * R/Q_circuit = eta_0 * ln(b/a) / pi^2
     *
     * Independent of frequency and cavity length.
     *============================================================================*/
    RoverQResult compute_r_over_q_analytical(double a, double b, double L);

    /*=============================================================================
     * Cleanup and output
     *============================================================================*/
    void r_over_q_free(RoverQResult* result);

    void r_over_q_print(const RoverQResult* result, const char* label);

    void r_over_q_print_comparison(
        const RoverQResult* analytical,
        const RoverQResult* numerical
    );

#ifdef __cplusplus
}
#endif

#endif /* R_OVER_Q_H */
