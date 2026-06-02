#ifndef Q_FACTOR_H
#define Q_FACTOR_H

#include "curlcurl_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*=============================================================================
     * Physical Constants
     *============================================================================*/
#define Q_C0          299792458.0                       /* Speed of light [m/s] */
#define Q_MU0         (4.0e-7 * M_PI)                  /* Permeability [H/m] */
#define Q_EPS0        (1.0 / (Q_MU0 * Q_C0 * Q_C0))   /* Permittivity [F/m] */
#define Q_ETA0        (Q_MU0 * Q_C0)                   /* Free-space impedance [Ohm] ~376.73 */
#define Q_SIGMA_CU    5.8e7                             /* Copper conductivity [S/m] */

     /*=============================================================================
      * Q Factor Result (wall losses)
      *
      * Q_0 = omega * W_stored / P_wall
      *
      * W_stored = eps_0 * integral |E|^2 r dr dphi dz   (= total energy at resonance)
      * P_wall   = (R_s / 2) * surface_integral |H_tan|^2 dS
      * R_s      = sqrt(omega * mu_0 / (2 * sigma))
      * H_tan    = curl(E)_tan / (omega * mu_0)  (from time-harmonic Maxwell)
      *============================================================================*/
    typedef struct {
        /* Mode properties */
        double k_squared;           /* Eigenvalue k^2 [1/m^2] */
        double frequency_Hz;        /* Resonant frequency [Hz] */
        double omega;               /* Angular frequency [rad/s] */
        double wavelength;          /* Free-space wavelength [m] */

        /* Energy and losses (in arbitrary normalization of E) */
        double W_stored;            /* Total stored energy [arb] */
        double P_wall_total;        /* Total wall power loss [arb] */
        double P_inner;             /* Loss on inner conductor (r=a) */
        double P_outer;             /* Loss on outer conductor (r=b_cavity) */
        double P_endplate_z0;       /* Loss on z=0 endplate */
        double P_endplate_zL;       /* Loss on z=L endplate */

        /* Material properties used */
        double conductivity;        /* Wall conductivity [S/m] */
        double R_s;                 /* Surface resistance [Ohm] */
        double skin_depth;          /* Skin depth [m] */

        /* Quality factor */
        double Q_0;                 /* Unloaded Q factor */

        /* Geometry factor G = Q_0 * R_s  [Ohm]
         * Material-independent figure of merit.
         * For any wall material: Q_0 = G / R_s */
        double G_factor;
    } QFactorResult;

    /*=============================================================================
     * Main Q factor computation (wall losses, CPU)
     *
     * Surface integrals use linear extrapolation from the two nearest H-field
     * samples to the actual wall location, giving O(dr^2) accuracy.
     *============================================================================*/
    QFactorResult compute_q_factor(
        const CurlCurlOperator* op,
        const double* eigenvector,
        double eigenvalue,
        double conductivity,
        double b_cavity,
        const PortConfig* apertures
    );

    /* Q factor for extended grid with z-pipes */
    QFactorResult compute_q_factor_extended(
        const CurlCurlOperator* op,
        const double* eigenvector,
        double eigenvalue,
        double conductivity,
        double b_cavity,
        const PortConfig* apertures,
        int k_endplate_z0,      /* k-index of physical z=0 endplate */
        int k_endplate_zL       /* k-index of physical z=L endplate */
    );

    /*=============================================================================
     * Analytical Q_0 for coaxial half-wave resonator (TEM mode)
     *============================================================================*/
    QFactorResult compute_q_analytical_coaxial_hwr(
        double a,
        double b,
        double L,
        double conductivity
    );

    /*=============================================================================
     * Surface loss integrals (exposed for testing/debugging)
     *============================================================================*/
    double surface_integral_inner(
        const CurlE* curlE, const GridParams* grid,
        const PortConfig* apertures
    );

    double surface_integral_outer(
        const CurlE* curlE, const GridParams* grid,
        int i_outer, double r_outer,
        const PortConfig* apertures
    );

    double surface_integral_endplate_z0(
        const CurlE* curlE, const GridParams* grid,
        int i_max_r, int i_max_half,
        const PortConfig* apertures
    );

    double surface_integral_endplate_zL(
        const CurlE* curlE, const GridParams* grid,
        int i_max_r, int i_max_half,
        const PortConfig* apertures
    );

    /*=============================================================================
     * Output
     *============================================================================*/
    void q_factor_print(const QFactorResult* result, const char* label);
    void q_factor_print_comparison(
        const QFactorResult* analytical,
        const QFactorResult* reference,
        const QFactorResult* full_model
    );

#ifdef __cplusplus
}
#endif

#endif /* Q_FACTOR_H */