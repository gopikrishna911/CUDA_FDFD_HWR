#ifndef CURLCURL_OPERATOR_H
#define CURLCURL_OPERATOR_H

#include "curl_E.h"
#include "curl_H.h"

typedef struct CurlCurlOperator CurlCurlOperator;

/*=============================================================================
 * Port Configuration for HWR Cavity
 *============================================================================*/

typedef enum {
    PORT_BEAM,
    PORT_VACUUM,
    PORT_POWER,
    PORT_PICKUP
} PortType;

typedef enum {
    SURFACE_INNER,        /* r = a (inner conductor) */
    SURFACE_OUTER,        /* r = b (outer conductor) */
    SURFACE_ENDPLATE_Z0,  /* z = 0 endplate */
    SURFACE_ENDPLATE_ZL   /* z = L endplate */
} PortSurface;

typedef struct {
    PortType type;
    PortSurface surface;
    double radius;
    double pos1;    /* phi (cylindrical) or r (endplate) */
    double pos2;    /* z (cylindrical) or phi (endplate) */
    const char* name;
} CavityPort;

typedef struct {
    int num_ports;
    int capacity;
    CavityPort* ports;
} PortConfig;

/* Port configuration functions */
void port_config_init(PortConfig* config);
void port_config_free(PortConfig* config);
void port_config_print(const PortConfig* config, const GridParams* grid);

void port_config_add_beam_apertures_single_pass(
    PortConfig* config,
    double z_center,
    double aperture_radius
);

void port_config_add_beam_apertures_multi_pass(
    PortConfig* config,
    double z_center,
    double aperture_radius,
    int num_passes
);

void port_config_add_vacuum_port(
    PortConfig* config,
    PortSurface surface,
    double r_center,
    double phi_center,
    double radius
);

int point_in_port_cylindrical(
    const PortConfig* config,
    const GridParams* grid,
    PortSurface surface,
    double phi,
    double z
);

int point_in_port_endplate(
    const PortConfig* config,
    const GridParams* grid,
    PortSurface surface,
    double r,
    double phi
);

/* New boundary condition function */
void apply_PEC_boundary_with_ports(
    EField* E,
    const GridParams* grid,
    const PortConfig* ports
);

/* New operator initialization with ports */
void curlcurl_op_init_with_ports(
    CurlCurlOperator* op,
    const GridParams* grid,
    const PortConfig* ports
);


/*=============================================================================
 * Curl-Curl Operator for Eigenvalue Problems
 * 
 * Wraps the curl_curl_E computation as a matrix-vector product:
 *     y = A * x
 * 
 * where:
 *     x, y = packed vectors containing all E-field components
 *     A    = curl-curl operator with PEC boundary conditions
 * 
 * This is used by eigensolvers to find k² such that:
 *     ∇ × ∇ × E = k² E
 *============================================================================*/

struct CurlCurlOperator{
    /* Grid parameters */
    GridParams grid;
    
    /* Working arrays (full size, including boundaries) */
    EField E_work;          /* Unpacked input */
    EField result_work;     /* Result of curl-curl */
    HField H_temp;          /* Intermediate for curl computation */
    
    /* Component sizes in full arrays */
    int size_Er;
    int size_Ephi;
    int size_Ez;
    
    /* Total DOFs (length of packed vector) */
    int n_total;
    
    /* Offsets in packed vector */
    int offset_Er;          /* = 0 */
    int offset_Ephi;        /* = size_Er */
    int offset_Ez;          /* = size_Er + size_Ephi */

    PortConfig ports;
    int has_ports;
    
};

/*=============================================================================
 * Operator lifecycle
 *============================================================================*/

/* Initialize operator (allocates internal storage) */
void curlcurl_op_init(CurlCurlOperator* op, const GridParams* grid);

/* Free operator resources */
void curlcurl_op_free(CurlCurlOperator* op);

/* Print operator info */
void curlcurl_op_print(const CurlCurlOperator* op);

/*=============================================================================
 * Core operation: y = A * x
 *============================================================================*/

/* Apply curl-curl operator: y = (∇×∇×) x */
void curlcurl_matvec(const CurlCurlOperator* op, const double* x, double* y);

/*=============================================================================
 * Boundary condition handling
 *============================================================================*/

/* Apply PEC boundary conditions (zero tangential E at boundaries) */
void apply_PEC_boundary(EField* E, const GridParams* grid);

/*=============================================================================
 * Pack/Unpack utilities
 *============================================================================*/

/* Pack E-field arrays into a single vector */
void pack_field(const EField* E, double* x, const CurlCurlOperator* op);

/* Unpack vector into E-field arrays */
void unpack_field(const double* x, EField* E, const CurlCurlOperator* op);

/*=============================================================================
 * Vector utilities for eigensolver
 *============================================================================*/

/* Dot product: result = x · y */
double vec_dot_product_weighted(const double* x, const double* y, const CurlCurlOperator* op);

/* Norm: result = ||x||₂ */
double vec_norm2_weighted(const double* x, const CurlCurlOperator* op);

/* Normalize: x = x / ||x||₂ */
void vec_normalize_weighted(double* x, const CurlCurlOperator* op);

/* Scale: x = alpha * x */
void vec_scale(double* x, double alpha, int n);

/* Copy: y = x */
void vec_copy(const double* x, double* y, int n);

/* AXPY: y = alpha * x + y */
void vec_axpy(double alpha, const double* x, double* y, int n);

/* Zero: x = 0 */
void vec_zero(double* x, int n);

/* Random initialization (for eigensolver starting vector) */
void vec_random(double* x, int n, unsigned int seed);

/*=============================================================================
 * Simple eigensolvers (for testing)
 *============================================================================*/

/* Power iteration: finds largest magnitude eigenvalue */
typedef struct {
    double eigenvalue;      /* Computed eigenvalue (k²) */
    double* eigenvector;    /* Computed eigenvector (allocated by caller) */
    int iterations;         /* Number of iterations used */
    double residual;        /* Final residual ||Ax - λx|| / ||x|| */
    int converged;          /* 1 if converged, 0 otherwise */
} EigenResult;

EigenResult power_iteration_weighted(
    const CurlCurlOperator* op,
    double* x,              /* Initial guess (overwritten with eigenvector) */
    int max_iter,
    double tol
);

/* Inverse iteration: finds eigenvalue closest to sigma */
EigenResult inverse_iteration(
    const CurlCurlOperator* op,
    double* x,              /* Initial guess (overwritten with eigenvector) */
    double sigma,           /* Shift (target eigenvalue) */
    int max_iter,
    double tol
);

/* Rayleigh quotient: λ = (x · Ax) / (x · x) */
double rayleigh_quotient_weighted(const CurlCurlOperator* op, const double* x);

/* Rayleigh Quotient Iteration: adaptive shift, cubic convergence */
EigenResult rayleigh_quotient_iteration(
    const CurlCurlOperator* op,
    double* x,              /* Initial guess (overwritten with eigenvector) */
    double sigma_init,      /* Initial shift hint */
    int max_iter,
    double tol
);

double compute_weighted_residual(
    const CurlCurlOperator* op,
    const double* Ax,
    const double* x,
    double lambda
);

void test_field_export(void);

void export_field_csv(
    const CurlCurlOperator* op,
    const double* x,
    const char* filename
);

void export_field_slice_rz(
    const CurlCurlOperator* op,
    const double* x,
    const char* filename
);

void export_field_vtk(
    const CurlCurlOperator* op,
    const double* x,
    const char* filename
);

#endif /* CURLCURL_OPERATOR_H */