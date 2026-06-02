#include "curlcurl_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*=============================================================================
 * Operator initialization
 *============================================================================*/
void curlcurl_op_init(CurlCurlOperator* op, const GridParams* grid) {
    op->grid = *grid;
    
    efield_alloc(&op->E_work, grid);
    efield_alloc(&op->result_work, grid);
    hfield_alloc(&op->H_temp, grid);
    
    op->size_Er = op->E_work.size_Er;
    op->size_Ephi = op->E_work.size_Ephi;
    op->size_Ez = op->E_work.size_Ez;
    
    op->offset_Er = 0;
    op->offset_Ephi = op->size_Er;
    op->offset_Ez = op->size_Er + op->size_Ephi;
    
    op->n_total = op->size_Er + op->size_Ephi + op->size_Ez;

    op->ports.num_ports = 0;
    op->ports.capacity = 0;
    op->ports.ports = NULL;
    op->has_ports = 0;
}

void curlcurl_op_free(CurlCurlOperator* op) {
    efield_free(&op->E_work);
    efield_free(&op->result_work);
    hfield_free(&op->H_temp);

    if (op->has_ports) {
        port_config_free(&op->ports);
        op->has_ports = 0;
    }

    op->n_total = 0;
}

void curlcurl_op_print(const CurlCurlOperator* op) {
    printf("Curl-Curl Operator:\n");
    printf("  Grid: Nr=%d, Nphi=%d, Nz=%d\n", 
           op->grid.Nr, op->grid.Nphi, op->grid.Nz);
    printf("  Domain: r=[%g, %g] m, z=[0, %g] m\n",
           op->grid.a, op->grid.b, op->grid.L);
    printf("  Component sizes:\n");
    printf("    Er:   %d\n", op->size_Er);
    printf("    Ephi: %d\n", op->size_Ephi);
    printf("    Ez:   %d\n", op->size_Ez);
    printf("  Total DOFs: %d\n", op->n_total);
    printf("  Vector offsets: Er=%d, Ephi=%d, Ez=%d\n",
           op->offset_Er, op->offset_Ephi, op->offset_Ez);
}

/*=============================================================================
 * Pack/Unpack
 *============================================================================*/
void pack_field(const EField* E, double* x, const CurlCurlOperator* op) {
    memcpy(x + op->offset_Er,   E->Er,   op->size_Er * sizeof(double));
    memcpy(x + op->offset_Ephi, E->Ephi, op->size_Ephi * sizeof(double));
    memcpy(x + op->offset_Ez,   E->Ez,   op->size_Ez * sizeof(double));
}

void unpack_field(const double* x, EField* E, const CurlCurlOperator* op) {
    memcpy(E->Er,   x + op->offset_Er,   op->size_Er * sizeof(double));
    memcpy(E->Ephi, x + op->offset_Ephi, op->size_Ephi * sizeof(double));
    memcpy(E->Ez,   x + op->offset_Ez,   op->size_Ez * sizeof(double));
}

/*=============================================================================
 * Apply PEC boundary conditions
 *============================================================================*/
void apply_PEC_boundary(EField* E, const GridParams* grid) {
    int Nr = grid->Nr;
    int Nphi = grid->Nphi;
    int Nz = grid->Nz;
    
    /* E_r = 0 at z=0 and z=L */
    for (int j = 0; j < Nphi; j++) {
        for (int i = 0; i < Nr; i++) {
            E->Er[idx_Er(grid, i, j, 0)] = 0.0;
            E->Er[idx_Er(grid, i, j, Nz)] = 0.0;
        }
    }
    
    /* E_phi = 0 at r=a, r=b, z=0, z=L */
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            E->Ephi[idx_Ephi(grid, 0, j, k)] = 0.0;
            E->Ephi[idx_Ephi(grid, Nr, j, k)] = 0.0;
        }
    }
    for (int j = 0; j < Nphi; j++) {
        for (int i = 0; i <= Nr; i++) {
            E->Ephi[idx_Ephi(grid, i, j, 0)] = 0.0;
            E->Ephi[idx_Ephi(grid, i, j, Nz)] = 0.0;
        }
    }
    
    /* E_z = 0 at r=a and r=b */
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            E->Ez[idx_Ez(grid, 0, j, k)] = 0.0;
            E->Ez[idx_Ez(grid, Nr, j, k)] = 0.0;
        }
    }
}

/*=============================================================================
 * Port Configuration Implementation
 *============================================================================*/

void port_config_init(PortConfig* config) {
    config->num_ports = 0;
    config->capacity = 16;
    config->ports = (CavityPort*)malloc(config->capacity * sizeof(CavityPort));
}

void port_config_free(PortConfig* config) {
    if (config->ports) {
        free(config->ports);
        config->ports = NULL;
    }
    config->num_ports = 0;
    config->capacity = 0;
}

static void port_config_add(PortConfig* config, const CavityPort* port) {
    if (config->num_ports >= config->capacity) {
        config->capacity *= 2;
        config->ports = (CavityPort*)realloc(config->ports,
                                              config->capacity * sizeof(CavityPort));
    }
    config->ports[config->num_ports] = *port;
    config->num_ports++;
}

void port_config_add_beam_apertures_single_pass(
    PortConfig* config,
    double z_center,
    double aperture_radius
) {
    CavityPort port;
    port.type = PORT_BEAM;
    port.radius = aperture_radius;
    port.pos2 = z_center;

    /* Inner conductor at φ = 0 */
    port.surface = SURFACE_INNER;
    port.pos1 = 0.0;
    port.name = "Beam Inner phi=0";
    port_config_add(config, &port);

    /* Inner conductor at φ = π */
    port.pos1 = M_PI;
    port.name = "Beam Inner phi=180";
    port_config_add(config, &port);

    /* Outer conductor at φ = 0 */
    port.surface = SURFACE_OUTER;
    port.pos1 = 0.0;
    port.name = "Beam Outer phi=0";
    port_config_add(config, &port);

    /* Outer conductor at φ = π */
    port.pos1 = M_PI;
    port.name = "Beam Outer phi=180";
    port_config_add(config, &port);
}

void port_config_add_beam_apertures_multi_pass(
    PortConfig* config,
    double z_center,
    double aperture_radius,
    int num_passes
) {
    CavityPort port;
    port.type = PORT_BEAM;
    port.radius = aperture_radius;
    port.pos2 = z_center;

    double dphi = M_PI / num_passes;

    for (int pass = 0; pass < num_passes; pass++) {
        double phi_entry = pass * dphi;
        double phi_exit = phi_entry + M_PI;
        if (phi_exit >= 2.0 * M_PI) {
            phi_exit -= 2.0 * M_PI;
        }

        /* Entry apertures */
        port.pos1 = phi_entry;
        port.surface = SURFACE_INNER;
        port.name = "Beam Inner Entry";
        port_config_add(config, &port);

        port.surface = SURFACE_OUTER;
        port.name = "Beam Outer Entry";
        port_config_add(config, &port);

        /* Exit apertures */
        port.pos1 = phi_exit;
        port.surface = SURFACE_INNER;
        port.name = "Beam Inner Exit";
        port_config_add(config, &port);

        port.surface = SURFACE_OUTER;
        port.name = "Beam Outer Exit";
        port_config_add(config, &port);
    }
}

void port_config_add_vacuum_port(
    PortConfig* config,
    PortSurface surface,
    double r_center,
    double phi_center,
    double radius
) {
    CavityPort port;
    port.type = PORT_VACUUM;
    port.surface = surface;
    port.radius = radius;
    port.pos1 = r_center;
    port.pos2 = phi_center;
    port.name = "Vacuum Port";
    port_config_add(config, &port);
}

void port_config_print(const PortConfig* config, const GridParams* /*grid*/) {
    const char* type_names[] = {"BEAM", "VACUUM", "POWER", "PICKUP"};
    const char* surface_names[] = {"Inner (r=a)", "Outer (r=b)", 
                                    "Endplate (z=0)", "Endplate (z=L)"};

    printf("\n");
    printf("======== PORT CONFIGURATION ========\n");
    printf("Total ports: %d\n", config->num_ports);

    for (int i = 0; i < config->num_ports; i++) {
        const CavityPort* p = &config->ports[i];

        printf("\nPort %d: %s\n", i + 1, p->name);
        printf("  Type:    %s\n", type_names[p->type]);
        printf("  Surface: %s\n", surface_names[p->surface]);
        printf("  Radius:  %.1f mm\n", p->radius * 1000);

        if (p->surface == SURFACE_INNER || p->surface == SURFACE_OUTER) {
            printf("  Position: phi = %.2f deg, z = %.4f m\n",
                   p->pos1 * 180.0 / M_PI, p->pos2);
        } else {
            printf("  Position: r = %.4f m, phi = %.2f deg\n",
                   p->pos1, p->pos2 * 180.0 / M_PI);
        }
    }
    printf("====================================\n");
}

int point_in_port_cylindrical(
    const PortConfig* config,
    const GridParams* grid,
    PortSurface surface,
    double phi,
    double z
) {
    if (!config || config->num_ports == 0) return 0;

    double r_surface;
    if (surface == SURFACE_INNER) {
        r_surface = grid->a;
    } else if (surface == SURFACE_OUTER) {
        r_surface = grid->b;
    } else {
        return 0;
    }

    for (int i = 0; i < config->num_ports; i++) {
        const CavityPort* p = &config->ports[i];

        if (p->surface != surface) continue;
        if (p->surface == SURFACE_ENDPLATE_Z0 || p->surface == SURFACE_ENDPLATE_ZL) {
            continue;
        }

        double phi_center = p->pos1;
        double z_center = p->pos2;

        double dphi = phi - phi_center;
        while (dphi > M_PI) dphi -= 2.0 * M_PI;
        while (dphi < -M_PI) dphi += 2.0 * M_PI;

        double arc_phi = r_surface * dphi;
        double dz = z - z_center;
        double dist_sq = arc_phi * arc_phi + dz * dz;

        if (dist_sq < p->radius * p->radius) {
            return 1;
        }
    }

    return 0;
}

int point_in_port_endplate(
    const PortConfig* config,
    const GridParams* /*grid*/,
    PortSurface surface,
    double r,
    double phi
) {
    if (!config || config->num_ports == 0) return 0;

    if (surface != SURFACE_ENDPLATE_Z0 && surface != SURFACE_ENDPLATE_ZL) {
        return 0;
    }

    for (int i = 0; i < config->num_ports; i++) {
        const CavityPort* p = &config->ports[i];

        if (p->surface != surface) continue;

        double r_center = p->pos1;
        double phi_center = p->pos2;

        double dphi = phi - phi_center;
        while (dphi > M_PI) dphi -= 2.0 * M_PI;
        while (dphi < -M_PI) dphi += 2.0 * M_PI;

        double dist_sq = r * r + r_center * r_center
                        - 2.0 * r * r_center * cos(dphi);

        if (dist_sq < p->radius * p->radius) {
            return 1;
        }
    }

    return 0;
}

/*=============================================================================
 * Boundary Conditions with Ports
 *============================================================================*/

void apply_PEC_boundary_with_ports(
    EField* E,
    const GridParams* grid,
    const PortConfig* ports
) {
    int Nr = grid->Nr;
    int Nphi = grid->Nphi;
    int Nz = grid->Nz;

    /* E_r = 0 at z = 0 and z = L (endplates) */
    for (int j = 0; j < Nphi; j++) {
        double phi = (j + 0.5) * grid->dphi;

        for (int i = 0; i < Nr; i++) {
            double r = grid->a + (i + 0.5) * grid->dr;

            if (!point_in_port_endplate(ports, grid, SURFACE_ENDPLATE_Z0, r, phi)) {
                E->Er[idx_Er(grid, i, j, 0)] = 0.0;
            }
            if (!point_in_port_endplate(ports, grid, SURFACE_ENDPLATE_ZL, r, phi)) {
                E->Er[idx_Er(grid, i, j, Nz)] = 0.0;
            }
        }
    }

    /* E_phi = 0 at z = 0 and z = L (endplates) */
    for (int j = 0; j < Nphi; j++) {
        double phi = (j + 0.5) * grid->dphi;

        for (int i = 0; i <= Nr; i++) {
            double r = grid->a + i * grid->dr;

            if (!point_in_port_endplate(ports, grid, SURFACE_ENDPLATE_Z0, r, phi)) {
                E->Ephi[idx_Ephi(grid, i, j, 0)] = 0.0;
            }
            if (!point_in_port_endplate(ports, grid, SURFACE_ENDPLATE_ZL, r, phi)) {
                E->Ephi[idx_Ephi(grid, i, j, Nz)] = 0.0;
            }
        }
    }

    /* E_phi = 0 at r = a and r = b (cylindrical conductors) */
    for (int k = 0; k <= Nz; k++) {
        double z = k * grid->dz;

        for (int j = 0; j < Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;

            if (!point_in_port_cylindrical(ports, grid, SURFACE_INNER, phi, z)) {
                E->Ephi[idx_Ephi(grid, 0, j, k)] = 0.0;
            }
            if (!point_in_port_cylindrical(ports, grid, SURFACE_OUTER, phi, z)) {
                E->Ephi[idx_Ephi(grid, Nr, j, k)] = 0.0;
            }
        }
    }

    /* E_z = 0 at r = a and r = b (cylindrical conductors) */
    for (int k = 0; k < Nz; k++) {
        double z = (k + 0.5) * grid->dz;

        for (int j = 0; j < Nphi; j++) {
            double phi = j * grid->dphi;

            if (!point_in_port_cylindrical(ports, grid, SURFACE_INNER, phi, z)) {
                E->Ez[idx_Ez(grid, 0, j, k)] = 0.0;
            }
            if (!point_in_port_cylindrical(ports, grid, SURFACE_OUTER, phi, z)) {
                E->Ez[idx_Ez(grid, Nr, j, k)] = 0.0;
            }
        }
    }
}

/*=============================================================================
 * Operator Initialization with Ports
 *============================================================================*/

void curlcurl_op_init_with_ports(
    CurlCurlOperator* op,
    const GridParams* grid,
    const PortConfig* ports
) {
    /* Initialize base operator */
    curlcurl_op_init(op, grid);

    /* Copy port configuration */
    if (ports && ports->num_ports > 0) {
        op->has_ports = 1;
        port_config_init(&op->ports);

        for (int i = 0; i < ports->num_ports; i++) {
            port_config_add(&op->ports, &ports->ports[i]);
        }
    }
}



/*=============================================================================
 * Core operation: y = A * x
 *============================================================================*/
void curlcurl_matvec(const CurlCurlOperator* op, const double* x, double* y) {
    CurlCurlOperator* op_mut = (CurlCurlOperator*)op;
    
    unpack_field(x, &op_mut->E_work, op);
    if (op->has_ports)
        apply_PEC_boundary_with_ports(&op_mut->E_work, &op->grid, &op->ports);
    else
        apply_PEC_boundary(&op_mut->E_work, &op->grid);


    compute_curl_curl_E_with_temp(&op_mut->E_work, &op_mut->result_work, &op_mut->H_temp, &op->grid);

    if (op->has_ports)
        apply_PEC_boundary_with_ports(&op_mut->result_work, &op->grid, &op->ports);
    else
        apply_PEC_boundary(&op_mut->result_work, &op->grid);

    pack_field(&op_mut->result_work, y, op);
}

/*=============================================================================
 * Vector utilities
 *============================================================================*/
/*=============================================================================
 * Weighted Inner Product for Cylindrical Coordinates
 * 
 * <E, F>_r = ∫ r * E·F * dr dφ dz
 * 
 * Each component weighted by r at its grid location:
 *   Er   at r_{i+1/2}
 *   Ephi at r_i  
 *   Ez   at r_i
 *============================================================================*/

double vec_dot_product_weighted(
    const double* x, 
    const double* y, 
    const CurlCurlOperator* op
) {
    double sum = 0.0;
    const GridParams* g = &op->grid;
    double dV = g->dr * g->dphi * g->dz;
    
    /* Er component: lives at (i+1/2, j, k+1/2), weight by r_{i+1/2} */
    for (int k = 0; k <= g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i < g->Nr; i++) {
                int idx = op->offset_Er + idx_Er(g, i, j, k);
                double r = g->a + (i + 0.5) * g->dr;  /* r_{i+1/2} */
                sum += r * x[idx] * y[idx] * dV;
            }
        }
    }
    
    /* Ephi component: lives at (i, j+1/2, k+1/2), weight by r_i */
    for (int k = 0; k <= g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i <= g->Nr; i++) {
                int idx = op->offset_Ephi + idx_Ephi(g, i, j, k);
                double r = g->a + i * g->dr;  /* r_i */
                sum += r * x[idx] * y[idx] * dV;
            }
        }
    }
    
    /* Ez component: lives at (i, j, k), weight by r_i */
    for (int k = 0; k < g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i <= g->Nr; i++) {
                int idx = op->offset_Ez + idx_Ez(g, i, j, k);
                double r = g->a + i * g->dr;  /* r_i */
                sum += r * x[idx] * y[idx] * dV;
            }
        }
    }
    
    return sum;
}

double vec_norm2_weighted(const double* x, const CurlCurlOperator* op) {
    return sqrt(vec_dot_product_weighted(x, x, op));
}

void vec_scale(double* x, double alpha, int n) {
    for (int i = 0; i < n; i++) {
        x[i] *= alpha;
    }
}

void vec_copy(const double* x, double* y, int n) {
    memcpy(y, x, n * sizeof(double));
}

void vec_axpy(double alpha, const double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

void vec_zero(double* x, int n) {
    memset(x, 0, n * sizeof(double));
}

void vec_random(double* x, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        x[i] = (double)rand() / RAND_MAX - 0.5;
    }
}

/*=============================================================================
 * Rayleigh quotient
 *============================================================================*/
double rayleigh_quotient_weighted(const CurlCurlOperator* op, const double* x) {
    double* Ax = (double*)malloc(op->n_total * sizeof(double));
    curlcurl_matvec(op, x, Ax);
    
    double xAx = vec_dot_product_weighted(x, Ax, op);
    double xx  = vec_dot_product_weighted(x, x, op);
    
    free(Ax);
    return xAx / xx;
}

void vec_normalize_weighted(double* x, const CurlCurlOperator* op) {
    double norm = vec_norm2_weighted(x, op);
    if (norm > 1e-14) {
        vec_scale(x, 1.0 / norm, op->n_total);
    }
}


/*=============================================================================
 * Compute weighted residual: ||Ax - λx||_r
 *============================================================================*/
double compute_weighted_residual(
    const CurlCurlOperator* op,
    const double* Ax,
    const double* x,
    double lambda
) {
    double residual = 0.0;
    const GridParams* g = &op->grid;
    double dV = g->dr * g->dphi * g->dz;
    
    /* Er contribution */
    for (int k = 0; k <= g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i < g->Nr; i++) {
                int idx = op->offset_Er + idx_Er(g, i, j, k);
                double r = g->a + (i + 0.5) * g->dr;
                double r_i = Ax[idx] - lambda * x[idx];
                residual += r * r_i * r_i * dV;
            }
        }
    }
    
    /* Ephi contribution */
    for (int k = 0; k <= g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i <= g->Nr; i++) {
                int idx = op->offset_Ephi + idx_Ephi(g, i, j, k);
                double r = g->a + i * g->dr;
                double r_i = Ax[idx] - lambda * x[idx];
                residual += r * r_i * r_i * dV;
            }
        }
    }
    
    /* Ez contribution */
    for (int k = 0; k < g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i <= g->Nr; i++) {
                int idx = op->offset_Ez + idx_Ez(g, i, j, k);
                double r = g->a + i * g->dr;
                double r_i = Ax[idx] - lambda * x[idx];
                residual += r * r_i * r_i * dV;
            }
        }
    }
    
    return sqrt(residual);
}


/*=============================================================================
 * Power iteration
 *============================================================================*/
EigenResult power_iteration_weighted(
    const CurlCurlOperator* op,
    double* x,
    int max_iter,
    double tol
) {
    EigenResult result;
    result.eigenvalue = 0.0;
    result.eigenvector = x;
    result.iterations = 0;
    result.residual = 1.0;
    result.converged = 0;
    
    int n = op->n_total;
    double* y = (double*)malloc(n * sizeof(double));
    
    /* Normalize initial guess with weighted norm */
    double norm = vec_norm2_weighted(x, op);
    if (norm < 1e-14) {
        vec_random(x, n, 12345);
        norm = vec_norm2_weighted(x, op);
    }
    vec_scale(x, 1.0 / norm, n);
    
    double lambda_old = 0.0;
    
    printf("  Power Iteration (weighted inner product):\n");
    printf("  %5s %18s %15s\n", "Iter", "Eigenvalue", "Change");
    printf("  ------------------------------------------\n");
    
    for (int iter = 0; iter < max_iter; iter++) {
        /* y = A * x */
        curlcurl_matvec(op, x, y);
        
        /* Rayleigh quotient: λ = <x, Ax>_r / <x, x>_r */
        /* Since x is normalized: <x,x>_r = 1, so λ = <x, y>_r */
        double lambda = vec_dot_product_weighted(x, y, op);
        
        /* Normalize y with weighted norm */
        norm = vec_norm2_weighted(y, op);
        if (norm < 1e-14) {
            printf("  Warning: Zero vector at iteration %d\n", iter);
            break;
        }
        vec_scale(y, 1.0 / norm, n);
        
        /* Compute weighted residual: ||Ax - λx||_r */
        double residual = 0.0;
        const GridParams* g = &op->grid;
        double dV = g->dr * g->dphi * g->dz;
        
        /* This is expensive but accurate - compute ||y*norm - λ*x||_r */
        double* Ax = (double*)malloc(n * sizeof(double));
        curlcurl_matvec(op, x, Ax);
        
        /* Er contribution */
        for (int k = 0; k <= g->Nz; k++) {
            for (int j = 0; j < g->Nphi; j++) {
                for (int i = 0; i < g->Nr; i++) {
                    int idx = op->offset_Er + idx_Er(g, i, j, k);
                    double r = g->a + (i + 0.5) * g->dr;
                    double r_i = Ax[idx] - lambda * x[idx];
                    residual += r * r_i * r_i * dV;
                }
            }
        }
        /* Ephi contribution */
        for (int k = 0; k <= g->Nz; k++) {
            for (int j = 0; j < g->Nphi; j++) {
                for (int i = 0; i <= g->Nr; i++) {
                    int idx = op->offset_Ephi + idx_Ephi(g, i, j, k);
                    double r = g->a + i * g->dr;
                    double r_i = Ax[idx] - lambda * x[idx];
                    residual += r * r_i * r_i * dV;
                }
            }
        }
        /* Ez contribution */
        for (int k = 0; k < g->Nz; k++) {
            for (int j = 0; j < g->Nphi; j++) {
                for (int i = 0; i <= g->Nr; i++) {
                    int idx = op->offset_Ez + idx_Ez(g, i, j, k);
                    double r = g->a + i * g->dr;
                    double r_i = Ax[idx] - lambda * x[idx];
                    residual += r * r_i * r_i * dV;
                }
            }
        }
        residual = sqrt(residual);
        free(Ax);
        
        result.eigenvalue = lambda;
        result.residual = residual;
        result.iterations = iter + 1;
        
        double change = fabs(lambda - lambda_old);
        
        if (iter % 10 == 0 || change < tol) {
            printf("  %5d %18.12f %15.8e\n", iter, lambda, change);
        }
        
        if (change < tol && iter > 0) {
            result.converged = 1;
            vec_copy(y, x, n);
            printf("  Converged!\n");
            break;
        }
        
        vec_copy(y, x, n);
        lambda_old = lambda;
    }
    
    free(y);
    return result;
}

/*=============================================================================
 * MINRES solver for (A - sigma*I) * z = b
 * 
 * Works for symmetric INDEFINITE matrices (unlike CG)
 *============================================================================*/
typedef struct {
    int iterations;
    double residual;
    int converged;
} LinearSolverResult;

static LinearSolverResult minres_solve_shifted(
    const CurlCurlOperator* op,
    double sigma,
    const double* b,
    double* x,
    int max_iter,
    double tol
) {
    LinearSolverResult result;
    result.iterations = 0;
    result.residual = 1.0;
    result.converged = 0;
    
    int n = op->n_total;
    
    /* Allocate working vectors */
    double* v_old = (double*)calloc(n, sizeof(double));
    double* v_cur = (double*)malloc(n * sizeof(double));
    double* v_new = (double*)malloc(n * sizeof(double));
    double* Av = (double*)malloc(n * sizeof(double));
    double* w_old = (double*)calloc(n, sizeof(double));
    double* w_cur = (double*)calloc(n, sizeof(double));
    double* w_new = (double*)malloc(n * sizeof(double));
    
    /* x = 0 initially */
    vec_zero(x, n);
    
    /* v_cur = b / ||b|| */
    double b_norm = vec_norm2_weighted(b, op);
    if (b_norm < 1e-14) {
        free(v_old); free(v_cur); free(v_new);
        free(Av); free(w_old); free(w_cur); free(w_new);
        result.converged = 1;
        result.residual = 0;
        return result;
    }
    
    for (int i = 0; i < n; i++) {
        v_cur[i] = b[i] / b_norm;
    }
    
    double beta_cur = b_norm;
    double eta = b_norm;
    
    double c_old = 1.0, c_cur = 1.0;
    double s_old = 0.0, s_cur = 0.0;
    
    for (int iter = 0; iter < max_iter; iter++) {
        /* Av = (A - sigma*I) * v_cur */
        curlcurl_matvec(op, v_cur, Av);
        for (int i = 0; i < n; i++) {
            Av[i] -= sigma * v_cur[i];
        }
        
        /* alpha = v_cur' * Av */
        double alpha = vec_dot_product_weighted(v_cur, Av, op);
        
        /* v_new = Av - alpha*v_cur - beta_cur*v_old */
        for (int i = 0; i < n; i++) {
            v_new[i] = Av[i] - alpha * v_cur[i] - beta_cur * v_old[i];
        }
        
        /* beta_new = ||v_new|| */
        double beta_new = vec_norm2_weighted(v_new, op);
        
        /* Normalize v_new */
        if (beta_new > 1e-14) {
            for (int i = 0; i < n; i++) {
                v_new[i] /= beta_new;
            }
        }
        
        /* Apply previous Givens rotations */
        double rho1 = s_old * beta_cur;
        double rho2 = c_old * c_cur * beta_cur + s_cur * alpha;
        // double rho3 = -s_old * s_cur * beta_cur + c_cur * alpha;
        
        /* Compute new Givens rotation */
        double rho3_bar = c_cur * alpha - c_old * s_cur * beta_cur;
        double gamma = sqrt(rho3_bar * rho3_bar + beta_new * beta_new);
        
        double c_new, s_new;
        if (gamma > 1e-14) {
            c_new = rho3_bar / gamma;
            s_new = beta_new / gamma;
        } else {
            c_new = 1.0;
            s_new = 0.0;
        }
        
        /* Update w vectors */
        for (int i = 0; i < n; i++) {
            w_new[i] = (v_cur[i] - rho2 * w_cur[i] - rho1 * w_old[i]);
            if (fabs(gamma) > 1e-14) {
                w_new[i] /= gamma;
            }
        }
        
        /* Update solution: x = x + c_new * eta * w_new */
        double update_coef = c_new * eta;
        vec_axpy(update_coef, w_new, x, n);
        
        /* Update eta */
        eta = -s_new * eta;
        
        /* Check convergence */
        result.residual = fabs(eta) / b_norm;
        result.iterations = iter + 1;
        
        if (result.residual < tol) {
            result.converged = 1;
            break;
        }
        
        /* Shift vectors for next iteration */
        double* temp;
        
        temp = v_old; v_old = v_cur; v_cur = v_new; v_new = temp;
        temp = w_old; w_old = w_cur; w_cur = w_new; w_new = temp;
        
        beta_cur = beta_new;
        c_old = c_cur; c_cur = c_new;
        s_old = s_cur; s_cur = s_new;
    }
    
    free(v_old); free(v_cur); free(v_new);
    free(Av); free(w_old); free(w_cur); free(w_new);
    
    return result;
}

/*=============================================================================
 * Rayleigh Quotient Iteration (RQI)
 * 
 * - Uses adaptive shift (current eigenvalue estimate)
 * - Cubic convergence near eigenvalue
 * - Better than fixed-shift inverse iteration
 *============================================================================*/
EigenResult rayleigh_quotient_iteration(
    const CurlCurlOperator* op,
    double* x,
    double sigma_init,
    int max_iter,
    double tol
) {
    EigenResult result;
    result.eigenvector = x;
    result.iterations = 0;
    result.residual = 1.0;
    result.converged = 0;
    
    int n = op->n_total;
    
    double* y = (double*)malloc(n * sizeof(double));
    double* Ax = (double*)malloc(n * sizeof(double));
    
    /* Normalize initial guess */
    double norm = vec_norm2_weighted(x, op);
    if (norm < 1e-14) {
        vec_random(x, n, 54321);
        norm = vec_norm2_weighted(x, op);
    }
    vec_scale(x, 1.0 / norm, n);
    
    /* Initial Rayleigh quotient */
    curlcurl_matvec(op, x, Ax);
    double sigma = vec_dot_product_weighted(x, Ax, op);
    
    /* If initial estimate is far off, use provided hint */
    if (fabs(sigma) < 1e-10 || sigma < 0) {
        sigma = sigma_init;
    }
    
    printf("  Rayleigh Quotient Iteration:\n");
    printf("  %5s %15s %15s %8s\n", "Iter", "Eigenvalue", "Residual", "LS its");
    printf("  -------------------------------------------------\n");
    
    for (int iter = 0; iter < max_iter; iter++) {
        /* Solve (A - sigma*I) * y = x using MINRES */
        LinearSolverResult ls = minres_solve_shifted(op, sigma, x, y, 2000, 1e-6);
        
        /* Normalize y */
        norm = vec_norm2_weighted(y, op);
        if (norm < 1e-14) {
            printf("  Warning: Zero vector at iteration %d\n", iter);
            break;
        }
        vec_scale(y, 1.0 / norm, n);
        
        /* Update x */
        vec_copy(y, x, n);
        
        /* Compute new Rayleigh quotient (this becomes the new shift) */
        curlcurl_matvec(op, x, Ax);
        double sigma_new = vec_dot_product_weighted(x, Ax, op);
        
        /* Compute weighted residual: ||Ax - sigma*x||_r */
        double residual = compute_weighted_residual(op, Ax, x, sigma_new);
                
        result.eigenvalue = sigma_new;
        result.residual = residual;
        result.iterations = iter + 1;
        
        printf("  %5d %15.10f %15.8e %8d\n", 
               iter, sigma_new, residual, ls.iterations);
        
        /* Check convergence (residual-based) */
        if (residual < tol) {
            result.converged = 1;
            printf("  Converged!\n");
            break;
        }
        
        /* Update shift for next iteration */
        sigma = sigma_new;
    }
    
    free(y);
    free(Ax);
    
    return result;
}

/*=============================================================================
 * Inverse iteration (using MINRES for indefinite systems)
 *============================================================================*/
EigenResult inverse_iteration(
    const CurlCurlOperator* op,
    double* x,
    double sigma,
    int max_iter,
    double tol
) {
    EigenResult result;
    result.eigenvalue = sigma;
    result.eigenvector = x;
    result.iterations = 0;
    result.residual = 1.0;
    result.converged = 0;
    
    int n = op->n_total;
    
    double* y = (double*)malloc(n * sizeof(double));
    double* Ax = (double*)malloc(n * sizeof(double));
    
    /* Normalize initial guess */
    double norm = vec_norm2_weighted(x, op);
    if (norm < 1e-14) {
        vec_random(x, n, 54321);
        norm = vec_norm2_weighted(x, op);
    }
    vec_scale(x, 1.0 / norm, n);
    
    printf("  Inverse iteration (sigma = %g):\n", sigma);
    printf("  %5s %15s %15s %8s\n", "Iter", "Eigenvalue", "Residual", "LS its");
    printf("  -------------------------------------------------\n");
    
    for (int iter = 0; iter < max_iter; iter++) {
        /* Solve (A - sigma*I) * y = x using MINRES */
        LinearSolverResult ls = minres_solve_shifted(op, sigma, x, y, 2000, 1e-8);
        
        /* Normalize y */
        norm = vec_norm2_weighted(y, op);
        if (norm < 1e-14) {
            printf("  Warning: Zero vector at iteration %d\n", iter);
            break;
        }
        vec_scale(y, 1.0 / norm, n);
        
        /* Compute Rayleigh quotient for eigenvalue estimate */
        curlcurl_matvec(op, y, Ax);
        double lambda = vec_dot_product_weighted(y, Ax, op);
        
        /* Compute weighted residual ||Ax - lambda*y||_r */
        double residual = compute_weighted_residual(op, Ax, y, lambda);
        
        result.eigenvalue = lambda;
        result.residual = residual;
        result.iterations = iter + 1;
        
        printf("  %5d %15.10f %15.8e %8d\n", 
               iter, lambda, residual, ls.iterations);
        
        /* Check convergence (residual-based) */
        if (residual < tol) {
            result.converged = 1;
            vec_copy(y, x, n);
            printf("  Converged!\n");
            break;
        }
        
        vec_copy(y, x, n);
    }
    
    free(y);
    free(Ax);
    
    return result;
}


/*=============================================================================
 * Test 6: Generate and export TEM mode field map
 *============================================================================*/
void test_field_export(void) {
    printf("\n");
    printf("============================================================\n");
    printf("Test 6: Export TEM mode field for visualization\n");
    printf("============================================================\n");
    
    GridParams grid;
    grid_init(&grid, 0.3333, 1.0, 1.39, 24, 16, 24);
    
    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);
    
    printf("Grid: Nr=%d, Nphi=%d, Nz=%d, DOFs=%d\n", 
           grid.Nr, grid.Nphi, grid.Nz, op.n_total);
    
    /* Start with TEM-like initial guess */
    double* x = (double*)malloc(op.n_total * sizeof(double));
    vec_zero(x, op.n_total);
    
    for (int k = 0; k <= grid.Nz; k++) {
        double z = k * grid.dz;
        double sin_piz_L = sin(M_PI * z / grid.L);
        for (int j = 0; j < grid.Nphi; j++) {
            for (int i = 0; i < grid.Nr; i++) {
                double r = r_at_i_half(&grid, i);
                int idx = op.offset_Er + idx_Er(&grid, i, j, k);
                x[idx] = sin_piz_L / r;
            }
        }
    }
    
    /* Refine with RQI */
    // double k2_target = (M_PI / grid.L) * (M_PI / grid.L);
    printf("\nFinding TEM mode with RQI...\n");
    // EigenResult result = rayleigh_quotient_iteration(&op, x, k2_target, 10, 1e-12);
    
    /* Normalize for nice visualization */
    double max_val = 0.0;
    for (int i = 0; i < op.n_total; i++) {
        if (fabs(x[i]) > max_val) max_val = fabs(x[i]);
    }
    if (max_val > 0) {
        vec_scale(x, 1.0 / max_val, op.n_total);
    }
    
    printf("\nExporting field data...\n");
    
    /* Export in different formats */
    export_field_slice_rz(&op, x, "tem_mode_rz.csv");
    export_field_vtk(&op, x, "tem_mode.vtk");
    export_field_csv(&op, x, "tem_mode.csv"); 
    
    /* Print some field values */
    printf("\nSample Er values at phi=0, various (r,z):\n");
    printf("%8s %8s %12s %12s\n", "r", "z", "Er (num)", "Er (exact)");
    printf("------------------------------------------------\n");
    
    int j = 0;
    for (int k = 0; k <= grid.Nz; k += grid.Nz/4) {
        double z = k * grid.dz;
        for (int i = 0; i < grid.Nr; i += grid.Nr/4) {
            double r = r_at_i_half(&grid, i);
            int idx = op.offset_Er + idx_Er(&grid, i, j, k);
            double Er_num = x[idx];
            double Er_exact = sin(M_PI * z / grid.L) / r / (1.0/grid.a);  /* normalized */
            printf("%8.4f %8.4f %12.6f %12.6f\n", r, z, Er_num, Er_exact);
        }
    }
    
    free(x);
    curlcurl_op_free(&op);
    
    printf("\nTest 6 completed - files ready for visualization\n");
}


/*=============================================================================
 * Field Export for Visualization
 *============================================================================*/

void export_field_csv(
    const CurlCurlOperator* op,
    const double* x,
    const char* filename
) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open %s for writing\n", filename);
        return;
    }
    
    const GridParams* g = &op->grid;
    
    /* Header */
    fprintf(fp, "r,phi,z,Er,Ephi,Ez,E_mag\n");
    
    /* We'll output on a regular grid by interpolating/sampling */
    /* For simplicity, output Er at its native locations */
    
    /* Export Er component */
    fprintf(fp, "# Er component (at r_{i+1/2}, phi_j, z_k)\n");
    for (int k = 0; k <= g->Nz; k++) {
        double z = k * g->dz;
        for (int j = 0; j < g->Nphi; j++) {
            double phi = (j + 0.5) * g->dphi;
            for (int i = 0; i < g->Nr; i++) {
                double r = g->a + (i + 0.5) * g->dr;
                int idx = op->offset_Er + idx_Er(g, i, j, k);
                double Er = x[idx];
                fprintf(fp, "%.6f,%.6f,%.6f,%.6e,0,0,%.6e\n", 
                        r, phi, z, Er, fabs(Er));
            }
        }
    }
    
    fclose(fp);
    printf("Field exported to %s\n", filename);
}

/* Export a 2D slice (r-z plane at phi=0) - most useful for TEM mode */
void export_field_slice_rz(
    const CurlCurlOperator* op,
    const double* x,
    const char* filename
) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open %s for writing\n", filename);
        return;
    }
    
    const GridParams* g = &op->grid;
    int j = 0;  /* phi = 0 slice */
    
    fprintf(fp, "# r-z slice of E-field at phi=0\n");
    fprintf(fp, "# Columns: r, z, Er, Ez\n");
    fprintf(fp, "r,z,Er,Ez\n");
    
    /* Output Er (lives at i+1/2, j, k) */
    for (int k = 0; k <= g->Nz; k++) {
        double z = k * g->dz;
        for (int i = 0; i < g->Nr; i++) {
            double r = g->a + (i + 0.5) * g->dr;
            int idx_er = op->offset_Er + idx_Er(g, i, j, k);
            
            /* Get Ez at nearby point (approximate) */
            double Ez = 0.0;
            if (k < g->Nz && i < g->Nr) {
                int idx_ez = op->offset_Ez + idx_Ez(g, i, j, k);
                Ez = x[idx_ez];
            }
            
            fprintf(fp, "%.6f,%.6f,%.6e,%.6e\n", r, z, x[idx_er], Ez);
        }
    }
    
    fclose(fp);
    printf("r-z slice exported to %s\n", filename);
}


void export_field_vtk(
    const CurlCurlOperator* op,
    const double* x,
    const char* filename
) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open %s\n", filename);
        return;
    }

    const GridParams* g = &op->grid;

    int Nx = g->Nr + 1;
    int Ny = g->Nphi + 1;
    int Nz = g->Nz + 1;
    int npoints = Nx * Ny * Nz;

    /* ---------- HEADER ---------- */
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "Interpolated E-field vectors (TEM mode)\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_GRID\n");
    fprintf(fp, "DIMENSIONS %d %d %d\n", Nx, Ny, Nz);

    /* ---------- POINT COORDINATES ---------- */
    fprintf(fp, "POINTS %d float\n", npoints);

    for (int k = 0; k <= g->Nz; k++) {
        double z = k * g->dz;
        for (int j = 0; j <= g->Nphi; j++) {
            double phi = j * g->dphi;
            for (int i = 0; i <= g->Nr; i++) {
                double r = g->a + i * g->dr;
                double xcart = r * cos(phi);
                double ycart = r * sin(phi);
                fprintf(fp, "%.6f %.6f %.6f\n", xcart, ycart, z);
            }
        }
    }

    /* ---------- POINT DATA (VECTORS) ---------- */
    fprintf(fp, "POINT_DATA %d\n", npoints);
    fprintf(fp, "VECTORS E float\n");

    for (int k = 0; k <= g->Nz; k++) {
        for (int j = 0; j <= g->Nphi; j++) {
            double phi = j * g->dphi;
            double c = cos(phi);
            double s = sin(phi);

            for (int i = 0; i <= g->Nr; i++) {

                /* --- interpolate Er to vertex --- */
                double Er = 0.0;
                if (i > 0 && i < g->Nr && k <= g->Nz) {
                    int idx1 = op->offset_Er + idx_Er(g, i - 1, j % g->Nphi, k);
                    int idx2 = op->offset_Er + idx_Er(g, i,     j % g->Nphi, k);
                    Er = 0.5 * (x[idx1] + x[idx2]);
                }

                /* --- interpolate Ephi --- */
                double Ephi = 0.0;
                if (j > 0 && j < g->Nphi && i < g->Nr) {
                    int idx1 = op->offset_Ephi + idx_Ephi(g, i, j - 1, k);
                    int idx2 = op->offset_Ephi + idx_Ephi(g, i, j % g->Nphi, k);
                    Ephi = 0.5 * (x[idx1] + x[idx2]);
                }

                /* --- interpolate Ez --- */
                double Ez = 0.0;
                if (k > 0 && k < g->Nz && i < g->Nr) {
                    int idx1 = op->offset_Ez + idx_Ez(g, i, j % g->Nphi, k - 1);
                    int idx2 = op->offset_Ez + idx_Ez(g, i, j % g->Nphi, k);
                    Ez = 0.5 * (x[idx1] + x[idx2]);
                }

                /* --- cylindrical → Cartesian --- */
                double Ex = Er * c - Ephi * s;
                double Ey = Er * s + Ephi * c;

                fprintf(fp, "%.6e %.6e %.6e\n", Ex, Ey, Ez);
            }
        }
    }

    fclose(fp);
    printf("VTK vector field written to %s\n", filename);
}