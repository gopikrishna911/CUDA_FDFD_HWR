#include "pipe_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*=============================================================================
 * PipeConfig lifecycle
 *============================================================================*/

void pipe_config_init(
    PipeConfig* config,
    double r_inner,
    double r_outer,
    double pipe_radius,
    double aperture_radius,
    double pipe_length,
    double taper_length    /* Can pass 0 — will compute from radii */
) {
    config->pipe_radius = pipe_radius;
    config->aperture_radius = aperture_radius;
    config->pipe_length = pipe_length;

    /* Fillet radius = difference in radii (quarter circle) */
    double R_fillet = aperture_radius - pipe_radius;
    config->taper_length = (taper_length > 0) ? taper_length : R_fillet;

    config->r_inner = r_inner;
    config->r_outer = r_outer;

    config->num_pipes = 0;
    config->capacity = 16;
    config->pipes = (PipeLocation*)malloc(config->capacity * sizeof(PipeLocation));
}

void pipe_config_free(PipeConfig* config) {
    if (config->pipes) {
        free(config->pipes);
        config->pipes = NULL;
    }
    config->num_pipes = 0;
}

void pipe_config_add(PipeConfig* config, double phi_center, double z_center) {
    if (config->num_pipes >= config->capacity) {
        config->capacity *= 2;
        config->pipes = (PipeLocation*)realloc(
            config->pipes, config->capacity * sizeof(PipeLocation));
    }
    config->pipes[config->num_pipes].phi_center = phi_center;
    config->pipes[config->num_pipes].z_center = z_center;
    config->num_pipes++;
}

void pipe_config_add_single_pass(PipeConfig* config, double z_center) {
    pipe_config_add(config, 0.0, z_center);       /* Entry at φ = 0 */
    pipe_config_add(config, M_PI, z_center);       /* Exit at φ = π */
}

void pipe_config_add_multi_pass(PipeConfig* config, double z_center, int num_passes) {
    double dphi = M_PI / num_passes;

    for (int pass = 0; pass < num_passes; pass++) {
        double phi_entry = pass * dphi;
        double phi_exit = phi_entry + M_PI;
        if (phi_exit >= 2.0 * M_PI) phi_exit -= 2.0 * M_PI;

        pipe_config_add(config, phi_entry, z_center);
        pipe_config_add(config, phi_exit, z_center);
    }
}

void pipe_config_print(const PipeConfig* config) {
    double R_fillet = config->aperture_radius - config->pipe_radius;

    printf("\n");
    printf("======== BEAM PIPE CONFIGURATION ========\n");
    printf("  Pipe radius:      %.1f mm\n", config->pipe_radius * 1000);
    printf("  Aperture radius:  %.1f mm\n", config->aperture_radius * 1000);
    printf("  Fillet radius:    %.1f mm (quarter-circle arc)\n", R_fillet * 1000);
    printf("  Pipe length:      %.1f mm (beyond fillet)\n",
        (config->pipe_length - R_fillet) * 1000);
    printf("  Total extension:  %.1f mm (fillet + pipe)\n",
        config->pipe_length * 1000);
    printf("  Inner conductor:  r = %.4f m\n", config->r_inner);
    printf("  Outer conductor:  r = %.4f m\n", config->r_outer);
    printf("  Number of pipes:  %d\n", config->num_pipes);

    printf("\n  Profile:\n");
    printf("    r = b:          aperture = %.1f mm\n",
        config->aperture_radius * 1000);
    printf("    r = b + %.0fmm:   pipe = %.1f mm (fillet ends)\n",
        R_fillet * 1000, config->pipe_radius * 1000);
    printf("    r = b + %.0fmm:  PEC cap (pipe ends)\n",
        config->pipe_length * 1000);

    for (int i = 0; i < config->num_pipes; i++) {
        printf("    Pipe %d: phi = %.2f deg, z = %.4f m\n",
            i + 1,
            config->pipes[i].phi_center * 180.0 / M_PI,
            config->pipes[i].z_center);
    }
    printf("=========================================\n");
}

/*=============================================================================
 * Pipe radius at radial position (with circular arc fillet)
 *
 * Quarter-circle arc connecting aperture at r=b to pipe:
 *
 *   R_fillet = aperture_radius - pipe_radius
 *   Arc center at (b + R_fillet, pipe_radius + R_fillet)
 *
 *   For r in [b, b + R_fillet]:
 *     ρ(r) = (pipe_radius + R_f) - sqrt(R_f² - (r - b - R_f)²)
 *
 *   Tangent at r = b:           vertical   (smooth wall junction)
 *   Tangent at r = b + R_f:     horizontal (smooth pipe junction)
 *============================================================================*/

double pipe_radius_at_r(const PipeConfig* config, double r) {
    double b = config->r_outer;

    if (r <= b) {
        /* Inside cavity: aperture size */
        return config->aperture_radius;
    }

    double R_f = config->aperture_radius - config->pipe_radius;
    double dr = r - b;  /* distance from outer wall */

    if (R_f < 1e-14) {
        /* No fillet: constant radius */
        return config->pipe_radius;
    }

    if (dr >= R_f) {
        /* Beyond fillet: constant pipe radius */
        return config->pipe_radius;
    }

    /* Quarter-circle arc fillet */
    double arg = R_f * R_f - (R_f - dr) * (R_f - dr);
    if (arg < 0.0) arg = 0.0;  /* Guard floating point */

    return (config->pipe_radius + R_f) - sqrt(arg);
}

/*=============================================================================
 * Point-in-vacuum test
 *============================================================================*/

int point_is_vacuum(const PipeConfig* config, double r, double phi, double z) {
    /* Inside cavity: always vacuum */
    if (r <= config->r_outer) {
        if (r >= config->r_inner) {
            return 1;   /* Cavity region */
        }
        return 0;       /* Inside inner conductor (shouldn't happen in our grid) */
    }

    /* Beyond pipe end: PEC cap */
    if (r >= config->r_outer + config->pipe_length) {
        return 0;
    }

    /* In pipe region: check each pipe */
    for (int i = 0; i < config->num_pipes; i++) {
        double phi_c = config->pipes[i].phi_center;
        double z_c = config->pipes[i].z_center;

        /* Angular distance (on the surface at radius r) */
        double dphi = phi - phi_c;
        while (dphi > M_PI) dphi -= 2.0 * M_PI;
        while (dphi < -M_PI) dphi += 2.0 * M_PI;

        double arc = r * dphi;
        double dz = z - z_c;

        double dist = sqrt(arc * arc + dz * dz);

        double R = pipe_radius_at_r(config, r);

        if (dist < R) {
            return 1;   /* Inside pipe: vacuum */
        }
    }

    /* Outside all pipes: PEC (outer conductor body) */
    return 0;
}


/*=============================================================================
 * Endcap Pipe Configuration
 *============================================================================*/

void endcap_pipe_config_init(EndcapPipeConfig* config) {
    config->num_pipes = 0;
    config->capacity = 4;
    config->pipes = (EndcapPipe*)malloc(config->capacity * sizeof(EndcapPipe));
    config->z0_extension = 0.0;
    config->zL_extension = 0.0;
}

void endcap_pipe_config_free(EndcapPipeConfig* config) {
    if (config->pipes) {
        free(config->pipes);
        config->pipes = NULL;
    }
    config->num_pipes = 0;
}

void endcap_pipe_config_add(
    EndcapPipeConfig* config,
    double r_center,
    double phi_center,
    double aperture_radius,
    double pipe_radius,
    double pipe_length,
    int is_z0
) {
    if (config->num_pipes >= config->capacity) {
        config->capacity *= 2;
        config->pipes = (EndcapPipe*)realloc(
            config->pipes, config->capacity * sizeof(EndcapPipe));
    }

    EndcapPipe* p = &config->pipes[config->num_pipes];
    p->r_center = r_center;
    p->phi_center = phi_center;
    p->aperture_radius = aperture_radius;
    p->pipe_radius = pipe_radius;
    p->pipe_length = pipe_length;
    p->is_z0 = is_z0;
    config->num_pipes++;

    /* Update maximum extensions */
    if (is_z0 && pipe_length > config->z0_extension)
        config->z0_extension = pipe_length;
    if (!is_z0 && pipe_length > config->zL_extension)
        config->zL_extension = pipe_length;
}

void endcap_pipe_config_print(const EndcapPipeConfig* config) {
    printf("\n");
    printf("======== ENDCAP PIPE CONFIGURATION ========\n");
    printf("  Number of endcap pipes: %d\n", config->num_pipes);
    printf("  z=0 extension: %.1f mm\n", config->z0_extension * 1000);
    printf("  z=L extension: %.1f mm\n", config->zL_extension * 1000);

    for (int i = 0; i < config->num_pipes; i++) {
        const EndcapPipe* p = &config->pipes[i];
        printf("\n  Endcap Pipe %d (%s):\n", i + 1,
            p->is_z0 ? "z=0" : "z=L");
        printf("    Position: r = %.4f m, phi = %.2f deg\n",
            p->r_center, p->phi_center * 180.0 / M_PI);
        printf("    Aperture radius: %.1f mm\n", p->aperture_radius * 1000);
        printf("    Pipe radius: %.1f mm\n", p->pipe_radius * 1000);
        printf("    Pipe length: %.1f mm\n", p->pipe_length * 1000);

        double k_c = 1.8412 / p->pipe_radius;
        double alpha = k_c;  /* Approximate: f << f_c */
        printf("    TE11 cutoff: %.1f MHz\n",
            k_c * 299792458.0 / (2.0 * M_PI) / 1e6);
        printf("    alpha = %.2f /m, decay length = %.1f mm\n",
            alpha, 1000.0 / alpha);
        printf("    Field at PEC cap: e^(-alpha*L) = %.4f%%\n",
            100.0 * exp(-alpha * p->pipe_length));
    }
    printf("============================================\n");
}

/*=============================================================================
 * Point-in-vacuum for endcap pipe region
 *
 * Coordinate system: z is in the EXTENDED grid coordinates
 *   z=0 in the extended grid corresponds to the PEC cap behind z=0 pipe
 *   The original cavity z=0 endplate is at z = z0_extension
 *   The original cavity z=L endplate is at z = z0_extension + L
 *
 * This function works in ORIGINAL coordinates:
 *   z < 0:           z=0 pipe region
 *   0 <= z <= L:     cavity (always vacuum)
 *   z > L:           z=L pipe region
 *============================================================================*/

int point_is_vacuum_endcap(
    const EndcapPipeConfig* config,
    double L_cavity,
    double r,
    double phi,
    double z
) {
    /* Inside cavity: always vacuum */
    if (z >= 0.0 && z <= L_cavity) {
        return 1;
    }

    /* In z=0 pipe region (z < 0) */
    if (z < 0.0) {
        double dz = -z;  /* Distance from endplate into pipe */

        for (int i = 0; i < config->num_pipes; i++) {
            const EndcapPipe* p = &config->pipes[i];
            if (!p->is_z0) continue;

            /* Beyond PEC cap */
            if (dz > p->pipe_length) continue;

            /* Distance from pipe axis on the endplate */
            double dphi = phi - p->phi_center;
            while (dphi > M_PI) dphi -= 2.0 * M_PI;
            while (dphi < -M_PI) dphi += 2.0 * M_PI;

            /* Exact distance on flat disk */
            double dist_sq = r * r + p->r_center * p->r_center
                - 2.0 * r * p->r_center * cos(dphi);

            /* Use aperture_radius at endplate, pipe_radius deep inside */
            /* Quarter-circle fillet (matching radial pipe style) */
            double R_f = p->aperture_radius - p->pipe_radius;
            double R_local;
            if (R_f < 1e-14 || dz >= R_f) {
                R_local = p->pipe_radius;
            }
            else {
                double arg = R_f * R_f - (R_f - dz) * (R_f - dz);
                if (arg < 0.0) arg = 0.0;
                R_local = (p->pipe_radius + R_f) - sqrt(arg);
            }

            if (dist_sq < R_local * R_local) {
                return 1;
            }
        }
        return 0;  /* PEC (endplate body) */
    }

    /* In z=L pipe region (z > L) */
    if (z > L_cavity) {
        double dz = z - L_cavity;  /* Distance from endplate into pipe */

        for (int i = 0; i < config->num_pipes; i++) {
            const EndcapPipe* p = &config->pipes[i];
            if (p->is_z0) continue;

            if (dz > p->pipe_length) continue;

            double dphi = phi - p->phi_center;
            while (dphi > M_PI) dphi -= 2.0 * M_PI;
            while (dphi < -M_PI) dphi += 2.0 * M_PI;

            double dist_sq = r * r + p->r_center * p->r_center
                - 2.0 * r * p->r_center * cos(dphi);

            /* Quarter-circle fillet (matching radial pipe style) */
            double R_f = p->aperture_radius - p->pipe_radius;
            double R_local;
            if (R_f < 1e-14 || dz >= R_f) {
                R_local = p->pipe_radius;
            }
            else {
                double arg = R_f * R_f - (R_f - dz) * (R_f - dz);
                if (arg < 0.0) arg = 0.0;
                R_local = (p->pipe_radius + R_f) - sqrt(arg);
            }

            if (dist_sq < R_local * R_local) {
                return 1;
            }
        }
        return 0;
    }

    return 0;
}

/*=============================================================================
 * Extended grid: radial + z-direction pipes
 *============================================================================*/

void grid_init_with_all_pipes(
    GridParams* grid,
    double a,
    double b,
    double L,
    double pipe_length_r,
    double pipe_length_z0,
    double pipe_length_zL,
    int Nr_cavity,
    int Nr_pipe,
    int Nphi,
    int Nz_cavity,
    int Nz_pipe_z0,
    int Nz_pipe_zL
) {
    double r_max = b + pipe_length_r;
    int Nr_total = Nr_cavity + Nr_pipe;

    double dz = L / Nz_cavity;

    double actual_z0 = Nz_pipe_z0 * dz;
    double actual_zL = Nz_pipe_zL * dz;
    double L_total = actual_z0 + L + actual_zL;
    int Nz_total = Nz_pipe_z0 + Nz_cavity + Nz_pipe_zL;

    //double L_total = pipe_length_z0 + L + pipe_length_zL;
    //int Nz_total = Nz_pipe_z0 + Nz_cavity + Nz_pipe_zL;

    /*
     * IMPORTANT: The grid now covers:
     *   r ∈ [a, b + pipe_length_r]
     *   z ∈ [0, L_total]
     *
     * Mapping to physical coordinates:
     *   z_physical = z_grid - pipe_length_z0
     *   z_grid = 0                    → z_phys = -pipe_length_z0 (PEC cap)
     *   z_grid = pipe_length_z0       → z_phys = 0 (original endplate)
     *   z_grid = pipe_length_z0 + L   → z_phys = L (original endplate)
     *   z_grid = L_total              → z_phys = L + pipe_length_zL (PEC cap)
     */

    grid_init(grid, a, r_max, L_total, Nr_total, Nphi, Nz_total);

    printf("  Extended grid (radial + axial pipes):\n");
    printf("    r ∈ [%.4f, %.4f] m (cavity) + [%.4f, %.4f] m (pipe)\n",
        a, b, b, r_max);
    printf("    z ∈ [%.4f, %.4f] m (z0-pipe) + [%.4f, %.4f] m (cavity)"
        " + [%.4f, %.4f] m (zL-pipe)\n",
        -pipe_length_z0, 0.0, 0.0, L, L, L + pipe_length_zL);
    printf("    Nr = %d (cavity: %d + pipe: %d)\n",
        Nr_total, Nr_cavity, Nr_pipe);
    printf("    Nz = %d (z0-pipe: %d + cavity: %d + zL-pipe: %d)\n",
        Nz_total, Nz_pipe_z0, Nz_cavity, Nz_pipe_zL);
    printf("    dr = %.2f mm,  dz = %.2f mm\n",
        grid->dr * 1000, grid->dz * 1000);
    printf("    z-offset: grid k=0 → z_phys = %.4f m\n", -pipe_length_z0);
    printf("    Cavity starts at k = %d\n", Nz_pipe_z0);
    printf("    Cavity ends   at k = %d\n", Nz_pipe_z0 + Nz_cavity);
}


/*=============================================================================
 * Material Mask
 *============================================================================*/

void material_mask_build(
    MaterialMask* mask,
    const PipeConfig* config,
    const GridParams* grid
) {
    int Nr = grid->Nr;
    int Nphi = grid->Nphi;
    int Nz = grid->Nz;

    /* Er at (i+1/2, j, k): i=0..Nr-1, j=0..Nphi-1, k=0..Nz */
    mask->size_Er = Nr * Nphi * (Nz + 1);
    mask->mask_Er = (int*)malloc(mask->size_Er * sizeof(int));

    mask->num_vacuum_Er = 0;
    for (int k = 0; k <= Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            for (int i = 0; i < Nr; i++) {
                double r = grid->a + (i + 0.5) * grid->dr;
                int idx = i + Nr * (j + Nphi * k);

                mask->mask_Er[idx] = point_is_vacuum(config, r, phi, z);
                mask->num_vacuum_Er += mask->mask_Er[idx];
            }
        }
    }

    /* Ephi at (i, j+1/2, k): i=0..Nr, j=0..Nphi-1, k=0..Nz */
    mask->size_Ephi = (Nr + 1) * Nphi * (Nz + 1);
    mask->mask_Ephi = (int*)malloc(mask->size_Ephi * sizeof(int));

    mask->num_vacuum_Ephi = 0;
    for (int k = 0; k <= Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            for (int i = 0; i <= Nr; i++) {
                double r = grid->a + i * grid->dr;
                int idx = i + (Nr + 1) * (j + Nphi * k);

                mask->mask_Ephi[idx] = point_is_vacuum(config, r, phi, z);
                mask->num_vacuum_Ephi += mask->mask_Ephi[idx];
            }
        }
    }

    /* Ez at (i, j, k+1/2): i=0..Nr, j=0..Nphi-1, k=0..Nz-1 */
    mask->size_Ez = (Nr + 1) * Nphi * Nz;
    mask->mask_Ez = (int*)malloc(mask->size_Ez * sizeof(int));

    mask->num_vacuum_Ez = 0;
    for (int k = 0; k < Nz; k++) {
        double z = (k + 0.5) * grid->dz;
        for (int j = 0; j < Nphi; j++) {
            double phi = j * grid->dphi;
            for (int i = 0; i <= Nr; i++) {
                double r = grid->a + i * grid->dr;
                int idx = i + (Nr + 1) * (j + Nphi * k);

                mask->mask_Ez[idx] = point_is_vacuum(config, r, phi, z);
                mask->num_vacuum_Ez += mask->mask_Ez[idx];
            }
        }
    }
}


/*=============================================================================
 * Combined material mask: radial pipes + endcap pipes
 *
 * Uses the extended grid coordinates where:
 *   z_physical = z_grid - z0_extension
 *============================================================================*/

void material_mask_build_full(
    MaterialMask* mask,
    const PipeConfig* radial_pipes,
    const EndcapPipeConfig* endcap_pipes,
    const GridParams* grid,
    double L_cavity,
    double z0_extension
) {
    int Nr = grid->Nr;
    int Nphi = grid->Nphi;
    int Nz = grid->Nz;

    /* Er at (i+1/2, j, k) */
    mask->size_Er = Nr * Nphi * (Nz + 1);
    mask->mask_Er = (int*)malloc(mask->size_Er * sizeof(int));
    mask->num_vacuum_Er = 0;

    for (int k = 0; k <= Nz; k++) {
        double z_grid = k * grid->dz;
        double z_phys = z_grid - z0_extension;

        for (int j = 0; j < Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            for (int i = 0; i < Nr; i++) {
                double r = grid->a + (i + 0.5) * grid->dr;
                int idx = i + Nr * (j + Nphi * k);
                int vac = 0;

                if (z_phys >= 0.0 && z_phys <= L_cavity) {
                    /* Inside original cavity z-range: use radial pipe logic */
                    vac = point_is_vacuum(radial_pipes, r, phi, z_phys);
                }
                else {
                    /* Outside cavity: only vacuum if inside an endcap pipe */
                    if (endcap_pipes) {
                        vac = point_is_vacuum_endcap(endcap_pipes, L_cavity,
                            r, phi, z_phys);
                    }
                }

                mask->mask_Er[idx] = vac;
                mask->num_vacuum_Er += vac;
            }
        }
    }

    /* Ephi at (i, j+1/2, k) */
    mask->size_Ephi = (Nr + 1) * Nphi * (Nz + 1);
    mask->mask_Ephi = (int*)malloc(mask->size_Ephi * sizeof(int));
    mask->num_vacuum_Ephi = 0;

    for (int k = 0; k <= Nz; k++) {
        double z_grid = k * grid->dz;
        double z_phys = z_grid - z0_extension;

        for (int j = 0; j < Nphi; j++) {
            double phi = (j + 0.5) * grid->dphi;
            for (int i = 0; i <= Nr; i++) {
                double r = grid->a + i * grid->dr;
                int idx = i + (Nr + 1) * (j + Nphi * k);
                int vac = 0;

                if (z_phys >= 0.0 && z_phys <= L_cavity) {
                    vac = point_is_vacuum(radial_pipes, r, phi, z_phys);
                }
                else {
                    if (endcap_pipes) {
                        vac = point_is_vacuum_endcap(endcap_pipes, L_cavity,
                            r, phi, z_phys);
                    }
                }

                mask->mask_Ephi[idx] = vac;
                mask->num_vacuum_Ephi += vac;
            }
        }
    }

    /* Ez at (i, j, k+1/2) */
    mask->size_Ez = (Nr + 1) * Nphi * Nz;
    mask->mask_Ez = (int*)malloc(mask->size_Ez * sizeof(int));
    mask->num_vacuum_Ez = 0;

    for (int k = 0; k < Nz; k++) {
        double z_grid = (k + 0.5) * grid->dz;
        double z_phys = z_grid - z0_extension;

        for (int j = 0; j < Nphi; j++) {
            double phi = j * grid->dphi;
            for (int i = 0; i <= Nr; i++) {
                double r = grid->a + i * grid->dr;
                int idx = i + (Nr + 1) * (j + Nphi * k);
                int vac = 0;

                if (z_phys >= 0.0 && z_phys <= L_cavity) {
                    vac = point_is_vacuum(radial_pipes, r, phi, z_phys);
                }
                else {
                    if (endcap_pipes) {
                        vac = point_is_vacuum_endcap(endcap_pipes, L_cavity,
                            r, phi, z_phys);
                    }
                }

                mask->mask_Ez[idx] = vac;
                mask->num_vacuum_Ez += vac;
            }
        }
    }
}


void material_mask_free(MaterialMask* mask) {
    if (mask->mask_Er) { free(mask->mask_Er);   mask->mask_Er = NULL; }
    if (mask->mask_Ephi) { free(mask->mask_Ephi); mask->mask_Ephi = NULL; }
    if (mask->mask_Ez) { free(mask->mask_Ez);   mask->mask_Ez = NULL; }
}

void material_mask_print_stats(const MaterialMask* mask, const GridParams* grid) {
    int total = mask->size_Er + mask->size_Ephi + mask->size_Ez;
    int vacuum = mask->num_vacuum_Er + mask->num_vacuum_Ephi + mask->num_vacuum_Ez;
    int pec = total - vacuum;

    double b = grid->a + grid->Nr * grid->dr;  /* This is now b + pipe_length */

    printf("\n  Material Mask Statistics:\n");
    printf("    Domain: r ∈ [%.4f, %.4f] m\n", grid->a, b);
    printf("    Er:   %d vacuum / %d total (%.1f%%)\n",
        mask->num_vacuum_Er, mask->size_Er,
        100.0 * mask->num_vacuum_Er / mask->size_Er);
    printf("    Ephi: %d vacuum / %d total (%.1f%%)\n",
        mask->num_vacuum_Ephi, mask->size_Ephi,
        100.0 * mask->num_vacuum_Ephi / mask->size_Ephi);
    printf("    Ez:   %d vacuum / %d total (%.1f%%)\n",
        mask->num_vacuum_Ez, mask->size_Ez,
        100.0 * mask->num_vacuum_Ez / mask->size_Ez);
    printf("    Total: %d vacuum, %d PEC (%.1f%% PEC)\n",
        vacuum, pec, 100.0 * pec / total);
}

/* Apply mask: zero E in PEC cells */
void apply_material_mask(EField* E, const MaterialMask* mask) {
    for (int i = 0; i < mask->size_Er; i++) {
        if (!mask->mask_Er[i]) E->Er[i] = 0.0;
    }
    for (int i = 0; i < mask->size_Ephi; i++) {
        if (!mask->mask_Ephi[i]) E->Ephi[i] = 0.0;
    }
    for (int i = 0; i < mask->size_Ez; i++) {
        if (!mask->mask_Ez[i]) E->Ez[i] = 0.0;
    }
}

/*=============================================================================
 * Extended grid initialization
 *============================================================================*/

void grid_init_with_pipes(
    GridParams* grid,
    double a,
    double b,
    double L,
    double pipe_length,
    int Nr_cavity,
    int Nr_pipe,
    int Nphi,
    int Nz
) {
    double r_max = b + pipe_length;
    int Nr_total = Nr_cavity + Nr_pipe;

    grid_init(grid, a, r_max, L, Nr_total, Nphi, Nz);

    printf("  Extended grid:\n");
    printf("    r ∈ [%.4f, %.4f] m (cavity) + [%.4f, %.4f] m (pipe)\n",
        a, b, b, r_max);
    printf("    Nr = %d (cavity: %d + pipe: %d)\n", Nr_total, Nr_cavity, Nr_pipe);
    printf("    dr = %.2f mm\n", grid->dr * 1000);
    printf("    Cavity cells: dr_cavity = %.2f mm\n", (b - a) / Nr_cavity * 1000);
    printf("    Pipe cells:   dr_pipe = %.2f mm\n", pipe_length / Nr_pipe * 1000);

    /* Check that dr is uniform (it is, since grid_init uses total Nr) */
    /* Note: dr = (r_max - a) / Nr_total */
    /* This means cavity and pipe have SAME dr */
    /* For better resolution, we might want different dr in pipe region */
    /* But uniform grid is simpler and works with existing curl code */
}

void material_mask_build_ibc(
    MaterialMask* ibc_mask,
    const MaterialMask* pec_mask,
    const GridParams* grid
) {
    int Nr = grid->Nr, Nphi = grid->Nphi, Nz = grid->Nz;
    int Nr1 = Nr + 1;

    /* Start with a copy of the PEC mask */
    ibc_mask->size_Er = pec_mask->size_Er;
    ibc_mask->size_Ephi = pec_mask->size_Ephi;
    ibc_mask->size_Ez = pec_mask->size_Ez;

    ibc_mask->mask_Er = (int*)malloc(ibc_mask->size_Er * sizeof(int));
    ibc_mask->mask_Ephi = (int*)malloc(ibc_mask->size_Ephi * sizeof(int));
    ibc_mask->mask_Ez = (int*)malloc(ibc_mask->size_Ez * sizeof(int));

    memcpy(ibc_mask->mask_Er, pec_mask->mask_Er, ibc_mask->size_Er * sizeof(int));
    memcpy(ibc_mask->mask_Ephi, pec_mask->mask_Ephi, ibc_mask->size_Ephi * sizeof(int));
    memcpy(ibc_mask->mask_Ez, pec_mask->mask_Ez, ibc_mask->size_Ez * sizeof(int));

    int flipped_Er = 0, flipped_Ephi = 0, flipped_Ez = 0;

    /* Er(i+½, j, k): tangential to φ-walls and z-walls.
     * Flip if PEC and has vacuum neighbor in φ or z direction. */
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr; i++) {
                int idx = i + Nr * (j + Nphi * k);
                if (pec_mask->mask_Er[idx] != 0) continue;  /* already vacuum */

                int jm1 = (j - 1 + Nphi) % Nphi;
                int jp1 = (j + 1) % Nphi;
                int has_vac = 0;

                /* φ neighbors */
                if (pec_mask->mask_Er[i + Nr * (jm1 + Nphi * k)]) has_vac = 1;
                if (pec_mask->mask_Er[i + Nr * (jp1 + Nphi * k)]) has_vac = 1;
                /* z neighbors */
                if (k > 0 && pec_mask->mask_Er[i + Nr * (j + Nphi * (k - 1))]) has_vac = 1;
                if (k < Nz && pec_mask->mask_Er[i + Nr * (j + Nphi * (k + 1))]) has_vac = 1;

                if (has_vac) {
                    ibc_mask->mask_Er[idx] = 1;
                    flipped_Er++;
                }
            }
        }
    }

    /* Ephi(i, j+½, k): tangential to r-walls and z-walls.
     * Flip if PEC and has vacuum neighbor in r or z direction. */
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i <= Nr; i++) {
                int idx = i + Nr1 * (j + Nphi * k);
                if (pec_mask->mask_Ephi[idx] != 0) continue;

                int has_vac = 0;
                /* r neighbors */
                if (i > 0 && pec_mask->mask_Ephi[(i - 1) + Nr1 * (j + Nphi * k)]) has_vac = 1;
                if (i < Nr && pec_mask->mask_Ephi[(i + 1) + Nr1 * (j + Nphi * k)]) has_vac = 1;
                /* z neighbors */
                if (k > 0 && pec_mask->mask_Ephi[i + Nr1 * (j + Nphi * (k - 1))]) has_vac = 1;
                if (k < Nz && pec_mask->mask_Ephi[i + Nr1 * (j + Nphi * (k + 1))]) has_vac = 1;

                if (has_vac) {
                    ibc_mask->mask_Ephi[idx] = 1;
                    flipped_Ephi++;
                }
            }
        }
    }

    /* Ez(i, j, k+½): tangential to r-walls and φ-walls.
     * Flip if PEC and has vacuum neighbor in r or φ direction. */
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i <= Nr; i++) {
                int idx = i + Nr1 * (j + Nphi * k);
                if (pec_mask->mask_Ez[idx] != 0) continue;

                int has_vac = 0;
                /* r neighbors */
                if (i > 0 && pec_mask->mask_Ez[(i - 1) + Nr1 * (j + Nphi * k)]) has_vac = 1;
                if (i < Nr && pec_mask->mask_Ez[(i + 1) + Nr1 * (j + Nphi * k)]) has_vac = 1;
                /* φ neighbors */
                int jm1 = (j - 1 + Nphi) % Nphi;
                int jp1 = (j + 1) % Nphi;
                if (pec_mask->mask_Ez[i + Nr1 * (jm1 + Nphi * k)]) has_vac = 1;
                if (pec_mask->mask_Ez[i + Nr1 * (jp1 + Nphi * k)]) has_vac = 1;

                if (has_vac) {
                    ibc_mask->mask_Ez[idx] = 1;
                    flipped_Ez++;
                }
            }
        }
    }

    printf("  IBC mask: flipped %d Er + %d Ephi + %d Ez = %d surface cells\n",
        flipped_Er, flipped_Ephi, flipped_Ez,
        flipped_Er + flipped_Ephi + flipped_Ez);
}