#ifndef PIPE_MODEL_H
#define PIPE_MODEL_H

#include "curl_E.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*=============================================================================
     * Beam Pipe Configuration
     *============================================================================*/

    typedef struct {
        double phi_center;      /* Azimuthal position of pipe axis */
        double z_center;        /* Axial position of pipe axis */
    } PipeLocation;

    typedef struct {
        /* Pipe geometry */
        double pipe_radius;         /* Inner pipe radius (25mm) */
        double aperture_radius;     /* Aperture radius at outer wall (35mm) */
        double pipe_length;         /* Pipe length beyond outer wall (50mm) */
        double taper_length;        /* Radial distance for fillet taper (10mm) */

        /* Cavity geometry (for reference) */
        double r_inner;             /* a - inner conductor */
        double r_outer;             /* b - outer conductor wall */

        /* Pipe locations */
        int num_pipes;
        int capacity;
        PipeLocation* pipes;
    } PipeConfig;

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
        double taper_length
    );

    void pipe_config_free(PipeConfig* config);

    void pipe_config_add(PipeConfig* config, double phi_center, double z_center);

    /* Add standard single-pass beam pipes (φ = 0 and φ = π) */
    void pipe_config_add_single_pass(PipeConfig* config, double z_center);

    /* Add multi-pass beam pipes */
    void pipe_config_add_multi_pass(PipeConfig* config, double z_center, int num_passes);

    void pipe_config_print(const PipeConfig* config);

    /*=============================================================================
     * Pipe radius at given radial position (includes fillet taper)
     *
     * Returns the pipe cross-section radius at radial position r:
     *   r <= r_outer:                   aperture_radius (35mm)
     *   r_outer < r < r_outer+taper:   linear taper from 35mm to 25mm
     *   r >= r_outer + taper:          pipe_radius (25mm)
     *============================================================================*/

    double pipe_radius_at_r(const PipeConfig* config, double r);

    /*=============================================================================
     * Point-in-pipe test
     *
     * Returns 1 if point (r, phi, z) is inside vacuum (cavity or pipe)
     * Returns 0 if point is inside PEC (outer conductor body)
     *============================================================================*/

    int point_is_vacuum(const PipeConfig* config, double r, double phi, double z);


    /*=============================================================================
    * Endcap Pipe Configuration (pipes through z=0 and z=L endplates)
    *============================================================================*/

    typedef struct {
        double r_center;        /* Radial position of pipe axis on endplate */
        double phi_center;      /* Azimuthal position of pipe axis */
        double aperture_radius; /* Aperture radius at the endplate surface */
        double pipe_radius;     /* Pipe radius (constant, no fillet for now) */
        double pipe_length;     /* Pipe extension beyond endplate */
        int is_z0;              /* 1 = z=0 endplate, 0 = z=L endplate */
    } EndcapPipe;

    typedef struct {
        int num_pipes;
        int capacity;
        EndcapPipe* pipes;

        /* Maximum extensions (computed from pipe configs) */
        double z0_extension;    /* Max pipe length extending below z=0 */
        double zL_extension;    /* Max pipe length extending above z=L */
    } EndcapPipeConfig;

    void endcap_pipe_config_init(EndcapPipeConfig* config);
    void endcap_pipe_config_free(EndcapPipeConfig* config);

    void endcap_pipe_config_add(
        EndcapPipeConfig* config,
        double r_center,
        double phi_center,
        double aperture_radius,
        double pipe_radius,
        double pipe_length,
        int is_z0
    );

    void endcap_pipe_config_print(const EndcapPipeConfig* config);

    /* Point-in-vacuum test for endcap pipe region */
    int point_is_vacuum_endcap(
        const EndcapPipeConfig* config,
        double L_cavity,        /* Original cavity length */
        double r,
        double phi,
        double z
    );

    /* Extended grid initialization including z-pipes */
    void grid_init_with_all_pipes(
        GridParams* grid,
        double a,
        double b,
        double L,
        double pipe_length_r,       /* Radial pipe extension beyond b */
        double pipe_length_z0,      /* z=0 pipe extension (below endplate) */
        double pipe_length_zL,      /* z=L pipe extension (above endplate) */
        int Nr_cavity,
        int Nr_pipe,
        int Nphi,
        int Nz_cavity,
        int Nz_pipe_z0,
        int Nz_pipe_zL
    );


    /*=============================================================================
     * Material Mask
     *
     * 3D arrays marking each E-field grid point as vacuum (1) or PEC (0).
     * Separate masks for Er, Ephi, Ez due to staggered grid.
     *============================================================================*/

    typedef struct {
        int* mask_Er;       /* size: Nr * Nphi * (Nz+1) */
        int* mask_Ephi;     /* size: (Nr+1) * Nphi * (Nz+1) */
        int* mask_Ez;       /* size: (Nr+1) * Nphi * Nz */

        int size_Er;
        int size_Ephi;
        int size_Ez;

        int num_vacuum_Er;
        int num_vacuum_Ephi;
        int num_vacuum_Ez;
    } MaterialMask;

    /* Build material mask from pipe configuration */
    void material_mask_build(
        MaterialMask* mask,
        const PipeConfig* config,
        const GridParams* grid
    );

    /* Combined material mask builder */
    void material_mask_build_full(
        MaterialMask* mask,
        const PipeConfig* radial_pipes,
        const EndcapPipeConfig* endcap_pipes,
        const GridParams* grid,
        double L_cavity,
        double z0_extension
    );

    void material_mask_free(MaterialMask* mask);
    void material_mask_print_stats(const MaterialMask* mask, const GridParams* grid);

    /* Apply mask to E-field: zero out PEC cells */
    void apply_material_mask(EField* E, const MaterialMask* mask);

    /*=============================================================================
     * Extended grid helper
     *
     * Creates a grid that extends beyond the outer conductor to include pipes
     *============================================================================*/

    void grid_init_with_pipes(
        GridParams* grid,
        double a,               /* inner radius */
        double b,               /* outer conductor radius */
        double L,               /* cavity length */
        double pipe_length,     /* pipe extension beyond b */
        int Nr_cavity,          /* radial cells in cavity region [a, b] */
        int Nr_pipe,            /* radial cells in pipe region [b, b+pipe_length] */
        int Nphi,
        int Nz
    );


    /* Build IBC-modified mask: surface PEC cells are unmasked (set to 1).
 * A "surface PEC cell" is a PEC cell (mask=0) with at least one vacuum
 * neighbor in a direction where the E-component is tangential to that wall.
 * These cells carry nonzero E_tan in IBC mode. */
    void material_mask_build_ibc(
        MaterialMask* ibc_mask,          /* output: new mask for IBC mode */
        const MaterialMask* pec_mask,    /* input: original PEC mask */
        const GridParams* grid
    );


#ifdef __cplusplus
}
#endif

#endif /* PIPE_MODEL_H */