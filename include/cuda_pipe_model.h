#ifndef CUDA_PIPE_MODEL_H
#define CUDA_PIPE_MODEL_H

#include "pipe_model.h"
#include "cuda_fields.h"
#include "cuda_operator.h"
#include "cuda_eigensolver.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*=============================================================================
     * GPU Material Mask (unchanged)
     *============================================================================*/

    typedef struct {
        int* d_mask_Er;
        int* d_mask_Ephi;
        int* d_mask_Ez;

        int size_Er;
        int size_Ephi;
        int size_Ez;
    } GPU_MaterialMask;

    int gpu_material_mask_init(GPU_MaterialMask* gpu_mask, const MaterialMask* cpu_mask);
    void gpu_material_mask_free(GPU_MaterialMask* gpu_mask);

    /*=============================================================================
     * IBC Boundary Weights for Pipe Walls
     *
     * Precomputed from the material mask: for each vacuum E-field cell that
     * has a PEC neighbor in a direction where E is tangential to the wall,
     * the weight = (surface_area / cell_volume) = 1/dr, 1/dz, or 1/(r*dphi).
     *
     * During the matvec, the IBC surface correction is:
     *   result += β * weight * E   (complex multiply, β = (1-j)/(2α))
     *
     * This handles: outer conductor, endplates, and all pipe curved walls
     * in one unified pass. Aperture holes are naturally excluded (both sides
     * vacuum → weight = 0). The inner conductor (i=0, grid edge) is handled
     * separately since there's no i=-1 mask cell.
     *============================================================================*/

    typedef struct {
        double* d_weight_Er;        /* size: same as mask_Er */
        double* d_weight_Ephi;      /* size: same as mask_Ephi */
        double* d_weight_Ez;        /* size: same as mask_Ez */

        int size_Er;
        int size_Ephi;
        int size_Ez;

        int num_boundary_Er;        /* count of nonzero weight cells */
        int num_boundary_Ephi;
        int num_boundary_Ez;
    } GPU_IBCWeights;

    /* Build IBC weights from material mask (CPU computation + GPU upload) */
    int gpu_ibc_weights_build(
        GPU_IBCWeights* w,
        const MaterialMask* pec_mask,    /* original PEC mask */
        const MaterialMask* ibc_mask,    /* IBC mask (surface cells unmasked) */
        const GridParams* grid
    );
    void gpu_ibc_weights_free(GPU_IBCWeights* weights);

    /*=============================================================================
     * GPU Operator with pipe model (extended for IBC)
     *============================================================================*/

    typedef struct {
        GPU_Operator base;
        GPU_MaterialMask mask;
        int has_mask;

        GPU_MaterialMask ibc_mask;      /* IBC mask (surface cells unmasked) */
        int has_ibc_mask;

        /* Interior endplate PEC enforcement (for PEC mode) */
        int k_endplate_z0;
        int k_endplate_zL;
        int* d_mask_Er_endplate_z0;
        int* d_mask_Er_endplate_zL;
        int* d_mask_Ephi_endplate_z0;
        int* d_mask_Ephi_endplate_zL;
        int has_interior_endplates;

        /* IBC boundary weights (for complex/IBC mode) */
        GPU_IBCWeights ibc_weights;
        int has_ibc_weights;
    } GPU_PipeOperator;

    /*=============================================================================
     * Lifecycle — PEC mode (unchanged)
     *============================================================================*/

    int gpu_pipe_operator_init(
        GPU_PipeOperator* pipe_op,
        const CurlCurlOperator* cpu_op,
        const MaterialMask* mask
    );

    int gpu_pipe_operator_set_endplates(
        GPU_PipeOperator* pipe_op,
        int k_z0,
        int k_zL,
        const EndcapPipeConfig* endcap_pipes,
        const GridParams* grid,
        double z0_extension
    );

    void gpu_pipe_operator_free(GPU_PipeOperator* pipe_op);

    /*=============================================================================
     * Lifecycle — IBC complex mode
     *
     * Allocates complex working arrays (has_imag=1) and precomputes IBC
     * boundary weights from the material mask.
     *============================================================================*/

    int gpu_pipe_operator_init_complex(
        GPU_PipeOperator* pipe_op,
        const CurlCurlOperator* cpu_op,
        const MaterialMask* mask
    );

    /*=============================================================================
     * PEC Matvec (unchanged)
     *============================================================================*/

    int gpu_pipe_matvec(
        const GPU_PipeOperator* pipe_op,
        const double* d_x,
        double* d_y
    );

    /*=============================================================================
     * IBC Complex Matvec: y = (∇×∇× + β·S) · x
     *
     * x, y are size 2*n_total = [real | imag].
     *
     * Flow:
     *   1. Unpack x → E (complex)
     *   2. PEC at pipe caps: i=Nr, k=0, k=Nz (but NOT i=0: inner conductor has IBC)
     *   3. Material mask (both re+im)
     *   4. ∇×∇×E (complex)
     *   5. IBC at inner conductor (i=0): result += β/dr · E  (with port masks)
     *   6. IBC from boundary weights: result += β · weight · E
     *   7. PEC at pipe caps (on result)
     *   8. Material mask on result (both re+im)
     *   9. Pack → y
     *
     * alpha = R_s/(ωμ₀) passed by caller.
     *============================================================================*/

    int gpu_pipe_matvec_complex(
        const GPU_PipeOperator* pipe_op,
        const double* d_x,
        double* d_y,
        double alpha
    );

    /* Apply material mask on GPU (real only, unchanged) */
    int gpu_apply_material_mask(
        GPU_EField* E,
        const GPU_MaterialMask* mask
    );

    /* Apply material mask to both real and imaginary parts */
    int gpu_apply_material_mask_complex(
        GPU_EField* E,
        const GPU_MaterialMask* mask
    );

    /*=============================================================================
     * Convenience: IBC Complex RQI for pipe model
     *
     * Wraps gpu_pipe_matvec_complex into the generic RQI interface.
     *============================================================================*/

    GPU_ComplexEigenResult gpu_rqi_complex_pipe(
        const GPU_PipeOperator* pipe_op,
        double* d_x,
        double sigma_init,
        double conductivity,
        int max_iter,
        double tol,
        int gmres_restart
    );

    /*=============================================================================
     * Perturbative IBC Q (single matvec, no iteration)
     *
     * Computes k²_IBC = ⟨x_PEC, A_IBC·x_PEC⟩ / ⟨x_PEC, x_PEC⟩
     * Cost: 1 complex matvec + 2 dot products. Runs in seconds.
     * Accuracy: O(perturbation²) — negligible vs O(dr) grid error.
     *============================================================================*/

    GPU_ComplexEigenResult gpu_ibc_perturbative_pipe(
        const GPU_PipeOperator* pipe_op,
        const double* d_x_pec,     /* PEC eigenvector on GPU, size n_real */
        double k2_pec,
        double conductivity
    );

#ifdef __cplusplus
}
#endif

#endif /* CUDA_PIPE_MODEL_H */
