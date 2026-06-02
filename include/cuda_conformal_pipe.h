#ifndef CUDA_CONFORMAL_PIPE_H
#define CUDA_CONFORMAL_PIPE_H

#include "cuda_pipe_model.h"
#include "cuda_conformal.h"
#include "conformal_geometry.h"

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * GPU Conformal Pipe Operator
 *
 * IBC mask approach for outer conductor + pipe walls.
 * Grid-plane IBC for endplates (avoids dz extension).
 *
 * Matvec flow:
 *   1.  Unpack
 *   2.  PEC at grid edges (i=Nr, k=0, k=Nz)
 *   3.  IBC mask (unmask surface cells)
 *   3b. Re-zero endplate surface cells (k=k_z0-1, k=k_zL+1)
 *   4.  Conformal curl-curl
 *   5.  Inner conductor IBC at i=0
 *   6a. Staircase IBC at flat walls (excl pipe cells)
 *   6b. Endplate IBC at k=k_z0, k=k_zL (grid-plane)
 *   6c. Conformal IBC at pipe walls
 *   7.  PEC at grid edges on result
 *   8.  IBC mask on result
 *   8b. Re-zero endplate surface cells on result
 *   9.  Pack
 *============================================================================*/

typedef struct {
    GPU_PipeOperator pipe_op;       /* Base: IBC mask + staircase weights */
    GPU_ConformalData conformal;    /* Pipe wall geometry */
    int has_conformal;

    /* Endplate grid-plane IBC */
    int k_endplate_z0;
    int k_endplate_zL;
    int* d_mask_Er_z0;      /* 1=pipe(skip), 0=wall(apply) at z0 endplate */
    int* d_mask_Ephi_z0;
    int* d_mask_Er_zL;
    int* d_mask_Ephi_zL;
    int has_endplate_ibc;
} GPU_ConformalPipeOperator;

int gpu_conformal_pipe_operator_init(
    GPU_ConformalPipeOperator* cop,
    const CurlCurlOperator* cpu_op,
    const MaterialMask* mask,
    const ConformalData* cd,
    int k_z0, int k_zL,
    const EndcapPipeConfig* endcap_pipes,
    const GridParams* grid,
    double z0_extension
);

void gpu_conformal_pipe_operator_free(GPU_ConformalPipeOperator* cop);

int gpu_conformal_pipe_matvec_complex(
    const GPU_ConformalPipeOperator* cop,
    const double* d_x, double* d_y, double alpha);

GPU_ComplexEigenResult gpu_rqi_complex_conformal_pipe(
    const GPU_ConformalPipeOperator* cop,
    double* d_x, double sigma_init, double conductivity,
    int max_iter, double tol, int gmres_restart);

GPU_ComplexEigenResult gpu_ibc_perturbative_conformal_pipe(
    const GPU_ConformalPipeOperator* cop,
    const double* d_x_pec, double k2_pec, double conductivity);

int gpu_conformal_pipe_print_ibc_breakdown(
    const GPU_ConformalPipeOperator* cop,
    const double* d_x,
    double k2_re,
    double conductivity);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_CONFORMAL_PIPE_H */
