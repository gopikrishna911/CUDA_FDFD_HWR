/*============================================================================
 * cpu_conformal.h
 *
 * Stage 2 of the CPU reference port: conformal Dey-Mittra subcell curls
 * and the full conformal-pipe complex matvec with IBC, mirroring
 *   cuda_conformal_curls.cu  ->  cpu_conformal.cpp (conformal curls)
 *   cuda_conformal_pipe.cu   ->  cpu_conformal.cpp (9-step pipe matvec)
 *
 * The geometry engine (conformal_geometry.cpp) and mask/pipe model
 * (pipe_model.cpp) are ALREADY pure host code, so this module reuses
 * ConformalData / MaterialMask / PipeConfig / EndcapPipeConfig directly
 * and only ports the compute kernels that consume them.
 *
 * Memory layout matches Stage 1: complex packed vector x[0..n-1]=real,
 * x[n..2n-1]=imag; each half ordered [Er | Ephi | Ez].
 *==========================================================================*/
#ifndef CPU_CONFORMAL_H
#define CPU_CONFORMAL_H

#include "curl_E.h"
#include "curlcurl_operator.h"
#include "pipe_model.h"
#include "conformal_geometry.h"
#include "cpu_reference.h"   /* ComplexEField, CpuComplexEigenResult, GMRES ws */

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * CPU staircase IBC weights (mirrors GPU_IBCWeights; built host-side by
 * the same logic as gpu_ibc_weights_build).
 *------------------------------------------------------------------------*/
typedef struct {
    double* weight_Er;     /* size_Er   */
    double* weight_Ephi;   /* size_Ephi */
    double* weight_Ez;     /* size_Ez   */
    int size_Er, size_Ephi, size_Ez;
    int num_boundary_Er, num_boundary_Ephi, num_boundary_Ez;
} CpuIBCWeights;

int  cpu_ibc_weights_build(CpuIBCWeights* w,
                           const MaterialMask* pec_mask,
                           const MaterialMask* ibc_mask,
                           const GridParams* grid);
void cpu_ibc_weights_free(CpuIBCWeights* w);

/*--------------------------------------------------------------------------
 * CPU conformal-pipe operator (mirrors GPU_ConformalPipeOperator).
 * Holds borrowed pointers to host ConformalData and the IBC MaterialMask
 * (owned by the caller), plus the staircase weights and endplate masks
 * it builds itself.
 *------------------------------------------------------------------------*/
typedef struct {
    const CurlCurlOperator* op;     /* borrowed: grid, sizes, offsets */
    const ConformalData*    cd;     /* borrowed: edge_frac, areas, ibc_weight */
    const MaterialMask*     ibc_mask;  /* borrowed: surface-cell-unmasked mask */
    int has_ibc_mask;

    CpuIBCWeights ibc_weights;      /* owned: staircase weights */
    int has_ibc_weights;

    /* Endplate grid-plane IBC */
    int k_endplate_z0;
    int k_endplate_zL;
    int* mask_Er_z0;     /* owned: 1=pipe(skip) 0=wall(apply); size Nr*Nphi   */
    int* mask_Ephi_z0;   /* owned: size (Nr+1)*Nphi */
    int* mask_Er_zL;
    int* mask_Ephi_zL;
    int has_endplate_ibc;

    /* Inner-conductor beam-port masks (Phase B1). 1 = inside a port
     * aperture (skip inner-conductor IBC there), 0 = solid conductor.
     * Built from op->ports via point_in_port_cylindrical. */
    int* port_mask_Ephi_inner;  /* owned: size Nphi*(Nz+1) */
    int* port_mask_Ez_inner;    /* owned: size Nphi*Nz     */
    int has_inner_port_masks;

    /* Complex work fields (re+im), allocated once */
    ComplexEField E_work;
    ComplexHField H_temp;
    ComplexEField result_work;
    int initialized;
} CpuConformalPipeOperator;

/* Build the operator. pec_mask is the original PEC mask, ibc_mask the
 * surface-unmasked IBC mask (both host, caller-owned and kept alive).
 * k_z0/k_zL are the endplate grid planes (or -1 to disable). endcap_pipes
 * (or NULL) supplies the endplate aperture masks, mirroring the GPU init. */
int cpu_conformal_pipe_operator_init(
    CpuConformalPipeOperator* cop,
    const CurlCurlOperator* cpu_op,
    const ConformalData* cd,
    const MaterialMask* pec_mask,
    const MaterialMask* ibc_mask,
    int k_z0, int k_zL,
    const EndcapPipeConfig* endcap_pipes,
    const GridParams* grid);

void cpu_conformal_pipe_operator_free(CpuConformalPipeOperator* cop);

/*--------------------------------------------------------------------------
 * Conformal curls (real; complex = run twice on re/im halves).
 * Output H written into the ComplexHField re/im arrays.
 *------------------------------------------------------------------------*/
void cpu_compute_curl_E_conformal_omp(const ComplexEField* E,
                                      ComplexHField* curlE,
                                      const GridParams* grid,
                                      const ConformalData* cd,
                                      int do_imag);
void cpu_compute_curl_H_conformal_omp(const ComplexHField* H,
                                      ComplexEField* curlH,
                                      const GridParams* grid,
                                      const ConformalData* cd,
                                      int do_imag);
void cpu_compute_curl_curl_E_conformal_complex_omp(
    const ComplexEField* E, ComplexEField* result,
    ComplexHField* temp, const GridParams* grid,
    const ConformalData* cd);

/*--------------------------------------------------------------------------
 * The 9-step conformal-pipe complex matvec: y = A_conformal_IBC x.
 * x, y are packed complex vectors of length 2*n_total.
 *------------------------------------------------------------------------*/
int cpu_conformal_pipe_matvec_complex_omp(
    const CpuConformalPipeOperator* cop,
    const double* x, double* y, double alpha);

/*--------------------------------------------------------------------------
 * Complex RQI driving the conformal-pipe matvec (mirrors
 * gpu_rqi_complex_conformal_pipe). Reuses the Stage 1 complex GMRES +
 * RQI machinery via a thin operator-agnostic path.
 *------------------------------------------------------------------------*/
CpuComplexEigenResult cpu_rqi_complex_conformal_pipe_omp(
    const CpuConformalPipeOperator* cop,
    double* x, double sigma_init, double conductivity,
    int max_iter, double tol, int gmres_restart);

#ifdef __cplusplus
}
#endif

#endif /* CPU_CONFORMAL_H */
