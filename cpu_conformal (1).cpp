/*============================================================================
 * cpu_conformal.cpp  —  Stage 2 CPU reference (conformal Dey-Mittra + pipe IBC)
 *
 * Ports cuda_conformal_curls.cu + cuda_conformal_pipe.cu to CPU/OpenMP.
 * Reuses the host geometry engine (ConformalData) and mask model
 * (MaterialMask) unchanged. See cpu_conformal.h for the API contract.
 *==========================================================================*/
#include "cpu_conformal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*--------------------------------------------------------------------------
 * Index helpers (identical convention to curl_E.h / cuda_conformal_curls.cu)
 *------------------------------------------------------------------------*/
static inline int ci_Er  (int Nr, int Nphi, int i, int j, int k){ return i + Nr      * (j + Nphi * k); }
static inline int ci_Ephi(int Nr, int Nphi, int i, int j, int k){ return i + (Nr + 1)* (j + Nphi * k); }
static inline int ci_Ez  (int Nr, int Nphi, int i, int j, int k){ return i + (Nr + 1)* (j + Nphi * k); }
static inline int ci_Hr  (int Nr, int Nphi, int i, int j, int k){ return i + (Nr + 1)* (j + Nphi * k); }
static inline int ci_Hphi(int Nr, int Nphi, int i, int j, int k){ return i + Nr      * (j + Nphi * k); }
static inline int ci_Hz  (int Nr, int Nphi, int i, int j, int k){ return i + Nr      * (j + Nphi * k); }
static inline int cper_j (int j, int Nphi){ return ((j % Nphi) + Nphi) % Nphi; }

/*==========================================================================
 * Conformal curl-E  (integral form:  H = (1/A_vac) * line_integral(frac*E*dl))
 *
 * do_imag selects which half (re or im) of the complex fields to operate on;
 * the caller invokes it twice for a complex curl (curl is linear).
 *==========================================================================*/
void cpu_compute_curl_E_conformal_omp(const ComplexEField* E,
                                      ComplexHField* curlE,
                                      const GridParams* grid,
                                      const ConformalData* cd,
                                      int do_imag)
{
    const int Nr = grid->Nr, Nphi = grid->Nphi, Nz = grid->Nz;
    const int Nr1 = Nr + 1;
    const double a = grid->a, dr = grid->dr, dphi = grid->dphi, dz = grid->dz;

    const EField* Ein = do_imag ? &E->im : &E->re;
    HField*       Hout= do_imag ? &curlE->im : &curlE->re;

    const double* Er   = Ein->Er;
    const double* Ephi = Ein->Ephi;
    const double* Ez   = Ein->Ez;

    /* (curl E)_r  at Hr nodes: Nr1 * Nphi * Nz */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr1; i++) {
                int gid = ci_Hr(Nr, Nphi, i, j, k);
                double A = cd->face_area_Hr[gid];
                if (A < 1e-30) { Hout->Hr[gid] = 0.0; continue; }
                double r_i = a + i * dr;
                int jp1 = cper_j(j + 1, Nphi);
                int e_Ez_jp1 = ci_Ez (Nr, Nphi, i, jp1, k);
                int e_Ez_j   = ci_Ez (Nr, Nphi, i, j,   k);
                int e_Ep_kp1 = ci_Ephi(Nr, Nphi, i, j, k + 1);
                int e_Ep_k   = ci_Ephi(Nr, Nphi, i, j, k);
                double li =
                    + cd->edge_frac_Ez[e_Ez_jp1]   * Ez[e_Ez_jp1]   * dz
                    - cd->edge_frac_Ez[e_Ez_j]     * Ez[e_Ez_j]     * dz
                    - cd->edge_frac_Ephi[e_Ep_kp1] * Ephi[e_Ep_kp1] * r_i * dphi
                    + cd->edge_frac_Ephi[e_Ep_k]   * Ephi[e_Ep_k]   * r_i * dphi;
                Hout->Hr[gid] = li / A;
            }
        }
    }

    /* (curl E)_phi  at Hphi nodes: Nr * Nphi * Nz */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr; i++) {
                int gid = ci_Hphi(Nr, Nphi, i, j, k);
                double A = cd->face_area_Hphi[gid];
                if (A < 1e-30) { Hout->Hphi[gid] = 0.0; continue; }
                int e_Er_kp1 = ci_Er(Nr, Nphi, i, j, k + 1);
                int e_Er_k   = ci_Er(Nr, Nphi, i, j, k);
                int e_Ez_ip1 = ci_Ez(Nr, Nphi, i + 1, j, k);
                int e_Ez_i   = ci_Ez(Nr, Nphi, i, j, k);
                double li =
                    + cd->edge_frac_Er[e_Er_kp1] * Er[e_Er_kp1] * dr
                    - cd->edge_frac_Er[e_Er_k]   * Er[e_Er_k]   * dr
                    - cd->edge_frac_Ez[e_Ez_ip1] * Ez[e_Ez_ip1] * dz
                    + cd->edge_frac_Ez[e_Ez_i]   * Ez[e_Ez_i]   * dz;
                Hout->Hphi[gid] = li / A;
            }
        }
    }

    /* (curl E)_z  at Hz nodes: Nr * Nphi * (Nz+1) */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < Nz + 1; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr; i++) {
                int gid = ci_Hz(Nr, Nphi, i, j, k);
                double A = cd->face_area_Hz[gid];
                if (A < 1e-30) { Hout->Hz[gid] = 0.0; continue; }
                double r_i   = a + i * dr;
                double r_ip1 = a + (i + 1) * dr;
                int jp1 = cper_j(j + 1, Nphi);
                int e_Ep_ip1 = ci_Ephi(Nr, Nphi, i + 1, j, k);
                int e_Ep_i   = ci_Ephi(Nr, Nphi, i, j, k);
                int e_Er_jp1 = ci_Er(Nr, Nphi, i, jp1, k);
                int e_Er_j   = ci_Er(Nr, Nphi, i, j, k);
                double li =
                    + cd->edge_frac_Ephi[e_Ep_ip1] * Ephi[e_Ep_ip1] * r_ip1 * dphi
                    - cd->edge_frac_Ephi[e_Ep_i]   * Ephi[e_Ep_i]   * r_i   * dphi
                    - cd->edge_frac_Er[e_Er_jp1]   * Er[e_Er_jp1]   * dr
                    + cd->edge_frac_Er[e_Er_j]     * Er[e_Er_j]     * dr;
                Hout->Hz[gid] = li / A;
            }
        }
    }
}

/*==========================================================================
 * Conformal curl-H  (E_dual = (1/A_dual_vac) * line_integral(H*dl))
 * H-edges are at cell centers; only the dual-face-area normalization differs
 * from the standard stencil.
 *==========================================================================*/
void cpu_compute_curl_H_conformal_omp(const ComplexHField* H,
                                      ComplexEField* curlH,
                                      const GridParams* grid,
                                      const ConformalData* cd,
                                      int do_imag)
{
    const int Nr = grid->Nr, Nphi = grid->Nphi, Nz = grid->Nz;
    const int Nr1 = Nr + 1;
    const double a = grid->a, dr = grid->dr, dphi = grid->dphi, dz = grid->dz;

    const HField* Hin = do_imag ? &H->im : &H->re;
    EField*       Eout= do_imag ? &curlH->im : &curlH->re;

    const double* Hr   = Hin->Hr;
    const double* Hphi = Hin->Hphi;
    const double* Hz   = Hin->Hz;

    /* (curl H)_r  -> Er nodes: Nr * Nphi * (Nz+1) */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < Nz + 1; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr; i++) {
                int gid = ci_Er(Nr, Nphi, i, j, k);
                double A = cd->dual_area_Er[gid];
                if (A < 1e-30) { Eout->Er[gid] = 0.0; continue; }
                double r_iph = a + (i + 0.5) * dr;
                int jm1 = cper_j(j - 1, Nphi);
                double Hz_j   = Hz[ci_Hz(Nr, Nphi, i, j,   k)];
                double Hz_jm1 = Hz[ci_Hz(Nr, Nphi, i, jm1, k)];
                double Hphi_k   = (k < Nz) ? Hphi[ci_Hphi(Nr, Nphi, i, j, k)]     : 0.0;
                double Hphi_km1 = (k > 0)  ? Hphi[ci_Hphi(Nr, Nphi, i, j, k - 1)] : 0.0;
                double li = + (Hz_j - Hz_jm1) * dz
                            - (Hphi_k - Hphi_km1) * r_iph * dphi;
                Eout->Er[gid] = li / A;
            }
        }
    }

    /* (curl H)_phi -> Ephi nodes: Nr1 * Nphi * (Nz+1) */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < Nz + 1; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr1; i++) {
                int gid = ci_Ephi(Nr, Nphi, i, j, k);
                double A = cd->dual_area_Ephi[gid];
                if (A < 1e-30) { Eout->Ephi[gid] = 0.0; continue; }
                double Hr_k   = (k < Nz) ? Hr[ci_Hr(Nr, Nphi, i, j, k)]     : 0.0;
                double Hr_km1 = (k > 0)  ? Hr[ci_Hr(Nr, Nphi, i, j, k - 1)] : 0.0;
                double Hz_i   = (i < Nr) ? Hz[ci_Hz(Nr, Nphi, i, j, k)]     : 0.0;
                double Hz_im1 = (i > 0)  ? Hz[ci_Hz(Nr, Nphi, i - 1, j, k)] : 0.0;
                double li = + (Hr_k - Hr_km1) * dr
                            - (Hz_i - Hz_im1) * dz;
                Eout->Ephi[gid] = li / A;
            }
        }
    }

    /* (curl H)_z -> Ez nodes: Nr1 * Nphi * Nz */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            for (int i = 0; i < Nr1; i++) {
                int gid = ci_Ez(Nr, Nphi, i, j, k);
                double A = cd->dual_area_Ez[gid];
                if (A < 1e-30) { Eout->Ez[gid] = 0.0; continue; }
                double r_iph = a + (i + 0.5) * dr;
                double r_imh = (i > 0) ? a + (i - 0.5) * dr : 0.0;
                int jm1 = cper_j(j - 1, Nphi);
                double rHphi_iph = (i < Nr) ? r_iph * Hphi[ci_Hphi(Nr, Nphi, i, j, k)]     : 0.0;
                double rHphi_imh = (i > 0)  ? r_imh * Hphi[ci_Hphi(Nr, Nphi, i - 1, j, k)] : 0.0;
                double Hr_j   = Hr[ci_Hr(Nr, Nphi, i, j,   k)];
                double Hr_jm1 = Hr[ci_Hr(Nr, Nphi, i, jm1, k)];
                double li = + (rHphi_iph - rHphi_imh) * dphi
                            - (Hr_j - Hr_jm1) * dr;
                Eout->Ez[gid] = li / A;
            }
        }
    }
}

void cpu_compute_curl_curl_E_conformal_complex_omp(
    const ComplexEField* E, ComplexEField* result,
    ComplexHField* temp, const GridParams* grid,
    const ConformalData* cd)
{
    /* curl-E (re, im) -> temp, then curl-H (re, im) -> result */
    cpu_compute_curl_E_conformal_omp(E, temp, grid, cd, 0);
    cpu_compute_curl_E_conformal_omp(E, temp, grid, cd, 1);
    cpu_compute_curl_H_conformal_omp(temp, result, grid, cd, 0);
    cpu_compute_curl_H_conformal_omp(temp, result, grid, cd, 1);
}

/*==========================================================================
 * Staircase IBC weights — host build, mirrors gpu_ibc_weights_build.
 *==========================================================================*/
int cpu_ibc_weights_build(CpuIBCWeights* w,
                          const MaterialMask* pec_mask,
                          const MaterialMask* ibc_mask,
                          const GridParams* grid)
{
    const int Nr = grid->Nr, Nphi = grid->Nphi, Nz = grid->Nz;
    const int Nr1 = Nr + 1;
    const double dr = grid->dr, dphi = grid->dphi, dz = grid->dz;

    w->size_Er   = ibc_mask->size_Er;
    w->size_Ephi = ibc_mask->size_Ephi;
    w->size_Ez   = ibc_mask->size_Ez;
    w->weight_Er   = (double*)calloc(w->size_Er,   sizeof(double));
    w->weight_Ephi = (double*)calloc(w->size_Ephi, sizeof(double));
    w->weight_Ez   = (double*)calloc(w->size_Ez,   sizeof(double));
    if (!w->weight_Er || !w->weight_Ephi || !w->weight_Ez) return -1;
    w->num_boundary_Er = w->num_boundary_Ephi = w->num_boundary_Ez = 0;

    /* Er(i+½, j, k): vacuum neighbors in φ and z */
    for (int k = 0; k <= Nz; k++)
    for (int j = 0; j < Nphi; j++)
    for (int i = 0; i < Nr;  i++) {
        int idx = i + Nr * (j + Nphi * k);
        if (pec_mask->mask_Er[idx] != 0) continue;
        if (ibc_mask->mask_Er[idx] != 1) continue;
        double wt = 0.0;
        double r = grid->a + (i + 0.5) * dr;
        double inv_rdphi = 1.0 / (r * dphi), inv_dz = 1.0 / dz;
        int jm1 = (j - 1 + Nphi) % Nphi, jp1 = (j + 1) % Nphi;
        if (pec_mask->mask_Er[i + Nr * (jm1 + Nphi * k)]) wt += inv_rdphi;
        if (pec_mask->mask_Er[i + Nr * (jp1 + Nphi * k)]) wt += inv_rdphi;
        if (k > 0  && pec_mask->mask_Er[i + Nr * (j + Nphi * (k - 1))]) wt += inv_dz;
        if (k < Nz && pec_mask->mask_Er[i + Nr * (j + Nphi * (k + 1))]) wt += inv_dz;
        if (wt > 0.0) { w->weight_Er[idx] = wt; w->num_boundary_Er++; }
    }

    /* Ephi(i, j+½, k): vacuum neighbors in r and z */
    for (int k = 0; k <= Nz; k++)
    for (int j = 0; j < Nphi; j++)
    for (int i = 0; i <= Nr;  i++) {
        int idx = i + Nr1 * (j + Nphi * k);
        if (pec_mask->mask_Ephi[idx] != 0) continue;
        if (ibc_mask->mask_Ephi[idx] != 1) continue;
        double wt = 0.0;
        double inv_dr = 1.0 / dr, inv_dz = 1.0 / dz;
        if (i > 0  && pec_mask->mask_Ephi[(i - 1) + Nr1 * (j + Nphi * k)]) wt += inv_dr;
        if (i < Nr && pec_mask->mask_Ephi[(i + 1) + Nr1 * (j + Nphi * k)]) wt += inv_dr;
        if (k > 0  && pec_mask->mask_Ephi[i + Nr1 * (j + Nphi * (k - 1))]) wt += inv_dz;
        if (k < Nz && pec_mask->mask_Ephi[i + Nr1 * (j + Nphi * (k + 1))]) wt += inv_dz;
        if (wt > 0.0) { w->weight_Ephi[idx] = wt; w->num_boundary_Ephi++; }
    }

    /* Ez(i, j, k+½): vacuum neighbors in r and φ */
    for (int k = 0; k < Nz; k++)
    for (int j = 0; j < Nphi; j++)
    for (int i = 0; i <= Nr; i++) {
        int idx = i + Nr1 * (j + Nphi * k);
        if (pec_mask->mask_Ez[idx] != 0) continue;
        if (ibc_mask->mask_Ez[idx] != 1) continue;
        double wt = 0.0;
        double r = grid->a + i * dr;
        double inv_dr = 1.0 / dr;
        double inv_rdphi = (r > 1e-14) ? 1.0 / (r * dphi) : 0.0;
        if (i > 0  && pec_mask->mask_Ez[(i - 1) + Nr1 * (j + Nphi * k)]) wt += inv_dr;
        if (i < Nr && pec_mask->mask_Ez[(i + 1) + Nr1 * (j + Nphi * k)]) wt += inv_dr;
        int jm1 = (j - 1 + Nphi) % Nphi, jp1 = (j + 1) % Nphi;
        if (pec_mask->mask_Ez[i + Nr1 * (jm1 + Nphi * k)]) wt += inv_rdphi;
        if (pec_mask->mask_Ez[i + Nr1 * (jp1 + Nphi * k)]) wt += inv_rdphi;
        if (wt > 0.0) { w->weight_Ez[idx] = wt; w->num_boundary_Ez++; }
    }

    printf("  IBC boundary weights (CPU): Er=%d Ephi=%d Ez=%d (total %d)\n",
           w->num_boundary_Er, w->num_boundary_Ephi, w->num_boundary_Ez,
           w->num_boundary_Er + w->num_boundary_Ephi + w->num_boundary_Ez);
    return 0;
}

void cpu_ibc_weights_free(CpuIBCWeights* w) {
    free(w->weight_Er);   w->weight_Er   = NULL;
    free(w->weight_Ephi); w->weight_Ephi = NULL;
    free(w->weight_Ez);   w->weight_Ez   = NULL;
}

/*==========================================================================
 * Operator lifecycle
 *==========================================================================*/
int cpu_conformal_pipe_operator_init(
    CpuConformalPipeOperator* cop,
    const CurlCurlOperator* cpu_op,
    const ConformalData* cd,
    const MaterialMask* pec_mask,
    const MaterialMask* ibc_mask,
    int k_z0, int k_zL,
    const GridParams* grid)
{
    memset(cop, 0, sizeof(*cop));
    cop->op = cpu_op;
    cop->cd = cd;
    cop->ibc_mask = ibc_mask;
    cop->has_ibc_mask = (ibc_mask != NULL);

    if (ibc_mask && pec_mask) {
        if (cpu_ibc_weights_build(&cop->ibc_weights, pec_mask, ibc_mask, grid) != 0)
            return -1;
        cop->has_ibc_weights = 1;
    }

    cop->k_endplate_z0 = k_z0;
    cop->k_endplate_zL = k_zL;
    /* Endplate masks: NULL => treat whole endplate plane as wall (apply IBC).
     * For benchmark cavities without endcap pipes this is correct; with
     * endcap pipes, the caller can later populate these from the pipe config
     * (1=pipe aperture -> skip). We leave them NULL here. */
    cop->has_endplate_ibc = (k_z0 >= 0 || k_zL >= 0);

    /* Inner-conductor beam-port masks (Phase B1): mirror build_port_masks'
     * inner cylindrical masks. 1 = port aperture -> skip inner IBC there.
     * Built from the base operator's PortConfig (op->ports), exactly as the
     * GPU does in cuda_operator.cu::build_port_masks. */
    cop->has_inner_port_masks = 0;
    if (cpu_op->has_ports && cpu_op->ports.num_ports > 0) {
        const PortConfig* ports = &cpu_op->ports;
        int Nr = grid->Nr, Nphi = grid->Nphi, Nz = grid->Nz;
        int cyl_size    = Nphi * (Nz + 1);
        int ez_cyl_size = Nphi * Nz;
        cop->port_mask_Ephi_inner = (int*)calloc(cyl_size,    sizeof(int));
        cop->port_mask_Ez_inner   = (int*)calloc(ez_cyl_size, sizeof(int));
        if (!cop->port_mask_Ephi_inner || !cop->port_mask_Ez_inner) return -1;

        /* Ephi_inner: phi=(j+0.5)dphi, z=k*dz, idx = j + Nphi*k */
        for (int k = 0; k <= Nz; k++) {
            double z = k * grid->dz;
            for (int j = 0; j < Nphi; j++) {
                double phi = (j + 0.5) * grid->dphi;
                if (point_in_port_cylindrical(ports, grid, SURFACE_INNER, phi, z))
                    cop->port_mask_Ephi_inner[j + Nphi * k] = 1;
            }
        }
        /* Ez_inner: phi=j*dphi, z=(k+0.5)dz, idx = j + Nphi*k */
        for (int k = 0; k < Nz; k++) {
            double z = (k + 0.5) * grid->dz;
            for (int j = 0; j < Nphi; j++) {
                double phi = j * grid->dphi;
                if (point_in_port_cylindrical(ports, grid, SURFACE_INNER, phi, z))
                    cop->port_mask_Ez_inner[j + Nphi * k] = 1;
            }
        }
        int nfree = 0;
        for (int i = 0; i < cyl_size; i++)    nfree += cop->port_mask_Ephi_inner[i];
        for (int i = 0; i < ez_cyl_size; i++) nfree += cop->port_mask_Ez_inner[i];
        cop->has_inner_port_masks = 1;
        printf("  Inner-conductor port masks (CPU): %d aperture cells "
               "(Ephi+Ez at i=0)\n", nfree);
    }

    cpu_cefield_alloc(&cop->E_work,      grid);
    cpu_chfield_alloc(&cop->H_temp,      grid);
    cpu_cefield_alloc(&cop->result_work, grid);
    cop->initialized = 1;
    return 0;
}

void cpu_conformal_pipe_operator_free(CpuConformalPipeOperator* cop) {
    if (!cop->initialized) return;
    if (cop->has_ibc_weights) cpu_ibc_weights_free(&cop->ibc_weights);
    free(cop->mask_Er_z0);   free(cop->mask_Ephi_z0);
    free(cop->mask_Er_zL);   free(cop->mask_Ephi_zL);
    free(cop->port_mask_Ephi_inner);
    free(cop->port_mask_Ez_inner);
    cpu_cefield_free(&cop->E_work);
    cpu_chfield_free(&cop->H_temp);
    cpu_cefield_free(&cop->result_work);
    cop->initialized = 0;
}

/*==========================================================================
 * Small in-place helpers for the matvec steps
 *==========================================================================*/

/* PEC at grid edges (complex): i=Nr (Ephi,Ez), k=0 & k=Nz (Er,Ephi). */
static void cpu_apply_pipe_cap_pec_complex(ComplexEField* E,
                                           int Nr, int Nphi, int Nz) {
    int Nr1 = Nr + 1;
    /* Ephi outer cap i=Nr : face Nphi*(Nz+1) */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k <= Nz; k++)
        for (int j = 0; j < Nphi; j++) {
            int idx = Nr + Nr1 * (j + Nphi * k);
            E->re.Ephi[idx] = 0.0; E->im.Ephi[idx] = 0.0;
        }
    /* Ez outer cap i=Nr : face Nphi*Nz */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < Nz; k++)
        for (int j = 0; j < Nphi; j++) {
            int idx = Nr + Nr1 * (j + Nphi * k);
            E->re.Ez[idx] = 0.0; E->im.Ez[idx] = 0.0;
        }
    /* Er endcaps k=0,Nz : face Nr*Nphi */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < Nphi; j++)
        for (int i = 0; i < Nr; i++) {
            int i0  = i + Nr * (j + Nphi * 0);
            int iNz = i + Nr * (j + Nphi * Nz);
            E->re.Er[i0]  = 0.0; E->im.Er[i0]  = 0.0;
            E->re.Er[iNz] = 0.0; E->im.Er[iNz] = 0.0;
        }
    /* Ephi endcaps k=0,Nz : face Nr1*Nphi */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < Nphi; j++)
        for (int i = 0; i < Nr1; i++) {
            int i0  = i + Nr1 * (j + Nphi * 0);
            int iNz = i + Nr1 * (j + Nphi * Nz);
            E->re.Ephi[i0]  = 0.0; E->im.Ephi[i0]  = 0.0;
            E->re.Ephi[iNz] = 0.0; E->im.Ephi[iNz] = 0.0;
        }
}

/* Apply material mask (complex): zero E where mask==0. */
static void cpu_apply_material_mask_complex(ComplexEField* E,
                                            const MaterialMask* m) {
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < m->size_Er; idx++)
        if (!m->mask_Er[idx])   { E->re.Er[idx]=0.0;   E->im.Er[idx]=0.0; }
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < m->size_Ephi; idx++)
        if (!m->mask_Ephi[idx]) { E->re.Ephi[idx]=0.0; E->im.Ephi[idx]=0.0; }
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < m->size_Ez; idx++)
        if (!m->mask_Ez[idx])   { E->re.Ez[idx]=0.0;   E->im.Ez[idx]=0.0; }
}

/* Re-zero endplate conductor-side surface cells (k_z0-1 and k_zL+1). */
static void cpu_rezero_endplate_surface(ComplexEField* E,
                                        int Nr, int Nphi, int Nz,
                                        int k_z0, int k_zL,
                                        const int* mEr_z0, const int* mEp_z0,
                                        const int* mEr_zL, const int* mEp_zL) {
    int Nr1 = Nr + 1;
    (void)Nz;
    /* z0 side: conductor-side layer is ks = k_z0 - 1 (needs k_z0 >= 1).
     * Only meaningful for an INTERIOR endplate (pipe region below it). */
    if (k_z0 >= 1) {
        int ks = k_z0 - 1;
        for (int j = 0; j < Nphi; j++)
            for (int i = 0; i < Nr; i++) {
                int g = i + Nr * j; int idx = i + Nr * (j + Nphi * ks);
                if (!mEr_z0 || !mEr_z0[g]) { E->re.Er[idx]=0.0; E->im.Er[idx]=0.0; }
            }
        for (int j = 0; j < Nphi; j++)
            for (int i = 0; i < Nr1; i++) {
                int g = i + Nr1 * j; int idx = i + Nr1 * (j + Nphi * ks);
                if (!mEp_z0 || !mEp_z0[g]) { E->re.Ephi[idx]=0.0; E->im.Ephi[idx]=0.0; }
            }
    }
    /* zL side: conductor-side layer is ks = k_zL + 1. Only valid when an
     * interior endplate sits below the grid top (k_zL + 1 <= Nz). For a
     * bare cavity whose endplate IS the grid edge (k_zL == Nz) there is no
     * layer beyond it — skip to avoid running off the Er/Ephi arrays. */
    if (k_zL >= 0 && k_zL + 1 <= Nz) {
        int ks = k_zL + 1;
        for (int j = 0; j < Nphi; j++)
            for (int i = 0; i < Nr; i++) {
                int g = i + Nr * j; int idx = i + Nr * (j + Nphi * ks);
                if (!mEr_zL || !mEr_zL[g]) { E->re.Er[idx]=0.0; E->im.Er[idx]=0.0; }
            }
        for (int j = 0; j < Nphi; j++)
            for (int i = 0; i < Nr1; i++) {
                int g = i + Nr1 * j; int idx = i + Nr1 * (j + Nphi * ks);
                if (!mEp_zL || !mEp_zL[g]) { E->re.Ephi[idx]=0.0; E->im.Ephi[idx]=0.0; }
            }
    }
}

/* result += (beta) * w * E   (complex), acting only where w>0.
 * If exclude!=NULL, skip cells where exclude[idx]!=0 (pipe-wall cells). */
static void cpu_ibc_weight_apply(double* res_re, double* res_im,
                                 const double* E_re, const double* E_im,
                                 const double* weight, const double* exclude,
                                 double beta_re, double beta_im, int n) {
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n; idx++) {
        double wv = weight[idx];
        if (wv > 0.0 && (!exclude || exclude[idx] == 0.0)) {
            double bw_re = beta_re * wv, bw_im = beta_im * wv;
            double er = E_re[idx], ei = E_im[idx];
            res_re[idx] += bw_re * er - bw_im * ei;
            res_im[idx] += bw_re * ei + bw_im * er;
        }
    }
}

/* Inner-conductor surface correction at i=0, with optional beam-port masks.
 * Where mask[gid]==1 (port aperture), the IBC is skipped — that cell is a
 * hole, not conductor. Mirrors ibc_inner_Ephi/Ez_kernel (mask && mask[gid]
 * -> return). idx layout: Ephi gid = j + Nphi*k (face Nphi*(Nz+1));
 * Ez gid = j + Nphi*k (face Nphi*Nz). */
static void cpu_ibc_inner_i0(ComplexEField* res, const ComplexEField* E,
                             int Nr, int Nphi, int Nz,
                             double bw_re, double bw_im,
                             const int* mask_Ephi, const int* mask_Ez) {
    int Nr1 = Nr + 1;
    /* Ephi at i=0: face Nphi*(Nz+1) */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k <= Nz; k++)
        for (int j = 0; j < Nphi; j++) {
            int gid = j + Nphi * k;
            if (mask_Ephi && mask_Ephi[gid]) continue;  /* skip port aperture */
            int idx = 0 + Nr1 * (j + Nphi * k);
            double er = E->re.Ephi[idx], ei = E->im.Ephi[idx];
            res->re.Ephi[idx] += bw_re * er - bw_im * ei;
            res->im.Ephi[idx] += bw_re * ei + bw_im * er;
        }
    /* Ez at i=0: face Nphi*Nz */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < Nz; k++)
        for (int j = 0; j < Nphi; j++) {
            int gid = j + Nphi * k;
            if (mask_Ez && mask_Ez[gid]) continue;       /* skip port aperture */
            int idx = 0 + Nr1 * (j + Nphi * k);
            double er = E->re.Ez[idx], ei = E->im.Ez[idx];
            res->re.Ez[idx] += bw_re * er - bw_im * ei;
            res->im.Ez[idx] += bw_re * ei + bw_im * er;
        }
}

/* Endplate grid-plane IBC at k=k_wall: Er (Nr*Nphi) and Ephi (Nr1*Nphi). */
static void cpu_ibc_surface_end(ComplexEField* res, const ComplexEField* E,
                                int Nr, int Nphi,
                                double bw_re, double bw_im, int k_wall,
                                const int* mEr, const int* mEp) {
    int Nr1 = Nr + 1;
    for (int j = 0; j < Nphi; j++)
        for (int i = 0; i < Nr; i++) {
            int g = i + Nr * j; if (mEr && mEr[g]) continue;
            int idx = i + Nr * (j + Nphi * k_wall);
            double er = E->re.Er[idx], ei = E->im.Er[idx];
            res->re.Er[idx] += bw_re * er - bw_im * ei;
            res->im.Er[idx] += bw_re * ei + bw_im * er;
        }
    for (int j = 0; j < Nphi; j++)
        for (int i = 0; i < Nr1; i++) {
            int g = i + Nr1 * j; if (mEp && mEp[g]) continue;
            int idx = i + Nr1 * (j + Nphi * k_wall);
            double er = E->re.Ephi[idx], ei = E->im.Ephi[idx];
            res->re.Ephi[idx] += bw_re * er - bw_im * ei;
            res->im.Ephi[idx] += bw_re * ei + bw_im * er;
        }
}

/* Conformal IBC at pipe walls: result += beta * ibc_weight * E (all components). */
static void cpu_apply_conformal_ibc(ComplexEField* res, const ComplexEField* E,
                                    double alpha, const ConformalData* cd) {
    double inv_2a = 1.0 / (2.0 * alpha);
    double beta_re =  inv_2a, beta_im = -inv_2a;
    cpu_ibc_weight_apply(res->re.Er, res->im.Er, E->re.Er, E->im.Er,
                         cd->ibc_weight_Er, NULL, beta_re, beta_im, cd->size_Er);
    cpu_ibc_weight_apply(res->re.Ephi, res->im.Ephi, E->re.Ephi, E->im.Ephi,
                         cd->ibc_weight_Ephi, NULL, beta_re, beta_im, cd->size_Ephi);
    cpu_ibc_weight_apply(res->re.Ez, res->im.Ez, E->re.Ez, E->im.Ez,
                         cd->ibc_weight_Ez, NULL, beta_re, beta_im, cd->size_Ez);
}

/*==========================================================================
 * The 9-step conformal-pipe complex matvec
 *==========================================================================*/
int cpu_conformal_pipe_matvec_complex_omp(
    const CpuConformalPipeOperator* cop,
    const double* x, double* y, double alpha)
{
    const CurlCurlOperator* op = cop->op;
    const GridParams* grid = &op->grid;
    const ConformalData* cd = cop->cd;
    int Nr = grid->Nr, Nphi = grid->Nphi, Nz = grid->Nz;

    /* mutable aliases (work fields are owned by the operator) */
    ComplexEField* E   = (ComplexEField*)&cop->E_work;
    ComplexHField* Ht  = (ComplexHField*)&cop->H_temp;
    ComplexEField* R   = (ComplexEField*)&cop->result_work;

    double inv_2a = 1.0 / (2.0 * alpha);
    double beta_re =  inv_2a, beta_im = -inv_2a;
    double bw_in_re = beta_re / grid->dr, bw_in_im = beta_im / grid->dr;
    double bw_en_re = beta_re / grid->dz, bw_en_im = beta_im / grid->dz;

    /* Step 1: unpack */
    cpu_unpack_field_complex_omp(x, E, op);

    /* Step 2: PEC at grid edges */
    cpu_apply_pipe_cap_pec_complex(E, Nr, Nphi, Nz);

    /* Step 3: IBC mask */
    if (cop->has_ibc_mask) cpu_apply_material_mask_complex(E, cop->ibc_mask);

    /* Step 3b: re-zero endplate conductor-side surface cells */
    if (cop->has_endplate_ibc)
        cpu_rezero_endplate_surface(E, Nr, Nphi, Nz,
            cop->k_endplate_z0, cop->k_endplate_zL,
            cop->mask_Er_z0, cop->mask_Ephi_z0,
            cop->mask_Er_zL, cop->mask_Ephi_zL);

    /* Step 4: conformal curl-curl */
    cpu_compute_curl_curl_E_conformal_complex_omp(E, R, Ht, grid, cd);

    /* Step 5: inner conductor IBC at i=0 (skip beam-port apertures) */
    cpu_ibc_inner_i0(R, E, Nr, Nphi, Nz, bw_in_re, bw_in_im,
                     cop->has_inner_port_masks ? cop->port_mask_Ephi_inner : NULL,
                     cop->has_inner_port_masks ? cop->port_mask_Ez_inner   : NULL);

    /* Step 6a: staircase IBC at flat walls, excluding pipe-wall cells */
    if (cop->has_ibc_weights) {
        const CpuIBCWeights* w = &cop->ibc_weights;
        cpu_ibc_weight_apply(R->re.Er, R->im.Er, E->re.Er, E->im.Er,
            w->weight_Er, cd->ibc_weight_Er, beta_re, beta_im, w->size_Er);
        cpu_ibc_weight_apply(R->re.Ephi, R->im.Ephi, E->re.Ephi, E->im.Ephi,
            w->weight_Ephi, cd->ibc_weight_Ephi, beta_re, beta_im, w->size_Ephi);
        cpu_ibc_weight_apply(R->re.Ez, R->im.Ez, E->re.Ez, E->im.Ez,
            w->weight_Ez, cd->ibc_weight_Ez, beta_re, beta_im, w->size_Ez);
    }

    /* Step 6b: endplate grid-plane IBC */
    if (cop->k_endplate_z0 >= 0)
        cpu_ibc_surface_end(R, E, Nr, Nphi, bw_en_re, bw_en_im,
                            cop->k_endplate_z0, cop->mask_Er_z0, cop->mask_Ephi_z0);
    if (cop->k_endplate_zL >= 0)
        cpu_ibc_surface_end(R, E, Nr, Nphi, bw_en_re, bw_en_im,
                            cop->k_endplate_zL, cop->mask_Er_zL, cop->mask_Ephi_zL);

    /* Step 6c: conformal IBC at pipe walls */
    cpu_apply_conformal_ibc(R, E, alpha, cd);

    /* Step 7: PEC at grid edges on result */
    cpu_apply_pipe_cap_pec_complex(R, Nr, Nphi, Nz);

    /* Step 8: IBC mask on result */
    if (cop->has_ibc_mask) cpu_apply_material_mask_complex(R, cop->ibc_mask);

    /* Step 8b: re-zero endplate surface cells on result */
    if (cop->has_endplate_ibc)
        cpu_rezero_endplate_surface(R, Nr, Nphi, Nz,
            cop->k_endplate_z0, cop->k_endplate_zL,
            cop->mask_Er_z0, cop->mask_Ephi_z0,
            cop->mask_Er_zL, cop->mask_Ephi_zL);

    /* Step 9: pack */
    cpu_pack_field_complex_omp(R, y, op);
    return 0;
}

/*==========================================================================
 * Complex GMRES(m) for the conformal-pipe operator.
 *
 * Line-for-line the same restarted Arnoldi / complex-Givens algorithm as
 * cpu_gmres_solve_complex_omp (Stage 1), but the matvec is the conformal
 * pipe operator. Reuses the Stage 1 CpuGmresWorkspace and the
 * operator-agnostic complex vector ops.
 *==========================================================================*/
static CpuLinSolveResult cpu_gmres_solve_conformal_omp(
    const CpuConformalPipeOperator* cop,
    double sigma_re, double sigma_im,
    const double* b, double* x,
    double alpha,
    int max_iter, double tol,
    CpuGmresWorkspace* ws)
{
    CpuLinSolveResult result = {0, 1.0, 0};
    int n_real = cop->op->n_total;
    int n      = 2 * n_real;
    int m      = ws->m;

    cpu_vec_zero_omp(x, n);

    double b_norm = cpu_vec_norm_flat_complex_omp(b, n_real);
    if (b_norm < 1e-14) { result.converged = 1; return result; }

    int    total_iters   = 0;
    double residual_norm = b_norm;

    while (total_iters < max_iter) {
        if (total_iters == 0) {
            cpu_vec_copy_omp(b, ws->V[0], n);
            residual_norm = b_norm;
        } else {
            cpu_conformal_pipe_matvec_complex_omp(cop, x, ws->w, alpha);
            cpu_vec_shift_complex_omp(ws->w, x, sigma_re, sigma_im, n_real);
            cpu_vec_copy_omp(b, ws->V[0], n);
            cpu_vec_axpy_omp(-1.0, ws->w, ws->V[0], n);
            residual_norm = cpu_vec_norm_flat_complex_omp(ws->V[0], n_real);
            if (residual_norm / b_norm < tol) {
                result.converged  = 1;
                result.residual   = residual_norm / b_norm;
                result.iterations = total_iters;
                return result;
            }
        }
        cpu_vec_scale_omp(ws->V[0], 1.0 / residual_norm, n);

        ws->g_re[0] = residual_norm; ws->g_im[0] = 0.0;
        for (int i = 1; i <= m; i++) { ws->g_re[i] = 0.0; ws->g_im[i] = 0.0; }
        memset(ws->H_re, 0, (size_t)(m + 1) * m * sizeof(double));
        memset(ws->H_im, 0, (size_t)(m + 1) * m * sizeof(double));

        int j_final = 0;

        for (int j = 0; j < m && total_iters < max_iter; j++) {
            total_iters++;

            cpu_conformal_pipe_matvec_complex_omp(cop, ws->V[j], ws->w, alpha);
            cpu_vec_shift_complex_omp(ws->w, ws->V[j], sigma_re, sigma_im, n_real);

            for (int i = 0; i <= j; i++) {
                double h_re, h_im;
                cpu_vec_dot_flat_complex_omp(ws->V[i], ws->w, &h_re, &h_im, n_real);
                ws->H_re[i * m + j] = h_re;
                ws->H_im[i * m + j] = h_im;
                cpu_vec_axpy_complex_omp(-h_re, -h_im, ws->V[i], ws->w, n_real);
            }

            double w_norm = cpu_vec_norm_flat_complex_omp(ws->w, n_real);
            ws->H_re[(j + 1) * m + j] = w_norm;
            ws->H_im[(j + 1) * m + j] = 0.0;

            if (w_norm > 1e-14) {
                cpu_vec_copy_omp(ws->w, ws->V[j + 1], n);
                cpu_vec_scale_omp(ws->V[j + 1], 1.0 / w_norm, n);
            } else {
                cpu_vec_zero_omp(ws->V[j + 1], n);
            }

            for (int i = 0; i < j; i++) {
                double a_re = ws->H_re[i * m + j], a_im = ws->H_im[i * m + j];
                double b_re = ws->H_re[(i + 1) * m + j], b_im = ws->H_im[(i + 1) * m + j];
                double c_re = ws->cs_re[i], c_im = ws->cs_im[i];
                double s_re = ws->sn_re[i], s_im = ws->sn_im[i];
                double na_re = (c_re * a_re + c_im * a_im) + (s_re * b_re + s_im * b_im);
                double na_im = (c_re * a_im - c_im * a_re) + (s_re * b_im - s_im * b_re);
                double nb_re = -(s_re * a_re - s_im * a_im) + (c_re * b_re - c_im * b_im);
                double nb_im = -(s_re * a_im + s_im * a_re) + (c_re * b_im + c_im * b_re);
                ws->H_re[i * m + j]       = na_re;
                ws->H_im[i * m + j]       = na_im;
                ws->H_re[(i + 1) * m + j] = nb_re;
                ws->H_im[(i + 1) * m + j] = nb_im;
            }

            {
                double a_re = ws->H_re[j * m + j], a_im = ws->H_im[j * m + j];
                double b_re = ws->H_re[(j + 1) * m + j], b_im = ws->H_im[(j + 1) * m + j];
                double r = sqrt(a_re*a_re + a_im*a_im + b_re*b_re + b_im*b_im);
                if (r > 1e-14) {
                    ws->cs_re[j] = a_re / r; ws->cs_im[j] = a_im / r;
                    ws->sn_re[j] = b_re / r; ws->sn_im[j] = b_im / r;
                } else {
                    ws->cs_re[j] = 1.0; ws->cs_im[j] = 0.0;
                    ws->sn_re[j] = 0.0; ws->sn_im[j] = 0.0;
                }
                ws->H_re[j * m + j]       = r;
                ws->H_im[j * m + j]       = 0.0;
                ws->H_re[(j + 1) * m + j] = 0.0;
                ws->H_im[(j + 1) * m + j] = 0.0;

                double gj_re  = ws->g_re[j],     gj_im  = ws->g_im[j];
                double gj1_re = ws->g_re[j + 1], gj1_im = ws->g_im[j + 1];
                double c_re = ws->cs_re[j], c_im = ws->cs_im[j];
                double s_re = ws->sn_re[j], s_im = ws->sn_im[j];
                ws->g_re[j] = (c_re*gj_re + c_im*gj_im) + (s_re*gj1_re + s_im*gj1_im);
                ws->g_im[j] = (c_re*gj_im - c_im*gj_re) + (s_re*gj1_im - s_im*gj1_re);
                ws->g_re[j + 1] = -(s_re*gj_re - s_im*gj_im) + (c_re*gj1_re - c_im*gj1_im);
                ws->g_im[j + 1] = -(s_re*gj_im + s_im*gj_re) + (c_re*gj1_im + c_im*gj1_re);
            }

            double res_est = sqrt(ws->g_re[j + 1] * ws->g_re[j + 1]
                                + ws->g_im[j + 1] * ws->g_im[j + 1]);
            result.residual   = res_est / b_norm;
            result.iterations = total_iters;
            j_final           = j + 1;

            if (result.residual < tol) { result.converged = 1; break; }
            if (w_norm < 1e-14) break;
        }

        for (int i = j_final - 1; i >= 0; i--) {
            double yi_re = ws->g_re[i], yi_im = ws->g_im[i];
            for (int k = i + 1; k < j_final; k++) {
                double h_re = ws->H_re[i * m + k], h_im = ws->H_im[i * m + k];
                double yk_re = ws->y_re[k], yk_im = ws->y_im[k];
                yi_re -= h_re * yk_re - h_im * yk_im;
                yi_im -= h_re * yk_im + h_im * yk_re;
            }
            double h_re = ws->H_re[i * m + i], h_im = ws->H_im[i * m + i];
            double denom = h_re * h_re + h_im * h_im;
            if (denom > 1e-28) {
                ws->y_re[i] = (yi_re * h_re + yi_im * h_im) / denom;
                ws->y_im[i] = (yi_im * h_re - yi_re * h_im) / denom;
            } else {
                ws->y_re[i] = 0.0; ws->y_im[i] = 0.0;
            }
        }

        for (int j = 0; j < j_final; j++) {
            cpu_vec_axpy_complex_omp(ws->y_re[j], ws->y_im[j],
                                     ws->V[j], x, n_real);
        }

        if (result.converged) break;
    }
    return result;
}

/*==========================================================================
 * Helpers: complex residual, Rayleigh quotient, alpha — conformal operator
 *==========================================================================*/
static double cpu_conf_residual(const CpuConformalPipeOperator* cop,
                                const double* Ax, const double* x,
                                double sigma_re, double sigma_im, double* temp) {
    int n_real = cop->op->n_total;
    int n      = 2 * n_real;
    cpu_vec_copy_omp(Ax, temp, n);
    cpu_vec_shift_complex_omp(temp, x, sigma_re, sigma_im, n_real);
    double r_norm = cpu_vec_norm_weighted_complex_omp(temp, cop->op);
    double x_norm = cpu_vec_norm_weighted_complex_omp(x,    cop->op);
    return (x_norm > 1e-28) ? r_norm / x_norm : r_norm;
}

static void cpu_conf_rayleigh(const CpuConformalPipeOperator* cop,
                              const double* x, const double* Ax,
                              double* sigma_re, double* sigma_im) {
    double xAx_re, xAx_im, xx_re, xx_im;
    cpu_vec_dot_weighted_complex_omp(x, Ax, &xAx_re, &xAx_im, cop->op);
    cpu_vec_dot_weighted_complex_omp(x, x,  &xx_re,  &xx_im,  cop->op);
    if (xx_re > 1e-28) { *sigma_re = xAx_re / xx_re; *sigma_im = xAx_im / xx_re; }
    else               { *sigma_re = 0.0; *sigma_im = 0.0; }
}

static inline double cpu_conf_alpha(double k2_re, double conductivity) {
    const double c0  = 299792458.0;
    const double mu0 = 4.0e-7 * M_PI;
    double omega = c0 * sqrt(fabs(k2_re));
    if (omega < 1.0) omega = 1.0;
    return 1.0 / sqrt(2.0 * conductivity * omega * mu0);
}

/*==========================================================================
 * Complex RQI for the conformal-pipe operator (mirrors
 * gpu_rqi_complex_conformal_pipe -> gpu_rqi_complex_ws).
 *==========================================================================*/
CpuComplexEigenResult cpu_rqi_complex_conformal_pipe_omp(
    const CpuConformalPipeOperator* cop,
    double* x, double sigma_init, double conductivity,
    int max_iter, double tol, int gmres_restart)
{
    CpuComplexEigenResult r;
    r.k2_re = 0.0; r.k2_im = 0.0;
    r.frequency_Hz = 0.0; r.Q_factor = 0.0;
    r.iterations = 0; r.residual = 1.0; r.converged = 0;

    int n_real = cop->op->n_total;
    int n      = 2 * n_real;
    const double c0 = 299792458.0;

    /* Workspace */
    CpuGmresWorkspace gws;
    if (cpu_gmres_workspace_init(&gws, n_real, gmres_restart) != 0) return r;
    double* y  = (double*)calloc((size_t)n, sizeof(double));
    double* Ax = (double*)calloc((size_t)n, sizeof(double));

    cpu_vec_normalize_weighted_complex_omp(x, cop->op);

    double alpha = cpu_conf_alpha(sigma_init, conductivity);
    cpu_conformal_pipe_matvec_complex_omp(cop, x, Ax, alpha);

    double sigma_re, sigma_im;
    cpu_conf_rayleigh(cop, x, Ax, &sigma_re, &sigma_im);
    sigma_re = sigma_init;   /* GPU override: fix GMRES shift at k2_PEC */
    sigma_im = 0.0;

    printf("\n  Complex RQI (conformal pipe, CPU, threads=%d):\n",
           cpu_omp_num_threads());
    printf("  %-5s %-15s %-12s %-12s %-12s %-6s\n",
           "Iter", "k2_re", "k2_im", "Q", "Residual", "GMRES");
    printf("  --------------------------------------------------"
           "----------------------\n");

    double sigma_re_old = sigma_re, sigma_im_old = sigma_im;

    for (int iter = 0; iter < max_iter; iter++) {
        alpha = cpu_conf_alpha(sigma_re, conductivity);
        int    gmres_maxiter = (iter == 0) ? 300  : 2000;
        double gmres_tol     = (iter == 0) ? 1e-1 : 1e-4;  /* tightened: Q depends
                                  on k2_im (~1e-4), which a loose 1e-1 inner solve
                                  leaves imprecise. 1e-4 pins it down so CPU/GPU
                                  agree on Q. Costs more GMRES iters per RQI step. */

        CpuLinSolveResult ls = cpu_gmres_solve_conformal_omp(
            cop, sigma_re, sigma_im, x, y, alpha,
            gmres_maxiter, gmres_tol, &gws);

        cpu_vec_normalize_weighted_complex_omp(y, cop->op);
        cpu_vec_copy_omp(y, x, n);

        alpha = cpu_conf_alpha(sigma_re, conductivity);
        cpu_conformal_pipe_matvec_complex_omp(cop, x, Ax, alpha);

        double sigma_re_new, sigma_im_new;
        cpu_conf_rayleigh(cop, x, Ax, &sigma_re_new, &sigma_im_new);

        double res = cpu_conf_residual(cop, Ax, x, sigma_re_new, sigma_im_new, y);

        double freq = (sigma_re_new > 0) ? c0 * sqrt(sigma_re_new) / (2.0*M_PI) : 0.0;
        double Q = (fabs(sigma_im_new) > 1e-30) ? -sigma_re_new / sigma_im_new : 0.0;

        r.k2_re = sigma_re_new; r.k2_im = sigma_im_new;
        r.frequency_Hz = freq;  r.Q_factor = Q;
        r.residual = res;       r.iterations = iter + 1;

        printf("  %-5d %15.10f %12.5e %12.1f %12.5e %6d\n",
               iter, sigma_re_new, sigma_im_new, Q, res, ls.iterations);

        double sigma_mag = sqrt(sigma_re_new*sigma_re_new + sigma_im_new*sigma_im_new);
        double d_re = sigma_re_new - sigma_re_old;
        double d_im = sigma_im_new - sigma_im_old;
        double d_mag = sqrt(d_re*d_re + d_im*d_im);
        double rel = (sigma_mag > 1e-28) ? d_mag / sigma_mag : d_mag;

        if (res < tol || (rel < tol && iter > 0)) {
            r.converged = 1;
            printf("  Converged!  f = %.6f MHz, Q = %.1f\n", freq / 1.0e6, Q);
            break;
        }
        sigma_re_old = sigma_re_new;
        sigma_im_old = sigma_im_new;
        /* GMRES shift stays fixed at sigma_init, per GPU code */
    }

    cpu_gmres_workspace_free(&gws);
    free(y); free(Ax);
    return r;
}
