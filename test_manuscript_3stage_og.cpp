/*============================================================================
 * test_manuscript_3stage.cpp
 *
 * Three-stage cavity solve for the manuscript's final results table.
 * Default grid: Nr=88 (cav 81 + pipe 7), Nphi=256, Nz_cav=84.
 *
 *   STAGE 1 (UNPERTURBED): bare coaxial cavity, no pipes, no ports.
 *   STAGE 2 (RADIAL PIPES ONLY): + radial Dey-Mittra beam pipes.
 *   STAGE 3 (FULL PRODUCTION): + endcap pipes + inner-conductor ports.
 *
 * For each stage we report:
 *   - frequency f [MHz]
 *   - Q from the surface integral (compute_q_factor_extended)
 *   - Q from the eigenvalue (=-Re(k^2)/Im(k^2)) for comparison
 *   - R/Q (linac and circuit; per crossing, per pass, total)
 *   - gap-voltage spread (%)
 *
 * Algorithm per stage (identical to test_conformal_ibc_full):
 *   Phase A: PEC pipe-operator RQI seed (run_pec_pipe_solver)
 *   Phase B: Complex IBC RQI (gpu_rqi_complex_conformal_pipe)
 *   Phase C: Q-surface + R/Q + voltage-spread post-processing
 *==========================================================================*/

#include "curlcurl_operator.h"
#include "pipe_model.h"
#include "conformal_geometry.h"
#include "cuda_fields.h"
#include "cuda_operator.h"
#include "cuda_pipe_model.h"
#include "cuda_conformal_pipe.h"
#include "cuda_eigensolver.h"
#include "cuda_vector_ops.h"
#include "q_factor.h"
#include "r_over_q.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static const double C0       = 299792458.0;
static const double SIGMA_CU = 5.8e7;

/* Forward declaration of the shift kernel used by inlined MINRES below
 * (defined in cuda_eigensolver.cu; not exported in any header). */
extern __global__ void shift_kernel(double*, const double*, double, int);

/*--------------------------------------------------------------------------
 * PEC pipe solver helper (verbatim from test_conformal_ibc.cpp, which works
 * — it inlines MINRES because the generic gpu_minres_solve_shifted_ws works
 * on GPU_Operator, not on GPU_PipeOperator).
 *------------------------------------------------------------------------*/
static double run_pec_pipe_solver(
    GPU_PipeOperator* pipe_op, const CurlCurlOperator* cpu_op,
    const GridParams* grid, double* d_x,
    int rqi_max, int minres_max, double tol)
{
    int n = cpu_op->n_total, blocks = (n+256-1)/256;
    EigensolverWorkspace ws; eigensolver_workspace_init(&ws, n);
    gpu_vec_normalize_weighted(d_x, cpu_op);
    double* d_Ax; cudaMalloc(&d_Ax, n*sizeof(double));
    gpu_pipe_matvec(pipe_op, d_x, d_Ax);
    double xAx, xx;
    gpu_vec_dot_weighted_ws(d_x, d_Ax, &xAx, cpu_op, &pipe_op->base.reduction_ws);
    gpu_vec_dot_weighted_ws(d_x, d_x,  &xx,  cpu_op, &pipe_op->base.reduction_ws);
    double sigma = xAx/xx;
    if (fabs(sigma)<1e-10 || sigma<0) sigma=(M_PI/grid->L)*(M_PI/grid->L);
    printf("    PEC Pipe RQI: sigma0=%.6f\n", sigma);
    for (int iter=0; iter<rqi_max; iter++) {
        gpu_vec_zero(ws.d_y,n); gpu_vec_zero(ws.minres_ws.d_v_old,n);
        gpu_vec_zero(ws.minres_ws.d_w_old,n); gpu_vec_zero(ws.minres_ws.d_w_cur,n);
        double b_norm; gpu_vec_dot_weighted_ws(d_x,d_x,&b_norm,cpu_op,&pipe_op->base.reduction_ws);
        b_norm=sqrt(b_norm); gpu_vec_copy(d_x,ws.minres_ws.d_v_cur,n);
        gpu_vec_scale(ws.minres_ws.d_v_cur,1.0/b_norm,n);
        double beta_cur=b_norm,eta=b_norm,c_old=1,c_cur=1,s_old=0,s_cur=0; int ls_iters=0;
        for (int ls=0; ls<minres_max; ls++) {
            gpu_pipe_matvec(pipe_op,ws.minres_ws.d_v_cur,ws.minres_ws.d_Av);
            shift_kernel<<<blocks,256>>>(ws.minres_ws.d_Av,ws.minres_ws.d_v_cur,sigma,n);
            double a; gpu_vec_dot_weighted_ws(ws.minres_ws.d_v_cur,ws.minres_ws.d_Av,&a,cpu_op,&pipe_op->base.reduction_ws);
            gpu_vec_copy(ws.minres_ws.d_Av,ws.minres_ws.d_v_new,n);
            gpu_vec_axpy(-a,ws.minres_ws.d_v_cur,ws.minres_ws.d_v_new,n);
            gpu_vec_axpy(-beta_cur,ws.minres_ws.d_v_old,ws.minres_ws.d_v_new,n);
            double bn; gpu_vec_dot_weighted_ws(ws.minres_ws.d_v_new,ws.minres_ws.d_v_new,&bn,cpu_op,&pipe_op->base.reduction_ws);
            bn=sqrt(bn); if(bn>1e-14) gpu_vec_scale(ws.minres_ws.d_v_new,1.0/bn,n);
            double r1=s_old*beta_cur, r2=c_old*c_cur*beta_cur+s_cur*a, r3=c_cur*a-c_old*s_cur*beta_cur;
            double g=sqrt(r3*r3+bn*bn), cn=1,sn=0;
            if(g>1e-14){cn=r3/g;sn=bn/g;}
            gpu_vec_copy(ws.minres_ws.d_v_cur,ws.minres_ws.d_w_new,n);
            gpu_vec_axpy(-r2,ws.minres_ws.d_w_cur,ws.minres_ws.d_w_new,n);
            gpu_vec_axpy(-r1,ws.minres_ws.d_w_old,ws.minres_ws.d_w_new,n);
            if(fabs(g)>1e-14) gpu_vec_scale(ws.minres_ws.d_w_new,1.0/g,n);
            gpu_vec_axpy(cn*eta,ws.minres_ws.d_w_new,ws.d_y,n);
            eta=-sn*eta; ls_iters=ls+1;
            if(fabs(eta)/b_norm<tol) break;
            double*t; t=ws.minres_ws.d_v_old;ws.minres_ws.d_v_old=ws.minres_ws.d_v_cur;
            ws.minres_ws.d_v_cur=ws.minres_ws.d_v_new;ws.minres_ws.d_v_new=t;
            t=ws.minres_ws.d_w_old;ws.minres_ws.d_w_old=ws.minres_ws.d_w_cur;
            ws.minres_ws.d_w_cur=ws.minres_ws.d_w_new;ws.minres_ws.d_w_new=t;
            beta_cur=bn;c_old=c_cur;c_cur=cn;s_old=s_cur;s_cur=sn;
        }
        gpu_vec_normalize_weighted(ws.d_y,cpu_op); gpu_vec_copy(ws.d_y,d_x,n);
        gpu_pipe_matvec(pipe_op,d_x,d_Ax);
        gpu_vec_dot_weighted_ws(d_x,d_Ax,&xAx,cpu_op,&pipe_op->base.reduction_ws);
        gpu_vec_dot_weighted_ws(d_x,d_x,&xx,cpu_op,&pipe_op->base.reduction_ws);
        double sn=xAx/xx;
        printf("      iter %d: k2=%.10f, MINRES=%d\n", iter, sn, ls_iters);
        double rel=fabs(sn-sigma)/fabs(sn); sigma=sn;
        if (rel<1e-8 && iter>0) { printf("    PEC converged.\n"); break; }
    }
    cudaFree(d_Ax);
    eigensolver_workspace_free(&ws);
    return sigma;
}

/*--------------------------------------------------------------------------
 * Stage configuration
 *------------------------------------------------------------------------*/
typedef struct {
    const char* label;
    int use_radial_pipes;
    int use_endcap_pipes;
    int use_inner_ports;
    double a, b, L;
    double pipe_radius, aperture_radius, pipe_length;
    int    num_passes;
    int Nr_cavity, Nr_pipe, Nphi, Nz_cavity;
} StageConfig;

typedef struct {
    double f_MHz;
    double Q_eig, Q_surf;
    double V_avg, V_spread_pct;
    double RoQ_linac_crossing, RoQ_circuit_crossing;
    double RoQ_linac_total,   RoQ_circuit_total;
} StageResult;

/*--------------------------------------------------------------------------
 * One stage: PEC + IBC + post-processing. Mirrors test_conformal_ibc_full.
 *------------------------------------------------------------------------*/
static StageResult run_stage(const StageConfig* sc)
{
    StageResult R = {0};
    printf("\n================================================================\n");
    printf("  %s\n", sc->label);
    printf("    radial_pipes=%d  endcap_pipes=%d  inner_ports=%d\n",
           sc->use_radial_pipes, sc->use_endcap_pipes, sc->use_inner_ports);
    printf("================================================================\n");

    /* --- radial pipe config (empty if disabled) --- */
    PipeConfig pipes;
    pipe_config_init(&pipes, sc->a, sc->b, sc->pipe_radius,
                     sc->aperture_radius, sc->pipe_length, 0.0);
    if (sc->use_radial_pipes)
        pipe_config_add_multi_pass(&pipes, sc->L / 2.0, sc->num_passes);

    /* --- endcap pipes (empty if disabled) --- */
    EndcapPipeConfig endcap;
    endcap_pipe_config_init(&endcap);
    if (sc->use_endcap_pipes) {
        endcap_pipe_config_add(&endcap, 0.85, M_PI/2.0,        0.105, 0.100, 0.28, 1);
        endcap_pipe_config_add(&endcap, 0.85, 3.0*M_PI/2.0,    0.095, 0.090, 0.25, 0);
    }

    /* --- grid (z-extended only when endcap pipes are present) --- */
    double pipe_len_r  = sc->pipe_length;  /* always extend past b (see comment below) */
    double pipe_len_z0 = sc->use_endcap_pipes ? endcap.z0_extension : 0.0;
    double pipe_len_zL = sc->use_endcap_pipes ? endcap.zL_extension : 0.0;
    /* Even stages without physical pipes need a small radial buffer (Nr_pipe
     * cells of PEC beyond b) so the outer conductor at r=b is an INTERIOR
     * surface, not a grid edge. Without this, Er cells at r=b don't exist
     * and the IBC mask gets 0 Er surface cells → wall loss is missing → Q
     * is wildly wrong. The buffer cells are solid PEC; they don't change
     * the physics, just ensure proper field resolution at the outer wall. */
    int Nr_pipe_eff = sc->Nr_pipe;  /* always use the buffer */
    double dz_cav = sc->L / sc->Nz_cavity;
    int Nz_pipe_z0 = sc->use_endcap_pipes ? (int)ceil(pipe_len_z0/dz_cav) : 0;
    int Nz_pipe_zL = sc->use_endcap_pipes ? (int)ceil(pipe_len_zL/dz_cav) : 0;

    GridParams grid;
    grid_init_with_all_pipes(&grid, sc->a, sc->b, sc->L,
        pipe_len_r, pipe_len_z0, pipe_len_zL,
        sc->Nr_cavity, Nr_pipe_eff, sc->Nphi,
        sc->Nz_cavity, Nz_pipe_z0, Nz_pipe_zL);

    double z0_offset = pipe_len_z0;
    int k_endplate_z0 = (Nz_pipe_z0 > 0) ? Nz_pipe_z0 : -1;
    int k_endplate_zL = (Nz_pipe_zL > 0) ? Nz_pipe_z0 + sc->Nz_cavity : -1;

    printf("  Grid: Nr=%d, Nphi=%d, Nz=%d  (DOFs ~ %lld)\n",
           grid.Nr, grid.Nphi, grid.Nz,
           (long long)((long long)grid.Nr * grid.Nphi * grid.Nz * 3));

    /* --- material mask (single combined mask as in production) --- */
    MaterialMask mask;
    material_mask_build_full(&mask, &pipes, &endcap, &grid, sc->L, z0_offset);
    material_mask_print_stats(&mask, &grid);

    /* --- conformal data (radial pipes only contribute cuts) --- */
    ConformalData cd;
    conformal_data_build(&cd, &pipes, &grid, sc->b, z0_offset);

    /* --- inner-conductor beam ports (stage 3 only) --- */
    PortConfig inner_ports;
    port_config_init(&inner_ports);
    if (sc->use_inner_ports) {
        double dphi_pass = M_PI / sc->num_passes;
        double z_grid = sc->L / 2.0 + z0_offset;
        for (int p = 0; p < sc->num_passes; p++) {
            double phi_entry = p * dphi_pass;
            double phi_exit  = phi_entry + M_PI;
            if (phi_exit >= 2.0 * M_PI) phi_exit -= 2.0 * M_PI;
            CavityPort port;
            port.type = PORT_BEAM; port.surface = SURFACE_INNER;
            port.radius = sc->aperture_radius; port.pos2 = z_grid;
            port.pos1 = phi_entry; port.name = "Inner Entry";
            if (inner_ports.num_ports >= inner_ports.capacity) {
                inner_ports.capacity = inner_ports.capacity ? inner_ports.capacity*2 : 4;
                inner_ports.ports = (CavityPort*)realloc(inner_ports.ports,
                    inner_ports.capacity * sizeof(CavityPort));
            }
            inner_ports.ports[inner_ports.num_ports++] = port;
            port.pos1 = phi_exit; port.name = "Inner Exit";
            if (inner_ports.num_ports >= inner_ports.capacity) {
                inner_ports.capacity *= 2;
                inner_ports.ports = (CavityPort*)realloc(inner_ports.ports,
                    inner_ports.capacity * sizeof(CavityPort));
            }
            inner_ports.ports[inner_ports.num_ports++] = port;
        }
    }

    /* --- base CPU operator (with ports for stage 3) --- */
    CurlCurlOperator cpu_op;
    if (sc->use_inner_ports)
        curlcurl_op_init_with_ports(&cpu_op, &grid, &inner_ports);
    else
        curlcurl_op_init(&cpu_op, &grid);
    int n  = cpu_op.n_total;
    int n2 = 2 * n;
    cuda_grid_init(&grid);

    /* --- TEM initial guess (z-shifted for extended grid) --- */
    double* h_x = (double*)calloc(n, sizeof(double));
    for (int k = 0; k <= grid.Nz; k++) {
        double z_phys = k * grid.dz - z0_offset;
        for (int j = 0; j < grid.Nphi; j++)
            for (int i = 0; i < grid.Nr; i++) {
                double r = grid.a + (i + 0.5) * grid.dr;
                if (r <= sc->b && z_phys >= 0.0 && z_phys <= sc->L) {
                    int idx = cpu_op.offset_Er + idx_Er(&grid, i, j, k);
                    h_x[idx] = sin(M_PI * z_phys / sc->L) / r;
                }
            }
    }

    /* ==================== PHASE A: PEC pipe RQI ==================== */
    printf("\n  Phase A: PEC pipe-operator RQI (seed for IBC)\n");
    GPU_PipeOperator gpu_pec;
    gpu_pipe_operator_init(&gpu_pec, &cpu_op, &mask);
    gpu_pipe_operator_set_endplates(&gpu_pec,
        k_endplate_z0, k_endplate_zL, &endcap, &grid, z0_offset);

    double* d_x;
    gpu_vector_alloc(&d_x, n);
    gpu_vector_to_device(d_x, h_x, n);
    clock_t t0 = clock();
    double k2_pec = run_pec_pipe_solver(&gpu_pec, &cpu_op, &grid, d_x,
                                        20, 3000, 1e-8);
    double t_pec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    double f_pec = C0 * sqrt(fabs(k2_pec)) / (2.0 * M_PI);
    gpu_vector_to_host(h_x, d_x, n);
    printf("  Phase A done: k2 = %.10f, f = %.6f MHz, time = %.1f s\n",
           k2_pec, f_pec/1e6, t_pec);

    /* ==================== PHASE B: complex IBC RQI ==================== */
    printf("\n  Phase B: Conformal IBC RQI (complex, seeded from PEC)\n");
    GPU_ConformalPipeOperator gpu_cfm;
    gpu_conformal_pipe_operator_init(&gpu_cfm, &cpu_op, &mask, &cd,
        k_endplate_z0, k_endplate_zL, &endcap, &grid, z0_offset);

    double* h_x_cx = (double*)calloc(n2, sizeof(double));
    memcpy(h_x_cx, h_x, n * sizeof(double));  /* real = PEC, imag = 0 */
    double* d_x_cx;
    gpu_vector_alloc(&d_x_cx, n2);
    gpu_vector_to_device(d_x_cx, h_x_cx, n2);
    t0 = clock();
    GPU_ComplexEigenResult ibc = gpu_rqi_complex_conformal_pipe(
        &gpu_cfm, d_x_cx, k2_pec, SIGMA_CU, 20, 1e-6, 50);
    double t_ibc = (double)(clock() - t0) / CLOCKS_PER_SEC;
    gpu_vector_to_host(h_x_cx, d_x_cx, n2);

    /* PHASE ROTATION (matches production field_export): the complex IBC
     * eigenvector has an arbitrary global phase e^{i*theta}. Q and R/Q use the
     * REAL part only, so we must first rotate the vector so the real part is
     * maximal (the physical standing-wave mode). Without this, an eigenvector
     * that happens to land near the imaginary axis gives a tiny, noise-
     * dominated real part and a garbage surface integral.
     *
     * Find the element of largest complex magnitude, take its phase theta,
     * and rotate every component by -theta:  re' = re*cos + im*sin. */
    {
        /* Global phase theta that MAXIMIZES the real-part energy:
         *   ||Re(e^{-i theta} x)||^2 = cos^2 A + sin(2 theta) B + sin^2 C
         *   with A=sum re^2, B=sum re*im, C=sum im^2
         *   => d/dtheta = 0  =>  tan(2 theta) = 2B/(A-C). */
        double A = 0.0, B = 0.0, Csum = 0.0;
        for (int i = 0; i < n; i++) {
            double re = h_x_cx[i], im = h_x_cx[n + i];
            A += re*re; B += re*im; Csum += im*im;
        }
        double theta = 0.5 * atan2(2.0*B, A - Csum);
        double ct = cos(theta), st = sin(theta);
        /* Pick the sign of theta that gives the LARGER real-part energy
         * (atan2 branch can land on the minimum). Test theta and theta+pi/2. */
        double E_re = ct*ct*A + 2*ct*st*B + st*st*Csum;
        double E_re_alt = st*st*A - 2*ct*st*B + ct*ct*Csum;
        if (E_re_alt > E_re) { theta += M_PI/2.0; ct = cos(theta); st = sin(theta); }
        double max_re = 0.0, max_im = 0.0;
        for (int i = 0; i < n; i++) {
            double re = h_x_cx[i], im = h_x_cx[n + i];
            double re_rot =  re*ct + im*st;   /* rotate by -theta */
            double im_rot = -re*st + im*ct;
            h_x_cx[i]     = re_rot;
            h_x_cx[n + i] = im_rot;
            if (fabs(re_rot) > max_re) max_re = fabs(re_rot);
            if (fabs(im_rot) > max_im) max_im = fabs(im_rot);
        }
        printf("  Phase rotation: theta = %.2f deg, "
               "max|Re| = %.4e, max|Im| = %.4e (ratio %.2e)\n",
               theta * 180.0 / M_PI, max_re, max_im,
               max_re > 0 ? max_im/max_re : 0.0);
    }
    double* h_x_ibc_re = h_x_cx;  /* real part now carries the physical mode */
    printf("  Phase B done: f = %.6f MHz, Q(eig) = %.1f, %s, time = %.1f s\n",
           ibc.frequency_Hz/1e6, ibc.Q_factor,
           ibc.converged ? "converged" : "NOT converged", t_ibc);

    /* ==================== PHASE C: surface Q + R/Q ==================== */
    printf("\n  Phase C: Q-surface, R/Q, voltage spread\n");

    /* Build an extended PortConfig for the Q-surface integral that includes
     * BOTH the inner-conductor beam ports (if any) AND the endcap pipe
     * apertures (if any). The latter are CavityPort entries with surface
     * type SURFACE_ENDPLATE_Z0/ZL, pos1=r_center, pos2=phi_center, radius=
     * aperture_radius. Without these, compute_q_factor_extended integrates
     * |H_tan|^2 over the entire endplate plane including the pipe holes,
     * overcounting the endplate loss and giving a Q that's ~40% too low. */
    PortConfig qf_ports;
    port_config_init(&qf_ports);
    /* Copy the inner-conductor ports (if any) */
    for (int p = 0; p < inner_ports.num_ports; p++) {
        if (qf_ports.num_ports >= qf_ports.capacity) {
            qf_ports.capacity = qf_ports.capacity ? qf_ports.capacity*2 : 4;
            qf_ports.ports = (CavityPort*)realloc(qf_ports.ports,
                qf_ports.capacity * sizeof(CavityPort));
        }
        qf_ports.ports[qf_ports.num_ports++] = inner_ports.ports[p];
    }
    /* Add endcap pipe apertures as endplate ports */
    if (sc->use_endcap_pipes) {
        CavityPort ep;
        ep.type = PORT_BEAM;
        /* z0 endcap: r=0.85, phi=pi/2, aperture_radius=0.105 */
        ep.surface = SURFACE_ENDPLATE_Z0;
        ep.pos1 = 0.85;       /* r_center on endplate */
        ep.pos2 = M_PI / 2.0; /* phi_center */
        ep.radius = 0.105;
        ep.name = "Endcap z0";
        if (qf_ports.num_ports >= qf_ports.capacity) {
            qf_ports.capacity = qf_ports.capacity ? qf_ports.capacity*2 : 4;
            qf_ports.ports = (CavityPort*)realloc(qf_ports.ports,
                qf_ports.capacity * sizeof(CavityPort));
        }
        qf_ports.ports[qf_ports.num_ports++] = ep;
        /* zL endcap: r=0.85, phi=3pi/2, aperture_radius=0.095 */
        ep.surface = SURFACE_ENDPLATE_ZL;
        ep.pos1 = 0.85;
        ep.pos2 = 3.0 * M_PI / 2.0;
        ep.radius = 0.095;
        ep.name = "Endcap zL";
        if (qf_ports.num_ports >= qf_ports.capacity) {
            qf_ports.capacity *= 2;
            qf_ports.ports = (CavityPort*)realloc(qf_ports.ports,
                qf_ports.capacity * sizeof(CavityPort));
        }
        qf_ports.ports[qf_ports.num_ports++] = ep;
    }

    QFactorResult qf = compute_q_factor_extended(
        &cpu_op, h_x_ibc_re, ibc.k2_re, SIGMA_CU,
        sc->b, qf_ports.num_ports > 0 ? &qf_ports : NULL,
        k_endplate_z0, k_endplate_zL);
    q_factor_print(&qf, sc->label);
    port_config_free(&qf_ports);

    RoverQResult roq = compute_r_over_q(
        &cpu_op, h_x_ibc_re, ibc.k2_re,
        sc->b, sc->L, z0_offset, sc->num_passes);
    r_over_q_print(&roq, sc->label);

    R.f_MHz                = ibc.frequency_Hz / 1e6;
    R.Q_eig                = ibc.Q_factor;
    R.Q_surf               = qf.Q_0;
    R.V_avg                = roq.V_gap_avg;
    R.V_spread_pct         = roq.V_gap_spread;
    R.RoQ_linac_crossing   = roq.R_over_Q_crossing_linac;
    R.RoQ_circuit_crossing = roq.R_over_Q_crossing_circuit;
    R.RoQ_linac_total      = roq.R_over_Q_total_linac;
    R.RoQ_circuit_total    = roq.R_over_Q_total_circuit;

    r_over_q_free(&roq);
    gpu_vector_free(d_x_cx); free(h_x_cx);
    gpu_conformal_pipe_operator_free(&gpu_cfm);
    gpu_vector_free(d_x);
    gpu_pipe_operator_free(&gpu_pec);
    free(h_x);
    conformal_data_free(&cd);
    material_mask_free(&mask);
    curlcurl_op_free(&cpu_op);
    port_config_free(&inner_ports);
    endcap_pipe_config_free(&endcap);
    pipe_config_free(&pipes);
    return R;
}

/*=========================================================================*/
int main(int argc, char** argv)
{
    printf("\n================================================================\n");
    printf("  MANUSCRIPT FINAL TABLE — THREE-STAGE CAVITY SOLVE\n");
    printf("================================================================\n");
    cuda_print_device_info();
    cuda_print_memory_info("startup");

    StageConfig base;
    memset(&base, 0, sizeof(base));
    base.a = 0.3333; base.b = 1.0; base.L = 1.395;
    base.pipe_radius = 0.0125; base.aperture_radius = 0.0175;
    base.pipe_length = 0.050; base.num_passes = 10;
    base.Nr_cavity = 81; base.Nr_pipe = 7;
    base.Nphi      = 256;
    base.Nz_cavity = 84;
    if (argc > 1) base.Nz_cavity = atoi(argv[1]);
    if (argc > 2) base.Nphi      = atoi(argv[2]);
    if (argc > 3) base.Nr_cavity = atoi(argv[3]);

    StageConfig s1 = base; s1.label = "STAGE 1: unperturbed coaxial cavity";
    s1.use_radial_pipes = 0; s1.use_endcap_pipes = 0; s1.use_inner_ports = 0;
    StageConfig s2 = base; s2.label = "STAGE 2: cavity + radial pipes only";
    s2.use_radial_pipes = 1; s2.use_endcap_pipes = 0; s2.use_inner_ports = 0;
    StageConfig s3 = base; s3.label = "STAGE 3: full production (radial+endcap+ports)";
    s3.use_radial_pipes = 1; s3.use_endcap_pipes = 1; s3.use_inner_ports = 1;

    StageResult r1 = run_stage(&s1);
    StageResult r2 = run_stage(&s2);
    StageResult r3 = run_stage(&s3);

    /* Analytical references */
    QFactorResult qa = compute_q_analytical_coaxial_hwr(base.a, base.b, base.L, SIGMA_CU);
    RoverQResult  ra = compute_r_over_q_analytical(base.a, base.b, base.L);
    double f_anal = C0 / (2.0 * base.L);

    /* --- final table --- */
    printf("\n\n================================================================\n");
    printf("  MANUSCRIPT FINAL RESULTS TABLE\n");
    printf("================================================================\n");
    printf("  Grid: Nr=%d, Nphi=%d, Nz_cav=%d\n",
           base.Nr_cavity + base.Nr_pipe, base.Nphi, base.Nz_cavity);

    printf("\n  %-38s %12s %12s %12s %12s\n",
           "Quantity", "Analytical", "Stage 1", "Stage 2", "Stage 3");
    printf("  %-38s %12s %12s %12s %12s\n",
           "", "(ideal HWR)", "(unpert.)", "(rad pipes)", "(full prod)");
    printf("  ----------------------------------------------------------------"
           "------------------------------------\n");
    printf("  %-38s %12.4f %12.4f %12.4f %12.4f\n",
           "Frequency [MHz]", f_anal/1e6, r1.f_MHz, r2.f_MHz, r3.f_MHz);
    printf("  %-38s %12.0f %12.0f %12.0f %12.0f\n",
           "Q (surface integral)",
           qa.Q_0, r1.Q_surf, r2.Q_surf, r3.Q_surf);
    printf("  %-38s %12s %12.0f %12.0f %12.0f\n",
           "Q (from eigenvalue, for ref)",
           "-", r1.Q_eig, r2.Q_eig, r3.Q_eig);
    printf("  %-38s %12.4f %12.4f %12.4f %12.4f\n",
           "R/Q linac per crossing [Ohm]",
           ra.R_over_Q_crossing_linac, r1.RoQ_linac_crossing,
           r2.RoQ_linac_crossing, r3.RoQ_linac_crossing);
    printf("  %-38s %12.4f %12.4f %12.4f %12.4f\n",
           "R/Q circuit per crossing [Ohm]",
           ra.R_over_Q_crossing_circuit, r1.RoQ_circuit_crossing,
           r2.RoQ_circuit_crossing, r3.RoQ_circuit_crossing);
    printf("  %-38s %12s %12.2f %12.2f %12.2f\n",
           "R/Q linac total (10 passes) [Ohm]", "-",
           r1.RoQ_linac_total, r2.RoQ_linac_total, r3.RoQ_linac_total);
    printf("  %-38s %12s %12.2f %12.2f %12.2f\n",
           "R/Q circuit total (10 passes) [Ohm]", "-",
           r1.RoQ_circuit_total, r2.RoQ_circuit_total, r3.RoQ_circuit_total);
    printf("  %-38s %12s %12.4f %12.4f %12.4f\n",
           "Gap voltage spread [%]", "-",
           r1.V_spread_pct, r2.V_spread_pct, r3.V_spread_pct);

    printf("\n================================================================\n");
    printf("  DONE. Stage 3 reproduces test_conformal_ibc_full.\n");
    printf("================================================================\n");
    return 0;
}
