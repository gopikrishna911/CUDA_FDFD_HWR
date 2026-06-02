/*=============================================================================
 * test_conformal_ibc.cpp — IBC mask + endplate grid-plane IBC
 *============================================================================*/

#include "cuda_conformal_pipe.h"
#include "cuda_pipe_model.h"
#include "cuda_eigensolver.h"
#include "conformal_geometry.h"
#include "pipe_model.h"
#include "curlcurl_operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>
#include "field_map_export.h"

#define C0          299792458.0
#define MU0         (4.0e-7 * M_PI)
#define SIGMA_CU    5.8e7

extern __global__ void shift_kernel(double*, const double*, double, int);

static double run_pec_pipe_solver(
    GPU_PipeOperator* pipe_op, const CurlCurlOperator* cpu_op,
    const GridParams* grid, double* d_x, int rqi_max, int minres_max, double tol
) {
    int n = cpu_op->n_total, blocks = (n+256-1)/256;
    EigensolverWorkspace ws; eigensolver_workspace_init(&ws, n);
    gpu_vec_normalize_weighted(d_x, cpu_op);
    double* d_Ax; cudaMalloc(&d_Ax, n*sizeof(double));
    gpu_pipe_matvec(pipe_op, d_x, d_Ax);
    double xAx, xx;
    gpu_vec_dot_weighted_ws(d_x, d_Ax, &xAx, cpu_op, &pipe_op->base.reduction_ws);
    gpu_vec_dot_weighted_ws(d_x, d_x, &xx, cpu_op, &pipe_op->base.reduction_ws);
    double sigma = xAx/xx;
    if (fabs(sigma)<1e-10||sigma<0) sigma=(M_PI/grid->L)*(M_PI/grid->L);
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
            if(fabs(eta)/b_norm<1e-6) break;
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
        printf("      iter %d: k2=%.10f, MINRES=%d\n",iter,sn,ls_iters);
        if(fabs(sn-sigma)/fabs(sigma)<tol&&iter>0){printf("    Converged.\n");sigma=sn;break;}
        sigma=sn;
    }
    cudaFree(d_Ax); eigensolver_workspace_free(&ws); return sigma;
}

int main(void) {
    printf("\n================================================================\n");
    printf("  CONFORMAL IBC — ENDPLATE GRID-PLANE FIX\n");
    printf("================================================================\n\n");
    cuda_print_device_info(); cuda_print_memory_info("startup");

    double a=0.3333,b=1.0,L=1.395;
    double pipe_radius=0.0125,aperture_radius=0.0175,pipe_length=0.050;
    int num_passes=10;
    double ec_z0_rc=0.85, ec_z0_phi=M_PI/2.0, ec_z0_ar=0.105, ec_z0_pr=0.100, ec_z0_pl=0.28;
    double ec_zL_rc=0.85, ec_zL_phi=3.0*M_PI/2.0, ec_zL_ar=0.095, ec_zL_pr=0.090, ec_zL_pl=0.25;

    PipeConfig pipes;
    pipe_config_init(&pipes,a,b,pipe_radius,aperture_radius,pipe_length,0.0);
    pipe_config_add_multi_pass(&pipes,L/2.0,num_passes);

    EndcapPipeConfig endcap_pipes;
    endcap_pipe_config_init(&endcap_pipes);
    endcap_pipe_config_add(&endcap_pipes,ec_z0_rc,ec_z0_phi,ec_z0_ar,ec_z0_pr,ec_z0_pl,1);
    endcap_pipe_config_add(&endcap_pipes,ec_zL_rc,ec_zL_phi,ec_zL_ar,ec_zL_pr,ec_zL_pl,0);

    int Nr_cavity=81, Nr_pipe=7, Nphi=256, Nz_cavity=84;
    double dz_target=L/Nz_cavity;
    int Nz_pipe_z0=(int)ceil(endcap_pipes.z0_extension/dz_target);
    int Nz_pipe_zL=(int)ceil(endcap_pipes.zL_extension/dz_target);

    GridParams grid;
    grid_init_with_all_pipes(&grid,a,b,L,pipe_length,
        endcap_pipes.z0_extension,endcap_pipes.zL_extension,
        Nr_cavity,Nr_pipe,Nphi,Nz_cavity,Nz_pipe_z0,Nz_pipe_zL);

    int k_z0=Nz_pipe_z0, k_zL=Nz_pipe_z0+Nz_cavity;
    double z0_offset=k_z0*grid.dz;
    printf("  Endplates: k_z0=%d, k_zL=%d (Nz=%d)\n",k_z0,k_zL,grid.Nz);
    printf("  Using snapped z0_offset=%.6f m (requested %.6f m)\n",
        z0_offset,endcap_pipes.z0_extension);

    PortConfig inner_ports; port_config_init(&inner_ports);
    double dphi_pass=M_PI/num_passes;
    for(int pass=0;pass<num_passes;pass++){
        double pe=pass*dphi_pass, px=pe+M_PI;
        if(px>=2*M_PI) px-=2*M_PI;
        CavityPort port; port.type=PORT_BEAM; port.surface=SURFACE_INNER;
        port.radius=aperture_radius; port.pos2=L/2.0+z0_offset;
        port.pos1=pe; port.name="Entry";
        if(inner_ports.num_ports>=inner_ports.capacity){
            inner_ports.capacity*=2;
            inner_ports.ports=(CavityPort*)realloc(inner_ports.ports,inner_ports.capacity*sizeof(CavityPort));}
        inner_ports.ports[inner_ports.num_ports++]=port;
        port.pos1=px; port.name="Exit";
        if(inner_ports.num_ports>=inner_ports.capacity){
            inner_ports.capacity*=2;
            inner_ports.ports=(CavityPort*)realloc(inner_ports.ports,inner_ports.capacity*sizeof(CavityPort));}
        inner_ports.ports[inner_ports.num_ports++]=port;
    }

    MaterialMask mask;
    material_mask_build_full(&mask,&pipes,&endcap_pipes,&grid,L,z0_offset);
    material_mask_print_stats(&mask,&grid);

    ConformalData cd;
    conformal_data_build(&cd,&pipes,&grid,b,z0_offset);
    conformal_data_print_stats(&cd,&grid);

    /* IBC unmask fixup (for outer conductor surface cells) */
    {
        MaterialMask ibc_mask;
        material_mask_build_ibc(&ibc_mask,&mask,&grid);
        conformal_data_apply_ibc_unmask(&cd,&mask,&ibc_mask,&grid);
        //conformal_data_zero_endplate_faces(&cd, &grid, k_z0, k_zL); //--------------------------------------------------------------------------
        material_mask_free(&ibc_mask);
    }

    CurlCurlOperator cpu_op;
    curlcurl_op_init_with_ports(&cpu_op,&grid,&inner_ports);
    int n=cpu_op.n_total;
    printf("  DOFs: %d (%.1f M)\n",n,n/1e6);
    cuda_grid_init(&grid);

    double* h_x=(double*)calloc(n,sizeof(double));
    for(int k=0;k<=grid.Nz;k++){
        double zp=k*grid.dz-z0_offset;
        for(int j=0;j<grid.Nphi;j++) for(int i=0;i<grid.Nr;i++){
            double r=grid.a+(i+0.5)*grid.dr;
            if(r<=b&&zp>=0&&zp<=L)
                h_x[cpu_op.offset_Er+idx_Er(&grid,i,j,k)]=sin(M_PI*zp/L)/r;
        }
    }

    /* PHASE 1: PEC */
    printf("\n================================================================\n");
    printf("  PHASE 1: PEC\n");
    printf("================================================================\n\n");
    double k2_pec,f_pec;
    {
        GPU_PipeOperator gpu_pec;
        gpu_pipe_operator_init(&gpu_pec,&cpu_op,&mask);
        gpu_pipe_operator_set_endplates(&gpu_pec,k_z0,k_zL,&endcap_pipes,&grid,z0_offset);
        double*d_x; gpu_vector_alloc(&d_x,n); gpu_vector_to_device(d_x,h_x,n);
        k2_pec=run_pec_pipe_solver(&gpu_pec,&cpu_op,&grid,d_x,20,3000,1e-8);
        f_pec=C0*sqrt(fabs(k2_pec))/(2*M_PI);
        gpu_vector_to_host(h_x,d_x,n);
        printf("\n  PEC: k2=%.10f, f=%.6f MHz\n\n",k2_pec,f_pec/1e6);
        gpu_vector_free(d_x); gpu_pipe_operator_free(&gpu_pec);
    }

    /* PHASE 2: Conformal IBC Perturbative */
    printf("================================================================\n");
    printf("  PHASE 2: CONFORMAL IBC PERTURBATIVE\n");
    printf("================================================================\n\n");
    double f_pert=0,Q_pert=0;
    {
        GPU_ConformalPipeOperator gpu_cfm;
        gpu_conformal_pipe_operator_init(&gpu_cfm,&cpu_op,&mask,&cd,
            k_z0,k_zL,&endcap_pipes,&grid,z0_offset);
        double*d_x; gpu_vector_alloc(&d_x,n); gpu_vector_to_device(d_x,h_x,n);
        GPU_ComplexEigenResult res=gpu_ibc_perturbative_conformal_pipe(&gpu_cfm,d_x,k2_pec,SIGMA_CU);
        f_pert=res.frequency_Hz; Q_pert=res.Q_factor;
        gpu_vector_free(d_x); gpu_conformal_pipe_operator_free(&gpu_cfm);
    }

    /* PHASE 3: Conformal IBC Iterative */
    printf("\n================================================================\n");
    printf("  PHASE 3: CONFORMAL IBC ITERATIVE\n");
    printf("================================================================\n\n");
    double f_iter=0,Q_iter=0;
    {
        GPU_ConformalPipeOperator gpu_cfm;
        gpu_conformal_pipe_operator_init(&gpu_cfm,&cpu_op,&mask,&cd,
            k_z0,k_zL,&endcap_pipes,&grid,z0_offset);
        int n2=2*n; double*h_cx=(double*)calloc(n2,sizeof(double));
        memcpy(h_cx,h_x,n*sizeof(double));
        double*d_cx; gpu_vector_alloc(&d_cx,n2); gpu_vector_to_device(d_cx,h_cx,n2);
        clock_t t0=clock();
        GPU_ComplexEigenResult res=gpu_rqi_complex_conformal_pipe(&gpu_cfm,d_cx,
            k2_pec,SIGMA_CU,30,1e-6,50);
        double t=(double)(clock()-t0)/CLOCKS_PER_SEC;
        f_iter=res.frequency_Hz; Q_iter=res.Q_factor;
        printf("\n  Iterative: k2=%.10f, f=%.6f MHz, Q=%.1f, res=%.2e, iters=%d, T=%.1fs\n\n",
            res.k2_re,f_iter/1e6,Q_iter,res.residual,res.iterations,t);
        gpu_conformal_pipe_print_ibc_breakdown(&gpu_cfm,d_cx,res.k2_re,SIGMA_CU);
        gpu_vector_free(d_cx); free(h_cx); gpu_conformal_pipe_operator_free(&gpu_cfm);
    }

    /* COMPARISON */
    printf("================================================================\n");
    printf("  COMPARISON\n");
    printf("================================================================\n\n");
    printf("    %-35s %12s %10s\n","Method","f(MHz)","Q");
    printf("    %-35s %12s %10s\n","------","------","---");
    printf("    %-35s %12.4f %10s\n","Analytical (ref cavity)",107.4525,"48202");
    printf("    %-35s %12.4f %10s\n","PEC (full model)",f_pec/1e6,"—");
    printf("    %-35s %12.4f %10.0f\n","Conformal IBC perturbative",f_pert/1e6,Q_pert);
    printf("    %-35s %12.4f %10.0f\n","Conformal IBC iterative",f_iter/1e6,Q_iter);
    printf("\n    f shift (iterative vs PEC): %+.4f%%\n",(f_iter-f_pec)/f_pec*100);
    printf("    Expected: ~0.001%% f shift, Q ~48000\n");
    printf("\n================================================================\n\n");

    free(h_x); port_config_free(&inner_ports); endcap_pipe_config_free(&endcap_pipes);
    pipe_config_free(&pipes); material_mask_free(&mask);
    conformal_data_free(&cd); curlcurl_op_free(&cpu_op);
    return 0;
}
