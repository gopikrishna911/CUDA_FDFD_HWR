/*=============================================================================
 * cpu_complex.cpp
 *
 * Complex-valued IBC path on the CPU.  Mirrors the GPU code:
 *   gpu_curlcurl_matvec_complex          (cuda_operator.cu)
 *   gpu_apply_IBC_surface_correction     (cuda_operator.cu)
 *   gpu_gmres_solve_shifted_complex_ws   (cuda_eigensolver.cu)
 *   gpu_rqi_complex_ws                   (cuda_eigensolver.cu)
 *
 * Vectors are packed as [real | imag], size 2 · n_total.
 * The real half is the standard PEC packed order [Er | Ephi | Ez].
 *
 * For the complex inner product we use the conjugate convention
 * ⟨x, y⟩ = Σ conj(x)·y, so
 *   Re⟨x, y⟩_r = Σ r · (x_re·y_re + x_im·y_im) · dV
 *   Im⟨x, y⟩_r = Σ r · (x_re·y_im − x_im·y_re) · dV
 *
 * IBC surface correction:
 *   β = (1 − j) / (2 α)              (with α from ibc_compute_alpha)
 *   bw_cyl_re =  1/(2α dr)   bw_cyl_im = −1/(2α dr)
 *   bw_end_re =  1/(2α dz)   bw_end_im = −1/(2α dz)
 * Applied without zeroing E_tan on conducting surfaces.
 *============================================================================*/

#include "cpu_reference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*=============================================================================
 * Complex field alloc/free
 *============================================================================*/
void cpu_cefield_alloc(ComplexEField* E, const GridParams* grid) {
    efield_alloc(&E->re, grid);
    efield_alloc(&E->im, grid);
}
void cpu_cefield_free(ComplexEField* E) {
    efield_free(&E->re); efield_free(&E->im);
}
void cpu_chfield_alloc(ComplexHField* H, const GridParams* grid) {
    hfield_alloc(&H->re, grid);
    hfield_alloc(&H->im, grid);
}
void cpu_chfield_free(ComplexHField* H) {
    hfield_free(&H->re); hfield_free(&H->im);
}

/*=============================================================================
 * Pack / unpack complex
 *============================================================================*/
void cpu_pack_field_complex_omp(const ComplexEField* E, double* x,
                                const CurlCurlOperator* op) {
    int n = op->n_total;
    /* real half */
    memcpy(x + op->offset_Er,   E->re.Er,   (size_t)op->size_Er   * sizeof(double));
    memcpy(x + op->offset_Ephi, E->re.Ephi, (size_t)op->size_Ephi * sizeof(double));
    memcpy(x + op->offset_Ez,   E->re.Ez,   (size_t)op->size_Ez   * sizeof(double));
    /* imag half */
    memcpy(x + n + op->offset_Er,   E->im.Er,   (size_t)op->size_Er   * sizeof(double));
    memcpy(x + n + op->offset_Ephi, E->im.Ephi, (size_t)op->size_Ephi * sizeof(double));
    memcpy(x + n + op->offset_Ez,   E->im.Ez,   (size_t)op->size_Ez   * sizeof(double));
}
void cpu_unpack_field_complex_omp(const double* x, ComplexEField* E,
                                  const CurlCurlOperator* op) {
    int n = op->n_total;
    memcpy(E->re.Er,   x + op->offset_Er,   (size_t)op->size_Er   * sizeof(double));
    memcpy(E->re.Ephi, x + op->offset_Ephi, (size_t)op->size_Ephi * sizeof(double));
    memcpy(E->re.Ez,   x + op->offset_Ez,   (size_t)op->size_Ez   * sizeof(double));
    memcpy(E->im.Er,   x + n + op->offset_Er,   (size_t)op->size_Er   * sizeof(double));
    memcpy(E->im.Ephi, x + n + op->offset_Ephi, (size_t)op->size_Ephi * sizeof(double));
    memcpy(E->im.Ez,   x + n + op->offset_Ez,   (size_t)op->size_Ez   * sizeof(double));
}

/*=============================================================================
 * Operator lifecycle
 *============================================================================*/
int cpu_complex_op_init(CpuComplexOperator* cop, const CurlCurlOperator* op) {
    cop->op = op;
    cpu_cefield_alloc(&cop->E_work,      &op->grid);
    cpu_cefield_alloc(&cop->result_work, &op->grid);
    cpu_chfield_alloc(&cop->H_temp,      &op->grid);
    return 0;
}
void cpu_complex_op_free(CpuComplexOperator* cop) {
    cpu_cefield_free(&cop->E_work);
    cpu_cefield_free(&cop->result_work);
    cpu_chfield_free(&cop->H_temp);
}

/*=============================================================================
 * Complex curl-curl: compute curl(curl(E)) on both real and imaginary parts.
 *
 * Identical to running the real curl-curl twice (the operator is real-linear,
 * so re and im decouple).  We reuse the OMP-parallel real routines.
 *============================================================================*/
static void cpu_compute_curl_curl_E_complex_omp(const ComplexEField* E,
                                                ComplexEField* result,
                                                ComplexHField* G,
                                                const GridParams* grid) {
    cpu_compute_curl_curl_E_omp(&E->re, &result->re, &G->re, grid);
    cpu_compute_curl_curl_E_omp(&E->im, &result->im, &G->im, grid);
}

/*=============================================================================
 * IBC surface correction: result += β·diag(w)·E at all 4 conducting walls.
 *
 * Mirrors gpu_apply_IBC_surface_correction.
 *  - cylinder walls (i=0, i=Nr) touch Ephi(i_wall, j+½, k) and Ez(i_wall, j, k+½)
 *  - endplates (k=0, k=Nz) touch Er(i+½, j, k_wall) and Ephi(i, j+½, k_wall)
 *
 * NB: we do NOT support port masks here (the user has not requested ports in
 * the IBC path yet).  If you need them, mirror build_port_masks().
 *============================================================================*/
static void cpu_apply_IBC_surface_correction(
    const ComplexEField* E, ComplexEField* result,
    double alpha, const GridParams* grid)
{
    const int Nr = grid->Nr, Nphi = grid->Nphi, Nz = grid->Nz;
    const double inv_2a   = 1.0 / (2.0 * alpha);
    const double bw_cyl_re =  inv_2a / grid->dr;
    const double bw_cyl_im = -inv_2a / grid->dr;
    const double bw_end_re =  inv_2a / grid->dz;
    const double bw_end_im = -inv_2a / grid->dz;

    /* ---- Inner conductor (i=0) ---- */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            int idx = idx_Ephi(grid, 0, j, k);
            double er = E->re.Ephi[idx], ei = E->im.Ephi[idx];
            result->re.Ephi[idx] += bw_cyl_re * er - bw_cyl_im * ei;
            result->im.Ephi[idx] += bw_cyl_re * ei + bw_cyl_im * er;
        }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            int idx = idx_Ez(grid, 0, j, k);
            double er = E->re.Ez[idx], ei = E->im.Ez[idx];
            result->re.Ez[idx] += bw_cyl_re * er - bw_cyl_im * ei;
            result->im.Ez[idx] += bw_cyl_re * ei + bw_cyl_im * er;
        }
    }

    /* ---- Outer conductor (i=Nr) ---- */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            int idx = idx_Ephi(grid, Nr, j, k);
            double er = E->re.Ephi[idx], ei = E->im.Ephi[idx];
            result->re.Ephi[idx] += bw_cyl_re * er - bw_cyl_im * ei;
            result->im.Ephi[idx] += bw_cyl_re * ei + bw_cyl_im * er;
        }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            int idx = idx_Ez(grid, Nr, j, k);
            double er = E->re.Ez[idx], ei = E->im.Ez[idx];
            result->re.Ez[idx] += bw_cyl_re * er - bw_cyl_im * ei;
            result->im.Ez[idx] += bw_cyl_re * ei + bw_cyl_im * er;
        }
    }

    /* ---- Endplate z=0 (k=0) ---- */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < Nphi; j++) {
        for (int i = 0; i < Nr; i++) {
            int idx = idx_Er(grid, i, j, 0);
            double er = E->re.Er[idx], ei = E->im.Er[idx];
            result->re.Er[idx] += bw_end_re * er - bw_end_im * ei;
            result->im.Er[idx] += bw_end_re * ei + bw_end_im * er;
        }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < Nphi; j++) {
        for (int i = 0; i <= Nr; i++) {
            int idx = idx_Ephi(grid, i, j, 0);
            double er = E->re.Ephi[idx], ei = E->im.Ephi[idx];
            result->re.Ephi[idx] += bw_end_re * er - bw_end_im * ei;
            result->im.Ephi[idx] += bw_end_re * ei + bw_end_im * er;
        }
    }

    /* ---- Endplate z=L (k=Nz) ---- */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < Nphi; j++) {
        for (int i = 0; i < Nr; i++) {
            int idx = idx_Er(grid, i, j, Nz);
            double er = E->re.Er[idx], ei = E->im.Er[idx];
            result->re.Er[idx] += bw_end_re * er - bw_end_im * ei;
            result->im.Er[idx] += bw_end_re * ei + bw_end_im * er;
        }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < Nphi; j++) {
        for (int i = 0; i <= Nr; i++) {
            int idx = idx_Ephi(grid, i, j, Nz);
            double er = E->re.Ephi[idx], ei = E->im.Ephi[idx];
            result->re.Ephi[idx] += bw_end_re * er - bw_end_im * ei;
            result->im.Ephi[idx] += bw_end_re * ei + bw_end_im * er;
        }
    }
}

/*=============================================================================
 * Complex IBC matvec: y = (∇×∇× + β·diag(w)) · x
 *
 * 1. unpack x → E_work (re + im); do NOT zero E_tan at walls
 * 2. curl-curl on both real and imag halves → result_work
 * 3. add IBC surface correction
 * 4. pack result_work → y
 *============================================================================*/
void cpu_curlcurl_matvec_complex_omp(const CpuComplexOperator* cop,
                                     const double* x, double* y, double alpha)
{
    CpuComplexOperator* m = (CpuComplexOperator*)cop;
    const CurlCurlOperator* op = cop->op;

    cpu_unpack_field_complex_omp(x, &m->E_work, op);
    cpu_compute_curl_curl_E_complex_omp(&m->E_work, &m->result_work,
                                        &m->H_temp, &op->grid);
    cpu_apply_IBC_surface_correction(&m->E_work, &m->result_work,
                                     alpha, &op->grid);
    cpu_pack_field_complex_omp(&m->result_work, y, op);
}

/*=============================================================================
 * Weighted complex inner products
 *
 * Re⟨x, y⟩_r = Σ r · (x_re·y_re + x_im·y_im) · dV
 * Im⟨x, y⟩_r = Σ r · (x_re·y_im − x_im·y_re) · dV
 *
 * x and y are packed as [real | imag], so x_re = x[idx], x_im = x[n+idx], etc.
 *============================================================================*/
void cpu_vec_dot_weighted_complex_omp(const double* x, const double* y,
                                      double* dot_re, double* dot_im,
                                      const CurlCurlOperator* op)
{
    const GridParams* g = &op->grid;
    const int n = op->n_total;
    const double dV = g->dr * g->dphi * g->dz;

    double sr_er = 0.0, si_er = 0.0;
    double sr_ph = 0.0, si_ph = 0.0;
    double sr_ez = 0.0, si_ez = 0.0;

    /* Er — weight r_{i+1/2} */
    #pragma omp parallel for collapse(2) reduction(+:sr_er,si_er) schedule(static)
    for (int k = 0; k <= g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i < g->Nr; i++) {
                int idx = op->offset_Er + idx_Er(g, i, j, k);
                double r = g->a + (i + 0.5) * g->dr;
                double xr = x[idx],     xi = x[idx + n];
                double yr = y[idx],     yi = y[idx + n];
                sr_er += r * (xr * yr + xi * yi);
                si_er += r * (xr * yi - xi * yr);
            }
        }
    }
    /* Ephi — weight r_i */
    #pragma omp parallel for collapse(2) reduction(+:sr_ph,si_ph) schedule(static)
    for (int k = 0; k <= g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i <= g->Nr; i++) {
                int idx = op->offset_Ephi + idx_Ephi(g, i, j, k);
                double r = g->a + i * g->dr;
                double xr = x[idx],     xi = x[idx + n];
                double yr = y[idx],     yi = y[idx + n];
                sr_ph += r * (xr * yr + xi * yi);
                si_ph += r * (xr * yi - xi * yr);
            }
        }
    }
    /* Ez — weight r_i */
    #pragma omp parallel for collapse(2) reduction(+:sr_ez,si_ez) schedule(static)
    for (int k = 0; k < g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i <= g->Nr; i++) {
                int idx = op->offset_Ez + idx_Ez(g, i, j, k);
                double r = g->a + i * g->dr;
                double xr = x[idx],     xi = x[idx + n];
                double yr = y[idx],     yi = y[idx + n];
                sr_ez += r * (xr * yr + xi * yi);
                si_ez += r * (xr * yi - xi * yr);
            }
        }
    }
    *dot_re = (sr_er + sr_ph + sr_ez) * dV;
    *dot_im = (si_er + si_ph + si_ez) * dV;
}

double cpu_vec_norm_weighted_complex_omp(const double* x,
                                         const CurlCurlOperator* op) {
    double re, im;
    cpu_vec_dot_weighted_complex_omp(x, x, &re, &im, op);
    /* Im part of ⟨x,x⟩ should be 0 in exact arithmetic; ignore. */
    return sqrt(re);
}

void cpu_vec_normalize_weighted_complex_omp(double* x,
                                            const CurlCurlOperator* op) {
    double nrm = cpu_vec_norm_weighted_complex_omp(x, op);
    if (nrm > 1e-14) cpu_vec_scale_omp(x, 1.0 / nrm, 2 * op->n_total);
}

/*=============================================================================
 * Flat (unweighted) complex inner products — used inside the GMRES Arnoldi
 * orthogonalisation, exactly as the GPU code does.
 *============================================================================*/
void cpu_vec_dot_flat_complex_omp(const double* x, const double* y,
                                  double* dot_re, double* dot_im, int n_real)
{
    double sr = 0.0, si = 0.0;
    #pragma omp parallel for reduction(+:sr,si) schedule(static)
    for (int i = 0; i < n_real; i++) {
        double xr = x[i],         xi = x[i + n_real];
        double yr = y[i],         yi = y[i + n_real];
        sr += xr * yr + xi * yi;
        si += xr * yi - xi * yr;
    }
    *dot_re = sr; *dot_im = si;
}

double cpu_vec_norm_flat_complex_omp(const double* x, int n_real) {
    double re, im;
    cpu_vec_dot_flat_complex_omp(x, x, &re, &im, n_real);
    return sqrt(re);
}

/*=============================================================================
 * Complex AXPY:    w += (cr + j·ci) · v        (v, w packed as [re | im])
 *   w_re += cr·v_re − ci·v_im
 *   w_im += cr·v_im + ci·v_re
 *============================================================================*/
void cpu_vec_axpy_complex_omp(double cr, double ci,
                              const double* v, double* w, int n_real)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_real; i++) {
        double vr = v[i], vi = v[i + n_real];
        w[i]          += cr * vr - ci * vi;
        w[i + n_real] += cr * vi + ci * vr;
    }
}

/*=============================================================================
 * Complex shift:    Av −= (σ_re + j·σ_im) · v       (Av, v packed [re|im])
 *============================================================================*/
void cpu_vec_shift_complex_omp(double* Av, const double* v,
                               double sigma_re, double sigma_im, int n_real)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_real; i++) {
        double vr = v[i], vi = v[i + n_real];
        Av[i]          -= sigma_re * vr - sigma_im * vi;
        Av[i + n_real] -= sigma_re * vi + sigma_im * vr;
    }
}

/*=============================================================================
 * GMRES(m) workspace
 *============================================================================*/
int cpu_gmres_workspace_init(CpuGmresWorkspace* ws, int n_real, int m) {
    ws->m      = m;
    ws->n_real = n_real;
    ws->initialized = 0;

    int n = 2 * n_real;
    ws->V = (double**)calloc((size_t)(m + 1), sizeof(double*));
    for (int j = 0; j <= m; j++) {
        ws->V[j] = (double*)calloc((size_t)n, sizeof(double));
        if (!ws->V[j]) return -1;
    }
    ws->w = (double*)calloc((size_t)n, sizeof(double));

    ws->H_re  = (double*)calloc((size_t)(m + 1) * m, sizeof(double));
    ws->H_im  = (double*)calloc((size_t)(m + 1) * m, sizeof(double));
    ws->cs_re = (double*)calloc((size_t)m, sizeof(double));
    ws->cs_im = (double*)calloc((size_t)m, sizeof(double));
    ws->sn_re = (double*)calloc((size_t)m, sizeof(double));
    ws->sn_im = (double*)calloc((size_t)m, sizeof(double));
    ws->g_re  = (double*)calloc((size_t)(m + 1), sizeof(double));
    ws->g_im  = (double*)calloc((size_t)(m + 1), sizeof(double));
    ws->y_re  = (double*)calloc((size_t)m, sizeof(double));
    ws->y_im  = (double*)calloc((size_t)m, sizeof(double));

    ws->initialized = 1;
    return 0;
}
void cpu_gmres_workspace_free(CpuGmresWorkspace* ws) {
    if (!ws->initialized) return;
    for (int j = 0; j <= ws->m; j++) free(ws->V[j]);
    free(ws->V);  free(ws->w);
    free(ws->H_re);  free(ws->H_im);
    free(ws->cs_re); free(ws->cs_im);
    free(ws->sn_re); free(ws->sn_im);
    free(ws->g_re);  free(ws->g_im);
    free(ws->y_re);  free(ws->y_im);
    ws->initialized = 0;
}

/*=============================================================================
 * GMRES(m) — complex shifted system   (A − σI) x = b
 *
 * Restarted Arnoldi with modified Gram–Schmidt and complex Givens rotations.
 * Mirrors gpu_gmres_solve_shifted_complex_ws line-for-line; uses flat
 * (unweighted) inner products for Arnoldi (as the GPU does, by design — the
 * Krylov subspace is the same, and flat ops are cheaper).
 *============================================================================*/
CpuLinSolveResult cpu_gmres_solve_complex_omp(
    const CpuComplexOperator* cop,
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
        /* r = b - (A - σI)·x */
        if (total_iters == 0) {
            cpu_vec_copy_omp(b, ws->V[0], n);
            residual_norm = b_norm;
        } else {
            cpu_curlcurl_matvec_complex_omp(cop, x, ws->w, alpha);
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

        /* g = [β, 0, ...] */
        ws->g_re[0] = residual_norm; ws->g_im[0] = 0.0;
        for (int i = 1; i <= m; i++) { ws->g_re[i] = 0.0; ws->g_im[i] = 0.0; }
        memset(ws->H_re, 0, (size_t)(m + 1) * m * sizeof(double));
        memset(ws->H_im, 0, (size_t)(m + 1) * m * sizeof(double));

        int j_final = 0;

        for (int j = 0; j < m && total_iters < max_iter; j++) {
            total_iters++;

            /* w = (A - σI) · v_j */
            cpu_curlcurl_matvec_complex_omp(cop, ws->V[j], ws->w, alpha);
            cpu_vec_shift_complex_omp(ws->w, ws->V[j], sigma_re, sigma_im, n_real);

            /* Modified Gram–Schmidt against v_0..v_j */
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

            /* Apply previous Givens rotations to column j */
            for (int i = 0; i < j; i++) {
                double a_re = ws->H_re[i * m + j], a_im = ws->H_im[i * m + j];
                double b_re = ws->H_re[(i + 1) * m + j], b_im = ws->H_im[(i + 1) * m + j];
                double c_re = ws->cs_re[i], c_im = ws->cs_im[i];
                double s_re = ws->sn_re[i], s_im = ws->sn_im[i];

                /* new_a = conj(c)·a + conj(s)·b */
                double na_re = (c_re * a_re + c_im * a_im) + (s_re * b_re + s_im * b_im);
                double na_im = (c_re * a_im - c_im * a_re) + (s_re * b_im - s_im * b_re);
                /* new_b = -s·a + c·b */
                double nb_re = -(s_re * a_re - s_im * a_im) + (c_re * b_re - c_im * b_im);
                double nb_im = -(s_re * a_im + s_im * a_re) + (c_re * b_im + c_im * b_re);

                ws->H_re[i * m + j]       = na_re;
                ws->H_im[i * m + j]       = na_im;
                ws->H_re[(i + 1) * m + j] = nb_re;
                ws->H_im[(i + 1) * m + j] = nb_im;
            }

            /* New Givens rotation for row j */
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

                /* Apply to g */
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

        /* Back-substitution: solve upper-triangular H · y = g (complex) */
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

        /* x += Σ y_j · v_j  (complex AXPY) */
        for (int j = 0; j < j_final; j++) {
            cpu_vec_axpy_complex_omp(ws->y_re[j], ws->y_im[j],
                                     ws->V[j], x, n_real);
        }

        if (result.converged) break;
    }

    return result;
}

/*=============================================================================
 * Complex residual ||A·x − σ·x||_r / ||x||_r,  σ complex
 *============================================================================*/
static double cpu_compute_complex_residual(
    const CpuComplexOperator* cop,
    const double* Ax, const double* x,
    double sigma_re, double sigma_im,
    double* temp)
{
    int n_real = cop->op->n_total;
    int n      = 2 * n_real;
    cpu_vec_copy_omp(Ax, temp, n);
    cpu_vec_shift_complex_omp(temp, x, sigma_re, sigma_im, n_real);
    double r_norm = cpu_vec_norm_weighted_complex_omp(temp, cop->op);
    double x_norm = cpu_vec_norm_weighted_complex_omp(x,    cop->op);
    return (x_norm > 1e-28) ? r_norm / x_norm : r_norm;
}

/*=============================================================================
 * Complex Rayleigh quotient   σ = ⟨x, A·x⟩_r / ⟨x, x⟩_r
 *============================================================================*/
static void cpu_compute_complex_rayleigh_quotient(
    const CpuComplexOperator* cop,
    const double* x, const double* Ax,
    double* sigma_re, double* sigma_im)
{
    double xAx_re, xAx_im, xx_re, xx_im;
    cpu_vec_dot_weighted_complex_omp(x, Ax, &xAx_re, &xAx_im, cop->op);
    cpu_vec_dot_weighted_complex_omp(x, x,  &xx_re,  &xx_im,  cop->op);
    if (xx_re > 1e-28) {
        *sigma_re = xAx_re / xx_re;
        *sigma_im = xAx_im / xx_re;
    } else {
        *sigma_re = 0.0; *sigma_im = 0.0;
    }
}

/*=============================================================================
 * α = 1 / sqrt(2 σ ω μ_0),    ω = c·sqrt(k²_re)
 * Same as ibc_compute_alpha() in cuda_operator.h.
 *============================================================================*/
static inline double cpu_ibc_compute_alpha(double k2_re, double conductivity) {
    const double c0  = 299792458.0;
    const double mu0 = 4.0e-7 * M_PI;
    double omega = c0 * sqrt(fabs(k2_re));
    if (omega < 1.0) omega = 1.0;
    return 1.0 / sqrt(2.0 * conductivity * omega * mu0);
}

/*=============================================================================
 * Complex eigensolver workspace
 *============================================================================*/
int cpu_complex_eigensolver_workspace_init(CpuComplexEigenWorkspace* ws,
                                           int n_real, int gmres_restart) {
    ws->n_real = n_real;
    ws->initialized = 0;
    if (cpu_gmres_workspace_init(&ws->gmres_ws, n_real, gmres_restart) != 0) return -1;
    ws->y  = (double*)calloc((size_t)(2 * n_real), sizeof(double));
    ws->Ax = (double*)calloc((size_t)(2 * n_real), sizeof(double));
    ws->initialized = 1;
    return 0;
}
void cpu_complex_eigensolver_workspace_free(CpuComplexEigenWorkspace* ws) {
    if (!ws->initialized) return;
    cpu_gmres_workspace_free(&ws->gmres_ws);
    free(ws->y); free(ws->Ax);
    ws->initialized = 0;
}

/*=============================================================================
 * Complex Rayleigh-Quotient Iteration  (mirrors gpu_rqi_complex_ws exactly)
 *
 * The GMRES shift σ is kept fixed at the user-provided sigma_init (= k²_PEC).
 * The Rayleigh quotient is still tracked each iteration for convergence and
 * to update α each step.
 *============================================================================*/
CpuComplexEigenResult cpu_rqi_complex_omp(
    const CpuComplexOperator* cop,
    double* x,
    double sigma_init,
    double conductivity,
    int max_iter, double tol,
    int gmres_restart,
    CpuComplexEigenWorkspace* ws)
{
    CpuComplexEigenResult r;
    r.k2_re = 0.0; r.k2_im = 0.0;
    r.frequency_Hz = 0.0; r.Q_factor = 0.0;
    r.iterations = 0; r.residual = 1.0; r.converged = 0;

    int n_real = cop->op->n_total;
    int n      = 2 * n_real;
    const double c0 = 299792458.0;
    (void)gmres_restart;  /* restart size is baked into ws->gmres_ws */

    cpu_vec_normalize_weighted_complex_omp(x, cop->op);

    double alpha = cpu_ibc_compute_alpha(sigma_init, conductivity);
    cpu_curlcurl_matvec_complex_omp(cop, x, ws->Ax, alpha);

    double sigma_re, sigma_im;
    cpu_compute_complex_rayleigh_quotient(cop, x, ws->Ax, &sigma_re, &sigma_im);
    /* GPU code overrides to (sigma_init, 0) — keep that behaviour exactly */
    sigma_re = sigma_init;
    sigma_im = 0.0;

    printf("\n  Complex RQI (IBC, CPU, threads=%d):\n", cpu_omp_num_threads());
    printf("  %-5s %-15s %-12s %-12s %-12s %-6s\n",
           "Iter", "k2_re", "k2_im", "Q", "Residual", "GMRES");
    printf("  --------------------------------------------------"
           "----------------------\n");

    double sigma_re_old = sigma_re, sigma_im_old = sigma_im;

    for (int iter = 0; iter < max_iter; iter++) {
        alpha = cpu_ibc_compute_alpha(sigma_re, conductivity);

        int    gmres_maxiter = (iter == 0) ? 300  : 2000;
        double gmres_tol     = (iter == 0) ? 5e-1 : 1e-1;

        CpuLinSolveResult ls = cpu_gmres_solve_complex_omp(
            cop, sigma_re, sigma_im,
            x, ws->y, alpha,
            gmres_maxiter, gmres_tol, &ws->gmres_ws);

        cpu_vec_normalize_weighted_complex_omp(ws->y, cop->op);
        cpu_vec_copy_omp(ws->y, x, n);

        alpha = cpu_ibc_compute_alpha(sigma_re, conductivity);
        cpu_curlcurl_matvec_complex_omp(cop, x, ws->Ax, alpha);

        double sigma_re_new, sigma_im_new;
        cpu_compute_complex_rayleigh_quotient(cop, x, ws->Ax,
                                              &sigma_re_new, &sigma_im_new);

        double res = cpu_compute_complex_residual(
            cop, ws->Ax, x, sigma_re_new, sigma_im_new, ws->y);

        double freq = (sigma_re_new > 0)
            ? c0 * sqrt(sigma_re_new) / (2.0 * M_PI) : 0.0;
        double Q = (fabs(sigma_im_new) > 1e-30)
            ? -sigma_re_new / sigma_im_new : 0.0;

        r.k2_re = sigma_re_new;
        r.k2_im = sigma_im_new;
        r.frequency_Hz = freq;
        r.Q_factor = Q;
        r.residual = res;
        r.iterations = iter + 1;

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
        /* Keep σ fixed at sigma_init for the GMRES shift, per GPU code */
    }
    return r;
}
