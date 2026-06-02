/*=============================================================================
 * cpu_reference.cpp
 *
 * Real-valued (PEC) CPU reference implementation with optional OpenMP.
 * Algorithmically identical to:
 *   - compute_curl_E       in curl_E.cpp
 *   - compute_curl_H       in curl_H.cpp
 *   - compute_curl_curl_E  in curl_H.cpp
 *   - curlcurl_matvec, apply_PEC_boundary, vec_dot_product_weighted, etc.
 *   - gpu_rqi_ws with gpu_minres_solve_shifted_ws as inner solver
 *
 * Loop bounds, index expressions, periodic-φ handling, r-weights and the
 * r=0 guard match the existing CPU code exactly. Parallel reductions use
 * #pragma omp parallel for reduction(+:sum) — these are bit-stable across
 * runs at fixed thread count but not bit-identical to a strict serial sum.
 * In practice the converged k² agrees with the serial path to < 1e-12.
 *============================================================================*/

#include "cpu_reference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*=============================================================================
 * OMP helpers
 *============================================================================*/
int cpu_omp_num_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
void cpu_omp_set_num_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

/*=============================================================================
 * Periodic φ index
 *============================================================================*/
static inline int periodic_j(int j, int Nphi) {
    return ((j % Nphi) + Nphi) % Nphi;
}

/*=============================================================================
 * Pack / Unpack (parallel memcpy is rarely a win; we keep memcpy)
 *============================================================================*/
void cpu_pack_field_omp(const EField* E, double* x, const CurlCurlOperator* op) {
    memcpy(x + op->offset_Er,   E->Er,   (size_t)op->size_Er   * sizeof(double));
    memcpy(x + op->offset_Ephi, E->Ephi, (size_t)op->size_Ephi * sizeof(double));
    memcpy(x + op->offset_Ez,   E->Ez,   (size_t)op->size_Ez   * sizeof(double));
}
void cpu_unpack_field_omp(const double* x, EField* E, const CurlCurlOperator* op) {
    memcpy(E->Er,   x + op->offset_Er,   (size_t)op->size_Er   * sizeof(double));
    memcpy(E->Ephi, x + op->offset_Ephi, (size_t)op->size_Ephi * sizeof(double));
    memcpy(E->Ez,   x + op->offset_Ez,   (size_t)op->size_Ez   * sizeof(double));
}

/*=============================================================================
 * PEC boundary — sets tangential E to zero at all conductor surfaces.
 * Identical to apply_PEC_boundary() in curlcurl_operator.cpp.
 *============================================================================*/
void cpu_apply_PEC_boundary_omp(EField* E, const GridParams* grid) {
    const int Nr = grid->Nr, Nphi = grid->Nphi, Nz = grid->Nz;

    /* E_r = 0 at z=0 and z=L */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < Nphi; j++) {
        for (int i = 0; i < Nr; i++) {
            E->Er[idx_Er(grid, i, j, 0)]  = 0.0;
            E->Er[idx_Er(grid, i, j, Nz)] = 0.0;
        }
    }

    /* E_phi = 0 at r=a, r=b */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k <= Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            E->Ephi[idx_Ephi(grid, 0,  j, k)] = 0.0;
            E->Ephi[idx_Ephi(grid, Nr, j, k)] = 0.0;
        }
    }
    /* E_phi = 0 at z=0 and z=L */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < Nphi; j++) {
        for (int i = 0; i <= Nr; i++) {
            E->Ephi[idx_Ephi(grid, i, j, 0)]  = 0.0;
            E->Ephi[idx_Ephi(grid, i, j, Nz)] = 0.0;
        }
    }

    /* E_z = 0 at r=a and r=b */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Nphi; j++) {
            E->Ez[idx_Ez(grid, 0,  j, k)] = 0.0;
            E->Ez[idx_Ez(grid, Nr, j, k)] = 0.0;
        }
    }
}

/*=============================================================================
 * curl E — three components.  Matches curl_E_r_at / curl_E_phi_at / curl_E_z_at
 * in curl_E.cpp formula-for-formula.  Output is stored into HField G (same
 * layout the GPU uses for its intermediate H_temp).
 *============================================================================*/
static inline double curl_Er_at(const EField* E, const GridParams* g, int i, int j, int k) {
    double r_i = g->a + i * g->dr;
    int jp1 = periodic_j(j + 1, g->Nphi);
    double dEz_dphi = (E->Ez[idx_Ez(g, i, jp1, k)] - E->Ez[idx_Ez(g, i, j, k)]) / g->dphi;
    double dEphi_dz = (E->Ephi[idx_Ephi(g, i, j, k + 1)] - E->Ephi[idx_Ephi(g, i, j, k)]) / g->dz;
    if (r_i < 1e-14) return -dEphi_dz;
    return dEz_dphi / r_i - dEphi_dz;
}
static inline double curl_Ephi_at(const EField* E, const GridParams* g, int i, int j, int k) {
    double dEr_dz = (E->Er[idx_Er(g, i, j, k + 1)] - E->Er[idx_Er(g, i, j, k)]) / g->dz;
    double dEz_dr = (E->Ez[idx_Ez(g, i + 1, j, k)] - E->Ez[idx_Ez(g, i, j, k)]) / g->dr;
    return dEr_dz - dEz_dr;
}
static inline double curl_Ez_at(const EField* E, const GridParams* g, int i, int j, int k) {
    double r_i   = g->a + i * g->dr;
    double r_ip1 = g->a + (i + 1) * g->dr;
    double r_iph = g->a + (i + 0.5) * g->dr;
    int jp1 = periodic_j(j + 1, g->Nphi);
    double drEphi_dr = (r_ip1 * E->Ephi[idx_Ephi(g, i + 1, j, k)]
                      - r_i   * E->Ephi[idx_Ephi(g, i,     j, k)]) / g->dr;
    double dEr_dphi = (E->Er[idx_Er(g, i, jp1, k)] - E->Er[idx_Er(g, i, j, k)]) / g->dphi;
    return drEphi_dr / r_iph - dEr_dphi / r_iph;
}

void cpu_compute_curl_E_omp(const EField* E, HField* G, const GridParams* grid) {
    /* (curl E)_r → G->Hr     (i, j+1/2, k+1/2),  i=0..Nr,  k=0..Nz-1 */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++)
                G->Hr[idx_Hr(grid, i, j, k)] = curl_Er_at(E, grid, i, j, k);
        }
    }
    /* (curl E)_phi → G->Hphi (i+1/2, j, k+1/2), i=0..Nr-1, k=0..Nz-1 */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++)
                G->Hphi[idx_Hphi(grid, i, j, k)] = curl_Ephi_at(E, grid, i, j, k);
        }
    }
    /* (curl E)_z → G->Hz    (i+1/2, j+1/2, k),  i=0..Nr-1, k=0..Nz */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k <= grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++)
                G->Hz[idx_Hz(grid, i, j, k)] = curl_Ez_at(E, grid, i, j, k);
        }
    }
}

/*=============================================================================
 * curl H — three components.  Matches curl_H_*_at in curl_H.cpp.  Output
 * stored into EField slot of the curl-curl result (same layout as input E).
 *============================================================================*/
static inline double curl_Hr_at(const HField* H, const GridParams* g, int i, int j, int k) {
    double r_iph = g->a + (i + 0.5) * g->dr;
    int jm1 = periodic_j(j - 1, g->Nphi);
    double Hz_j   = H->Hz[idx_Hz(g, i, j,   k)];
    double Hz_jm1 = H->Hz[idx_Hz(g, i, jm1, k)];
    double dHz_dphi = (Hz_j - Hz_jm1) / g->dphi;

    double Hphi_kph = 0.0, Hphi_kmh = 0.0;
    if (k < g->Nz) Hphi_kph = H->Hphi[idx_Hphi(g, i, j, k)];
    if (k > 0)     Hphi_kmh = H->Hphi[idx_Hphi(g, i, j, k - 1)];
    double dHphi_dz = (Hphi_kph - Hphi_kmh) / g->dz;
    return dHz_dphi / r_iph - dHphi_dz;
}
static inline double curl_Hphi_at(const HField* H, const GridParams* g, int i, int j, int k) {
    double Hr_kph = 0.0, Hr_kmh = 0.0;
    if (k < g->Nz) Hr_kph = H->Hr[idx_Hr(g, i, j, k)];
    if (k > 0)     Hr_kmh = H->Hr[idx_Hr(g, i, j, k - 1)];
    double dHr_dz = (Hr_kph - Hr_kmh) / g->dz;

    double Hz_iph = 0.0, Hz_imh = 0.0;
    if (i < g->Nr) Hz_iph = H->Hz[idx_Hz(g, i,     j, k)];
    if (i > 0)     Hz_imh = H->Hz[idx_Hz(g, i - 1, j, k)];
    double dHz_dr = (Hz_iph - Hz_imh) / g->dr;
    return dHr_dz - dHz_dr;
}
static inline double curl_Hz_at(const HField* H, const GridParams* g, int i, int j, int k) {
    double r_i   = g->a + i * g->dr;
    double r_iph = g->a + (i + 0.5) * g->dr;
    double r_imh = (i > 0) ? (g->a + (i - 0.5) * g->dr) : 0.0;
    int jm1 = periodic_j(j - 1, g->Nphi);

    double rHphi_iph = 0.0, rHphi_imh = 0.0;
    if (i < g->Nr) rHphi_iph = r_iph * H->Hphi[idx_Hphi(g, i,     j, k)];
    if (i > 0)     rHphi_imh = r_imh * H->Hphi[idx_Hphi(g, i - 1, j, k)];
    double drHphi_dr = (rHphi_iph - rHphi_imh) / g->dr;

    double Hr_j   = H->Hr[idx_Hr(g, i, j,   k)];
    double Hr_jm1 = H->Hr[idx_Hr(g, i, jm1, k)];
    double dHr_dphi = (Hr_j - Hr_jm1) / g->dphi;

    if (r_i < 1e-14) return drHphi_dr * 2.0;
    return drHphi_dr / r_i - dHr_dphi / r_i;
}

void cpu_compute_curl_H_omp(const HField* H, EField* curlH, const GridParams* grid) {
    /* (curl H)_r → curlH->Er   (i+1/2, j, k),  i=0..Nr-1, k=0..Nz */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k <= grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++)
                curlH->Er[idx_Er(grid, i, j, k)] = curl_Hr_at(H, grid, i, j, k);
        }
    }
    /* (curl H)_phi → curlH->Ephi (i, j+1/2, k), i=0..Nr, k=0..Nz */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k <= grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++)
                curlH->Ephi[idx_Ephi(grid, i, j, k)] = curl_Hphi_at(H, grid, i, j, k);
        }
    }
    /* (curl H)_z → curlH->Ez   (i, j, k+1/2),  i=0..Nr, k=0..Nz-1 */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < grid->Nz; k++) {
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i <= grid->Nr; i++)
                curlH->Ez[idx_Ez(grid, i, j, k)] = curl_Hz_at(H, grid, i, j, k);
        }
    }
}

/*=============================================================================
 * curl-curl: result = curl(curl(E))
 *============================================================================*/
void cpu_compute_curl_curl_E_omp(const EField* E, EField* result,
                                 HField* G_temp, const GridParams* grid) {
    cpu_compute_curl_E_omp(E, G_temp, grid);
    cpu_compute_curl_H_omp(G_temp, result, grid);
}

/*=============================================================================
 * PEC matvec y = A·x  (real)
 * Mirrors curlcurl_matvec() in curlcurl_operator.cpp and gpu_curlcurl_matvec.
 *============================================================================*/
void cpu_curlcurl_matvec_omp(const CurlCurlOperator* op,
                             const double* x, double* y) {
    CurlCurlOperator* op_mut = (CurlCurlOperator*)op;

    /* 1. Unpack */
    cpu_unpack_field_omp(x, &op_mut->E_work, op);

    /* 2. PEC boundary on input (with or without port masks) */
    if (op->has_ports)
        apply_PEC_boundary_with_ports(&op_mut->E_work, &op->grid, &op->ports);
    else
        cpu_apply_PEC_boundary_omp(&op_mut->E_work, &op->grid);

    /* 3. Curl-curl */
    cpu_compute_curl_curl_E_omp(&op_mut->E_work, &op_mut->result_work,
                                &op_mut->H_temp, &op->grid);

    /* 4. PEC boundary on result */
    if (op->has_ports)
        apply_PEC_boundary_with_ports(&op_mut->result_work, &op->grid, &op->ports);
    else
        cpu_apply_PEC_boundary_omp(&op_mut->result_work, &op->grid);

    /* 5. Pack */
    cpu_pack_field_omp(&op_mut->result_work, y, op);
}

/*=============================================================================
 * Weighted vector operations  ⟨x, y⟩_r = Σ r · x · y · dV
 *
 * Per-component r-weights:
 *   Er   at (i+1/2, j, k)     → r = a + (i+0.5)·dr
 *   Ephi at (i,     j+1/2, k) → r = a + i·dr
 *   Ez   at (i, j, k+1/2)     → r = a + i·dr
 *============================================================================*/
double cpu_vec_dot_weighted_omp(const double* x, const double* y,
                                const CurlCurlOperator* op) {
    const GridParams* g = &op->grid;
    const double dV = g->dr * g->dphi * g->dz;

    double sum_er = 0.0, sum_eph = 0.0, sum_ez = 0.0;

    /* Er block — weight r_{i+1/2} */
    #pragma omp parallel for collapse(2) reduction(+:sum_er) schedule(static)
    for (int k = 0; k <= g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i < g->Nr; i++) {
                int idx = op->offset_Er + idx_Er(g, i, j, k);
                double r = g->a + (i + 0.5) * g->dr;
                sum_er += r * x[idx] * y[idx];
            }
        }
    }
    /* Ephi block — weight r_i */
    #pragma omp parallel for collapse(2) reduction(+:sum_eph) schedule(static)
    for (int k = 0; k <= g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i <= g->Nr; i++) {
                int idx = op->offset_Ephi + idx_Ephi(g, i, j, k);
                double r = g->a + i * g->dr;
                sum_eph += r * x[idx] * y[idx];
            }
        }
    }
    /* Ez block — weight r_i */
    #pragma omp parallel for collapse(2) reduction(+:sum_ez) schedule(static)
    for (int k = 0; k < g->Nz; k++) {
        for (int j = 0; j < g->Nphi; j++) {
            for (int i = 0; i <= g->Nr; i++) {
                int idx = op->offset_Ez + idx_Ez(g, i, j, k);
                double r = g->a + i * g->dr;
                sum_ez += r * x[idx] * y[idx];
            }
        }
    }
    return (sum_er + sum_eph + sum_ez) * dV;
}

double cpu_vec_norm_weighted_omp(const double* x, const CurlCurlOperator* op) {
    return sqrt(cpu_vec_dot_weighted_omp(x, x, op));
}

void cpu_vec_normalize_weighted_omp(double* x, const CurlCurlOperator* op) {
    double nrm = cpu_vec_norm_weighted_omp(x, op);
    if (nrm > 1e-14) cpu_vec_scale_omp(x, 1.0 / nrm, op->n_total);
}

void cpu_vec_axpy_omp(double alpha, const double* x, double* y, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) y[i] += alpha * x[i];
}
void cpu_vec_scale_omp(double* x, double alpha, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) x[i] *= alpha;
}
void cpu_vec_copy_omp(const double* x, double* y, int n) {
    memcpy(y, x, (size_t)n * sizeof(double));
}
void cpu_vec_zero_omp(double* x, int n) {
    memset(x, 0, (size_t)n * sizeof(double));
}

/*=============================================================================
 * MINRES for real symmetric (A − σI) under the weighted inner product ⟨·,·⟩_r.
 *
 * This is a line-for-line mirror of gpu_minres_solve_shifted_ws in
 * cuda_eigensolver.cu (Paige–Saunders).  Note: the GPU code computes the
 * Lanczos β as a *weighted* norm of v_new, then divides v_new by that β.
 * This keeps every iterate orthonormal in the weighted inner product.
 *============================================================================*/
typedef struct {
    int    converged;
    int    iterations;
    double residual;
} CpuMinresResult;

static CpuMinresResult cpu_minres_shifted(
    const CurlCurlOperator* op, double sigma,
    const double* b, double* x,
    int max_iter, double tol)
{
    CpuMinresResult r = {0, 0, 1.0};
    int n = op->n_total;

    /* x = 0 */
    cpu_vec_zero_omp(x, n);

    double* v_old = (double*)calloc((size_t)n, sizeof(double));
    double* v_cur = (double*)calloc((size_t)n, sizeof(double));
    double* v_new = (double*)calloc((size_t)n, sizeof(double));
    double* Av    = (double*)calloc((size_t)n, sizeof(double));
    double* w_old = (double*)calloc((size_t)n, sizeof(double));
    double* w_cur = (double*)calloc((size_t)n, sizeof(double));
    double* w_new = (double*)calloc((size_t)n, sizeof(double));

    double b_norm = sqrt(cpu_vec_dot_weighted_omp(b, b, op));
    if (b_norm < 1e-14) {
        r.converged = 1; r.residual = 0.0;
        free(v_old); free(v_cur); free(v_new); free(Av);
        free(w_old); free(w_cur); free(w_new);
        return r;
    }

    /* v_cur = b / ||b||_r */
    cpu_vec_copy_omp(b, v_cur, n);
    cpu_vec_scale_omp(v_cur, 1.0 / b_norm, n);

    double beta_cur = b_norm;
    double eta      = b_norm;
    double c_old = 1.0, c_cur = 1.0;
    double s_old = 0.0, s_cur = 0.0;

    for (int it = 0; it < max_iter; it++) {
        /* Av = (A − σI) v_cur */
        cpu_curlcurl_matvec_omp(op, v_cur, Av);
        cpu_vec_axpy_omp(-sigma, v_cur, Av, n);

        double alpha = cpu_vec_dot_weighted_omp(v_cur, Av, op);

        /* v_new = Av − α v_cur − β_cur v_old */
        cpu_vec_copy_omp(Av, v_new, n);
        cpu_vec_axpy_omp(-alpha,    v_cur, v_new, n);
        cpu_vec_axpy_omp(-beta_cur, v_old, v_new, n);

        double beta_new = sqrt(cpu_vec_dot_weighted_omp(v_new, v_new, op));
        if (beta_new > 1e-14) cpu_vec_scale_omp(v_new, 1.0 / beta_new, n);

        /* Apply previous rotation, then new rotation (Paige–Saunders form) */
        double rho1     = s_old * beta_cur;
        double rho2     = c_old * c_cur * beta_cur + s_cur * alpha;
        double rho3_bar = c_cur * alpha - c_old * s_cur * beta_cur;
        double gamma    = sqrt(rho3_bar * rho3_bar + beta_new * beta_new);

        double c_new, s_new;
        if (gamma > 1e-14) { c_new = rho3_bar / gamma; s_new = beta_new / gamma; }
        else               { c_new = 1.0;              s_new = 0.0; }

        /* w_new = (v_cur − ρ2 w_cur − ρ1 w_old) / γ */
        cpu_vec_copy_omp(v_cur, w_new, n);
        cpu_vec_axpy_omp(-rho2, w_cur, w_new, n);
        cpu_vec_axpy_omp(-rho1, w_old, w_new, n);
        if (fabs(gamma) > 1e-14) cpu_vec_scale_omp(w_new, 1.0 / gamma, n);

        /* x += c_new · η · w_new */
        cpu_vec_axpy_omp(c_new * eta, w_new, x, n);
        eta = -s_new * eta;

        r.residual   = fabs(eta) / b_norm;
        r.iterations = it + 1;
        if (r.residual < tol) { r.converged = 1; break; }

        /* Rotate buffers */
        double* tmp;
        tmp = v_old; v_old = v_cur; v_cur = v_new; v_new = tmp;
        tmp = w_old; w_old = w_cur; w_cur = w_new; w_new = tmp;

        beta_cur = beta_new;
        c_old = c_cur; c_cur = c_new;
        s_old = s_cur; s_cur = s_new;
    }

    free(v_old); free(v_cur); free(v_new); free(Av);
    free(w_old); free(w_cur); free(w_new);
    return r;
}

/*=============================================================================
 * Real RQI driver — mirrors gpu_rqi_ws.
 *
 * Algorithm:
 *   1. Normalize x
 *   2. Loop:
 *      a) sigma = Rayleigh quotient ⟨x, A x⟩_r / ⟨x, x⟩_r
 *      b) Solve (A − σ I) y = x  via MINRES
 *      c) Normalize y, set x = y
 *      d) Compute residual ||A x − σ x||_r / ||x||_r
 *      e) Check convergence
 *============================================================================*/
EigenResult cpu_rqi_minres_omp(const CurlCurlOperator* op,
                               double* x,
                               double sigma_init,
                               int max_iter, double tol)
{
    EigenResult r;
    r.eigenvalue  = 0.0;
    r.eigenvector = x;
    r.iterations  = 0;
    r.residual    = 1.0;
    r.converged   = 0;

    int n = op->n_total;
    double* Ax = (double*)malloc((size_t)n * sizeof(double));
    double* y  = (double*)malloc((size_t)n * sizeof(double));
    double* r_vec = (double*)malloc((size_t)n * sizeof(double));

    cpu_vec_normalize_weighted_omp(x, op);

    /* Initial shift from the seed's Rayleigh quotient — mirrors gpu_rqi_ws.
     * Fall back to sigma_init only if the RQ is ~0 or negative. */
    cpu_curlcurl_matvec_omp(op, x, Ax);
    double xAx = cpu_vec_dot_weighted_omp(x, Ax, op);
    double xx  = cpu_vec_dot_weighted_omp(x, x, op);
    double sigma = xAx / xx;
    if (fabs(sigma) < 1e-10 || sigma < 0) sigma = sigma_init;

    printf("\n  Real RQI (MINRES inner, CPU, threads=%d):\n", cpu_omp_num_threads());
    printf("  %-5s %-18s %-14s %-12s %-8s\n",
           "Iter", "Eigenvalue", "Δλ", "Residual", "Inner");
    printf("  -----------------------------------------------------------\n");

    double lambda_old = sigma;

    for (int it = 0; it < max_iter; it++) {
        /* Inner solve (A − σI) y = x via MINRES with the CURRENT fixed shift.
         * Matches gpu_rqi_ws: maxiter=1000, tol=1e-6 every iteration. */
        CpuMinresResult ms = cpu_minres_shifted(op, sigma, x, y, 1000, 1e-6);

        /* Normalize y, take it as the new x */
        cpu_vec_normalize_weighted_omp(y, op);
        cpu_vec_copy_omp(y, x, n);

        /* Ax for Rayleigh quotient + residual */
        cpu_curlcurl_matvec_omp(op, x, Ax);
        xAx = cpu_vec_dot_weighted_omp(x, Ax, op);
        xx  = cpu_vec_dot_weighted_omp(x, x, op);
        double sigma_new = xAx / xx;

        /* Residual ||Ax − σ_new x||_r / ||x||_r */
        cpu_vec_copy_omp(Ax, r_vec, n);
        cpu_vec_axpy_omp(-sigma_new, x, r_vec, n);
        double res = cpu_vec_norm_weighted_omp(r_vec, op) /
                     cpu_vec_norm_weighted_omp(x, op);

        r.eigenvalue = sigma_new;
        r.residual   = res;
        r.iterations = it + 1;

        printf("  %-5d %-18.10f %-14.3e %-12.3e %-8d\n",
               it, sigma_new, sigma_new - lambda_old, res, ms.iterations);

        if (res < tol) { r.converged = 1; printf("  Converged!\n"); break; }

        /* Shift updates AFTER the solve, mirroring the GPU. */
        sigma = sigma_new;
        lambda_old = sigma_new;
    }

    free(Ax); free(y); free(r_vec);
    return r;
}
