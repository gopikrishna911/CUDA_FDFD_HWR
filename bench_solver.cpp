/*=============================================================================
 * bench_solver.cpp
 *
 * Wall-clock benchmark of the rhodotron eigensolver on the CPU paths
 * provided in cpu_reference.* and cpu_complex.* :
 *
 *   1) Real PEC RQI (MINRES inner)        ← cpu_rqi_minres_omp
 *   2) Complex IBC RQI (GMRES inner)      ← cpu_rqi_complex_omp
 *
 * For every grid size we time:
 *   - matvec only (1000 applications)
 *   - one full eigensolve
 *
 * Default mode runs with the CPU code compiled twice — once without OpenMP
 * (single-threaded reference) and once with OpenMP.  This single binary
 * supports both: pass --threads=1 for the single-threaded baseline, or
 * --threads=N for OpenMP with N threads.
 *
 * Optional GPU comparison: if compiled with -DENABLE_GPU and linked against
 * the project's CUDA objects, the same problem runs through the GPU
 * eigensolver and its time is reported alongside.
 *
 * Output: human-readable table + machine-readable CSV (--csv=bench.csv)
 *
 * Usage:
 *   ./bench_solver [--threads=N] [--csv=bench.csv] [--matvec-iters=1000]
 *                  [--max-rqi=20] [--rqi-tol=1e-6] [--mode=pec|ibc|both]
 *                  [--sizes=16,32,48,64,96]
 *
 *============================================================================*/

#include "cpu_reference.h"
#include "curl_E.h"
#include "curl_H.h"
#include "curlcurl_operator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <string>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------------ */
/*  Wall-clock helper (chrono, monotonic, ns precision)                     */
/* ------------------------------------------------------------------------ */
static double now_seconds() {
    using clk = std::chrono::steady_clock;
    static const auto t0 = clk::now();
    return std::chrono::duration<double>(clk::now() - t0).count();
}

/* ------------------------------------------------------------------------ */
/*  Default cavity parameters — typical Rhodotron HWR proportions           */
/* ------------------------------------------------------------------------ */
struct Cavity {
    /* Production Rhodotron HWR geometry (from test_conformal_ibc_full.cpp).
     * Half-wave coax fundamental: f = c/(2L) = 299.79e6/(2*1.395) = 107.5 MHz. */
    double a = 0.3333; /* inner radius [m]  */
    double b = 1.0;    /* outer radius [m]  */
    double L = 1.395;  /* length      [m]   */
};

/* ------------------------------------------------------------------------ */
/*  Grid sizes to sweep                                                     */
/* ------------------------------------------------------------------------ */
struct BenchSize {
    int Nr, Nphi, Nz;
};

/* ------------------------------------------------------------------------ */
/*  CLI parsing                                                             */
/* ------------------------------------------------------------------------ */
struct CliArgs {
    int    threads      = -1;     /* -1 = leave OMP default */
    int    matvec_iters = 200;
    int    max_rqi      = 8;
    double rqi_tol      = 1e-6;
    std::string csv_path = "";
    std::string mode     = "both"; /* pec | ibc | both */
    std::vector<BenchSize> sizes  = {{16,16,16},{24,24,24},{32,32,32},{48,48,48}};
    int    gmres_restart = 30;
    double conductivity  = 5.8e7;  /* copper */
    std::string seed_mode = "tem"; /* tem | random | mixed */
};

static int parse_int_after_eq(const char* s) { return atoi(strchr(s, '=') + 1); }
static double parse_double_after_eq(const char* s) { return atof(strchr(s, '=') + 1); }

static std::vector<BenchSize> parse_sizes(const std::string& s) {
    std::vector<BenchSize> out;
    std::string cur;
    for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == ',') {
            if (!cur.empty()) {
                int N = atoi(cur.c_str());
                if (N > 0) out.push_back({N, N, N});
                cur.clear();
            }
        } else cur += s[i];
    }
    return out;
}

static CliArgs parse_args(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; i++) {
        const char* s = argv[i];
        if      (!strncmp(s, "--threads=",      10)) a.threads       = parse_int_after_eq(s);
        else if (!strncmp(s, "--matvec-iters=", 15)) a.matvec_iters  = parse_int_after_eq(s);
        else if (!strncmp(s, "--max-rqi=",      10)) a.max_rqi       = parse_int_after_eq(s);
        else if (!strncmp(s, "--rqi-tol=",      10)) a.rqi_tol       = parse_double_after_eq(s);
        else if (!strncmp(s, "--csv=",           6)) a.csv_path      = strchr(s, '=') + 1;
        else if (!strncmp(s, "--mode=",          7)) a.mode          = strchr(s, '=') + 1;
        else if (!strncmp(s, "--sizes=",         8)) a.sizes         = parse_sizes(strchr(s, '=') + 1);
        else if (!strncmp(s, "--gmres-restart=", 16)) a.gmres_restart = parse_int_after_eq(s);
        else if (!strncmp(s, "--conductivity=",  15)) a.conductivity  = parse_double_after_eq(s);
        else if (!strncmp(s, "--seed-mode=",      12)) a.seed_mode     = strchr(s, '=') + 1;
        else if (!strcmp(s, "-h") || !strcmp(s, "--help")) {
            printf("Usage: %s [options]\n"
                   "  --threads=N           Set OpenMP threads (default: auto)\n"
                   "  --matvec-iters=K      Matvec timing iterations (default: 200)\n"
                   "  --max-rqi=K           Outer RQI iterations (default: 8)\n"
                   "  --rqi-tol=X           RQI tolerance (default: 1e-6)\n"
                   "  --mode=pec|ibc|both   Which solver to benchmark (default: both)\n"
                   "  --sizes=16,32,48,64   Grid sizes to sweep (Nr=Nphi=Nz=N)\n"
                   "  --gmres-restart=M     GMRES restart parameter (default: 30)\n"
                   "  --conductivity=σ      Wall σ [S/m] (default: 5.8e7 Cu)\n"
                   "  --seed-mode=MODE      Eigensolver initial guess (default: tem)\n"
                   "                          tem    = physical TEM mode (converges in 1-2 iters;\n"
                   "                                   use for matvec timing / correctness)\n"
                   "                          random = random start (forces many RQI/Krylov\n"
                   "                                   iters; use to benchmark the inner solver)\n"
                   "                          mixed  = TEM + perturbation (a few iters; realistic)\n"
                   "  --csv=path            Append CSV results to path\n",
                   argv[0]);
            exit(0);
        }
    }
    return a;
}

/* ------------------------------------------------------------------------ */
/*  Random fill — used ONLY for matvec throughput timing (vector content     */
/*  is irrelevant to timing). NOT used to seed the eigensolver.              */
/* ------------------------------------------------------------------------ */
static void fill_random_real(double* x, const CurlCurlOperator* op,
                             unsigned seed) {
    int n = op->n_total;
    srand(seed);
    for (int i = 0; i < n; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
    cpu_vec_normalize_weighted_omp(x, op);
}
static void fill_random_complex(double* x, const CurlCurlOperator* op,
                                unsigned seed) {
    int n_real = op->n_total;
    srand(seed);
    for (int i = 0; i < n_real; i++)            x[i] = (double)rand() / RAND_MAX - 0.5;
    for (int i = 0; i < n_real; i++) x[i + n_real] = 0.0;
    cpu_vec_normalize_weighted_complex_omp(x, op);
}

/* ------------------------------------------------------------------------ */
/*  Physical TEM initial guess: Er = sin(pi z / L) / r  on the Er nodes.     */
/*  This is the half-wave coaxial TEM mode shape and mirrors exactly the     */
/*  seed used in test_conformal_ibc_full.cpp (z0_offset = 0 here, since the  */
/*  bare benchmark grid has no endcap z-extension). Its Rayleigh quotient    */
/*  starts near the fundamental, so RQI converges to the physical mode       */
/*  instead of chasing the spectral bulk.                                    */
/* ------------------------------------------------------------------------ */
static void make_tem_guess_real(double* x, const CurlCurlOperator* op,
                                const GridParams* grid, const Cavity& c) {
    int n = op->n_total;
    for (int i = 0; i < n; i++) x[i] = 0.0;
    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;                 /* z0_offset = 0 */
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                double r = grid->a + (i + 0.5) * grid->dr;
                if (r <= c.b && z >= 0.0 && z <= c.L) {
                    int idx = op->offset_Er + idx_Er(grid, i, j, k);
                    x[idx] = sin(M_PI * z / c.L) / r;
                }
            }
        }
    }
    cpu_vec_normalize_weighted_omp(x, op);
}
static void make_tem_guess_complex(double* x, const CurlCurlOperator* op,
                                   const GridParams* grid, const Cavity& c) {
    int n_real = op->n_total;
    /* real part = TEM shape, imag part = 0 */
    make_tem_guess_real(x, op, grid, c);              /* fills [0..n_real) + normalizes */
    for (int i = 0; i < n_real; i++) x[i + n_real] = 0.0;
    cpu_vec_normalize_weighted_complex_omp(x, op);
}

/* ------------------------------------------------------------------------ */
/*  Seed dispatchers — choose the eigensolver initial guess by mode.        */
/*    tem    : physical TEM mode  -> 1-2 RQI iters (timing matvec / verify)  */
/*    random : random start       -> many iters    (stress the inner solver) */
/*    mixed  : TEM + 20% random    -> a few iters    (realistic warm start)   */
/* ------------------------------------------------------------------------ */
static void seed_real(double* x, const CurlCurlOperator* op,
                      const GridParams* grid, const Cavity& c,
                      const std::string& mode) {
    if (mode == "random") {
        fill_random_real(x, op, 42);
    } else if (mode == "mixed") {
        int n = op->n_total;
        make_tem_guess_real(x, op, grid, c);          /* TEM into x, normalized */
        srand(123);
        for (int i = 0; i < n; i++)
            x[i] += 0.2 * ((double)rand() / RAND_MAX - 0.5);
        cpu_vec_normalize_weighted_omp(x, op);
    } else { /* tem (default) */
        make_tem_guess_real(x, op, grid, c);
    }
}
static void seed_complex(double* x, const CurlCurlOperator* op,
                         const GridParams* grid, const Cavity& c,
                         const std::string& mode) {
    int n_real = op->n_total;
    if (mode == "random") {
        fill_random_complex(x, op, 42);
    } else if (mode == "mixed") {
        make_tem_guess_real(x, op, grid, c);           /* fills real half */
        srand(123);
        for (int i = 0; i < n_real; i++)
            x[i] += 0.2 * ((double)rand() / RAND_MAX - 0.5);
        for (int i = 0; i < n_real; i++) x[i + n_real] = 0.0;
        cpu_vec_normalize_weighted_complex_omp(x, op);
    } else { /* tem (default) */
        make_tem_guess_complex(x, op, grid, c);
    }
}

/* ------------------------------------------------------------------------ */
/*  Analytical TEM coaxial frequency (used as RQI starting shift)           */
/* ------------------------------------------------------------------------ */
static double tem_coax_k2(const Cavity& c) {
    /* k_z = π/L for the half-wave TEM mode in a shorted coax */
    double kz = M_PI / c.L;
    return kz * kz;
}

/* ------------------------------------------------------------------------ */
/*  One benchmark row: PEC real path                                        */
/* ------------------------------------------------------------------------ */
struct BenchRow {
    int    Nr, Nphi, Nz;
    int    n_total;
    int    threads;
    double matvec_time_s;
    double matvec_per_iter_us;
    double eigensolve_time_s;
    int    rqi_iters;
    double k2_converged;
    double frequency_MHz;
    int    converged;
};

static BenchRow run_pec(const Cavity& c, BenchSize sz, const CliArgs& args) {
    BenchRow row;
    row.Nr = sz.Nr; row.Nphi = sz.Nphi; row.Nz = sz.Nz;
    row.threads = cpu_omp_num_threads();
    row.matvec_time_s = 0.0;
    row.matvec_per_iter_us = 0.0;
    row.eigensolve_time_s = 0.0;
    row.rqi_iters = 0;
    row.k2_converged = 0.0;
    row.frequency_MHz = 0.0;
    row.converged = 0;

    GridParams grid;
    grid_init(&grid, c.a, c.b, c.L, sz.Nr, sz.Nphi, sz.Nz);
    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);
    row.n_total = op.n_total;

    /* -- matvec timing (vector content irrelevant) -- */
    std::vector<double> x(op.n_total), y(op.n_total);
    fill_random_real(x.data(), &op, 42);

    /* Warmup */
    cpu_curlcurl_matvec_omp(&op, x.data(), y.data());

    double t0 = now_seconds();
    for (int i = 0; i < args.matvec_iters; i++) {
        cpu_curlcurl_matvec_omp(&op, x.data(), y.data());
    }
    double t1 = now_seconds();
    row.matvec_time_s     = t1 - t0;
    row.matvec_per_iter_us = (t1 - t0) * 1.0e6 / args.matvec_iters;

    /* -- one full eigensolve (seed per --seed-mode) -- */
    seed_real(x.data(), &op, &grid, c, args.seed_mode);
    double sigma0 = tem_coax_k2(c);

    t0 = now_seconds();
    EigenResult er = cpu_rqi_minres_omp(&op, x.data(), sigma0,
                                        args.max_rqi, args.rqi_tol);
    t1 = now_seconds();
    row.eigensolve_time_s = t1 - t0;
    row.rqi_iters         = er.iterations;
    row.k2_converged      = er.eigenvalue;
    row.frequency_MHz     = (er.eigenvalue > 0)
                            ? 299.792458 * sqrt(er.eigenvalue) / (2.0 * M_PI)
                            : 0.0;
    row.converged         = er.converged;

    curlcurl_op_free(&op);
    return row;
}

/* ------------------------------------------------------------------------ */
/*  Benchmark row: IBC complex path                                         */
/* ------------------------------------------------------------------------ */
struct BenchRowIBC : BenchRow {
    double Q_factor;
    double k2_im;
};

static BenchRowIBC run_ibc(const Cavity& c, BenchSize sz, const CliArgs& args) {
    BenchRowIBC row;
    row.Nr = sz.Nr; row.Nphi = sz.Nphi; row.Nz = sz.Nz;
    row.threads = cpu_omp_num_threads();
    row.matvec_time_s = 0.0;
    row.matvec_per_iter_us = 0.0;
    row.eigensolve_time_s = 0.0;
    row.rqi_iters = 0;
    row.k2_converged = 0.0;
    row.frequency_MHz = 0.0;
    row.converged = 0;
    row.Q_factor = 0.0;
    row.k2_im = 0.0;

    GridParams grid;
    grid_init(&grid, c.a, c.b, c.L, sz.Nr, sz.Nphi, sz.Nz);
    CurlCurlOperator op;
    curlcurl_op_init(&op, &grid);
    row.n_total = op.n_total;

    CpuComplexOperator cop;
    cpu_complex_op_init(&cop, &op);

    int n_real = op.n_total;
    std::vector<double> x(2 * n_real), y(2 * n_real);
    fill_random_complex(x.data(), &op, 42);

    double sigma0 = tem_coax_k2(c);
    double alpha  = 1.0 / sqrt(2.0 * args.conductivity
                               * (299792458.0 * sqrt(sigma0))
                               * (4.0e-7 * M_PI));

    /* Warmup */
    cpu_curlcurl_matvec_complex_omp(&cop, x.data(), y.data(), alpha);

    double t0 = now_seconds();
    for (int i = 0; i < args.matvec_iters; i++) {
        cpu_curlcurl_matvec_complex_omp(&cop, x.data(), y.data(), alpha);
    }
    double t1 = now_seconds();
    row.matvec_time_s      = t1 - t0;
    row.matvec_per_iter_us = (t1 - t0) * 1.0e6 / args.matvec_iters;

    /* Full complex eigensolve (seed per --seed-mode) */
    seed_complex(x.data(), &op, &grid, c, args.seed_mode);
    CpuComplexEigenWorkspace ws;
    cpu_complex_eigensolver_workspace_init(&ws, n_real, args.gmres_restart);

    t0 = now_seconds();
    CpuComplexEigenResult er = cpu_rqi_complex_omp(
        &cop, x.data(), sigma0, args.conductivity,
        args.max_rqi, args.rqi_tol, args.gmres_restart, &ws);
    t1 = now_seconds();

    row.eigensolve_time_s = t1 - t0;
    row.rqi_iters         = er.iterations;
    row.k2_converged      = er.k2_re;
    row.k2_im             = er.k2_im;
    row.frequency_MHz     = er.frequency_Hz / 1.0e6;
    row.Q_factor          = er.Q_factor;
    row.converged         = er.converged;

    cpu_complex_eigensolver_workspace_free(&ws);
    cpu_complex_op_free(&cop);
    curlcurl_op_free(&op);
    return row;
}

/* ------------------------------------------------------------------------ */
/*  Pretty-print + CSV emit                                                 */
/* ------------------------------------------------------------------------ */
static void print_header_pec() {
    printf("\n=== PEC (real) results ===\n");
    printf("%-8s %-10s %-7s %-12s %-13s %-12s %-5s %-12s %-9s\n",
           "Grid", "DOFs", "Thr",
           "Matvec [µs]", "Eigensolve[s]", "k²", "Iter", "f [MHz]", "Conv");
    printf("---------------------------------------------------------------"
           "-------------------\n");
}
static void print_row_pec(const BenchRow& r) {
    char g[16]; snprintf(g, sizeof(g), "%dx%dx%d", r.Nr, r.Nphi, r.Nz);
    printf("%-8s %-10d %-7d %-12.3f %-13.3f %-12.6f %-5d %-12.4f %-9s\n",
           g, r.n_total, r.threads,
           r.matvec_per_iter_us, r.eigensolve_time_s,
           r.k2_converged, r.rqi_iters,
           r.frequency_MHz, r.converged ? "yes" : "no");
}
static void print_header_ibc() {
    printf("\n=== IBC (complex) results ===\n");
    printf("%-8s %-10s %-7s %-12s %-13s %-12s %-5s %-10s %-10s %-9s\n",
           "Grid", "DOFs", "Thr",
           "Matvec [µs]", "Eigensolve[s]", "Re(k²)", "Iter", "f [MHz]", "Q", "Conv");
    printf("---------------------------------------------------------------"
           "-------------------------------\n");
}
static void print_row_ibc(const BenchRowIBC& r) {
    char g[16]; snprintf(g, sizeof(g), "%dx%dx%d", r.Nr, r.Nphi, r.Nz);
    printf("%-8s %-10d %-7d %-12.3f %-13.3f %-12.6f %-5d %-10.4f %-10.1f %-9s\n",
           g, r.n_total, r.threads,
           r.matvec_per_iter_us, r.eigensolve_time_s,
           r.k2_converged, r.rqi_iters,
           r.frequency_MHz, r.Q_factor, r.converged ? "yes" : "no");
}

static void csv_append_pec(FILE* fp, const BenchRow& r) {
    fprintf(fp, "PEC,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%d,%.10f,%.6f,%d\n",
            r.Nr, r.Nphi, r.Nz, r.n_total, r.threads,
            r.matvec_time_s, r.matvec_per_iter_us,
            r.eigensolve_time_s, r.rqi_iters,
            r.k2_converged, r.frequency_MHz, r.converged);
}
static void csv_append_ibc(FILE* fp, const BenchRowIBC& r) {
    fprintf(fp, "IBC,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%d,%.10f,%.6e,%.6f,%.3f,%d\n",
            r.Nr, r.Nphi, r.Nz, r.n_total, r.threads,
            r.matvec_time_s, r.matvec_per_iter_us,
            r.eigensolve_time_s, r.rqi_iters,
            r.k2_converged, r.k2_im, r.frequency_MHz, r.Q_factor, r.converged);
}

/* ------------------------------------------------------------------------ */
/*  Main                                                                    */
/* ------------------------------------------------------------------------ */
int main(int argc, char** argv) {
    CliArgs args = parse_args(argc, argv);
    if (args.threads > 0) cpu_omp_set_num_threads(args.threads);

    Cavity cav;

    printf("======================================================\n");
    printf("  Rhodotron eigensolver CPU benchmark\n");
    printf("  Cavity:  a=%.3f  b=%.3f  L=%.3f  [m]\n",
           cav.a, cav.b, cav.L);
    printf("  Threads (OMP max): %d\n", cpu_omp_num_threads());
    printf("  Mode: %s\n", args.mode.c_str());
    printf("  Seed: %s\n", args.seed_mode.c_str());
    printf("  Grid sizes: ");
    for (auto& s : args.sizes) { printf("%d ", s.Nr); }
    printf("\n");
    printf("======================================================\n");

    FILE* csv = nullptr;
    if (!args.csv_path.empty()) {
        csv = fopen(args.csv_path.c_str(), "a");
        if (csv) {
            fseek(csv, 0, SEEK_END);
            if (ftell(csv) == 0) {
                fprintf(csv, "mode,Nr,Nphi,Nz,n_total,threads,"
                            "matvec_total_s,matvec_per_us,eigensolve_s,"
                            "rqi_iters,k2_re,k2_im_or_freqMHz,freqMHz,Q,converged\n");
            }
        }
    }

    std::vector<BenchRow>    pec_rows;
    std::vector<BenchRowIBC> ibc_rows;

    if (args.mode == "pec" || args.mode == "both") {
        for (auto sz : args.sizes) {
            auto row = run_pec(cav, sz, args);
            pec_rows.push_back(row);
            if (csv) csv_append_pec(csv, row);
        }
        print_header_pec();
        for (auto& r : pec_rows) print_row_pec(r);
    }

    if (args.mode == "ibc" || args.mode == "both") {
        for (auto sz : args.sizes) {
            auto row = run_ibc(cav, sz, args);
            ibc_rows.push_back(row);
            if (csv) csv_append_ibc(csv, row);
        }
        print_header_ibc();
        for (auto& r : ibc_rows) print_row_ibc(r);
    }

    if (csv) fclose(csv);
    return 0;
}
