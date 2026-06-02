/*============================================================================
 * bench_gpu_xcheck.cu
 *
 * GPU cross-check harness for the CPU reference benchmark.
 *
 * This mirrors run_pec() / run_ibc() from bench_solver.cpp EXACTLY:
 *   - identical geometry (a=0.3333, b=1.0, L=1.395)
 *   - identical physical TEM initial guess  Er = sin(pi z / L) / r
 *   - identical bare-coax PEC / perturbative-IBC operator
 *   - identical RQI / complex-RQI algorithm
 * ...but calls the GPU functions (gpu_curlcurl_matvec, gpu_rqi_ws,
 * gpu_curlcurl_matvec_complex, gpu_rqi_complex) instead of the CPU ones.
 *
 * Purpose: confirm the CPU port and the GPU agree on converged k² to
 * ~1e-10 (limited only by parallel-reduction order). Any larger gap
 * means a real discrepancy between the two operators.
 *
 * Build (add to your Makefile or compile directly):
 *   nvcc -O3 -arch=sm_86 --use_fast_math -I$(CUDA_PATH)/include \
 *        bench_gpu_xcheck.cu curl_E.o curl_H.o curlcurl_operator.o \
 *        cuda_fields.o cuda_vector_ops.o cuda_curls.o cuda_operator.o \
 *        cuda_eigensolver.o -o bench_gpu_xcheck -lcudart -lm
 *
 * Run (use the SAME flags as bench_serial):
 *   ./bench_gpu_xcheck --sizes=16,24,32,48 --mode=both --csv=gpu_xcheck.csv
 *==========================================================================*/

#include "curl_E.h"
#include "curl_H.h"
#include "curlcurl_operator.h"
#include "cuda_operator.h"
#include "cuda_eigensolver.h"
#include "cuda_fields.h"
#include "cuda_vector_ops.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <chrono>

#define CUDA_CHECK(call) do {                                            \
    cudaError_t _e = (call);                                             \
    if (_e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(_e));                                 \
        exit(1);                                                         \
    }                                                                    \
} while (0)

/* ------------------------------------------------------------------------ */
/*  Geometry — MUST match bench_solver.cpp                                  */
/* ------------------------------------------------------------------------ */
struct Cavity {
    double a = 0.3333;
    double b = 1.0;
    double L = 1.395;
};

struct BenchSize { int Nr, Nphi, Nz; };

struct CliArgs {
    int    matvec_iters = 200;
    int    max_rqi      = 8;
    double rqi_tol      = 1e-6;
    std::string csv_path = "";
    std::string mode     = "both";
    std::vector<BenchSize> sizes = {{16,16,16},{24,24,24},{32,32,32},{48,48,48}};
    int    gmres_restart = 30;
    double conductivity  = 5.8e7;
};

static int    parse_int_after_eq(const char* s)    { return atoi(strchr(s, '=') + 1); }
static double parse_double_after_eq(const char* s) { return atof(strchr(s, '=') + 1); }

static std::vector<BenchSize> parse_sizes(const std::string& s) {
    std::vector<BenchSize> out; std::string cur;
    for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == ',') {
            if (!cur.empty()) { int N = atoi(cur.c_str());
                if (N > 0) out.push_back({N,N,N}); cur.clear(); }
        } else cur += s[i];
    }
    return out;
}

static CliArgs parse_args(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; i++) {
        const char* s = argv[i];
        if      (!strncmp(s,"--matvec-iters=",15)) a.matvec_iters = parse_int_after_eq(s);
        else if (!strncmp(s,"--max-rqi=",10))      a.max_rqi      = parse_int_after_eq(s);
        else if (!strncmp(s,"--rqi-tol=",10))      a.rqi_tol      = parse_double_after_eq(s);
        else if (!strncmp(s,"--csv=",6))           a.csv_path     = strchr(s,'=')+1;
        else if (!strncmp(s,"--mode=",7))          a.mode         = strchr(s,'=')+1;
        else if (!strncmp(s,"--sizes=",8))         a.sizes        = parse_sizes(strchr(s,'=')+1);
        else if (!strncmp(s,"--gmres-restart=",16))a.gmres_restart= parse_int_after_eq(s);
        else if (!strncmp(s,"--conductivity=",15)) a.conductivity = parse_double_after_eq(s);
        else if (!strcmp(s,"-h")||!strcmp(s,"--help")) {
            printf("Usage: %s [--sizes=..] [--mode=pec|ibc|both] [--matvec-iters=K]\n"
                   "          [--max-rqi=K] [--rqi-tol=X] [--gmres-restart=M]\n"
                   "          [--conductivity=s] [--csv=path]\n", argv[0]);
            exit(0);
        }
    }
    return a;
}

/* ------------------------------------------------------------------------ */
/*  Physical TEM initial guess — identical to bench_solver.cpp              */
/* ------------------------------------------------------------------------ */
static void make_tem_guess_real(double* h_x, const CurlCurlOperator* op,
                                const GridParams* grid, const Cavity& c) {
    int n = op->n_total;
    for (int i = 0; i < n; i++) h_x[i] = 0.0;
    for (int k = 0; k <= grid->Nz; k++) {
        double z = k * grid->dz;
        for (int j = 0; j < grid->Nphi; j++) {
            for (int i = 0; i < grid->Nr; i++) {
                double r = grid->a + (i + 0.5) * grid->dr;
                if (r <= c.b && z >= 0.0 && z <= c.L) {
                    int idx = op->offset_Er + idx_Er(grid, i, j, k);
                    h_x[idx] = sin(M_PI * z / c.L) / r;
                }
            }
        }
    }
    /* NOTE: gpu_rqi_ws normalizes internally; no host-side normalize needed. */
}

static double tem_coax_k2(const Cavity& c) {
    double kz = M_PI / c.L;
    return kz * kz;
}

static double now_seconds() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
        steady_clock::now().time_since_epoch()).count();
}

/* ------------------------------------------------------------------------ */
/*  Result rows (same columns as the CPU harness)                           */
/* ------------------------------------------------------------------------ */
struct Row {
    int Nr, Nphi, Nz, n_total;
    double matvec_us, eigensolve_s, k2_re, k2_im, freq_MHz, Q;
    int iters, converged;
};

/* ------------------------------------------------------------------------ */
/*  PEC (real) GPU run                                                      */
/* ------------------------------------------------------------------------ */
static Row run_pec_gpu(const Cavity& c, BenchSize sz, const CliArgs& args) {
    Row row; memset(&row, 0, sizeof(row));
    row.Nr = sz.Nr; row.Nphi = sz.Nphi; row.Nz = sz.Nz;

    GridParams grid;
    grid_init(&grid, c.a, c.b, c.L, sz.Nr, sz.Nphi, sz.Nz);
    CurlCurlOperator cpu_op;
    curlcurl_op_init(&cpu_op, &grid);
    cuda_grid_init(&grid);
    int n = cpu_op.n_total;
    row.n_total = n;

    GPU_Operator gpu_op;
    if (gpu_operator_init(&gpu_op, &cpu_op) != 0) {
        fprintf(stderr, "gpu_operator_init failed\n"); exit(1);
    }

    double* d_x = nullptr; double* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)n * sizeof(double)));

    std::vector<double> h_x(n);
    make_tem_guess_real(h_x.data(), &cpu_op, &grid, c);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)n*sizeof(double),
                          cudaMemcpyHostToDevice));

    /* -- matvec timing (GPU kernel time via events) -- */
    gpu_curlcurl_matvec(&gpu_op, d_x, d_y);     /* warmup */
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < args.matvec_iters; i++)
        gpu_curlcurl_matvec(&gpu_op, d_x, d_y);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    row.matvec_us = (double)ms * 1.0e3 / args.matvec_iters;  /* ms -> us */

    /* -- one full eigensolve (re-seed TEM, RQI normalizes internally) -- */
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)n*sizeof(double),
                          cudaMemcpyHostToDevice));
    EigensolverWorkspace ws;
    eigensolver_workspace_init(&ws, n);

    double sigma0 = tem_coax_k2(c);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = now_seconds();
    GPU_EigenResult er = gpu_rqi_ws(&gpu_op, d_x, sigma0,
                                    args.max_rqi, args.rqi_tol, &ws);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();

    row.eigensolve_s = t1 - t0;
    row.iters        = er.iterations;
    row.k2_re        = er.eigenvalue;
    row.k2_im        = 0.0;
    row.freq_MHz     = (er.eigenvalue > 0)
                       ? 299.792458 * sqrt(er.eigenvalue) / (2.0 * M_PI) : 0.0;
    row.Q            = 0.0;
    row.converged    = er.converged;

    eigensolver_workspace_free(&ws);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_x); cudaFree(d_y);
    gpu_operator_free(&gpu_op);
    curlcurl_op_free(&cpu_op);
    return row;
}

/* ------------------------------------------------------------------------ */
/*  IBC (complex) GPU run                                                   */
/* ------------------------------------------------------------------------ */
static Row run_ibc_gpu(const Cavity& c, BenchSize sz, const CliArgs& args) {
    Row row; memset(&row, 0, sizeof(row));
    row.Nr = sz.Nr; row.Nphi = sz.Nphi; row.Nz = sz.Nz;

    GridParams grid;
    grid_init(&grid, c.a, c.b, c.L, sz.Nr, sz.Nphi, sz.Nz);
    CurlCurlOperator cpu_op;
    curlcurl_op_init(&cpu_op, &grid);
    cuda_grid_init(&grid);
    int n = cpu_op.n_total;
    row.n_total = n;

    GPU_Operator gpu_op;
    if (gpu_operator_init_complex(&gpu_op, &cpu_op) != 0) {
        fprintf(stderr, "gpu_operator_init_complex failed\n"); exit(1);
    }

    double* d_x = nullptr; double* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)2*n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)2*n * sizeof(double)));

    /* TEM seed in real half, zero imag half */
    std::vector<double> h_x(2*n, 0.0);
    make_tem_guess_real(h_x.data(), &cpu_op, &grid, c);   /* fills [0..n) */
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)2*n*sizeof(double),
                          cudaMemcpyHostToDevice));

    double sigma0 = tem_coax_k2(c);
    double alpha  = 1.0 / sqrt(2.0 * args.conductivity
                               * (299792458.0 * sqrt(sigma0))
                               * (4.0e-7 * M_PI));

    /* -- matvec timing -- */
    gpu_curlcurl_matvec_complex(&gpu_op, d_x, d_y, alpha);  /* warmup */
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < args.matvec_iters; i++)
        gpu_curlcurl_matvec_complex(&gpu_op, d_x, d_y, alpha);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    row.matvec_us = (double)ms * 1.0e3 / args.matvec_iters;

    /* -- full complex eigensolve (re-seed) -- */
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)2*n*sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = now_seconds();
    GPU_ComplexEigenResult er = gpu_rqi_complex(
        &gpu_op, d_x, sigma0, args.conductivity,
        args.max_rqi, args.rqi_tol, args.gmres_restart);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();

    row.eigensolve_s = t1 - t0;
    row.iters        = er.iterations;
    row.k2_re        = er.k2_re;
    row.k2_im        = er.k2_im;
    row.freq_MHz     = er.frequency_Hz / 1e6;
    row.Q            = er.Q_factor;
    row.converged    = er.converged;

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_x); cudaFree(d_y);
    gpu_operator_free(&gpu_op);
    curlcurl_op_free(&cpu_op);
    return row;
}

/* ------------------------------------------------------------------------ */
int main(int argc, char** argv) {
    CliArgs args = parse_args(argc, argv);
    Cavity cav;

    int dev = 0; cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        printf("  GPU: %s\n", prop.name);

    printf("======================================================\n");
    printf("  Rhodotron eigensolver GPU cross-check\n");
    printf("  Cavity:  a=%.3f  b=%.3f  L=%.3f  [m]\n", cav.a, cav.b, cav.L);
    printf("  Mode: %s\n", args.mode.c_str());
    printf("  Grid sizes: ");
    for (auto& s : args.sizes) { printf("%d ", s.Nr); }
    printf("\n======================================================\n");

    FILE* csv = nullptr;
    if (!args.csv_path.empty()) {
        csv = fopen(args.csv_path.c_str(), "a");
        if (csv) fprintf(csv, "mode,Nr,Nphi,Nz,n_total,device,"
                              "matvec_per_us,eigensolve_s,rqi_iters,"
                              "k2_re,k2_im,freqMHz,Q,converged\n");
    }

    bool do_pec = (args.mode == "pec" || args.mode == "both");
    bool do_ibc = (args.mode == "ibc" || args.mode == "both");

    if (do_pec) {
        printf("\n=== PEC (real) GPU results ===\n");
        printf("%-8s %-10s %-12s %-13s %-12s %-5s %-12s %-8s\n",
               "Grid","DOFs","Matvec[us]","Eigensolve[s]","k2","Iter",
               "f[MHz]","Conv");
        printf("--------------------------------------------------"
               "------------------------------\n");
        for (auto& s : args.sizes) {
            Row r = run_pec_gpu(cav, s, args);
            printf("%dx%dx%d %-10d %-12.3f %-13.4f %-12.6f %-5d %-12.4f %-8s\n",
                   r.Nr,r.Nphi,r.Nz,r.n_total,r.matvec_us,r.eigensolve_s,
                   r.k2_re,r.iters,r.freq_MHz,r.converged?"yes":"no");
            if (csv) fprintf(csv,"PEC,%d,%d,%d,%d,%s,%.6e,%.6e,%d,"
                                 "%.13f,%.6e,%.6f,%.1f,%d\n",
                             r.Nr,r.Nphi,r.Nz,r.n_total,prop.name,
                             r.matvec_us,r.eigensolve_s,r.iters,
                             r.k2_re,r.k2_im,r.freq_MHz,r.Q,r.converged);
        }
    }

    if (do_ibc) {
        printf("\n=== IBC (complex) GPU results ===\n");
        printf("%-8s %-10s %-12s %-13s %-12s %-5s %-10s %-10s %-8s\n",
               "Grid","DOFs","Matvec[us]","Eigensolve[s]","Re(k2)","Iter",
               "f[MHz]","Q","Conv");
        printf("--------------------------------------------------"
               "--------------------------------------------\n");
        for (auto& s : args.sizes) {
            Row r = run_ibc_gpu(cav, s, args);
            printf("%dx%dx%d %-10d %-12.3f %-13.4f %-12.6f %-5d %-10.4f %-10.1f %-8s\n",
                   r.Nr,r.Nphi,r.Nz,r.n_total,r.matvec_us,r.eigensolve_s,
                   r.k2_re,r.iters,r.freq_MHz,r.Q,r.converged?"yes":"no");
            if (csv) fprintf(csv,"IBC,%d,%d,%d,%d,%s,%.6e,%.6e,%d,"
                                 "%.13f,%.6e,%.6f,%.1f,%d\n",
                             r.Nr,r.Nphi,r.Nz,r.n_total,prop.name,
                             r.matvec_us,r.eigensolve_s,r.iters,
                             r.k2_re,r.k2_im,r.freq_MHz,r.Q,r.converged);
        }
    }

    if (csv) fclose(csv);
    return 0;
}
