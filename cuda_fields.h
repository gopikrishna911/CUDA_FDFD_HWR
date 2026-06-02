#ifndef CUDA_FIELDS_H
#define CUDA_FIELDS_H

#include "curl_E.h"
#include "curl_H.h"

#ifdef __cplusplus
extern "C" {
#endif

    /*=============================================================================
     * GPU Field Structures
     * Same layout as CPU EField/HField but pointers are DEVICE pointers.
     *
     * IBC EXTENSION: Each struct now carries optional imaginary-part arrays.
     * When has_imag == 0, the _im pointers are NULL and behaviour is
     * identical to the original PEC code.  When has_imag == 1, every
     * component has a paired imaginary array of the same size.
     *============================================================================*/

    typedef struct {
        /* Real part (same as original) */
        double* Er;         /* Device pointer */
        double* Ephi;       /* Device pointer */
        double* Ez;         /* Device pointer */

        /* Imaginary part (IBC extension, NULL when has_imag == 0) */
        double* Er_im;
        double* Ephi_im;
        double* Ez_im;

        int size_Er;
        int size_Ephi;
        int size_Ez;

        int has_imag;       /* 0 = real-only (PEC), 1 = complex (IBC) */
    } GPU_EField;

    typedef struct {
        /* Real part */
        double* Hr;         /* Device pointer */
        double* Hphi;       /* Device pointer */
        double* Hz;         /* Device pointer */

        /* Imaginary part (IBC extension, NULL when has_imag == 0) */
        double* Hr_im;
        double* Hphi_im;
        double* Hz_im;

        int size_Hr;
        int size_Hphi;
        int size_Hz;

        int has_imag;
    } GPU_HField;

    /*=============================================================================
     * GPU Initialization & Info
     *============================================================================*/

     /* Copy grid parameters to GPU constant memory */
    int cuda_grid_init(const GridParams* grid);

    /* Print GPU device info */
    void cuda_print_device_info(void);

    /* Print current GPU memory usage */
    void cuda_print_memory_info(const char* label);

    /*=============================================================================
     * GPU E-Field Memory Management
     *
     * Original functions allocate REAL part only (has_imag = 0).
     * New _complex variants allocate BOTH real and imaginary (has_imag = 1).
     *============================================================================*/

    /* Allocate real part only (backward compatible) */
    int  gpu_efield_alloc(GPU_EField* E, const GridParams* grid);

    /* Allocate real + imaginary parts for IBC */
    int  gpu_efield_alloc_complex(GPU_EField* E, const GridParams* grid);

    void gpu_efield_free(GPU_EField* E);
    int  gpu_efield_zero(GPU_EField* E);

    /* CPU → GPU (real part only, backward compatible) */
    int gpu_efield_to_device(GPU_EField* dst, const EField* src);

    /* GPU → CPU (real part only, backward compatible) */
    int gpu_efield_to_host(EField* dst, const GPU_EField* src);

    /* CPU → GPU (imaginary part: src_im is a CPU EField holding imag data) */
    int gpu_efield_im_to_device(GPU_EField* dst, const EField* src_im);

    /* GPU → CPU (imaginary part) */
    int gpu_efield_im_to_host(EField* dst_im, const GPU_EField* src);

    /*=============================================================================
     * GPU H-Field Memory Management
     *============================================================================*/

    /* Allocate real part only */
    int  gpu_hfield_alloc(GPU_HField* H, const GridParams* grid);

    /* Allocate real + imaginary parts for IBC */
    int  gpu_hfield_alloc_complex(GPU_HField* H, const GridParams* grid);

    void gpu_hfield_free(GPU_HField* H);
    int  gpu_hfield_zero(GPU_HField* H);

    /* CPU → GPU */
    int gpu_hfield_to_device(GPU_HField* dst, const HField* src);

    /* GPU → CPU */
    int gpu_hfield_to_host(HField* dst, const GPU_HField* src);

    /*=============================================================================
     * GPU Packed Vector Management
     * (For eigensolver: single flat array containing all E components)
     *
     * PEC layout (real-only):
     *   x[0 .. size_Er-1]                           = Er
     *   x[size_Er .. size_Er+size_Ephi-1]            = Ephi
     *   x[size_Er+size_Ephi .. n-1]                  = Ez
     *   (n = n_total = size_Er + size_Ephi + size_Ez)
     *
     * IBC layout (complex):
     *   x[0       .. n-1]       = real part  (same layout as PEC)
     *   x[n       .. 2n-1]      = imaginary part (same sub-layout)
     *   Total length = 2 * n_total
     *============================================================================*/

    int  gpu_vector_alloc(double** d_vec, int n);
    void gpu_vector_free(double* d_vec);
    int  gpu_vector_zero(double* d_vec, int n);

    /* CPU → GPU */
    int gpu_vector_to_device(double* d_dst, const double* h_src, int n);

    /* GPU → CPU */
    int gpu_vector_to_host(double* h_dst, const double* d_src, int n);

    /*=============================================================================
     * GPU Pack/Unpack — Real part only (backward compatible)
     *============================================================================*/

    int gpu_pack_field(const GPU_EField* E, double* d_x,
        int offset_Er, int offset_Ephi, int offset_Ez);

    int gpu_unpack_field(const double* d_x, GPU_EField* E,
        int offset_Er, int offset_Ephi, int offset_Ez);

    /*=============================================================================
     * GPU Pack/Unpack — Complex (real + imaginary)
     *
     * Packs/unpacks both real and imaginary parts.
     * Real part goes to offsets [offset_Er, offset_Ephi, offset_Ez].
     * Imaginary part goes to [offset_Er + n_real, ...] where n_real is
     * the size of the real-only packed vector.
     *============================================================================*/

    int gpu_pack_field_complex(const GPU_EField* E, double* d_x,
        int offset_Er, int offset_Ephi, int offset_Ez,
        int n_real);

    int gpu_unpack_field_complex(const double* d_x, GPU_EField* E,
        int offset_Er, int offset_Ephi, int offset_Ez,
        int n_real);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_FIELDS_H */
