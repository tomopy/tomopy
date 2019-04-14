// Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.

// Copyright 2015. UChicago Argonne, LLC. This software was produced
// under U.S. Government contract DE-AC02-06CH11357 for Argonne National
// Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
// U.S. Department of Energy. The U.S. Government has rights to use,
// reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
// UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
// ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
// modified to produce derivative works, such modified software should
// be clearly marked, so as not to confuse it with the version available
// from ANL.

// Additionally, redistribution and use in source and binary forms, with
// or without modification, are permitted provided that the following
// conditions are met:

//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.

//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in
//       the documentation and/or other materials provided with the
//       distribution.

//     * Neither the name of UChicago Argonne, LLC, Argonne National
//       Laboratory, ANL, the U.S. Government, nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
// Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//  ---------------------------------------------------------------
//   TOMOPY implementation

// Possible speedups:
//   * Profile code and check adding SIMD to various functions (from OpenMP)

//#define WRITE_FILES
#define _USE_MATH_DEFINES

// Use X/Open-7, where posix_memalign is introduced
#define _XOPEN_SOURCE 700

#include "gridrec.hh"
#include "mkl.h"
#include <complex>

#if defined(TOMOPY_CXX_GRIDREC)
using namespace std::literals::complex_literals;
#endif

#if defined(_MSC_VER)
#    if defined(__LIKELY)
#        undef __LIKELY
#    endif
#    define __LIKELY(EXPR) EXPR
#endif

//===========================================================================//

void
cxx_gridrec(const float* data, int dy, int dt, int dx, const float* center,
            const float* theta, float* recon, int ngridx, int ngridy, const char* fname,
            const float* filter_par)
{
#if defined(TOMOPY_CXX_GRIDREC)
    int    s, p, iu, iv;
    int    j;
    float *sine, *cose, *wtbl, *winv;

    float *work, *work2;

    filter_func filter = get_filter(fname);

    const float          C      = 7.0;
    const float          nt     = 20.0;
    const float          lambda = 0.99998546;
    const unsigned int   L      = (int) (2 * C / M_PI);
    const int            ltbl   = 512;
    int                  pdim;
    std::complex<float>* sino, *filphase, *filphase_iter = NULL, **H;
    std::complex<float>**U_d, **V_d;
    float *              J_z, *P_z;

    const float coefs[11] = { 0.5767616E+02,  -0.8931343E+02, 0.4167596E+02,
                              -0.1053599E+02, 0.1662374E+01,  -0.1780527E-00,
                              0.1372983E-01,  -0.7963169E-03, 0.3593372E-04,
                              -0.1295941E-05, 0.3817796E-07 };

    // Compute pdim = next power of 2 >= dx
    for(pdim = 16; pdim < dx; pdim *= 2)
        ;

    const int pdim2 = pdim >> 1;
    const int M02   = pdim2 - 1;
    const int M2    = pdim2;

    unsigned char filter2d = filter_is_2d(fname);

    // Allocate storage for various arrays.
    sino = cxx_malloc_vector_c(pdim);
    if(!filter2d)
    {
        filphase      = cxx_malloc_vector_c(pdim2);
        filphase_iter = filphase;
    }
    else
    {
        filphase = cxx_malloc_vector_c(dt * (pdim2));
    }
    __ASSSUME_64BYTES_ALIGNED(filphase);
    H = cxx_malloc_matrix_c(pdim, pdim);
    __ASSSUME_64BYTES_ALIGNED(H);
    wtbl = cxx_malloc_vector_f(ltbl + 1);
    __ASSSUME_64BYTES_ALIGNED(wtbl);
    winv = cxx_malloc_vector_f(pdim - 1);
    __ASSSUME_64BYTES_ALIGNED(winv);
    J_z = cxx_malloc_vector_f(pdim2 * dt);
    __ASSSUME_64BYTES_ALIGNED(J_z);
    P_z = cxx_malloc_vector_f(pdim2 * dt);
    __ASSSUME_64BYTES_ALIGNED(P_z);
    U_d = cxx_malloc_matrix_c(dt, pdim);
    __ASSSUME_64BYTES_ALIGNED(U_d);
    V_d = cxx_malloc_matrix_c(dt, pdim);
    __ASSSUME_64BYTES_ALIGNED(V_d);
    work = cxx_malloc_vector_f(L + 1);
    __ASSSUME_64BYTES_ALIGNED(work);
    work2 = cxx_malloc_vector_f(L + 1);
    __ASSSUME_64BYTES_ALIGNED(work2);

    // Set up table of sines and cosines.
    set_trig_tables(dt, theta, &sine, &cose);
    __ASSSUME_64BYTES_ALIGNED(sine);
    __ASSSUME_64BYTES_ALIGNED(cose);

    // Set up PSWF lookup tables.
    set_pswf_tables(C, nt, lambda, coefs, ltbl, M02, wtbl, winv);

    DFTI_DESCRIPTOR_HANDLE reverse_1d;
    MKL_LONG               length_1d = (MKL_LONG) pdim;
    DftiCreateDescriptor(&reverse_1d, DFTI_SINGLE, DFTI_COMPLEX, 1, length_1d);
    DftiSetValue(reverse_1d, DFTI_THREAD_LIMIT,
                 1); /* FFT should run sequentially to avoid oversubscription */
    DftiCommitDescriptor(reverse_1d);
    DFTI_DESCRIPTOR_HANDLE forward_2d;
    MKL_LONG               length_2d[2] = { (MKL_LONG) pdim, (MKL_LONG) pdim };
    DftiCreateDescriptor(&forward_2d, DFTI_SINGLE, DFTI_COMPLEX, 2, length_2d);
    DftiSetValue(forward_2d, DFTI_THREAD_LIMIT,
                 1); /* FFT should run sequentially to avoid oversubscription */
    DftiCommitDescriptor(forward_2d);

    for(p = 0; p < dt; p++)
    {
        for(j = 1; j < pdim2; j++)
        {
            U_d[p][j] = j * cose[p] + M2;
            V_d[p][j] = j * sine[p] + M2;
        }
    }

    float       U, V;
    const float L2      = (int) (C / M_PI);
    const float tblspcg = 2 * ltbl / L;
    int         iul, iuh, ivl, ivh;
    int         k, k2;

    // For each slice.
    for(s = 0; s < dy; s += 2)
    {
        // Set up table of combined filter-phase factors.
        cxx_set_filter_tables(dt, pdim, center[s], filter, filter_par, filphase,
                              filter2d);

        // First clear the array H
        memset(H[0], 0, pdim * pdim * sizeof(H[0][0]));
        // for(int i = 0; i < pdim * pdim; ++i)
        //    *(H[i]) = std::complex<float>();

        // Loop over the dt projection angles. For each angle, do the following:

        //     1. Copy the real projection data from the two slices into the
        //      real and imaginary parts of the first dx elements of the
        //      complex array, sino[].  Set the remaining pdim-dx elements
        //      to zero (zero-padding).

        //     2. Carry out a (1D) Fourier transform on the complex data.
        //      This results in transform data that is arranged in
        //      "wrap-around" order, with non-negative spatial frequencies
        //      occupying the first half, and negative frequencies the second
        //      half, of the array, sino[].

        //     3. Multiply each element of the 1-D transform by a complex,
        //      frequency dependent factor, filphase[].  These factors were
        //      precomputed as part of recofour1((float*)sino-1,pdim,1);n_init()
        //      and combine the tomographic filtering with a phase factor which
        //      shifts the origin in configuration space to the projection of
        //      the rotation axis as defined by the parameter, "center".  If a
        //      region of interest (ROI) centered on a different origin has
        //      been specified [(X0,Y0)!=(0,0)], multiplication by an
        //      additional phase factor, dependent on angle as well as
        //      frequency, is required.

        //     4. For each data element, find the Cartesian coordinates,
        //      <U,V>, of the corresponding point in the 2D frequency plane,
        //      in  units of the spacing in the MxM rectangular grid placed
        //      thereon; then calculate the upper and lower limits in each
        //      coordinate direction of the integer coordinates for the
        //      grid points contained in an LxL box centered on <U,V>.
        //      Using a precomputed table of the (1-D) convolving function,
        //      W, calculate the contribution of this data element to the
        //      (2-D) convolvent (the 2_D convolvent is the product of
        //      1_D convolvents in the X and Y directions) at each of these
        //      grid points, and update the complex 2D array H accordingly.

        // At the end of Phase 1, the array H[][] contains data arranged in
        // "natural", rather than wrap-around order -- that is, the origin in
        // the spatial frequency plane is situated in the middle, rather than
        // at the beginning, of the array, H[][].  This simplifies the code
        // for carrying out the convolution (step 4 above), but necessitates
        // an additional correction -- See Phase 3 below.

        std::complex<float> Cdata1, Cdata2;

        // For each projection
        for(p = 0; p < dt; p++)
        {
            float              sine_p = sine[p], cose_p = cose[p];
            const unsigned int j0 = dx * (p + s * dt), delta_index = dx * dt;

            __PRAGMA_SIMD_VECREMAINDER
            for(j = 0; j < dx; j++)
            {
                // Add data from both slices
                float              second_sino = 0.0;
                const unsigned int index       = j + j0;
                if(__LIKELY((s + 1) < dy))
                {
                    second_sino = data[index + delta_index];
                }
                sino[j] = data[index] + 1if * second_sino;
            }

            __PRAGMA_SIMD_VECREMAINDER
            for(j = dx; j < pdim; j++)
            {
                // Zero fill the rest of the array
                sino[j] = 0.0;
            }

            DftiComputeBackward(reverse_1d, sino);

            if(filter2d)
                filphase_iter = filphase + pdim2 * p;

            // For each FFT(projection)
            for(j = 1; j < pdim2; j++)
            {
                Cdata1 = filphase_iter[j] * sino[j];
                Cdata2 = std::conj<float>(filphase_iter[j]) * sino[pdim - j];

                U = j * cose_p + M2;
                V = j * sine_p + M2;

                // Note freq space origin is at (M2,M2), but we
                // offset the indices U, V, etc. to range from 0 to M-1.
                iul = ceilf(U - L2);
                iuh = floorf(U + L2);
                ivl = ceilf(V - L2);
                ivh = floorf(V + L2);
                if(iul < 1)
                    iul = 1;
                if(iuh >= pdim)
                    iuh = pdim - 1;
                if(ivl < 1)
                    ivl = 1;
                if(ivh >= pdim)
                    ivh = pdim - 1;

                // Note aliasing value (at index=0) is forced to zero.
                __PRAGMA_SIMD_VECREMAINDER_VECLEN8
                for(iv = ivl, k = 0; iv <= ivh; iv++, k++)
                {
                    work[k] = wtbl[(int) roundf(fabsf(V - iv) * tblspcg)];
                }

                __PRAGMA_SIMD_VECREMAINDER_VECLEN8
                for(iu = iul, k = 0; iu <= iuh; iu++, k++)
                {
                    work2[k] = wtbl[(int) roundf(fabsf(U - iu) * tblspcg)];
                }

                __PRAGMA_OMP_SIMD_COLLAPSE
                for(iu = iul, k2 = 0; iu <= iuh; iu++, k2++)
                {
                    for(iv = ivl, k = 0; iv <= ivh; iv++, k++)
                    {
                        const float rtmp    = work2[k2];
                        const float convolv = rtmp * work[k];
                        H[iu][iv] += convolv * Cdata1;
                        H[pdim - iu][pdim - iv] += convolv * Cdata2;
                    }
                }
            }
        }

        // Carry out a 2D inverse FFT on the array H.

        // At the conclusion of this phase, the configuration
        // space data is arranged in wrap-around order with the origin
        // (center of reconstructed images) situated at the start of the
        // array.  The first (resp. second) half of the array contains the
        // lower, Y<0 (resp, upper Y>0) part of the image, and within each row
        // of the array, the first (resp. second) half contains data for the
        // right [X>0] (resp. left [X<0]) half of the image.

        DftiComputeForward(forward_2d, H[0]);

        // Copy the real and imaginary parts of the complex data from H[][],
        // into the output buffers for the two reconstructed real images,
        // simultaneously carrying out a final multiplicative correction.
        // The correction factors are taken from the array, winv[], previously
        // computed in set_pswf_tables(), and consist logically of three parts,
        // namely:

        //  1. A positive real factor, corresponding to the reciprocal
        //     of the inverse Fourier transform, of the convolving
        //     function, W, and

        //  2. Multiplication by the cell size, (1/D1)^2, in 2D frequency
        //     space.  This correctly normalizes the 2D inverse FFT carried
        //     out in Phase 2.  (Note that all quantities are expressed in
        //     units in which the detector spacing is one.)

        //  3. A sign change for the "odd-numbered" elements (in a
        //     checkerboard pattern) of the array.  This compensates
        //     for the fact that the 2-D Fourier transform (Phase 2)
        //     started with a frequency array in which the zero frequency
        //     point appears in the middle of the array instead of at
        //     its start.

        // Only the elements in the square M0xM0 subarray of H[][], centered
        // about the origin, are utilized.  The other elements are not part of
        // the actual region being reconstructed and are discarded.  Because of
        // the wrap-around ordering, the subarray must actually be taken from
        // the four corners" of the 2D array, H[][] -- See Phase 2 description,
        // above.

        // The final data corresponds physically to the linear X-ray absorption
        // coefficient expressed in units of the inverse detector spacing -- to
        // convert to inverse cm (say), one must divide the data by the detector
        // spacing in cm.

        int       ustart, vstart, ufin, vfin;
        const int padx    = (pdim - ngridx) / 2;
        const int pady    = (pdim - ngridy) / 2;
        const int offsetx = M02 + 1 - padx;
        const int offsety = M02 + 1 - pady;
        const int islc1   = s * ngridx * ngridy;        // index slice 1
        const int islc2   = (s + 1) * ngridx * ngridy;  // index slice 2

        ustart = pdim - offsety;
        ufin   = pdim;
        j      = 0;
        while(j < ngridy)
        {
            for(iu = ustart; iu < ufin; j++, iu++)
            {
                const float corrn_u = winv[j + pady];
                vstart              = pdim - offsetx;
                vfin                = pdim;
                k                   = 0;
                while(k < ngridx)
                {
                    __PRAGMA_SIMD
                    for(iv = vstart; iv < vfin; k++, iv++)
                    {
                        const float corrn = corrn_u * winv[k + padx];
                        recon[islc1 + ngridy * (ngridx - 1 - k) + j] =
                            corrn * std::real<float>(H[iu][iv]);
                        if(__LIKELY((s + 1) < dy))
                        {
                            recon[islc2 + ngridy * (ngridx - 1 - k) + j] =
                                corrn * std::imag<float>(H[iu][iv]);
                        }
                    }
                    if(k < ngridx)
                    {
                        vstart = 0;
                        vfin   = ngridx - offsetx;
                    }
                }
            }
            if(j < ngridy)
            {
                ustart = 0;
                ufin   = ngridy - offsety;
            }
        }
    }

    cxx_free_vector_f(sine);
    cxx_free_vector_f(cose);
    cxx_free_vector_c(sino);
    cxx_free_vector_f(wtbl);
    cxx_free_vector_c(filphase);
    cxx_free_vector_f(winv);
    cxx_free_vector_f(work);
    cxx_free_matrix_c(H);
    cxx_free_vector_f(J_z);
    cxx_free_vector_f(P_z);
    cxx_free_matrix_c(U_d);
    cxx_free_matrix_c(V_d);
    DftiFreeDescriptor(&reverse_1d);
    DftiFreeDescriptor(&forward_2d);
    return;
#else
    throw std::runtime_error("Error! TOMOPY_CXX_GRIDREC was disabled at compile time.");
#endif
}

//===========================================================================//

void
cxx_set_filter_tables(int dt, int pd, float center, filter_func pf,
                      const float* filter_par, std::complex<float>* A,
                      unsigned char filter2d)
{
#if defined(TOMOPY_CXX_GRIDREC)
    // Set up the complex array, filphase[], each element of which
    // consists of a real filter factor [obtained from the function,
    // pf(...)], multiplying a complex phase factor (derived from the
    // parameter, center}.  See Phase 1 comments.

    const float norm  = M_PI / pd / dt;
    const float rtmp1 = 2 * M_PI * center / pd;
    int         j, i;
    int         pd2 = pd / 2;
    float       x;

    if(!filter2d)
    {
        for(j = 0; j < pd2; j++)
        {
            A[j] = pf((float) j / pd, j, 0, pd2, filter_par);
        }

        __PRAGMA_SIMD
        for(j = 0; j < pd2; j++)
        {
            x = j * rtmp1;
            A[j] *= (cosf(x) - 1if * sinf(x)) * norm;
        }
    }
    else
    {
        for(i = 0; i < dt; i++)
        {
            int j0 = i * pd2;

            for(j = 0; j < pd2; j++)
            {
                A[j0 + j] = pf((float) j / pd, j, i, pd2, filter_par);
            }

            __PRAGMA_SIMD
            for(j = 0; j < pd2; j++)
            {
                x = j * rtmp1;
                A[j0 + j] *= (cosf(x) - 1if * sinf(x)) * norm;
            }
        }
    }
#else
    throw std::runtime_error("Error! TOMOPY_CXX_GRIDREC was disabled at compile time.");
#endif
}

//===========================================================================//

float*
cxx_malloc_vector_f(size_t n)
{
    return new float[n];
}

//===========================================================================//

void
cxx_free_vector_f(float*& v)
{
    delete[] v;
    v = nullptr;
}

//===========================================================================//

std::complex<float>*
cxx_malloc_vector_c(size_t n)
{
    return new std::complex<float>[n];
}

//===========================================================================//

void
cxx_free_vector_c(std::complex<float>*& v)
{
    delete[] v;
    v = nullptr;
}

//===========================================================================//
void*
cxx_malloc_64bytes_aligned(size_t sz)
{
#if defined(__MINGW32__)
    return __mingw_aligned_malloc(sz, 64);
#elif defined(_MSC_VER)
    void* r = _aligned_malloc(sz, 64);
    return r;
#else
    void* r   = NULL;
    int   err = posix_memalign(&r, 64, sz);
    return (err) ? NULL : r;
#endif
}

//===========================================================================//

std::complex<float>**
cxx_malloc_matrix_c(size_t nr, size_t nc)
{
    std::complex<float>** m = nullptr;
    size_t                i;

    // Allocate pointers to rows,
    m = (std::complex<float>**) cxx_malloc_64bytes_aligned(nr *
                                                           sizeof(std::complex<float>*));

    /* Allocate rows and set the pointers to them */
    m[0] = cxx_malloc_vector_c(nr * nc);

    for(i = 1; i < nr; i++)
    {
        m[i] = m[i - 1] + nc;
    }
    return m;
}

//===========================================================================//

void
cxx_free_matrix_c(std::complex<float>**& m)
{
    cxx_free_vector_c(m[0]);
#if defined(__MINGW32__)
    __mingw_aligned_free(m);
#elif defined(_MSC_VER)
    _aligned_free(m);
#else
    free(m);
#endif
    m = nullptr;
}

//===========================================================================//
