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

// Possible speedups:
//   * Profile code and check adding SIMD to various functions (from OpenMP)

//#define WRITE_FILES
#define _USE_MATH_DEFINES

// Use X/Open-7, where posix_memalign is introduced
#define _XOPEN_SOURCE 700

#include "mkl.h"
#include "filters.h"
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define __LIKELY(x) __builtin_expect(!!(x), 1)
#ifdef __INTEL_COMPILER
#    define __PRAGMA_SIMD _Pragma("simd assert")
#    define __PRAGMA_SIMD_VECREMAINDER _Pragma("simd assert, vecremainder")
#    define __PRAGMA_SIMD_VECREMAINDER_VECLEN8                                           \
        _Pragma("simd assert, vecremainder, vectorlength(8)")
#    define __PRAGMA_OMP_SIMD_COLLAPSE _Pragma("omp simd collapse(2)")
#    define __PRAGMA_IVDEP _Pragma("ivdep")
#    define __ASSSUME_64BYTES_ALIGNED(x) __assume_aligned((x), 64)
#else
#    define __PRAGMA_SIMD
#    define __PRAGMA_SIMD_VECREMAINDER
#    define __PRAGMA_SIMD_VECREMAINDER_VECLEN8
#    define __PRAGMA_OMP_SIMD_COLLAPSE
#    define __PRAGMA_IVDEP
#    define __ASSSUME_64BYTES_ALIGNED(x)
#endif

inline float*
malloc_vector_f(size_t n)
{
    return (float*) malloc(n * sizeof(float));
}

void
set_pswf_tables(float C, int nt, float lambda, const float* coefs, int ltbl, int linv,
                float* wtbl, float* winv)
{
    // Set up lookup tables for convolvent (used in Phase 1 of
    // do_recon()), and for the final correction factor (used in
    // Phase 3).

    int         i;
    float       norm;
    const float fac   = (float) ltbl / (linv + 0.5);
    const float polyz = legendre(nt, coefs, 0.);

    wtbl[0] = 1.0;
    for(i = 1; i <= ltbl; i++)
    {
        wtbl[i] = legendre(nt, coefs, (float) i / ltbl) / polyz;
    }

    // Note the final result at end of Phase 3 contains the factor,
    // norm^2.  This incorporates the normalization of the 2D
    // inverse FFT in Phase 2 as well as scale factors involved
    // in the inverse Fourier transform of the convolvent.
    norm = sqrt(M_PI / 2 / C / lambda) / 1.2;

    winv[linv] = norm / wtbl[0];
    __PRAGMA_IVDEP
    for(i = 1; i <= linv; i++)
    {
        // Minus sign for alternate entries
        // corrects for "natural" data layout
        // in array H at end of Phase 1.
        norm           = -norm;
        winv[linv + i] = winv[linv - i] = norm / wtbl[(int) roundf(i * fac)];
    }
}

float
legendre(int n, const float* coefs, float x)
{
    // Compute SUM(coefs(k)*P(2*k,x), for k=0,n/2)
    // where P(j,x) is the jth Legendre polynomial.
    // x must be between -1 and 1.
    float penult, last, cur, y, mxlast;

    y      = coefs[0];
    penult = 1.0;
    last   = x;
    for(int j = 2; j <= n; j++)
    {
        mxlast = -(x * last);
        cur    = -(2 * mxlast + penult) + (penult + mxlast) / j;
        // cur = (x*(2*j-1)*last-(j-1)*penult)/j;
        if(!(j & 1))  // if j is even
        {
            y += cur * coefs[j >> 1];
        }

        penult = last;
        last   = cur;
    }
    return y;
}

void
set_trig_tables(int dt, const float* theta, float** sine, float** cose)
{
    // Set up tables of sines and cosines.
    float *s, *c;

    *sine = s = malloc_vector_f(dt);
    __ASSSUME_64BYTES_ALIGNED(s);
    *cose = c = malloc_vector_f(dt);
    __ASSSUME_64BYTES_ALIGNED(c);

    __PRAGMA_SIMD
    for(int j = 0; j < dt; j++)
    {
        s[j] = sinf(theta[j]);
        c[j] = cosf(theta[j]);
    }
}

// No filter
float
filter_none(float x, int i, int j, int fwidth, const float* pars)
{
    return 1.0;
}

// Shepp-Logan filter
float
filter_shepp(float x, int i, int j, int fwidth, const float* pars)
{
    if(i == 0)
        return 0.0;
    return fabsf(2 * x) * (sinf(M_PI * x) / (M_PI * x));
}

// Cosine filter
float
filter_cosine(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2 * x) * (cosf(M_PI * x));
}

// Hann filter
float
filter_hann(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2 * x) * 0.5 * (1. + cosf(2 * M_PI * x / pars[0]));
}

// Hamming filter
float
filter_hamming(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2 * x) * (0.54 + 0.46 * cosf(2 * M_PI * x / pars[0]));
}

// Ramlak filter
float
filter_ramlak(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2 * x);
}

// Parzen filter
float
filter_parzen(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2 * x) * pow(1 - fabs(x) / pars[0], 3);
}

// Butterworth filter
float
filter_butterworth(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2 * x) / (1 + pow(x / pars[0], 2 * pars[1]));
}

// Custom filter
float
filter_custom(float x, int i, int j, int fwidth, const float* pars)
{
    return pars[i];
}

// Custom 2D filter
float
filter_custom2d(float x, int i, int j, int fwidth, const float* pars)
{
    return pars[j * fwidth + i];
}

float (*get_filter(const char* name))(float, int, int, int, const float*)
{
    struct
    {
        const char* name;
        float (*const fp)(float, int, int, int, const float*);
    } fltbl[] = { { "none", filter_none },       { "shepp", filter_shepp },  // Default
                  { "cosine", filter_cosine },   { "hann", filter_hann },
                  { "hamming", filter_hamming }, { "ramlak", filter_ramlak },
                  { "parzen", filter_parzen },   { "butterworth", filter_butterworth },
                  { "custom", filter_custom },   { "custom2d", filter_custom2d } };

    for(int i = 0; i < 10; i++)
    {
        if(!strncmp(name, fltbl[i].name, 16))
        {
            return fltbl[i].fp;
        }
    }
    return fltbl[1].fp;
}

unsigned char
filter_is_2d(const char* name)
{
    if(!strncmp(name, "custom2d", 16))
        return 1;
    return 0;
}