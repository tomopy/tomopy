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

#include <math.h>
#include <string.h>

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

void
set_filter_tables(int dt, int pd, float center, filter_func pf,
                      const float* filter_par, complex* A,
                      unsigned char filter2d)
{
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
            A[j] *= (cosf(x) - I * sinf(x)) * norm;
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
                A[j0 + j] *= (cosf(x) - I * sinf(x)) * norm;
            }
        }
    }
}