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

#include <math.h>
#include <string.h>

#ifndef M_PI
#    define M_PI 3.14159265359f
#endif

typedef float (*const filter_func)(float, int, int, int, const float*);

// No filter
float
filter_none(float x, int i, int j, int fwidth, const float* pars)
{
    return 1.0f;
}

// Shepp-Logan filter
float
filter_shepp(float x, int i, int j, int fwidth, const float* pars)
{
    if(i == 0)
        return 0.0f;
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
    return fabsf(2 * x) * 0.5f * (1.0f + cosf(2 * M_PI * x / pars[0]));
}

// Hamming filter
float
filter_hamming(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2 * x) * (0.54f + 0.46f * cosf(2 * M_PI * x / pars[0]));
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
    return fabsf(2 * x) * powf(1.0f - fabsf(x) / pars[0], 3);
}

// Butterworth filter
float
filter_butterworth(float x, int i, int j, int fwidth, const float* pars)
{
    return fabsf(2 * x) / (1.0f + powf(x / pars[0], 2 * pars[1]));
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
