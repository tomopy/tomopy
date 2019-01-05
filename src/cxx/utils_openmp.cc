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

#include "gpu.hh"
#include <cstdint>

BEGIN_EXTERN_C
#include "utils_openmp.h"
END_EXTERN_C

//============================================================================//

void
openmp_preprocessing(int ry, int rz, int num_pixels, float center, float* mov,
                     float* gridx, float* gridy)
{
    for(int i = 0; i <= ry; ++i)
    {
        gridx[i] = -ry * 0.5f + i;
    }

    for(int i = 0; i <= rz; ++i)
    {
        gridy[i] = -rz * 0.5f + i;
    }

    *mov = ((float) num_pixels - 1) * 0.5f - center;
    if(*mov - floor(*mov) < 0.01f)
    {
        *mov += 0.01f;
    }
    *mov += 0.5;
}

//============================================================================//

int
openmp_calc_quadrant(float theta_p)
{
    // here we cast the float to an integer and rescale the integer to
    // near INT_MAX to retain the precision. This method was tested
    // on 1M random random floating points between -2*pi and 2*pi and
    // was found to produce a speed up of:
    //
    //  - 14.5x (Intel i7 MacBook)
    //  - 2.2x  (NERSC KNL)
    //  - 1.5x  (NERSC Edison)
    //  - 1.7x  (NERSC Haswell)
    //
    // with a 0.0% incorrect quadrant determination rate
    //
    const int32_t ipi_c   = 340870420;
    int32_t       theta_i = (int32_t)(theta_p * ipi_c);
    theta_i += (theta_i < 0) ? (2.0f * M_PI * ipi_c) : 0;

    return ((theta_i >= 0 && theta_i < 0.5f * M_PI * ipi_c) ||
            (theta_i >= 1.0f * M_PI * ipi_c && theta_i < 1.5f * M_PI * ipi_c))
               ? 1
               : 0;
}

//============================================================================//

void
openmp_calc_coords(int ry, int rz, float xi, float yi, float sin_p, float cos_p,
                   const float* gridx, const float* gridy, float* coordx,
                   float* coordy)
{
    float srcx, srcy, detx, dety;
    float slope, islope;
    int   n;

    srcx = xi * cos_p - yi * sin_p;
    srcy = xi * sin_p + yi * cos_p;
    detx = -xi * cos_p - yi * sin_p;
    dety = -xi * sin_p + yi * cos_p;

    slope  = (srcy - dety) / (srcx - detx);
    islope = (srcx - detx) / (srcy - dety);

#pragma omp simd
    for(n = 0; n <= ry; n++)
    {
        coordy[n] = slope * (gridx[n] - srcx) + srcy;
    }
#pragma omp simd
    for(n = 0; n <= rz; n++)
    {
        coordx[n] = islope * (gridy[n] - srcy) + srcx;
    }
}

//============================================================================//

void
openmp_trim_coords(int ry, int rz, const float* coordx, const float* coordy,
                   const float* gridx, const float* gridy, int* asize,
                   float* ax, float* ay, int* bsize, float* bx, float* by)
{
    *asize         = 0;
    *bsize         = 0;
    float gridx_gt = gridx[0] + 0.01f;
    float gridx_le = gridx[ry] - 0.01f;

    for(int n = 0; n <= rz; ++n)
    {
        if(coordx[n] >= gridx_gt && coordx[n] <= gridx_le)
        {
            ax[*asize] = coordx[n];
            ay[*asize] = gridy[n];
            ++(*asize);
        }
    }

    float gridy_gt = gridy[0] + 0.01f;
    float gridy_le = gridy[rz] - 0.01f;

    for(int n = 0; n <= ry; ++n)
    {
        if(coordy[n] >= gridy_gt && coordy[n] <= gridy_le)
        {
            bx[*bsize] = gridx[n];
            by[*bsize] = coordy[n];
            ++(*bsize);
        }
    }
}

//============================================================================//

void
openmp_sort_intersections(int ind_condition, int asize, const float* ax,
                          const float* ay, int bsize, const float* bx,
                          const float* by, int* csize, float* coorx,
                          float* coory)
{
    int i = 0, j = 0, k = 0;
    if(ind_condition == 0)
    {
        while(i < asize && j < bsize)
        {
            if(ax[asize - 1 - i] < bx[j])
            {
                coorx[k] = ax[asize - 1 - i];
                coory[k] = ay[asize - 1 - i];
                ++i;
            }
            else
            {
                coorx[k] = bx[j];
                coory[k] = by[j];
                ++j;
            }
            ++k;
        }

        while(i < asize)
        {
            coorx[k] = ax[asize - 1 - i];
            coory[k] = ay[asize - 1 - i];
            ++i;
            ++k;
        }
        while(j < bsize)
        {
            coorx[k] = bx[j];
            coory[k] = by[j];
            ++j;
            ++k;
        }

        (*csize) = asize + bsize;
    }
    else
    {
        while(i < asize && j < bsize)
        {
            if(ax[i] < bx[j])
            {
                coorx[k] = ax[i];
                coory[k] = ay[i];
                ++i;
            }
            else
            {
                coorx[k] = bx[j];
                coory[k] = by[j];
                ++j;
            }
            ++k;
        }

        while(i < asize)
        {
            coorx[k] = ax[i];
            coory[k] = ay[i];
            ++i;
            ++k;
        }
        while(j < bsize)
        {
            coorx[k] = bx[j];
            coory[k] = by[j];
            ++j;
            ++k;
        }
        (*csize) = asize + bsize;
    }
}

//============================================================================//

void
openmp_calc_dist(int ry, int rz, int csize, const float* coorx,
                 const float* coory, int* indi, float* dist)
{
    const int _size = csize - 1;

    //------------------------------------------------------------------------//
    //              calculate dist
    //------------------------------------------------------------------------//
    {
        float _diffx[_size];
        float _diffy[_size];

#pragma omp simd
        for(int n = 0; n < _size; ++n)
        {
            _diffx[n] = (coorx[n + 1] - coorx[n]) * (coorx[n + 1] - coorx[n]);
        }

#pragma omp simd
        for(int n = 0; n < _size; ++n)
        {
            _diffy[n] = (coory[n + 1] - coory[n]) * (coory[n + 1] - coory[n]);
        }

#pragma omp simd
        for(int n = 0; n < _size; ++n)
        {
            dist[n] = sqrtf(_diffx[n] + _diffy[n]);
        }
    }

    //------------------------------------------------------------------------//
    //              calculate indi
    //------------------------------------------------------------------------//

    float _midx[_size];
    float _midy[_size];
    float _x1[_size];
    float _x2[_size];
    int   _i1[_size];
    int   _i2[_size];
    int   _indx[_size];
    int   _indy[_size];

#pragma omp simd
    for(int n = 0; n < _size; ++n)
    {
        _midx[n] = 0.5f * (coorx[n + 1] + coorx[n]);
    }
#pragma omp simd
    for(int n = 0; n < _size; ++n)
    {
        _midy[n] = 0.5f * (coory[n + 1] + coory[n]);
    }
#pragma omp simd
    for(int n = 0; n < _size; ++n)
    {
        _x1[n] = _midx[n] + 0.5f * ry;
    }
#pragma omp simd
    for(int n = 0; n < _size; ++n)
    {
        _x2[n] = _midy[n] + 0.5f * rz;
    }
#pragma omp simd
    for(int n = 0; n < _size; ++n)
    {
        _i1[n] = (int) (_midx[n] + 0.5f * ry);
    }
#pragma omp simd
    for(int n = 0; n < _size; ++n)
    {
        _i2[n] = (int) (_midy[n] + 0.5f * rz);
    }
#pragma omp simd
    for(int n = 0; n < _size; ++n)
    {
        _indx[n] = _i1[n] - (_i1[n] > _x1[n]);
    }
#pragma omp simd
    for(int n = 0; n < _size; ++n)
    {
        _indy[n] = _i2[n] - (_i2[n] > _x2[n]);
    }
#pragma omp simd
    for(int n = 0; n < _size; ++n)
    {
        indi[n] = _indy[n] + (_indx[n] * rz);
    }
}

//============================================================================//

void
openmp_calc_simdata(int s, int p, int d, int ry, int rz, int dt, int dx,
                    int csize, const int* indi, const float* dist,
                    const float* model, float* simdata)
{
    int index_model = s * ry * rz;
    int index_data  = d + p * dx + s * dt * dx;
    for(int n = 0; n < csize - 1; ++n)
    {
        simdata[index_data] += model[indi[n] + index_model] * dist[n];
    }
}

//============================================================================//