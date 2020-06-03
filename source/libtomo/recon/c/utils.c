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

#include "utils.h"
#include <float.h>
#include <stdint.h>

// for windows build
#ifdef WIN32
#    ifdef PY3K
void
PyInit_libtomopy(void)
{
}
#    else
void
initlibtomopy(void)
{
}
#    endif
#endif

//======================================================================================//

void
preprocessing(int ry, int rz, int num_pixels, float center, float* mov, float* gridx,
              float* gridy)
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

//======================================================================================//

int
calc_quadrant(float theta_p)
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

//======================================================================================//

void
calc_coords(int ry, int rz, float xi, float yi, float sin_p, float cos_p,
            const float* gridx, const float* gridy, float* coordx, float* coordy)
{
    float srcx   = xi * cos_p - yi * sin_p;
    float srcy   = xi * sin_p + yi * cos_p;
    float detx   = -xi * cos_p - yi * sin_p;
    float dety   = -xi * sin_p + yi * cos_p;
    float slope  = (srcy - dety) / (srcx - detx);
    float islope = (srcx - detx) / (srcy - dety);

#pragma omp simd
    for(int n = 0; n <= rz; ++n)
    {
        coordx[n] = islope * (gridy[n] - srcy) + srcx;
    }
#pragma omp simd
    for(int n = 0; n <= ry; ++n)
    {
        coordy[n] = slope * (gridx[n] - srcx) + srcy;
    }
}

//======================================================================================//

void
trim_coords(int ry, int rz, const float* coordx, const float* coordy, const float* gridx,
            const float* gridy, int* asize, float* ax, float* ay, int* bsize, float* bx,
            float* by)
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

//======================================================================================//

void
sort_intersections(int ind_condition, int asize, const float* ax, const float* ay,
                   int bsize, const float* bx, const float* by, int* csize, float* coorx,
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

//======================================================================================//

void
calc_dist(int ry, int rz, int csize, const float* coorx, const float* coory, int* indi,
          float* dist)
{
    if(csize < 2)
        return;
    const int _size = csize - 1;

    //------------------------------------------------------------------------//
    //              calculate dist
    //------------------------------------------------------------------------//
    {
        float* _diffx = malloc(_size * sizeof(float));
        float* _diffy = malloc(_size * sizeof(float));

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

        free(_diffx);
        free(_diffy);
    }

    //------------------------------------------------------------------------//
    //              calculate indi
    //------------------------------------------------------------------------//

    int* _indx = malloc(_size * sizeof(int));
    int* _indy = malloc(_size * sizeof(int));

#pragma omp simd
    for(int n = 0; n < _size; ++n)
    {
        float _midx = 0.5f * (coorx[n + 1] + coorx[n]);
        float _x1   = _midx + 0.5f * ry;
        float _i1   = (int) (_midx + 0.5f * ry);
        _indx[n]    = _i1 - (_i1 > _x1);
    }
#pragma omp simd
    for(int n = 0; n < _size; ++n)
    {
        float _midy = 0.5f * (coory[n + 1] + coory[n]);
        float _x2   = _midy + 0.5f * rz;
        float _i2   = (int) (_midy + 0.5f * rz);
        _indy[n]    = _i2 - (_i2 > _x2);
    }
#pragma omp simd
    for(int n = 0; n < _size; ++n)
    {
        indi[n] = _indy[n] + (_indx[n] * rz);
    }

    free(_indx);
    free(_indy);
}

//======================================================================================//

void
calc_dist2(int ry, int rz, int csize, const float* coorx, const float* coory, int* indx,
           int* indy, float* dist)
{
#pragma omp simd
    for(int n = 0; n < csize - 1; ++n)
    {
        float diffx = coorx[n + 1] - coorx[n];
        float diffy = coory[n + 1] - coory[n];
        dist[n]     = sqrt(diffx * diffx + diffy * diffy);
    }

#pragma omp simd
    for(int n = 0; n < csize - 1; ++n)
    {
        float midx = (coorx[n + 1] + coorx[n]) * 0.5f;
        float midy = (coory[n + 1] + coory[n]) * 0.5f;
        float x1   = midx + ry * 0.5f;
        float x2   = midy + rz * 0.5f;
        int   i1   = (int) (midx + ry * 0.5f);
        int   i2   = (int) (midy + rz * 0.5f);
        indx[n]    = i1 - (i1 > x1);
        indy[n]    = i2 - (i2 > x2);
    }
}

//======================================================================================//

void
calc_simdata(int s, int p, int d, int ry, int rz, int dt, int dx, int csize,
             const int* indi, const float* dist, const float* model, float* simdata)
{
    int index_model = s * ry * rz;
    int index_data  = d + p * dx + s * dt * dx;
    for(int n = 0; n < csize - 1; ++n)
    {
        simdata[index_data] += model[indi[n] + index_model] * dist[n];
    }
}

//======================================================================================//

void
calc_simdata2(int s, int p, int d, int ry, int rz, int dt, int dx, int csize,
              const int* indx, const int* indy, const float* dist, float vx, float vy,
              const float* modelx, const float* modely, float* simdata)
{
    int n;

    for(n = 0; n < csize - 1; n++)
    {
        simdata[d + p * dx + s * dt * dx] +=
            (modelx[indy[n] + indx[n] * rz + s * ry * rz] * vx +
             modely[indy[n] + indx[n] * rz + s * ry * rz] * vy) *
            dist[n];
    }
}

//======================================================================================//

void
calc_simdata3(int s, int p, int d, int ry, int rz, int dt, int dx, int csize,
              const int* indx, const int* indy, const float* dist, float vx, float vy,
              const float* modelx, const float* modely, const float* modelz, int axis,
              float* simdata)
{
    int n;

    if(axis == 0)
    {
        for(n = 0; n < csize - 1; n++)
        {
            simdata[d + p * dx + s * dt * dx] +=
                (modelx[indy[n] + indx[n] * rz + s * ry * rz] * vx +
                 modely[indy[n] + indx[n] * rz + s * ry * rz] * vy) *
                dist[n];
        }
    }
    else if(axis == 1)
    {
        for(n = 0; n < csize - 1; n++)
        {
            simdata[d + p * dx + s * dt * dx] +=
                (modely[s + indx[n] * rz + indy[n] * ry * rz] * vx +
                 modelz[s + indx[n] * rz + indy[n] * ry * rz] * vy) *
                dist[n];
        }
    }
    else if(axis == 2)
    {
        for(n = 0; n < csize - 1; n++)
        {
            simdata[d + p * dx + s * dt * dx] +=
                (modelx[indx[n] + s * rz + indy[n] * ry * rz] * vx +
                 modelz[indx[n] + s * rz + indy[n] * ry * rz] * vy) *
                dist[n];
        }
    }
}

//======================================================================================//
