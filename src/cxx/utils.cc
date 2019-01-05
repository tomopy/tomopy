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

#include "utils.hh"

//============================================================================//

// unsigned ncount = 0;
// unsigned modulo = 1;

//============================================================================//

void
cxx_preprocessing(int ry, int rz, int num_pixels, float center, float& mov,
                  farray_t& gridx, farray_t& gridy)
{
    for(int i = 0; i <= ry; ++i)
    {
        gridx[i] = -ry * 0.5f + i;
    }

    for(int i = 0; i <= rz; ++i)
    {
        gridy[i] = -rz * 0.5f + i;
    }

    mov = ((float) num_pixels - 1) * 0.5f - center;
    if(mov - floor(mov) < 0.01f)
    {
        mov += 0.01f;
    }
    mov += 0.5;
}

//============================================================================//

void
cxx_calc_coords(int ry, int rz, float xi, float yi, float sin_p, float cos_p,
                const farray_t& gridx, const farray_t& gridy, farray_t& coordx,
                farray_t& coordy)
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
cxx_trim_coords(int ry, int rz, const farray_t& coordx, const farray_t& coordy,
                const farray_t& gridx, const farray_t& gridy, farray_t& ax,
                farray_t& ay, farray_t& bx, farray_t& by)
{
    int   asize    = 0;
    int   bsize    = 0;
    float gridx_gt = gridx[0] + 0.01f;
    float gridx_le = gridx[ry] - 0.01f;

    for(int n = 0; n <= rz; ++n)
    {
        if(coordx[n] >= gridx_gt && coordx[n] <= gridx_le)
        {
            ax[asize] = coordx[n];
            ay[asize] = gridy[n];
            ++(asize);
        }
    }
    ax.resize(asize, 0.0f);
    ay.resize(asize, 0.0f);

    float gridy_gt = gridy[0] + 0.01f;
    float gridy_le = gridy[rz] - 0.01f;

    for(int n = 0; n <= ry; ++n)
    {
        if(coordy[n] >= gridy_gt && coordy[n] <= gridy_le)
        {
            bx[bsize] = gridx[n];
            by[bsize] = coordy[n];
            ++(bsize);
        }
    }
    bx.resize(bsize, 0.0f);
    by.resize(bsize, 0.0f);
}

//============================================================================//

void
cxx_sort_intersections(const int& ind_condition, const farray_t& ax,
                       const farray_t& ay, const farray_t& bx,
                       const farray_t& by, int& csize, farray_t& coorx,
                       farray_t& coory)
{
    uintmax_t i = 0, j = 0, k = 0;
    auto      _asize = ax.size();
    auto      _bsize = bx.size();
    if(ind_condition == 0)
    {
        for(; i < _asize && j < _bsize;)
        {
            if(ax[ax.size() - 1 - i] < bx[j])
            {
                coorx[k] = ax[ax.size() - 1 - i];
                coory[k] = ay[ax.size() - 1 - i];
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

        uintmax_t nitr = (_asize - i);
        if(nitr > 0)
        {
#pragma omp simd
            for(uintmax_t nn = 0; nn < nitr; ++nn)
                coorx[k + nn] = ax[ax.size() - 1 - i - nn];
#pragma omp simd
            for(uintmax_t nn = 0; nn < nitr; ++nn)
                coory[k + nn] = ay[ax.size() - 1 - i - nn];
            i += nitr;
            k += nitr;
        }

        nitr = (_bsize - j);
        if(nitr > 0)
        {
#pragma omp simd
            for(uintmax_t nn = 0; nn < nitr; ++nn)
                coorx[k + nn] = bx[j + nn];
#pragma omp simd
            for(uintmax_t nn = 0; nn < nitr; ++nn)
                coory[k + nn] = by[j + nn];
            j += nitr;
            k += nitr;
        }
    }
    else
    {
        for(; i < _asize && j < _bsize;)
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

        uintmax_t nitr = (_asize - i);
        if(nitr > 0)
        {
#pragma omp simd
            for(uintmax_t nn = 0; nn < nitr; ++nn)
                coorx[k + nn] = ax[i + nn];
#pragma omp simd
            for(uintmax_t nn = 0; nn < nitr; ++nn)
                coory[k + nn] = ay[i + nn];
            i += nitr;
            k += nitr;
        }

        nitr = (_bsize - j);
        if(nitr > 0)
        {
#pragma omp simd
            for(uintmax_t nn = 0; nn < nitr; ++nn)
                coorx[k + nn] = bx[j + nn];
#pragma omp simd
            for(uintmax_t nn = 0; nn < nitr; ++nn)
                coory[k + nn] = by[j + nn];
            j += nitr;
            k += nitr;
        }
    }
    csize = ax.size() + bx.size();
}

//============================================================================//

float
cxx_calc_sum_sqr(const farray_t& dist)
{
    float sum_sqr = 0.0f;
    //#pragma omp simd reduction(+:sum_sqr)
    for(uintmax_t n = 0; n < dist.size(); ++n)
        sum_sqr += dist[n] * dist[n];
    return sum_sqr;
}

//============================================================================//

void
cxx_calc_dist(int ry, int rz, int csize, const farray_t& coorx,
              const farray_t& coory, iarray_t& indi, farray_t& dist)
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
cxx_calc_dist2(int ry, int rz, int csize, const farray_t& coorx,
               const farray_t& coory, iarray_t& indx, iarray_t& indy,
               farray_t& dist)
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
        float midx = (coorx[n + 1] + coorx[n]) * 0.5;
        float midy = (coory[n + 1] + coory[n]) * 0.5;
        float x1   = midx + ry * 0.5;
        float x2   = midy + rz * 0.5;
        int   i1   = (int) (midx + ry * 0.5);
        int   i2   = (int) (midy + rz * 0.5);
        indx[n]    = i1 - (i1 > x1);
        indy[n]    = i2 - (i2 > x2);
    }
}

//============================================================================//

void
cxx_calc_simdata(int s, int p, int d, int ry, int rz, int dt, int dx, int csize,
                 const iarray_t& indi, const farray_t& dist, const float* model,
                 farray_t& simdata)
{
    int index_model = s * ry * rz;
    int index_data  = d + p * dx + s * dt * dx;
    for(int n = 0; n < csize - 1; ++n)
    {
        simdata[index_data] += model[indi[n] + index_model] * dist[n];
    }
}

//============================================================================//

void
cxx_calc_simdata2(int s, int p, int d, int ry, int rz, int dt, int dx,
                  int csize, const iarray_t& indx, const iarray_t& indy,
                  const farray_t& dist, float vx, float vy,
                  const farray_t& modelx, const farray_t& modely,
                  farray_t& simdata)
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

//============================================================================//

void
cxx_calc_simdata3(int s, int p, int d, int ry, int rz, int dt, int dx,
                  int csize, const iarray_t& indx, const iarray_t& indy,
                  const farray_t& dist, float vx, float vy,
                  const farray_t& modelx, const farray_t& modely,
                  const farray_t& modelz, int axis, farray_t& simdata)
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

//============================================================================//

farray_t
cxx_expand(const farray_t& arr_i, const int& factor)
{
    farray_t arr_o(arr_i.size() * factor, 0.0);
    for(uint64_t i = 0; i < arr_i.size(); ++i)
    {
        for(uint64_t off = 0; off < factor; ++off)
        {
            arr_o.at(i * factor + off) = arr_i.at(i);
        }
    }
    return arr_o;
}

//============================================================================//

farray_t
cxx_compress(const farray_t& arr_i, const int& factor)
{
    farray_t arr_o(arr_i.size() / factor, 0.0);
    for(uint64_t i = 0; i < arr_o.size(); ++i)
    {
        for(uint64_t off = 0; off < factor; ++off)
        {
            arr_o.at(i) += arr_i.at(i * factor + off);
        }
        arr_o.at(i) /= factor;
    }
    return arr_o;
}

//============================================================================//

float
cxx_rotate_x(const float x, const float y, const float theta)
{
    return x * cosf(theta) - y * sinf(theta);
}

//============================================================================//

float
cxx_rotate_y(const float x, const float y, const float theta)
{
    return x * sinf(theta) + y * cosf(theta);
}

//============================================================================//

bool
bounded(const farray_t& obj, int nx, int ix, int iy)
{
    auto idx = [&](int _x, int _y) { return _y * nx + _x; };
    return (ix >= 0 && iy >= 0 && idx(ix, iy) >= 0 && idx(ix, iy) < obj.size());
}

//============================================================================//

float
compute_neighbors(const farray_t& obj, float theta, int nx, int ix, int iy)
{
    float value = 0.0f;
    int   nvals = 0;
    auto  idx   = [&](int _x, int _y) { return _y * nx + _x; };

    if(bounded(obj, nx, ix - 1, iy))
    {
        ++nvals;
        value += powf(cosf(theta), 2.0f) * obj.at(idx(ix - 1, iy));
    }

    if(bounded(obj, nx, ix, iy + 1))
    {
        ++nvals;
        value += powf(sinf(theta), 2.0f) * obj.at(idx(ix, iy + 1));
    }

    return (nvals > 0) ? (value * (1.0f - powf(cos(theta), 2.0f)))
                       : (powf(sinf(theta), 2.0f) * obj.at(idx(ix, iy)));
}

//============================================================================//

farray_t
cxx_rotate(const farray_t& obj, const float theta, const int nx, const int ny)
{
    farray_t rot(nx * ny, 0.0);
    float    xoff = round(nx / 2.0);
    float    yoff = round(ny / 2.0);
    float    xop  = (nx % 2 == 0) ? 0.5 : 0.0;
    float    yop  = (ny % 2 == 0) ? 0.5 : 0.0;

    for(int i = 0; i < nx; ++i)
    {
        for(int j = 0; j < ny; ++j)
        {
            // indices in 2D
            float rx = float(i) - xoff + xop;
            float ry = float(j) - yoff + yop;
            // transformation
            float tx = cxx_rotate_x(rx, ry, theta);
            float ty = cxx_rotate_y(rx, ry, theta);
            // indices in 2D
            float x = (tx + xoff - xop);
            float y = (ty + yoff - yop);
            // index in 1D array
            int   rz       = j * nx + i;
            float _obj_val = 0.0f;
            // within bounds
            if(rz < rot.size())
            {
                int   x1   = floor(tx + xoff - xop);
                int   y1   = floor(ty + yoff - yop);
                int   x2   = x1 + 1;
                int   y2   = y1 + 1;
                float fxy1 = 0.0f;
                float fxy2 = 0.0f;
                if(y1 * nx + x1 < obj.size())
                    fxy1 += (x2 - x) * obj.at(y1 * nx + x1);
                if(y1 * nx + x2 < obj.size())
                    fxy1 += (x - x1) * obj.at(y1 * nx + x2);
                if(y2 * nx + x1 < obj.size())
                    fxy2 += (x2 - x) * obj.at(y2 * nx + x1);
                if(y2 * nx + x2 < obj.size())
                    fxy2 += (x - x1) * obj.at(y2 * nx + x2);
                _obj_val = (y2 - y) * fxy1 + (y - y1) * fxy2;
            }
            rot.at(rz) += _obj_val;
        }
    }
    return rot;
}

//============================================================================//
