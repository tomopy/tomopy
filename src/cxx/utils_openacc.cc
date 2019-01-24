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
#include "utils.hh"
#include <cstdint>

BEGIN_EXTERN_C
#include "utils_openacc.h"
END_EXTERN_C

//======================================================================================//

float*
openacc_rotate(const float* src, const float theta, const int nx, const int ny)
{
    float* dst      = new float[nx * ny];
    float  xoff     = round(nx / 2.0);
    float  yoff     = round(ny / 2.0);
    float  xop      = (nx % 2 == 0) ? 0.5 : 0.0;
    float  yop      = (ny % 2 == 0) ? 0.5 : 0.0;
    int    src_size = nx * ny;
    memset(dst, 0, nx * ny * sizeof(float));

    for(int j = 0; j < ny; ++j)
    {
        for(int i = 0; i < nx; ++i)
        {
            // indices in 2D
            float rx = float(i) - xoff + xop;
            float ry = float(j) - yoff + yop;
            // transformation
            float tx = rx * cosf(theta) + -ry * sinf(theta);
            float ty = rx * sinf(theta) + ry * cosf(theta);
            // indices in 2D
            float x = (tx + xoff - xop);
            float y = (ty + yoff - yop);
            // index in 1D array
            int rz = j * nx + i;
            if(rz < 0 || rz >= src_size)
                continue;
            // within bounds
            int   x1   = floor(tx + xoff - xop);
            int   y1   = floor(ty + yoff - yop);
            int   x2   = x1 + 1;
            int   y2   = y1 + 1;
            float fxy1 = 0.0f;
            float fxy2 = 0.0f;
            if(x1 >= 0 && y1 >= 0 && y1 * nx + x1 < src_size)
                fxy1 += (x2 - x) * src[y1 * nx + x1];
            if(x2 >= 0 && y1 >= 0 && y1 * nx + x2 < src_size)
                fxy1 += (x - x1) * src[y1 * nx + x2];
            if(x1 >= 0 && y2 >= 0 && y2 * nx + x1 < src_size)
                fxy2 += (x2 - x) * src[y2 * nx + x1];
            if(x2 >= 0 && y2 >= 0 && y2 * nx + x2 < src_size)
                fxy2 += (x - x1) * src[y2 * nx + x2];
            dst[rz] += (y2 - y) * fxy1 + (y - y1) * fxy2;
        }
    }
    return dst;
}

//======================================================================================//

void
openacc_compute_projection(int dt, int dx, int ngridx, int ngridy, const float* data,
                           const float* theta, int s, int p, float* simdata,
                           float* update, float* recon_off)
{
    // needed for recon to output at proper orientation
    float pi_offset = 0.5f * (float) M_PI;
    float fngridx   = ngridx;
    float theta_p   = fmodf(theta[p] + pi_offset, 2.0f * (float) M_PI);

    // Rotate object
    float* recon_rot = openacc_rotate(recon_off, -theta_p, ngridx, ngridy);

    for(int d = 0; d < dx; d++)
    {
        int    pix_offset = d * ngridx;  // pixel offset
        int    idx_data   = d + p * dx + s * dt * dx;
        float* _simdata   = simdata + idx_data;
        float* _recon_rot = recon_rot + pix_offset;
        float  _sim       = 0.0f;

        // Calculate simulated data by summing up along x-axis
        for(int n = 0; n < ngridx; n++)
            _sim += _recon_rot[n];

        // update shared simdata array
        *_simdata += _sim;

        // Make update by backprojecting error along x-axis
        float upd = (data[idx_data] - *_simdata) / fngridx;
        for(int n = 0; n < ngridx; n++)
            _recon_rot[n] += upd;
    }
    // Back-Rotate object
    float* recon_tmp = openacc_rotate(recon_rot, theta_p, ngridx, ngridy);

    static Mutex _mutex;
    _mutex.lock();
    // update shared update array
    for(int i = 0; i < (ngridx * ngridy); ++i)
        update[i] += recon_tmp[i];
    _mutex.unlock();

    delete[] recon_rot;
    delete[] recon_tmp;
}

//======================================================================================//
