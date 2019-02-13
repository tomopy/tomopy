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

#include "stripe.h"

void
remove_stripe_sf(float* data, int dx, int dy, int dz, int size, int istart, int iend)
{
    int    i, j, k, p, s;
    float* avrage_row;
    float* smooth_row;

    // For each slice.
    for(s = istart; s < iend; s++)
    {
        avrage_row = (float*) calloc(dz, sizeof(float));
        smooth_row = (float*) calloc(dz, sizeof(float));

        // For each pixel.
        for(j = 0; j < dz; j++)
        {
            // For each projection.
            for(p = 0; p < dx; p++)
            {
                avrage_row[j] += data[j + s * dz + p * dy * dz] / dx;
            }
        }

        // We have now computed the average row of the sinogram.
        // Smooth it
        for(i = 0; i < dz; i++)
        {
            smooth_row[i] = 0;
            for(j = 0; j < size; j++)
            {
                k = i + j - size / 2;
                if(k < 0)
                    k = 0;
                if(k > dz - 1)
                    k = dz - 1;
                smooth_row[i] += avrage_row[k];
            }
            smooth_row[i] /= size;
        }

        // For each projection.
        for(p = 0; p < dx; p++)
        {
            // Subtract this difference from each row in sinogram.
            for(j = 0; j < dz; j++)
            {
                data[j + s * dz + p * dy * dz] -= (avrage_row[j] - smooth_row[j]);
            }
        }

        free(avrage_row);
        free(smooth_row);
    }
}
