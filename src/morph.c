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

#include "morph.h"


DLL void 
apply_pad(
    float* data, int dx, int dy, int dz, 
    int axis, int npad, float* out) 
{
    int n, m, i1, i2, j1, j2, k;

    if (axis == 0)
    {
        for (m = 0; m < dx; m++) 
        {
            i1 = m * (dy * dz);
            i2 = ((int)(npad-dx)/2 + m) * (dy * dz);
            for (n = 0; n < dy; n++) 
            {
                j1 = n * dz;
                j2 = n * dz;
                for (k = 0; k < dz; k++) 
                {
                    out[i2+j2+k] = data[i1+j1+k];
                }
            }
        }
    }
    else if (axis == 1)
    {
        for (m = 0; m < dx; m++) 
        {
            i1 = m * (dy * dz);
            i2 = m * (npad * dz);
            for (n = 0; n < dy; n++) 
            {
                j1 = n * dz;
                j2 = ((int)(npad-dy)/2 + n) * dz;
                for (k = 0; k < dz; k++) 
                {
                    out[i2+j2+k] = data[i1+j1+k];
                }
            }
        }
    }
    else if (axis == 2)
    {
        for (m = 0; m < dx; m++) 
        {
            i1 = m * (dy * dz);
            i2 = (int)(npad-dz)/2 + m * (npad * dy);
            for (n = 0; n < dy; n++) 
            {
                j1 = n * dz;
                j2 = n * npad;
                for (k = 0; k < dz; k++) 
                {
                    out[i2+j2+k] = data[i1+j1+k];
                }
            }
        }
    }
}


DLL void 
downsample(
    float* data, int dx, int dy, int dz,
    int level, int axis, float* out) 
{
    int m, n, k, i, j, p, ind;
    int binsize;
    
    binsize = pow(2, level);
    
    if (axis == 0)
    {
        dx /= binsize;
        for (m = 0, ind = 0; m < dx; m++) 
        {
            i = m * (dy * dz);
            for (p = 0; p < binsize; p++) 
            {
                for (n = 0; n < dy; n++) 
                {
                    j = n * dz;
                    for (k = 0; k < dz; k++)
                    {
                        out[i+j+k] += data[ind]/binsize;
                        ind++;
                    }
                }
            }
        }
    }
    else if (axis == 1)
    {
        dy /= binsize;
        for (m = 0, ind = 0; m < dx; m++) 
        {
            i = m * (dy * dz);
            for (n = 0; n < dy; n++) 
            {
                j = n * dz;
                for (p = 0; p < binsize; p++) 
                {
                    for (k = 0; k < dz; k++)
                    {
                        out[i+j+k] += data[ind]/binsize;
                        ind++;
                    }
                }
            }
        }
    }
    else if (axis == 2)
    {
        dz /= binsize;
        for (m = 0, ind = 0; m < dx; m++) 
        {
            i = m * (dy * dz);
            for (n = 0; n < dy; n++) 
            {
                j = n * dz;
                for (k = 0; k < dz; k++) 
                {
                    for (p = 0; p < binsize; p++) 
                    {
                        out[i+j+k] += data[ind]/binsize;
                        ind++;
                    }
                }
            }
        }
    }
}


DLL void 
upsample(
    float* data, int dx, int dy, int dz,
    int level, int axis, float* out) {

    int m, n, k, i, j, p, ind;
    int binsize;
    
    binsize = pow(2, level);
    
    if (axis == 0)
    {
        for (m = 0, ind = 0; m < dx; m++) 
        {
            i = m * (dy * dz);
            for (p = 0; p < binsize; p++) 
            {
                for (n = 0; n < dy; n++) 
                {
                    j = n * dz;
                    for (k = 0; k < dz; k++)
                    {
                        out[ind] = data[i+j+k];
                        ind++;
                    }
                }
            }
        }
    }
    else if (axis == 1)
    {
        for (m = 0, ind = 0; m < dx; m++) 
        {
            i = m * (dy * dz);
            for (n = 0; n < dy; n++) 
            {
                j = n * dz;
                for (p = 0; p < binsize; p++) 
                {
                    for (k = 0; k < dz; k++)
                    {
                        out[ind] = data[i+j+k];
                        ind++;
                    }
                }
            }
        }
    }
    else if (axis == 2)
    {
        for (m = 0, ind = 0; m < dx; m++) 
        {
            i = m * (dy * dz);
            for (n = 0; n < dy; n++) 
            {
                j = n * dz;
                for (k = 0; k < dz; k++) 
                {
                    for (p = 0; p < binsize; p++) 
                    {
                        out[ind] = data[i+j+k];
                        ind++;
                    }
                }
            }
        }
    }
}