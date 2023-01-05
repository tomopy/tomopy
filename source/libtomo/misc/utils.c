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

#include <float.h>
#include <math.h>
#include <stdint.h>

#include "utils.h"

// for windows build
#ifdef WIN32
#    ifdef PY3K
void
PyInit_libtomo(void)
{
}
#    else
void
initlibtomo(void)
{
}
#    endif
#endif

/* Copy Image (float) */
void
copyIm(float* A, float* U, long dimX, long dimY, long dimZ)
{
    long j;
    for(j = 0; j < dimX * dimY * dimZ; j++)
        U[j] = A[j];
    return;
}

/* Copy Image -unsigned char (8bit)*/
void
copyIm_unchar(unsigned char* A, unsigned char* U, int dimX, int dimY, int dimZ)
{
    int j;
    for(j = 0; j < dimX * dimY * dimZ; j++)
        U[j] = A[j];
    return;
}

/* Copy Image - unsigned short (16bit)*/
void
copyIm_unshort(unsigned short* A, unsigned short* U, int dimX, int dimY, int dimZ)
{
    int j;
    for(j = 0; j < dimX * dimY * dimZ; j++)
        U[j] = A[j];
    return;
}

/* sorting using bubble method (float)*/
void
sort_bubble_float(float* x, int n_size)
{
    int   i, j;
    float temp;

    for(i = 0; i < n_size - 1; i++)
    {
        for(j = 0; j < n_size - i - 1; j++)
        {
            if(x[j] > x[j + 1])
            {
                temp     = x[j];
                x[j]     = x[j + 1];
                x[j + 1] = temp;
            }
        }
    }
    return;
}

/* sorting using bubble method (uint16)*/
void
sort_bubble_uint16(unsigned short* x, int n_size)
{
    int            i, j;
    unsigned short temp;

    for(i = 0; i < n_size - 1; i++)
    {
        for(j = 0; j < n_size - i - 1; j++)
        {
            if(x[j] > x[j + 1])
            {
                temp     = x[j];
                x[j]     = x[j + 1];
                x[j + 1] = temp;
            }
        }
    }
    return;
}

void
quicksort_float(float* x, int first, int last)
{
    int   i, j, pivot;
    float temp;

    if(first < last)
    {
        pivot = first;
        i     = first;
        j     = last;

        while(i < j)
        {
            while(x[i] <= x[pivot] && i < last)
                i++;
            while(x[j] > x[pivot])
                j--;
            if(i < j)
            {
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }

        temp     = x[pivot];
        x[pivot] = x[j];
        x[j]     = temp;
        quicksort_float(x, first, j - 1);
        quicksort_float(x, j + 1, last);
    }
    return;
}

void
quicksort_uint16(unsigned short* x, int first, int last)
{
    int            i, j, pivot;
    unsigned short temp;

    if(first < last)
    {
        pivot = first;
        i     = first;
        j     = last;

        while(i < j)
        {
            while(x[i] <= x[pivot] && i < last)
                i++;
            while(x[j] > x[pivot])
                j--;
            if(i < j)
            {
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }

        temp     = x[pivot];
        x[pivot] = x[j];
        x[j]     = temp;
        quicksort_uint16(x, first, j - 1);
        quicksort_uint16(x, j + 1, last);
    }
    return;
}
