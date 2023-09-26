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

// C-module for median filtration and dezingering (3D and 2D cases)
// Original author: Daniil Kazantsev, Diamond Light Source Ltd.

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "libtomo/median_filt3d.h"

int floatcomp(const void* elem1, const void* elem2)
{
    if(*(const float*)elem1 < *(const float*)elem2)
        return -1;
    return *(const float*)elem1 > *(const float*)elem2;
}
int uint16comp(const void* elem1, const void* elem2)
{
    if(*(const unsigned short*)elem1 < *(const unsigned short*)elem2)
        return -1;
    return *(const unsigned short*)elem1 > *(const unsigned short*)elem2;
}


void
medfilt3D_float(float* Input, float* Output, int radius, int sizefilter_total,
                float mu_threshold, long i, long j, long k, size_t index, size_t dimX,
                size_t dimY, size_t dimZ)
{
    float*    ValVec;
    long      i_m;
    long      j_m;
    long      k_m;
    long      i1;
    long      j1;
    long      k1;
    long long counter;
    int       midval;
    size_t    index1;
    midval = (sizefilter_total / 2);
    ValVec = (float*) calloc(sizefilter_total, sizeof(float));

    /* filling the allocated vector with the neighbouring values */
    counter = 0LL;
    for(i_m = -radius; i_m <= radius; i_m++)
    {
        i1 = i + i_m;
        if((i1 < 0) || (i1 >= dimX))
            i1 = i;
        for(j_m = -radius; j_m <= radius; j_m++)
        {
            j1 = j + j_m;
            if((j1 < 0) || (j1 >= dimY))
                j1 = j;
            for(k_m = -radius; k_m <= radius; k_m++)
            {
                k1 = k + k_m;
                if((k1 < 0) || (k1 >= dimZ))
                    k1 = k;
                index1 = dimX * dimY * (size_t)k1 + (size_t)j1 * dimX + (size_t)i1;
                ValVec[counter] = Input[index1];
                counter++;
            }
        }
    }

    qsort(ValVec, sizefilter_total, sizeof(float), floatcomp);

    if(mu_threshold == 0.0F)
    {
        /* perform median filtration */
        Output[index] = ValVec[midval];
    }
    else
    {
        /* perform dezingering */
        if(fabsf(Input[index] - ValVec[midval]) >= mu_threshold)
            Output[index] = ValVec[midval];
    }
    free(ValVec);
}

void
medfilt2D_float(float* Input, float* Output, int radius, int sizefilter_total,
                float mu_threshold, long i, long j, size_t index, size_t dimX, size_t dimY)
{
    float*    ValVec;
    long      i_m;
    long      j_m;
    long      i1;
    long      j1;
    long long counter;
    int       midval;
    size_t    index1;
    midval = (sizefilter_total / 2);
    ValVec = (float*) calloc(sizefilter_total, sizeof(float));

    /* filling the allocated vector with the neighbouring values */
    counter = 0LL;
    for(i_m = -radius; i_m <= radius; i_m++)
    {
        i1 = i + i_m;
        if((i1 < 0) || (i1 >= dimX))
            i1 = i;
        for(j_m = -radius; j_m <= radius; j_m++)
        {
            j1 = j + j_m;
            if((j1 < 0) || (j1 >= dimY))
                j1 = j;
            index1 = (size_t)(j1) * dimX + (size_t)(i1);
            ValVec[counter] = Input[index1];
            counter++;
        }
    }
    qsort(ValVec, sizefilter_total, sizeof(float), floatcomp);

    if(mu_threshold == 0.0F)
    {
        /* perform median filtration */
        Output[index] = ValVec[midval];
    }
    else
    {
        /* perform dezingering */
        if(fabsf(Input[index] - ValVec[midval]) >= mu_threshold)
            Output[index] = ValVec[midval];
    }
    free(ValVec);
}

void
medfilt3D_uint16(unsigned short* Input, unsigned short* Output, int radius,
                 int sizefilter_total, float mu_threshold, long i, long j, long k,
                 size_t index, size_t dimX, size_t dimY, size_t dimZ)
{
    unsigned short* ValVec;
    long            i_m;
    long            j_m;
    long            k_m;
    long            i1;
    long            j1;
    long            k1;
    long long       counter;
    int             midval;
    size_t          index1;
    midval = (sizefilter_total / 2);
    ValVec = (unsigned short*) calloc(sizefilter_total, sizeof(unsigned short));

    /* filling the allocated vector with the neighbouring values */
    counter = 0LL;
    for(i_m = -radius; i_m <= radius; i_m++)
    {
        i1 = i + i_m;
        if((i1 < 0) || (i1 >= dimX))
            i1 = i;
        for(j_m = -radius; j_m <= radius; j_m++)
        {
            j1 = j + j_m;
            if((j1 < 0) || (j1 >= dimY))
                j1 = j;
            for(k_m = -radius; k_m <= radius; k_m++)
            {
                k1 = k + k_m;
                if((k1 < 0) || (k1 >= dimZ))
                    k1 = k;
                index1 = dimX * dimY * (size_t)k1 + (size_t)j1 * dimX + (size_t)i1;
                ValVec[counter] = Input[index1];
                counter++;
            }
        }
    }

    qsort(ValVec, sizefilter_total, sizeof(unsigned short), uint16comp);

    if(mu_threshold == 0.0F)
    {
        /* perform median filtration */
        Output[index] = ValVec[midval];
    }
    else
    {
        /* perform dezingering */
        if(abs(Input[index] - ValVec[midval]) >= mu_threshold)
            Output[index] = ValVec[midval];
    }
    free(ValVec);
}

void
medfilt2D_uint16(unsigned short* Input, unsigned short* Output, int radius,
                 int sizefilter_total, float mu_threshold, long i, long j, size_t index,
                 size_t dimX, size_t dimY)
{
    unsigned short* ValVec;
    long            i_m;
    long            j_m;
    long            i1;
    long            j1;
    long long       counter;
    int             midval;
    size_t          index1;
    midval = (sizefilter_total / 2);
    ValVec = (unsigned short*) calloc(sizefilter_total, sizeof(unsigned short));

    /* filling the allocated vector with the neighbouring values */
    counter = 0LL;
    for(i_m = -radius; i_m <= radius; i_m++)
    {
        i1 = i + i_m;
        if((i1 < 0) || (i1 >= dimX))
            i1 = i;
        for(j_m = -radius; j_m <= radius; j_m++)
        {
            j1 = j + j_m;
            if((j1 < 0) || (j1 >= dimY))
                j1 = j;
            index1 = (size_t)(j1) * dimX + (size_t)(i1);
            ValVec[counter] = Input[index1];
            counter++;
        }
    }

    qsort(ValVec, sizefilter_total, sizeof(unsigned short), uint16comp);

    if(mu_threshold == 0.0F)
    {
        /* perform median filtration */
        Output[index] = ValVec[midval];
    }
    else
    {
        /* perform dezingering */
        if(abs(Input[index] - ValVec[midval]) >= mu_threshold)
            Output[index] = ValVec[midval];
    }
    free(ValVec);
}

DLL int
medianfilter_main_float(float* Input, float* Output, int radius, float mu_threshold,
                        int ncores, int dimX, int dimY, int dimZ)
{
    int       sizefilter_total;
    int       diameter;
    long      i;
    long      j;
    long      k;
    size_t    index;
    size_t    totalvoxels;

    totalvoxels = (size_t)(dimX) * (size_t)(dimY) * (size_t)(dimZ);
    diameter = (2 * radius + 1); /* diameter of the filter's kernel */

    /* dealing here with a custom given number of cpu threads */
    if(ncores > 0)
    {
        // Explicitly disable dynamic teams
        omp_set_dynamic(0);
        // Use a number of threads for all consecutive parallel regions
        omp_set_num_threads(ncores);
    }
    if(dimZ == 0)
    /* 2D filtering */
    {
        sizefilter_total = (int) (powf(diameter, 2));
#pragma omp parallel for shared(Input, Output) private(i, j, index)
        for(j = 0; j < dimY; j++)
        {
            for(i = 0; i < dimX; i++)
            {
                index = (size_t)(j) * dimX + (size_t)(i);
                medfilt2D_float(Input, Output, radius, sizefilter_total, mu_threshold, i,
                                j, index, (size_t) dimX, (size_t) dimY);
            }
        }
    }
    else
    /* 3D filtering */
    {
        sizefilter_total = (int) (powf(diameter, 3));
#pragma omp parallel for shared(Input, Output) private(i, j, k, index)
        for(k = 0; k < dimZ; k++)
        {
            for(j = 0; j < dimY; j++)
            {
                for(i = 0; i < dimX; i++)
                {
                    index = dimX * dimY * (size_t)(k) + (size_t)(j) * dimX + (size_t)(i);
                    medfilt3D_float(Input, Output, radius, sizefilter_total, mu_threshold,
                                    i, j, k, index, (size_t) dimX, (size_t) dimY,
                                    (size_t) dimZ);
                }
            }
        }
    }
    return 0;
}

DLL int
medianfilter_main_uint16(unsigned short* Input, unsigned short* Output, int radius,
                         float mu_threshold, int ncores, int dimX, int dimY, int dimZ)

{
    int       sizefilter_total;
    int       diameter;
    long      i;
    long      j;
    long      k;
    size_t    index;
    size_t    totalvoxels;

    totalvoxels = (size_t)(dimX) * (size_t)(dimY) * (size_t)(dimZ);
    diameter = (2 * radius + 1); /* diameter of the filter's kernel */

    /* dealing here with a custom given number of cpu threads */
    if(ncores > 0)
    {
        // Explicitly disable dynamic teams
        omp_set_dynamic(0);
        // Use a number of threads for all consecutive parallel regions
        omp_set_num_threads(ncores);
    }
    if(dimZ == 0)
    /* 2D filtering */
    {
        sizefilter_total = (int) (powf(diameter, 2));
#pragma omp parallel for shared(Input, Output) private(i, j, index)
        for(j = 0; j < dimY; j++)
        {
            for(i = 0; i < dimX; i++)
            {
                index = (size_t)(j) * dimX + (size_t)(i);
                medfilt2D_uint16(Input, Output, radius, sizefilter_total, mu_threshold, i,
                                 j, index, (size_t) dimX,  (size_t) dimY);
            }
        }
    }
    else
    /* 3D filtering */
    {
        sizefilter_total = (int) (powf(diameter, 3));
#pragma omp parallel for shared(Input, Output) private(i, j, k, index)
        for(k = 0; k < dimZ; k++)
        {
            for(j = 0; j < dimY; j++)
            {
                for(i = 0; i < dimX; i++)
                {
                    index = dimX * dimY * (size_t)(k) + (size_t)(j) * dimX + (size_t)(i);
                    medfilt3D_uint16(Input, Output, radius, sizefilter_total,
                                     mu_threshold, i, j, k, index,(size_t) dimX,
                                     (size_t)dimY, (size_t)dimZ);
                }
            }
        }
    }
    return 0;
}
