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

// C-module for detecting and emphasising stripes present in the data (3D case)
// Original author: Daniil Kazantsev, Diamond Light Source Ltd.

#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "libtomo/stripes_detect3d.h"
#include "../misc/utils.h"


void
ratio_mean_stride3d(float* input, float* output,
                    long i, long j, long k, long long index,
                    long dimX, long dimY, long dimZ)
{
    float mean_plate; 
    float mean_horiz;
    int radius = 3;
    int diameter = 2*radius + 1;
    long      i_m;
    long      j_m;
    long      k_m;
    long      i1;
    long      j1;
    long      k1;
    
    /* calculate mean of gradientX in a 2D plate parallel to stripes direction */
    mean_plate = 0.0f;
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
            mean_plate += fabsf(input[((dimX * dimY) * k1 + j1 * dimX + i)]);
        }
    }
    mean_plate /= 49.0f;
    
    /* calculate mean of gradientX in a direction orthogonal to stripes direction */    
    mean_horiz = 0.0f;
    for(i_m = 1; i_m <= diameter; i_m++)
    {
        i1 = i + i_m;
        if (i1 >= dimX) 
            i1 = i;
        mean_horiz += fabsf(input[((dimX * dimY) * k + j * dimX + i1)]);
    }
    mean_horiz /= 7.0f;

    /* calculate the ratio between two means assuming that the mean 
    orthogonal to stripes direction should be larger than the mean 
    parallel to it */
    if ((mean_horiz > mean_plate) && (mean_horiz != 0.0f))
    {        
        output[index] = mean_plate/mean_horiz;
    }
    if ((mean_horiz < mean_plate) && (mean_plate != 0.0f))
    {        
        output[index] = mean_horiz/mean_plate;
    }    
}

void
vertical_median_stride3d(float* input, float* output,
                        int window_halflength_vertical, 
                        int window_fulllength,
                        int midval_window_index,
                        long i, long j, long k, long long index,
                        long dimX, long dimY, long dimZ)
{
    int       counter;
    long      k_m;
    long      k1;
    float* _values;
    
    _values = (float*) calloc(window_fulllength, sizeof(float));    
    
    counter = 0;
    for(k_m = -window_halflength_vertical; k_m <= window_halflength_vertical; k_m++)
    {
        k1 = k + k_m;
        if((k1 < 0) || (k1 >= dimZ))
            k1 = k;
        _values[counter] = input[((dimX * dimY) * k1 + j * dimX + i)];
        counter++;
    }
    quicksort_float(_values, 0, window_fulllength-1);
    output[index] = _values[midval_window_index];

    free (_values);
}

DLL int
stripesdetect3d_main_float(float* Input, float* Output, int window_halflength_vertical,
                        int ncores, int dimX, int dimY, int dimZ)
{
    long      i;
    long      j;
    long      k;
    long long index;
    long long totalvoxels;

    float* gradient3d_x_arr;
    float* mean_ratio3d_arr;

    totalvoxels = (long long) (dimX*dimY*dimZ);    

    int window_fulllength = (int)(2*window_halflength_vertical + 1); 
    int midval_window_index = (int)(0.5f*window_fulllength) - 1;

    gradient3d_x_arr = calloc(totalvoxels, sizeof(float));
    mean_ratio3d_arr = calloc(totalvoxels, sizeof(float));

    /* dealing here with a custom given number of cpu threads */
    if(ncores > 0)
    {
        // Explicitly disable dynamic teams
        omp_set_dynamic(0);
        // Use a number of threads for all consecutive parallel regions
        omp_set_num_threads(ncores);
    }

    /* Take the gradient in the horizontal direction, axis = 0*/
    gradient3D(Input, gradient3d_x_arr, (long) (dimX), (long) (dimY), (long) (dimZ), 0, 1);
    
    /* Here we calculate the ratio between the mean in a small 2D neighbourhood parallel to the stripe 
    and the mean orthogonal to the stripe. The gradient variation in the direction orthogonal to the
    stripe is expected to be large (a jump), while in parallel direction small. Therefore at the edges
    of a stripe we should get a ratio small/large or large/small. */
#pragma omp parallel for shared(gradient3d_x_arr, mean_ratio3d_arr) private(i, j, k, index)
        for(k = 0; k < dimZ; k++)
        {
            for(j = 0; j < dimY; j++)
            {
                for(i = 0; i < dimX; i++)
                {
                    index = ((dimX * dimY) * k + j * dimX + i);
                    ratio_mean_stride3d(gradient3d_x_arr, mean_ratio3d_arr,  i, j, k, index, (long) (dimX), (long) (dimY), (long) (dimZ));
                }
            }
        }
    /* We process the resulting ratio map with a vertical median filter which removes 
    small outliers of clusters */
#pragma omp parallel for shared(mean_ratio3d_arr, Output) private(i, j, k, index)
        for(k = 0; k < dimZ; k++)
        {
            for(j = 0; j < dimY; j++)
            {
                for(i = 0; i < dimX; i++)
                {
                    index = ((dimX * dimY) * k + j * dimX + i);
                    vertical_median_stride3d(mean_ratio3d_arr, Output, 
                                             window_halflength_vertical, 
                                             window_fulllength,
                                             midval_window_index,
                                             i, j, k, index, (long) (dimX), (long) (dimY), (long) (dimZ));
                }
            }
        }

    free(gradient3d_x_arr);
    free(mean_ratio3d_arr);
    return 0;
}

DLL int
stripesmask3d_main_float(float* Input, unsigned short* Output,
                         float threshold_val,
                         int stripe_length_min, int stripe_depth_min,
                         int ncores, int dimX, int dimY, int dimZ)
{
    long      i;
    long      j;
    long      k;
    long long index;
    long long totalvoxels;
    totalvoxels = (long long) (dimX*dimY*dimZ);    

    /* dealing here with a custom given number of cpu threads */
    if(ncores > 0)
    {
        // Explicitly disable dynamic teams
        omp_set_dynamic(0);
        // Use a number of threads for all consecutive parallel regions
        omp_set_num_threads(ncores);
    }

    /* First step is to mask all the values in the given weights input image    
       that are bellow a given threshold_val */
#pragma omp parallel for shared(Input, Output) private(i, j, k, index)
        for(k = 0; k < dimZ; k++)
        {
            for(j = 0; j < dimY; j++)
            {
                for(i = 0; i < dimX; i++)
                {
                    index = ((dimX * dimY) * k + j * dimX + i);
                    if (Input[index] <= threshold_val) 
                    {
                        Output[index] = 1;
                    }
                    
                }
            }
        }

    return 0;
}
