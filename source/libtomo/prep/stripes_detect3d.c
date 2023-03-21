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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libtomo/stripe.h"

/********************************************************************/
/**********************Supporting Functions**************************/
/********************************************************************/
/* Calculate the forward difference derrivative of the 3D input in the
direction of the "axis" parameter using the step_size in pixels to skip pixels
(i.e. step_size = 1 is the classical gradient)
axis = 0: horizontal direction
axis = 1: depth direction
axis = 2: vertical direction
*/
void
gradient3D_local(const float* input, float* output, long dimX, long dimY, long dimZ,
                 int axis, int step_size)
{
    long   i;
    long   j;
    long   k;
    long   i1;
    long   j1;
    long   k1;
    size_t index;

#pragma omp parallel for shared(input, output) private(i, j, k, i1, j1, k1, index)
    for(j = 0; j < dimY; j++)
    {
        for(i = 0; i < dimX; i++)
        {
            for(k = 0; k < dimZ; k++)
            {
                index = (size_t) (dimX * dimY * k) + (size_t) (j * dimX + i);
                /* Forward differences */
                if(axis == 0)
                {
                    i1 = i + step_size;
                    if(i1 >= dimX)
                        i1 = i - step_size;
                    output[index] =
                        input[(size_t) (dimX * dimY * k) + (size_t) (j * dimX + i1)] -
                        input[index];
                }
                else if(axis == 1)
                {
                    j1 = j + step_size;
                    if(j1 >= dimY)
                        j1 = j - step_size;
                    output[index] =
                        input[(size_t) (dimX * dimY * k) + (size_t) (j1 * dimX + i)] -
                        input[index];
                }
                else
                {
                    k1 = k + step_size;
                    if(k1 >= dimZ)
                        k1 = k - step_size;
                    output[index] =
                        input[(size_t) (dimX * dimY * k1) + (size_t) (j * dimX + i)] -
                        input[index];
                }
            }
        }
    }
}

void
ratio_mean_stride3d(float* input, float* output, int radius, long i, long j, long k,
                    long dimX, long dimY, long dimZ)
{
    float  mean_plate;
    float  mean_horiz;
    float  mean_horiz2;
    float  min_val;
    int    diameter          = 2 * radius + 1;
    int    all_pixels_window = diameter * diameter;
    long   i_m;
    long   j_m;
    long   k_m;
    long   i1;
    long   j1;
    long   k1;
    size_t index;
    size_t newindex;

    index = (size_t) (dimX * dimY * k) + (size_t) (j * dimX + i);

    min_val = 0.0F;
    /* calculate mean of gradientX in a 2D plate parallel to stripes direction */
    mean_plate = 0.0F;
    for(j_m = -radius; j_m <= radius; j_m++)
    {
        j1 = j + j_m;
        if((j1 < 0) || (j1 >= dimY))
            j1 = j - j_m;
        for(k_m = -radius; k_m <= radius; k_m++)
        {
            k1 = k + k_m;
            if((k1 < 0) || (k1 >= dimZ))
                k1 = k - k_m;
            newindex = (size_t) (dimX * dimY * k1) + (size_t) (j1 * dimX + i);
            mean_plate += fabsf(input[newindex]);
        }
    }
    mean_plate /= (float) (all_pixels_window);

    /* calculate mean of gradientX in a 2D plate orthogonal to stripes direction */
    mean_horiz = 0.0F;
    for(j_m = -1; j_m <= 1; j_m++)
    {
        j1 = j + j_m;
        if((j1 < 0) || (j1 >= dimY))
            j1 = j - j_m;
        for(i_m = 1; i_m <= radius; i_m++)
        {
            i1 = i + i_m;
            if(i1 >= dimX)
                i1 = i - i_m;
            newindex = (size_t) (dimX * dimY * k) + (size_t) (j1 * dimX + i1);
            mean_horiz += fabsf(input[newindex]);
        }
    }
    mean_horiz /= (float) (radius * 3);

    /* Calculate another mean symmetrically */
    mean_horiz2 = 0.0F;
    for(j_m = -1; j_m <= 1; j_m++)
    {
        j1 = j + j_m;
        if((j1 < 0) || (j1 >= dimY))
            j1 = j - j_m;
        for(i_m = -radius; i_m <= -1; i_m++)
        {
            i1 = i + i_m;
            if(i1 < 0)
                i1 = i - i_m;
            newindex = (size_t) (dimX * dimY * k) + (size_t) (j1 * dimX + i1);
            mean_horiz2 += fabsf(input[newindex]);
        }
    }
    mean_horiz2 /= (float) (radius * 3);

    /* calculate the ratio between two means assuming that the mean
    orthogonal to stripes direction should be larger than the mean
    parallel to it */
    if((mean_horiz >= mean_plate) && (mean_horiz != 0.0F))
        output[index] = mean_plate / mean_horiz;
    if((mean_horiz < mean_plate) && (mean_plate != 0.0F))
        output[index] = mean_horiz / mean_plate;
    if((mean_horiz2 >= mean_plate) && (mean_horiz2 != 0.0F))
        min_val = mean_plate / mean_horiz2;
    if((mean_horiz2 < mean_plate) && (mean_plate != 0.0F))
        min_val = mean_horiz2 / mean_plate;

    /* accepting the smallest value */
    if(output[index] > min_val)
        output[index] = min_val;
}

int
floatcomp(const void* elem1, const void* elem2)
{
    if(*(const float*) elem1 < *(const float*) elem2)
        return -1;
    return *(const float*) elem1 > *(const float*) elem2;
}

void
vertical_median_stride3d(const float* input, float* output,
                         int window_halflength_vertical, int window_fulllength,
                         int midval_window_index, long i, long j, long k, long dimX,
                         long dimY, long dimZ)
{
    int    counter;
    long   k_m;
    long   k1;
    size_t index;

    index = (size_t) (dimX * dimY * k) + (size_t) (j * dimX + i);

    float* _values;
    _values = (float*) calloc(window_fulllength, sizeof(float));

    counter = 0;
    for(k_m = -window_halflength_vertical; k_m <= window_halflength_vertical; k_m++)
    {
        k1 = k + k_m;
        if((k1 < 0) || (k1 >= dimZ))
            k1 = k - k_m;
        _values[counter] = input[(size_t) (dimX * dimY * k1) + (size_t) (j * dimX + i)];
        counter++;
    }
    qsort(_values, window_fulllength, sizeof(float), floatcomp);
    output[index] = _values[midval_window_index];

    free(_values);
}

void
mean_stride3d(const float* input, float* output, long i, long j, long k, long dimX,
              long dimY, long dimZ)
{
    /* a 3d mean to enusre a more stable gradient */
    long   i1;
    long   i2;
    long   j1;
    long   j2;
    long   k1;
    long   k2;
    float  val1;
    float  val2;
    float  val3;
    float  val4;
    float  val5;
    float  val6;
    size_t index;

    index = (size_t) (dimX * dimY * k) + (size_t) (j * dimX + i);

    i1 = i - 1;
    i2 = i + 1;
    j1 = j - 1;
    j2 = j + 1;
    k1 = k - 1;
    k2 = k + 1;

    if(i1 < 0)
        i1 = i2;
    if(i2 >= dimX)
        i2 = i1;
    if(j1 < 0)
        j1 = j2;
    if(j2 >= dimY)
        j2 = j1;
    if(k1 < 0)
        k1 = k2;
    if(k2 >= dimZ)
        k2 = k1;

    val1 = input[(size_t) (dimX * dimY * k) + (size_t) (j * dimX + i1)];
    val2 = input[(size_t) (dimX * dimY * k) + (size_t) (j * dimX + i2)];
    val3 = input[(size_t) (dimX * dimY * k) + (size_t) (j1 * dimX + i)];
    val4 = input[(size_t) (dimX * dimY * k) + (size_t) (j2 * dimX + i)];
    val5 = input[(size_t) (dimX * dimY * k1) + (size_t) (j * dimX + i)];
    val6 = input[(size_t) (dimX * dimY * k2) + (size_t) (j * dimX + i)];

    output[index] = 0.1428F * (input[index] + val1 + val2 + val3 + val4 + val5 + val6);
}

void
remove_inconsistent_stripes(const bool* mask, bool* out, int stripe_length_min,
                            int stripe_depth_min, float sensitivity, int switch_dim,
                            long i, long j, long k, long dimX, long dimY, long dimZ)
{
    int    counter_vert_voxels;
    int    counter_depth_voxels;
    int    halfstripe_length = stripe_length_min / 2;
    int    halfstripe_depth  = stripe_depth_min / 2;
    long   k_m;
    long   k1;
    long   y_m;
    long   y1;
    size_t index;
    index = (size_t) (dimX * dimY * k) + (size_t) (j * dimX + i);

    int threshold_vertical = (int) ((0.01F * sensitivity) * stripe_length_min);
    int threshold_depth    = (int) ((0.01F * sensitivity) * stripe_depth_min);

    /* start by considering vertical features */
    if(switch_dim == 0)
    {
        if(mask[index] == true)
        {
            counter_vert_voxels = 0;
            for(k_m = -halfstripe_length; k_m <= halfstripe_length; k_m++)
            {
                k1 = k + k_m;
                if((k1 < 0) || (k1 >= dimZ))
                    k1 = k - k_m;
                if(mask[(size_t) (dimX * dimY * k1) + (size_t) (j * dimX + i)] == true)
                    counter_vert_voxels++;
            }
            if(counter_vert_voxels < threshold_vertical)
                out[index] = false;
        }
    }
    else
    {
        /*
         Considering the depth of features an removing the deep ones
         Here we assume that the stripes do not normally extend far
         in the depth dimension compared to the features that belong to a
         sample.
        */
        if(mask[index] == true)
        {
            if(stripe_depth_min != 0)
            {
                counter_depth_voxels = 0;
                for(y_m = -halfstripe_depth; y_m <= halfstripe_depth; y_m++)
                {
                    y1 = j + y_m;
                    if((y1 < 0) || (y1 >= dimY))
                        y1 = j - y_m;
                    if(mask[(size_t) (dimX * dimY * k) + (size_t) (y1 * dimX + i)] ==
                       true)
                        counter_depth_voxels++;
                }
                if(counter_depth_voxels > threshold_depth)
                    out[index] = false;
            }
        }
    }
}

void
remove_short_stripes(const bool* mask, bool* out, int stripe_length_min, long i, long j,
                     long k, long dimX, long dimY, long dimZ)
{
    int    counter_vert_voxels;
    int    halfstripe_length = stripe_length_min / 2;
    long   k_m;
    long   k1;
    size_t index;
    index = (size_t) (dimX * dimY * k) + (size_t) (j * dimX + i);

    if(mask[index] == true)
    {
        counter_vert_voxels = 0;
        for(k_m = -halfstripe_length; k_m <= halfstripe_length; k_m++)
        {
            k1 = k + k_m;
            if((k1 < 0) || (k1 >= dimZ))
                k1 = k - k_m;
            if(mask[(size_t) (dimX * dimY * k1) + (size_t) (j * dimX + i)] == true)
                counter_vert_voxels++;
        }
        if(counter_vert_voxels < halfstripe_length)
            out[index] = false;
    }
}

void
merge_stripes(const bool* mask, bool* out, int stripe_length_min, int stripe_width_min,
              long i, long j, long k, long dimX, long dimY, long dimZ)
{
    int halfstripe_width = stripe_width_min / 2;
    int vertical_length  = 2 * stripe_width_min;

    long   x;
    long   x_l;
    long   x_r;
    long   k_u;
    long   k_d;
    int    mask_left;
    int    mask_right;
    int    mask_up;
    int    mask_down;
    size_t index;
    index = (size_t) (dimX * dimY * k) + (size_t) (j * dimX + i);

    if(mask[index] == false)
    {
        /* checking if there is a mask to the left of False */
        mask_left = 0;
        for(x = -halfstripe_width; x <= 0; x++)
        {
            x_l = i + x;
            if(x_l < 0)
                x_l = i - x;
            if(mask[(size_t) (dimX * dimY * k) + (size_t) (j * dimX + x_l)] == true)
            {
                mask_left = 1;
                break;
            }
        }
        /* checking if there is a mask to the right of False */
        mask_right = 0;
        for(x = 0; x <= halfstripe_width; x++)
        {
            x_r = i + x;
            if(x_r >= dimX)
                x_r = i - x;
            if(mask[(size_t) (dimX * dimY * k) + (size_t) (j * dimX + x_r)] == true)
            {
                mask_right = 1;
                break;
            }
        }
        /* now if there is a mask from the left and from the right side of True value make
         * it True */
        if((mask_left == 1) && (mask_right == 1))
            out[index] = true;

        /* perform vertical merging */
        if(out[index] == false)
        {
            /* checking if there is a mask up of True */
            mask_up = 0;
            for(x = -vertical_length; x <= 0; x++)
            {
                k_u = k + x;
                if(k_u < 0)
                    k_u = k - x;
                if(mask[(size_t) (dimX * dimY * k_u) + (size_t) (j * dimX + i)] == true)
                {
                    mask_up = 1;
                    break;
                }
            }
            /* checking if there is a mask down of False */
            mask_down = 0;
            for(x = 0; x <= vertical_length; x++)
            {
                k_d = k + x;
                if(k_d >= dimZ)
                    k_d = k - x;
                if(mask[(size_t) (dimX * dimY * k_d) + (size_t) (j * dimX + i)] == true)
                {
                    mask_down = 1;
                    break;
                }
            }
            /* now if there is a mask above and bellow of the False make it True */
            if((mask_up == 1) && (mask_down == 1))
                out[index] = true;
        }
    }
}

/********************************************************************/
/*************************stripesdetect3d****************************/
/********************************************************************/
DLL int
stripesdetect3d_main_float(float* Input, float* Output, int window_halflength_vertical,
                           int ratio_radius, int ncores, int dimX, int dimY, int dimZ)
{
    long      i;
    long      j;
    long      k;
    long long totalvoxels;
    totalvoxels = (long long) ((long) (dimX) * (long) (dimY) * (long) (dimZ));

    int window_fulllength   = (2 * window_halflength_vertical + 1);
    int midval_window_index = (int) (0.5F * window_fulllength) - 1;

    float* temp3d_arr;
    temp3d_arr = malloc(totalvoxels * sizeof(float));
    if(temp3d_arr == NULL)
    {
        printf("Memory allocation of the 'temp3d_arr' array in "
               "'stripesdetect3d_main_float' failed");
    }

    /* dealing here with a custom given number of cpu threads */
    if(ncores > 0)
    {
        // Explicitly disable dynamic teams
        omp_set_dynamic(0);
        // Use a number of threads for all consecutive parallel regions
        omp_set_num_threads(ncores);
    }

/* Perform a gentle (6-stencil) 3d mean smoothing of the data to ensure more stability in
 * the gradient calculation */
#pragma omp parallel for shared(temp3d_arr) private(i, j, k)
    for(k = 0; k < dimZ; k++)
    {
        for(j = 0; j < dimY; j++)
        {
            for(i = 0; i < dimX; i++)
            {
                mean_stride3d(Input, temp3d_arr, i, j, k, dimX, dimY, dimZ);
            }
        }
    }

    /* Take the gradient in the horizontal direction, axis = 0, step = 2*/
    gradient3D_local(Input, Output, dimX, dimY, dimZ, 0, 2);

    /*
    Here we calculate a ratio between the mean in a small 2D neighbourhood parallel to the
    stripe and the mean orthogonal to the stripe. The gradient variation in the direction
    orthogonal to the stripe is expected to be large (a jump), while in parallel direction
    small. Therefore at the edges of a stripe we should get a ratio small/large or
    large/small.
    */
#pragma omp parallel for shared(Output, temp3d_arr) private(i, j, k)
    for(k = 0; k < dimZ; k++)
    {
        for(j = 0; j < dimY; j++)
        {
            for(i = 0; i < dimX; i++)
            {
                ratio_mean_stride3d(Output, temp3d_arr, ratio_radius, i, j, k, dimX, dimY,
                                    dimZ);
            }
        }
    }

    /*
    We process the resulting ratio map with a vertical median filter which removes
    inconsistent from longer stripes features
    */
#pragma omp parallel for shared(temp3d_arr, Output) private(i, j, k)
    for(k = 0; k < dimZ; k++)
    {
        for(j = 0; j < dimY; j++)
        {
            for(i = 0; i < dimX; i++)
            {
                vertical_median_stride3d(temp3d_arr, Output, window_halflength_vertical,
                                         window_fulllength, midval_window_index, i, j, k,
                                         dimX, dimY, dimZ);
            }
        }
    }

    free(temp3d_arr);
    return 0;
}

/********************************************************************/
/*************************stripesmask3d******************************/
/********************************************************************/
DLL int
stripesmask3d_main_float(float* Input, bool* Output, float threshold_val,
                         int stripe_length_min, int stripe_depth_min,
                         int stripe_width_min, float sensitivity, int ncores, int dimX,
                         int dimY, int dimZ)
{
    long   i;
    long   j;
    long   k;
    int    iter_merge;
    int    switch_dim;
    size_t index;
    size_t totalvoxels = (long) (dimX) * (long) (dimY) * (long) (dimZ);

    bool* mask;
    mask = malloc(totalvoxels * sizeof(bool));
    if(mask == NULL)
    {
        printf(
            "Memory allocation of the 'mask' array in 'stripesmask3d_main_float' failed");
    }

    /* dealing here with a custom given number of cpu threads */
    if(ncores > 0)
    {
        // Explicitly disable dynamic teams
        omp_set_dynamic(0);
        // Use a number of threads for all consecutive parallel regions
        omp_set_num_threads(ncores);
    }

    /*
    First step is to mask all the values in the given weights input image
    that are bellow a given "threshold_val" parameter
    */
#pragma omp parallel for shared(Input, mask) private(i, j, k, index)
    for(k = 0; k < dimZ; k++)
    {
        for(j = 0; j < dimY; j++)
        {
            for(i = 0; i < dimX; i++)
            {
                index = (size_t) (dimX * dimY * k) + (size_t) (j * dimX + i);
                if(Input[index] <= threshold_val)
                {
                    mask[index] = true;
                }
                else
                {
                    mask[index] = false;
                }
            }
        }
    }

    /* Copy mask to output */
    memcpy(Output, mask, totalvoxels * sizeof(bool));

    /* the depth consistency for features  */
    switch_dim = 1;
#pragma omp parallel for shared(mask, Output) private(i, j, k)
    for(k = 0; k < dimZ; k++)
    {
        for(j = 0; j < dimY; j++)
        {
            for(i = 0; i < dimX; i++)
            {
                remove_inconsistent_stripes(mask, Output, stripe_length_min,
                                            stripe_depth_min, sensitivity, switch_dim, i,
                                            j, k, dimX, dimY, dimZ);
            }
        }
    }
    /* Copy output to mask */
    memcpy(mask, Output, totalvoxels * sizeof(bool));

    /*
    Now we need to remove stripes that are shorter than "stripe_length_min" parameter
    or inconsistent otherwise. For every pixel we will run a 1D vertical window to count
    nonzero values in the mask. We also check for the depth of the mask's value,
    assuming that the stripes are normally shorter in depth compare to the features that
    belong to true data.
    */

    /*continue by including longer vertical features and discarding the shorter ones */
    switch_dim = 0;
#pragma omp parallel for shared(mask, Output) private(i, j, k)
    for(k = 0; k < dimZ; k++)
    {
        for(j = 0; j < dimY; j++)
        {
            for(i = 0; i < dimX; i++)
            {
                remove_inconsistent_stripes(mask, Output, stripe_length_min,
                                            stripe_depth_min, sensitivity, switch_dim, i,
                                            j, k, dimX, dimY, dimZ);
            }
        }
    }
    /* Copy output to mask */
    memcpy(mask, Output, totalvoxels * sizeof(bool));

    /* now we clean the obtained mask if the features do not hold our assumptions about
     * the lengths */

#pragma omp parallel for shared(mask, Output) private(i, j, k)
    for(k = 0; k < dimZ; k++)
    {
        for(j = 0; j < dimY; j++)
        {
            for(i = 0; i < dimX; i++)
            {
                remove_short_stripes(mask, Output, stripe_length_min, i, j, k, dimX, dimY,
                                     dimZ);
            }
        }
    }

    /* Copy output to mask */
    memcpy(mask, Output, totalvoxels * sizeof(bool));

    /*
    We can merge stripes together if they are relatively close to each other
    horizontally and vertically. We do that iteratively.
    */
    for(iter_merge = 0; iter_merge < stripe_width_min; iter_merge++)
    {
#pragma omp parallel for shared(mask, Output) private(i, j, k)
        for(k = 0; k < dimZ; k++)
        {
            for(j = 0; j < dimY; j++)
            {
                for(i = 0; i < dimX; i++)
                {
                    merge_stripes(mask, Output, stripe_length_min, stripe_width_min, i, j,
                                  k, dimX, dimY, dimZ);
                }
            }
        }
        /* Copy output to mask */
        memcpy(mask, Output, totalvoxels * sizeof(bool));
    }

    free(mask);
    return 0;
}
