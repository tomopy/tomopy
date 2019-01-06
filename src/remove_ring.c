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

// Original author: Justin Blair

#include "remove_ring.h"

#define INT_MODE_WRAP 0
#define INT_MODE_REFLECT 1

void
remove_ring(float* data, float center_x, float center_y, int dx, int dy, int dz,
            float thresh_max, float thresh_min, float threshold, int angular_min,
            int ring_width, int int_mode, int istart, int iend)
{
    int     pol_width  = 0;
    int     pol_height = 0;
    int     m_rad      = 30;
    int     r_scale    = 1;
    int     ang_scale  = 1;
    int     m_azi;
    float** polar_image = 0;
    float** ring_image  = 0;
    float** image       = (float**) calloc(dy, sizeof(float*));

    // For each reconstructed slice
    for(int s = istart; s < iend; s++)
    {
        // Fill in reconstructed slice data array into reshaped 2D array
        image[0] = data + s * dy * dx;
        for(int i = 1; i < dy; i++)
        {
            image[i] = image[i - 1] + dx;
        }
        // Translate Image to Polar Coordinates
        polar_image =
            polar_transform(image, center_x, center_y, dx, dy, &pol_width, &pol_height,
                            thresh_max, thresh_min, r_scale, ang_scale, ring_width);
        m_azi = ceil((float) pol_height / 360.0) * angular_min;
        m_rad = 2 * ring_width + 1;

        // Call Ring Algorithm
        ring_filter(&polar_image, pol_height, pol_width, threshold, m_rad, m_azi,
                    ring_width, int_mode);

        // Translate Ring-Image to Cartesian Coordinates
        ring_image = inverse_polar_transform(polar_image, center_x, center_y, pol_width,
                                             pol_height, dx, dy, r_scale, ring_width);

        // Subtract Ring-Image from Image
        for(int row = 0; row < dy; row++)
        {
            for(int col = 0; col < dx; col++)
            {
                image[row][col] -= ring_image[row][col];
            }
        }

        // Flatten 2D filtered array
        for(int j = 0; j < dy; j++)
        {
            for(int i = 0; i < dx; i++)
            {
                data[i + (j * dx) + s * dy * dx] = image[j][i];
            }
        }

        free(polar_image[0]);
        free(polar_image);

        free(ring_image[0]);
        free(ring_image);
    }

    free(image);

    return;
}

int
min_distance_to_edge(float center_x, float center_y, int width, int height)
{
    int* dist = calloc(4, sizeof(int));
    dist[0]   = center_x + 1;
    dist[1]   = center_y + 1;
    dist[2]   = width - center_x;
    dist[3]   = height - center_y;
    int min   = dist[0];
    for(int i = 1; i < 4; i++)
    {
        if(min > dist[i])
        {
            min = dist[i];
        }
    }
    free(dist);
    return min;
}

int
iroundf(float x)
{
    return (x != 0.0) ? floor(x + 0.5) : 0;
}

float**
polar_transform(float** image, float center_x, float center_y, int width, int height,
                int* p_pol_width, int* p_pol_height, float thresh_max, float thresh_min,
                int r_scale, int ang_scale, int overhang)
{
    int max_r      = min_distance_to_edge(center_x, center_y, width, height);
    int pol_width  = r_scale * max_r;
    int pol_height = iroundf((float) ang_scale * 2.0 * PI * (float) max_r);
    *p_pol_width   = pol_width;
    *p_pol_height  = pol_height;

    float*  image_block = (float*) calloc(pol_height * pol_width, sizeof(float));
    float** polar_image = (float**) calloc(pol_height, sizeof(float*));
    polar_image[0]      = image_block;
    for(int i = 1; i < pol_height; i++)
    {
        polar_image[i] = polar_image[i - 1] + pol_width;
    }
    for(int row = 0; row < pol_height; row++)
    {
        for(int r = 0; r <= pol_width - r_scale; r++)
        {
            float theta = (float) row * 2.0 * PI / (float) pol_height;
            float fl_x =
                (float) r * cos(theta + (PI / (float) pol_height)) / (float) r_scale;
            float fl_y =
                (float) r * sin(theta + (PI / (float) pol_height)) / (float) r_scale;
            int x = iroundf(fl_x + center_x);
            int y = iroundf(fl_y + center_y);

            polar_image[row][r] = image[y][x];
            if(polar_image[row][r] > thresh_max)
            {
                polar_image[row][r] = thresh_max;
            }
            else if(polar_image[row][r] < thresh_min)
            {
                polar_image[row][r] = thresh_min;
            }
        }
    }

    return polar_image;
}

float**
inverse_polar_transform(float** polar_image, float center_x, float center_y,
                        int pol_width, int pol_height, int width, int height, int r_scale,
                        int over_hang)
{
    float*  image_block = (float*) calloc(height * width, sizeof(float));
    float** cart_image  = (float**) calloc(height, sizeof(float*));
    cart_image[0]       = image_block;
    for(int i = 1; i < height; i++)
    {
        cart_image[i] = cart_image[i - 1] + width;
    }
    for(int row = 0; row < height; row++)
    {
        for(int col = 0; col < width; col++)
        {
            float theta = atan2((float) (row - center_y),
                                (float) (col - center_x) - (PI / (float) pol_height));
            if(theta < 0)
            {
                theta += 2.0 * PI;
            }
            int pol_row = iroundf(theta * (float) pol_height / (2.0 * PI));
            int pol_col =
                iroundf((float) r_scale *
                        sqrtf(((float) row - center_y) * ((float) row - center_y) +
                              ((float) col - center_x) * ((float) col - center_x)));
            if(pol_row < pol_height && pol_col < pol_width)
            {
                cart_image[row][col] = polar_image[pol_row][pol_col];
            }
            else
            {
                cart_image[row][col] = 0.0;
            }
        }
    }
    return cart_image;
}

void
swap_float(float* arr, int index1, int index2)
{
    float store_value = arr[index1];
    arr[index1]       = arr[index2];
    arr[index2]       = store_value;
    return;
}

void
swap_integer(int* arr, int index1, int index2)
{
    int store_value = arr[index1];
    arr[index1]     = arr[index2];
    arr[index2]     = store_value;
    return;
}

int
partition(float* median_array, int left, int right, int pivot_index)
{
    float pivot_value = median_array[pivot_index];
    swap_float(median_array, pivot_index, right);
    int store_index = left;
    for(int i = left; i < right; i++)
    {
        if(median_array[i] <= pivot_value)
        {
            swap_float(median_array, i, store_index);
            store_index += 1;
        }
    }
    swap_float(median_array, store_index, right);
    return store_index;
}

int
partition_2_arrays(float* median_array, int* position_array, int left, int right,
                   int pivot_index)
{
    float pivot_value = median_array[pivot_index];
    swap_float(median_array, pivot_index, right);
    swap_integer(position_array, pivot_index, right);
    int store_index = left;
    for(int i = left; i < right; i++)
    {
        if(median_array[i] <= pivot_value)
        {
            swap_float(median_array, i, store_index);
            swap_integer(position_array, i, store_index);
            store_index += 1;
        }
    }
    swap_float(median_array, store_index, right);
    swap_integer(position_array, store_index, right);
    return store_index;
}

void
quick_sort(float* median_array, int left, int right)
{
    if(left < right)
    {
        int pivot_index     = (int) ((left + right) / 2);
        int new_pivot_index = partition(median_array, left, right, pivot_index);
        quick_sort(median_array, left, new_pivot_index - 1);
        quick_sort(median_array, new_pivot_index + 1, right);
    }
}

void
quick_sort_2_arrays(float* median_array, int* position_array, int left, int right)
{
    if(left < right)
    {
        int pivot_index = (int) ((left + right) / 2);
        int new_pivot_index =
            partition_2_arrays(median_array, position_array, left, right, pivot_index);
        quick_sort_2_arrays(median_array, position_array, left, new_pivot_index - 1);
        quick_sort_2_arrays(median_array, position_array, new_pivot_index + 1, right);
    }
    return;
}

void
bubble_2_arrays(float* median_array, int* position_array, int index, int length)
{
    if(index > 0 && index < length - 1)
    {
        if(median_array[index] < median_array[index - 1])
        {
            swap_float(median_array, index, index - 1);
            swap_integer(position_array, index, index - 1);
            bubble_2_arrays(median_array, position_array, index - 1, length);
        }
        else if(median_array[index] > median_array[index + 1])
        {
            swap_float(median_array, index, index + 1);
            swap_integer(position_array, index, index + 1);
            bubble_2_arrays(median_array, position_array, index + 1, length);
        }
    }
    else if(index == 0)
    {
        if(median_array[index] > median_array[index + 1])
        {
            swap_float(median_array, index, index + 1);
            swap_integer(position_array, index, index + 1);
            bubble_2_arrays(median_array, position_array, index + 1, length);
        }
    }
    else if(index == length - 1)
    {
        if(median_array[index] < median_array[index - 1])
        {
            swap_float(median_array, index, index - 1);
            swap_integer(position_array, index, index - 1);
            bubble_2_arrays(median_array, position_array, index - 1, length);
        }
    }
    return;
}

void
median_filter_fast_1D(float*** filtered_image, float*** image, int start_row,
                      int start_col, int end_row, int end_col, char axis, int kernel_rad,
                      int filter_width, int width, int height)
{
    int    row, col;
    float* median_array   = (float*) calloc(2 * kernel_rad + 1, sizeof(float));
    int*   position_array = (int*) calloc(2 * kernel_rad + 1, sizeof(int));
    if(axis == 'x')
    {
        for(row = start_row; row <= end_row; row++)
        {
            col = start_col;
            for(int n = -kernel_rad; n < kernel_rad + 1; n++)
            {
                int adjusted_col = col + n;
                int adjusted_row = row;
                if(adjusted_col < 0)
                {
                    adjusted_col = -adjusted_col;
                    if(row < height / 2)
                    {
                        adjusted_row += height / 2;
                    }
                    else
                    {
                        adjusted_row -= height / 2;
                    }
                    median_array[n + kernel_rad] = image[0][adjusted_row][adjusted_col];
                }
                else
                {
                    median_array[n + kernel_rad] = image[0][row][adjusted_col];
                }
                position_array[n + kernel_rad] = n + kernel_rad;
            }
            // Sort the array
            quick_sort_2_arrays(median_array, position_array, 0, 2 * kernel_rad);
            filtered_image[0][row][col] = median_array[kernel_rad];

            // Roll filter along the rest of the row
            for(col = start_col + 1; col <= end_col; col++)
            {
                float next_value     = 0.0;
                int   next_value_col = col + kernel_rad;
                if(next_value_col < width)
                {
                    next_value = image[0][row][next_value_col];
                }
                int last_value_index = 0;
                for(int i = 0; i < 2 * kernel_rad + 1; i++)
                {
                    position_array[i] -= 1;
                    if(position_array[i] < 0)
                    {
                        last_value_index  = i;
                        position_array[i] = 2 * kernel_rad;
                        median_array[i]   = next_value;
                    }
                }
                bubble_2_arrays(median_array, position_array, last_value_index,
                                2 * kernel_rad + 1);
                filtered_image[0][row][col] = median_array[kernel_rad];
            }
        }
    }
    else if(axis == 'y')
    {
        for(col = start_col; col <= end_col; col++)
        {
            row = start_row;
            for(int n = -kernel_rad; n < kernel_rad + 1; n++)
            {
                int adjusted_row = row + n;
                int adjusted_col = col;
                if(adjusted_row < 0)
                {
                    // Handle edge cases
                    adjusted_row += height;
                    median_array[n + kernel_rad] = image[0][adjusted_row][adjusted_col];
                }
                else
                {
                    median_array[n + kernel_rad] = image[0][adjusted_row][adjusted_col];
                }
                position_array[n + kernel_rad] = n + kernel_rad;
            }
            // Sort the array
            quick_sort_2_arrays(median_array, position_array, 0, 2 * kernel_rad);
            filtered_image[0][row][col] = median_array[kernel_rad];

            // Roll filter along the rest of the col
            for(row = start_row + 1; row <= end_row; row++)
            {
                float next_value     = 0.0;
                int   next_value_row = row + kernel_rad;
                if(next_value_row < height)
                {
                    next_value = image[0][next_value_row][col];
                }
                int last_value_index = 0;
                for(int i = 0; i < 2 * kernel_rad + 1; i++)
                {
                    position_array[i] -= 1;
                    if(position_array[i] < 0)
                    {
                        last_value_index  = i;
                        position_array[i] = 2 * kernel_rad;
                        median_array[i]   = next_value;
                    }
                }
                bubble_2_arrays(median_array, position_array, last_value_index,
                                2 * kernel_rad + 1);
                filtered_image[0][row][col] = median_array[kernel_rad];
            }
        }
    }
    free(median_array);
    free(position_array);
    return;
}

/* Runs slightly faster than the above mean filter, but floating-point rounding
 * causes errors on the order of 1E-10. Should be small enough error to not care
 * about, but be careful...
 */
void
mean_filter_fast_1D(float*** filtered_image, float*** image, int start_row, int start_col,
                    int end_row, int end_col, int int_mode, int kernel_rad, int width,
                    int height)
{
    long double mean = 0, sum = 0, previous_sum = 0,
                num_elems = (double) (2 * kernel_rad + 1);
    int row, col;
    if(int_mode == INT_MODE_WRAP)
    {
        // iterate over each row of the image subset
        for(col = start_col; col <= end_col; col++)
        {
            sum = 0;
            // calculate average of first element of the column
            for(int n = -kernel_rad; n < (kernel_rad + 1); n++)
            {
                row = n + start_row;
                if(row < 0)
                {
                    row += height;
                }
                else if(row >= height)
                {
                    row -= height;
                }
                sum += image[0][row][col];
            }
            mean                              = sum / num_elems;
            filtered_image[0][start_row][col] = mean;
            previous_sum                      = sum;

            for(row = start_row + 1; row <= end_row; row++)
            {
                int last_row = (row - 1) - (kernel_rad);
                int next_row = row + (kernel_rad);
                if(last_row < 0)
                {
                    last_row += height;
                }
                if(next_row >= height)
                {
                    next_row -= height;
                }
                sum = previous_sum - image[0][last_row][col] + image[0][next_row][col];
                if(image[0][row][col] != 0)
                {
                    filtered_image[0][row][col] = sum / num_elems;
                }
                else
                {
                    filtered_image[0][row][col] = 0.0;
                }
                previous_sum = sum;
            }
        }
    }
    else if(int_mode == INT_MODE_REFLECT)
    {
        // iterate over each column of the image subset
        for(col = start_col; col <= end_col; col++)
        {
            sum = 0;
            // calculate average of first element of the column
            for(int n = -kernel_rad; n < (kernel_rad + 1); n++)
            {
                row = n;
                if(row < 0)
                {
                    row = -row;
                }
                else if(row >= height / 2)
                {
                    row = height / 2 - (row - height / 2) - 2;
                }
                sum += image[0][row][col];
            }
            mean                      = sum / num_elems;
            filtered_image[0][0][col] = mean;
            previous_sum              = sum;

            for(row = 1; row < height / 2; row++)
            {
                int last_row = (row - 1) - (kernel_rad);
                int next_row = row + (kernel_rad);
                if(last_row < 0)
                {
                    last_row = -last_row;
                }
                if(next_row >= height / 2)
                {
                    next_row = height / 2 - (next_row - height / 2) - 2;
                }
                sum = previous_sum - image[0][last_row][col] + image[0][next_row][col];
                if(image[0][row][col] != 0)
                {
                    filtered_image[0][row][col] = sum / num_elems;
                }
                else
                {
                    filtered_image[0][row][col] = 0.0;
                }
                previous_sum = sum;
            }

            sum = 0;
            // calculate average of first element of the column
            for(int n = -kernel_rad; n < (kernel_rad + 1); n++)
            {
                row = n + height / 2;
                if(row < height / 2)
                {
                    row = height / 2 + (height / 2 - row);
                }
                else if(row >= height)
                {
                    row = height - (row - height) - 2;
                }
                sum += image[0][row][col];
            }
            mean                               = sum / num_elems;
            filtered_image[0][height / 2][col] = mean;
            previous_sum                       = sum;

            for(row = height / 2 + 1; row < height; row++)
            {
                int last_row = (row - 1) - (kernel_rad);
                int next_row = row + (kernel_rad);
                if(last_row < height / 2)
                {
                    last_row = height / 2 + (height / 2 - last_row);
                }
                if(next_row >= height)
                {
                    next_row = height - (next_row - height) - 2;
                }
                sum = previous_sum - image[0][last_row][col] + image[0][next_row][col];
                if(image[0][row][col] != 0)
                {
                    filtered_image[0][row][col] = sum / num_elems;
                }
                else
                {
                    filtered_image[0][row][col] = 0.0;
                }
                previous_sum = sum;
            }
        }
    }
    return;
}

void
ring_filter(float*** polar_image, int pol_height, int pol_width, float threshold,
            int m_rad, int m_azi, int ring_width, int int_mode)
{
    float*  image_block    = (float*) calloc(pol_height * pol_width, sizeof(float));
    float** filtered_image = (float**) calloc(pol_height, sizeof(float*));
    filtered_image[0]      = image_block;
    for(int i = 1; i < pol_height; i++)
    {
        filtered_image[i] = filtered_image[i - 1] + pol_width;
    }

    median_filter_fast_1D(&filtered_image, polar_image, 0, 0, pol_height - 1,
                          pol_width / 3 - 1, 'x', m_rad / 3, ring_width, pol_width,
                          pol_height);
    median_filter_fast_1D(&filtered_image, polar_image, 0, pol_width / 3, pol_height - 1,
                          2 * pol_width / 3 - 1, 'x', 2 * m_rad / 3, ring_width,
                          pol_width, pol_height);
    median_filter_fast_1D(&filtered_image, polar_image, 0, 2 * pol_width / 3,
                          pol_height - 1, pol_width - 1, 'x', m_rad, ring_width,
                          pol_width, pol_height);

    // subtract filtered image from polar image to get difference image & do
    // last thresholding

    for(int row = 0; row < pol_height; row++)
    {
        for(int col = 0; col < pol_width; col++)
        {
            polar_image[0][row][col] -= filtered_image[row][col];
            if(polar_image[0][row][col] > threshold ||
               polar_image[0][row][col] < -threshold)
            {
                polar_image[0][row][col] = 0;
            }
        }
    }

    /* Do Azimuthal filter #2 (faster mean, does whole column in one call)
     * using different kernel sizes for the different regions of the image
     * (based on radius)
     */

    mean_filter_fast_1D(&filtered_image, polar_image, 0, 0, pol_height - 1,
                        pol_width / 3 - 1, int_mode, m_azi / 3, pol_width, pol_height);
    mean_filter_fast_1D(&filtered_image, polar_image, 0, pol_width / 3, pol_height - 1,
                        2 * pol_width / 3 - 1, int_mode, 2 * m_azi / 3, pol_width,
                        pol_height);
    mean_filter_fast_1D(&filtered_image, polar_image, 0, 2 * pol_width / 3,
                        pol_height - 1, pol_width - 1, int_mode, m_azi, pol_width,
                        pol_height);

    // Set "polar_image" to the fully filtered data
    for(int row = 0; row < pol_height; row++)
    {
        for(int col = 0; col < pol_width; col++)
        {
            polar_image[0][row][col] = filtered_image[row][col];
        }
    }

    free(filtered_image[0]);
    free(filtered_image);
    return;
}
