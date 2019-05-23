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

// Module for ring removal in reconstructed domain

#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef WIN32
#    define DLL __declspec(dllexport)
#else
#    define DLL
#endif

#define PI 3.14159265359

DLL void
remove_ring(float* data, float center_x, float center_y, int dx, int dy, int dz,
            float thresh_max, float thresh_min, float threshold, int angular_min,
            int ring_width, int int_mode, int istart, int iend);

int
min_distance_to_edge(float center_x, float center_y, int width, int height);

int
iroundf(float x);

float**
polar_transform(float** image, float center_x, float center_y, int width, int height,
                int* p_pol_width, int* p_pol_height, float thresh_max, float thresh_min,
                int r_scale, int ang_scale, int overhang);

float**
inverse_polar_transform(float** polar_image, float center_x, float center_y,
                        int pol_width, int pol_height, int width, int height, int r_scale,
                        int over_hang);

void
swap_float(float* arr, int index1, int index2);

void
swap_integer(int* arr, int index1, int index2);

int
partition(float* median_array, int left, int right, int pivot_index);

int
partition_2_arrays(float* median_array, int* position_array, int left, int right,
                   int pivot_index);

void
quick_sort(float* median_array, int left, int right);

void
quick_sort_2_arrays(float* median_array, int* position_array, int left, int right);

void
bubble_2_arrays(float* median_array, int* position_array, int index, int length);

void
median_filter_fast_1D(float*** filtered_image, float*** image, int start_row,
                      int start_col, int end_row, int end_col, char axis, int kernel_rad,
                      int filter_width, int width, int height);

void
mean_filter_fast_1D(float*** filtered_image, float*** image, int start_row, int start_col,
                    int end_row, int end_col, int int_mode, int kernel_rad, int width,
                    int height);

void
ring_filter(float*** polar_image, int pol_height, int pol_width, float threshold,
            int m_rad, int m_azi, int ring_width, int int_mode);
