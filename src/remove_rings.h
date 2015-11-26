#ifndef _remove_rings_h
#define _remove_rings_h

#include <math.h> 
#include <stdlib.h>
#include <stdio.h>

#ifdef WIN32
#define DLL_declspec(dllexport)
#else
#define DLL
#endif

#define PI 3.14159265359
#pragma once

void DLL
remove_rings(
		float* data,
		float center_x, 
		float center_y,
		int dx,
		int dy,
		int dz,
		float thresh_max,
		float thresh_min,
		float threshold, 
		int angular_min,
		int ring_width,
		int istart,
		int iend);

int
min_distance_to_edge(
	float center_x, float center_y,
	int width, int height);

int
iroundf(float x);

float**
polar_transform(
		float** image, float center_x, float center_y,
		int width, int height, int* p_pol_width,
		int* p_pol_height, float thresh_max, float thresh_min,
		int r_scale, int ang_scale, int overhang);

float**
polar_transform_bilinear(
	float** image, float center_x, float center_y, int width,
	int height, int* p_pol_width, int* p_pol_height, float thresh_max,
    float thresh_min, int r_scale, int ang_scale, int overhang);

float**
inverse_polar_transform(
	float** polar_image, float center_x, float center_y, int pol_width,
	int  pol_height, int width, int height, int r_scale, int over_hang);

float**
inverse_polar_transform_bilinear(
	float** polar_image, float center_x, float center_y, int pol_width,
    int pol_height, int width, int height, int r_scale, int overhang);

void
swap_float(
	float* arr, int index1, int index2);

void
swap_integer(
	int* arr, int index1, int index2);

int
partition(
	float* median_array, int left, int right, int pivot_index);

int
partition_2_arrays(
	float* median_array, int* position_array, int left, int right,
	int pivot_index);

void
quick_sort(
	float* median_array, int left, int right);

void
quick_sort_2_arrays(
	float* median_array, int* position_array, int left, int right);

void
bubble_2_arrays(
	float* median_array, int* position_array, int index, int length);

void
median_filter_fast_1D(
	float *** filtered_image, float*** image, int start_row,
	int start_col, int end_row, int end_col, char axis, 
	int kernel_rad,	int filter_width, int width, int height);

void
median_filter_1D(
	float *** filtered_image, float*** image, int start_row,
	int start_col, int end_row, int end_col, char axis, 
	int kernel_rad,	int filter_width, int width, int height);

void
mean_filter_fast_1D(
	float*** filtered_image, float*** image,
 	int start_row, int start_col, int end_row, int end_col,
	char axis, int kernel_rad, int width, int height);

void
mean_filter_1D(
	float*** filtered_image, float*** image, int start_row,
	int start_col, int end_row, int end_col, char axis, 
	int kernel_rad, int height, int width);

void
ring_filter(
	float*** polar_image, int pol_height, int pol_width,
	float threshold, int m_rad, int m_azi, int ring_width);

#endif