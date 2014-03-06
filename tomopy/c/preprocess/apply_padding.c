#include <stdio.h>
#include <math.h>


void apply_padding(float* data, int* num_projections, 
                  int* num_slices, int* num_pixels, 
                  int* num_pad, float* padded_data) {
    
    int n, m, i, j, jp, iproj;
    int pad_width = (int)(*num_pad-*num_pixels)/2;
    
    for (m = 0; m < *num_projections; m++) {
        iproj = m * (*num_pixels * *num_slices);

        for (n = 0; n < *num_slices; n++) {
            i = iproj + n * *num_pixels;

            for (j = 0; j < *num_pixels; j++) {
                jp = pad_width + iproj + (n * *num_pad);
                padded_data[j+jp] = data[i+j];
            }
        }
    }
}