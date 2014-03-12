#include <stdio.h>
#include <math.h>


void apply_padding(float* data, int num_projections, 
                   int num_slices, int num_pixels, 
                   int num_pad, float* padded_data) {
    
    int n, m, i, j, k, iproj, ipproj;
    int pad_width = (int)(num_pad-num_pixels)/2;
    
    for (m = 0; m < num_projections; m++) {
        iproj = m * (num_pixels * num_slices);
        ipproj = pad_width + m * (num_pad * num_slices);

        for (n = 0; n < num_slices; n++) {
            i = iproj + n * num_pixels;
            j = ipproj + n * num_pad;

            for (k = 0; k < num_pixels; k++) {
                padded_data[j+k] = data[i+k];
            }
        }
    }
}
