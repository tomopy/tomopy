#include <stdio.h>


void correct_drift(float* data, int num_projections, 
                   int num_slices, int num_pixels, int air_pixels) {
                       
    int n, m, i, j, iproj;
    double air_left, air_right, air_slope, air;

    for (m = 0; m < num_projections; m++) {
        iproj = m * (num_pixels * num_slices);
            
        for (n = 0; n < num_slices; n++) {
            i = iproj + n * num_pixels;

            for (j = 0, air_left = 0, air_right = 0; j < air_pixels; j++) {
                air_left += data[i+j];
                air_right += data[i+num_pixels-1-j];
            }
            
            air_left /= (float)air_pixels;
            air_right /= (float)air_pixels;
            
            if (air_left <= 0.) {
                air_left = 1.;
            }
            if (air_right <= 0.) {
                air_right = 1.;
            }
            
            air_slope = (air_right - air_left) / (num_pixels - 1);

            for (j = 0; j < num_pixels; j++) {
                air = air_left + air_slope*j;
                data[i+j] = data[i+j] / air;
            }
        }
    }
}








