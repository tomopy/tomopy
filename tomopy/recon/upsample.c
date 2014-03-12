#include <stdio.h>
#include <math.h>


void upsample2d(float* data, int num_slices, int num_pixels,
                int level, float* upsampled_data) {

    long m, n, k, i, p, q, iproj, ind;
    int binsize;
    
    binsize = pow(2, level);
    
    for (m = 0, ind = 0; m < num_slices; m++) {
        iproj = m * (num_pixels * num_pixels);

        for (n = 0; n < num_pixels; n++) {
            i = iproj + n * num_pixels;

            for (q = 0; q < binsize; q++) {

                for (k = 0; k < num_pixels; k++) {
				
                    for (p = 0; p < binsize; p++) {
                        upsampled_data[ind] = data[i+k];
                        ind++;
                    }
                }
            }
        }
    }
}


void upsample3d(float* data, int num_slices, int num_pixels,
                int level, float* upsampled_data) {

    int m, n, k, i, p, q, j, iproj, ind;
    int binsize;

    binsize = pow(2, level);

    for (m = 0, ind = 0; m < num_slices; m++) {
        iproj = m * (num_pixels * num_pixels);
    
        for (j = 0; j < binsize; j++) {

            for (n = 0; n < num_pixels; n++) {
                i = iproj + n * num_pixels;

                for (q = 0; q < binsize; q++) {

                    for (k = 0; k < num_pixels; k++) {
				
                        for (p = 0; p < binsize; p++) {
                            upsampled_data[ind] = data[i+k];
                            ind++;
                        }
                    }
                }
            }
        }
    }
}