#include <stdio.h>
#include <math.h>


void downsample2d(float* data, int num_projections,
                  int num_slices, int num_pixels,
                  int level, float* downsampled_data) {

    int m, n, k, i, p, q, iproj, ind;
    int binsize;
    
    binsize = pow(2, level);
    
    num_pixels /= binsize;

    for (m = 0, ind = 0; m < num_projections; m++) {
        iproj = m * (num_pixels * num_slices);
        
	for (n = 0; n < num_slices; n++) {
	    i = iproj + n * num_pixels;
	    
            for (k = 0; k < num_pixels; k++) {
                        
                for (p = 0; p < binsize; p++) {
                    downsampled_data[i+k] += data[ind]/binsize;
                    ind++;
                }
            }
        }
    }
}

void downsample3d(float* data, int num_projections,
                  int num_slices, int num_pixels,
                  int level, float* downsampled_data) {

    int m, n, k, i, p, q, iproj, ind;
    int binsize, binsize2;
    
    binsize = pow(2, level);
    binsize2 = binsize * binsize;
    
    num_slices /= binsize;
    num_pixels /= binsize;

    for (m = 0, ind = 0; m < num_projections; m++) {
        iproj = m * (num_pixels * num_slices);
        
	for (n = 0; n < num_slices; n++) {
	    i = iproj + n * num_pixels;
			
	    for (q = 0; q < binsize; q++) {

		for (k = 0; k < num_pixels; k++) {
		            
		    for (p = 0; p < binsize; p++) {
			downsampled_data[i+k] += data[ind]/binsize2;
			ind++;
		    }
		}
            }
        }
    }
}