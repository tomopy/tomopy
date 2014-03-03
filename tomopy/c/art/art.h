#include <iostream>
#include <math.h>

const double PI = 3.141592653589793238462;

class Art
{
    
public:
    Art(float *data, float *theta, float *center, 
        int *num_projections, int *num_slices, int *num_pixels, 
        int *num_grid, int *num_air);
        
    ~Art();

    void reconstruct(float *recon, int *iters, int *slices_start, int *slices_end);
    
private:
    // Constructor inputs.
    float *data_; // Tomographic data. [projections, slices, pixels]
    float *theta_; // Projection angles in radians.
    float center_; // Position of rotation axis on pixels axis.
    int num_projections_; // Number of projections.
    int num_slices_; // Number of slices.
    int num_pixels_; // Number of pixels.
    int num_grid_; // Grid size of the reconstructed slices.
    int num_air_; // Number of air (edge) pixels for correcting sinogram.
    
    // Constructor parameters derived from input.
    int padded_data_size; // Data size after padding.
    float *padded_data; // Padded data.
    int pad_size; // Size of the padding at one side of data.
    float *air; // Data correction based on the data border values.
    float *gridx, *gridy; // Reconstruction grid axes.
    float *coordx, *coordy;
    float *ax, *ay, *bx, *by;
    float *coorx, *coory;
    float *leng, *leng2;
    float diffx, diffy;
    float midx, midy;
    int *indx, *indy;
    int *indi;
};