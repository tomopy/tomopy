#include <iostream>
#include <math.h>

const double PI = 3.141592653589793238462;

class Art
{

public:
    Art(int *num_projections, int *num_slices, int *num_pixels, int *num_grid, float *data);
    void reconstruct(int *iters, float *center, float *theta, float *recon);
    
private:
    int num_projections_;
    int num_slices_;
    int num_pixels_;
    int num_grid_;
    float *data_;
    float *gridx_, *gridy_;
    float *coordx, *coordy;
    float *ax, *ay, *bx, *by;
    float *coorx, *coory;
    float *leng, *leng2;
    float diffx, diffy;
    float midx, midy;
    int *indi;
};