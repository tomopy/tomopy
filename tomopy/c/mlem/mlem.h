#include <iostream>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <iterator>
#include <cstdlib>

#define CEIL(VARIABLE) ( (VARIABLE - (int)VARIABLE)<0 ? (int)VARIABLE : (int)VARIABLE+1 )
#define FLOOR(VARIABLE) ( (VARIABLE - (int)VARIABLE)>0 ? (int)VARIABLE : (int)VARIABLE-1 )

const double PI = 3.141592653589793238462;

class Mlem
{

public:
    Mlem(int *num_projections, int *num_slices, int *num_pixels, float *data);
    void reconstruct(int *iters, float *center, float *theta, float *recon);
    
private:
    int num_projections_;
    int num_slices_;
    int num_pixels_;
    float *data_;
    float *center_;
    float *theta_;
    float *gridx_, *gridy_, *gridz_;
    float *coordx, *coordy;
    float *ax, *ay, *bx, *by;
    float *coorx, *coory;
    float *leng;
    float diffx, diffy;
    float midx, midy;
    int *indx_, *indy_;
    int *indi;
    float *suma, *sumay;
};