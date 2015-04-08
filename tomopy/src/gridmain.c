#include "gridrec.h"

void test_gridrec(
    float *data, float* theta, float center,
    int num_projs, int num_slices, int num_pixels,
    float *recon)
{
    sg_struct sgStruct;
    grid_struct gridStruct;

    sgStruct.n_ang = num_projs;
    sgStruct.n_det = num_pixels;
    sgStruct.angles = theta;
    sgStruct.center = 662;
    get_pswf(4.0, &gridStruct.pswf); // < -- effect of C?
    gridStruct.filter = get_filter(gridStruct.fname);

    float ***data3d = convert(data, num_projs, num_slices, num_pixels);
    float ***recon3d = convert(recon, num_slices, 2048, 2048);
    
    gridrec(data3d, &sgStruct, recon3d, &gridStruct);
}