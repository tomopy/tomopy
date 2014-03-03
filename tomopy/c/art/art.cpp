#include "art.h"
using namespace std;

extern "C"
{
    Art* create(int *num_projections, int *num_slices, int *num_pixels, int *num_grid, float *data) {
        return new Art(num_projections, num_slices, num_pixels, num_grid, data);
    }
    
    void reconstruct(Art *Art, int *iters, float *center, float *theta, float *recon) {
        Art->reconstruct(iters, center, theta, recon);
    }
    
} // extern "C"


Art::Art(int *num_projections, int *num_slices, int *num_pixels, int *num_grid, float *data) :
num_projections_(*num_projections),
num_slices_(*num_slices),
num_pixels_(*num_pixels),
num_grid_(*num_grid),
data_(data)
{
    gridx_ = new float[num_grid_+1];
    gridy_ = new float[num_grid_+1];
    
    for (int m = 0; m <= num_grid_; m++) {
        gridx_[m] = -float(num_grid_)/2 + m;
        gridy_[m] = -float(num_grid_)/2 + m;
    }
    
    coordx = new float[num_grid_+1];
    coordy = new float[num_grid_+1];
    
    ax = new float[num_grid_+1];
    ay = new float[num_grid_+1];
    bx = new float[num_grid_+1];
    by = new float[num_grid_+1];
    
    coorx = new float[2*num_grid_];
    coory = new float[2*num_grid_];
    
    leng = new float[2*num_grid_];
    leng2 = new float[2*num_grid_];
    
    indx_ = new int[2*num_grid_];
    indy_ = new int[2*num_grid_];
    indi = new int[2*num_grid_];
}

void Art::reconstruct(int *iters, float *center, float *theta, float *recon)
{
    int m, n, k, q, i, j, t;
    float xi, yi;
    float slope, islope;
    int alen, blen, len;
    int ii, io, a2;
    float simdata;
    float srcx, srcy, detx, dety;
    
    float mov = num_pixels_/2 - *center;
    if (mov-ceil(mov) < 1e-2) {
        mov += 1e-2;
    }
    
//    for (t = 0; t < num_grid_*num_grid_*num_slices_; t++) {
//        cout << t << "   " << recon[t] << endl;
//    }
    
    for (t = 0; t < *iters; t++) {
        
        for (q = 0; q < num_projections_; q++) {
            
            for (m = 0; m < num_pixels_; m++) {
                
                xi = -1e6;
                yi = -float(num_pixels_-1)/2 + m;
                srcx = xi * cos(theta[q]) - yi * sin(theta[q]);
                srcy = xi * sin(theta[q]) + yi * cos(theta[q]);
                detx = -xi * cos(theta[q]) - yi * sin(theta[q]);
                dety = -xi * sin(theta[q]) + yi * cos(theta[q]);
                
                slope = (srcy - dety) / (srcx - detx);
                islope = 1 / slope;
                
                for (n = 0; n <= num_grid_; n++) {
                    coordx[n] = islope * (gridy_[n] - srcy) + srcx;
                    coordy[n] = slope * (gridx_[n] - srcx) + srcy;
                }
                
                alen = 0;
                blen = 0;
                for (n = 0; n <= num_grid_; n++) {
                    if (coordx[n] > gridx_[0]) {
                        if (coordx[n] < gridx_[num_grid_]) {
                            ax[alen] = coordx[n];
                            ay[alen] = gridy_[n];
                            alen++;
                        }
                    }
                    if (coordy[n] > gridy_[0]) {
                        if (coordy[n] < gridy_[num_grid_]) {
                            bx[blen] = gridx_[n];
                            by[blen] = coordy[n];
                            blen++;
                        }
                    }
                }
                len = alen+blen;
                
                
                
                i = 0;
                j = 0;
                k = 0;
                if ((theta[q] >= 0 && theta[q] < PI/2) || (theta[q] >= PI && theta[q] < 3*PI/2)) {
                    
                    while (i < alen && j < blen)
                    {
                        if (ax[i] < bx[j]) {
                            coorx[k] = ax[i];
                            coory[k] = ay[i];
                            i++;
                            k++;
                        } else {
                            coorx[k] = bx[j];
                            coory[k] = by[j];
                            j++;
                            k++;
                        }
                        
                    }
                    
                    while (i < alen) {
                        coorx[k] = ax[i];
                        coory[k] = ay[i];
                        i++;
                        k++;
                    }
                    
                    while (j < blen) {
                        coorx[k] = bx[j];
                        coory[k] = by[j];
                        j++;
                        k++;
                    }
                    
                } else {
                    while (i < alen && j < blen)
                    {
                        if (ax[alen-1-i] < bx[j]) {
                            coorx[k] = ax[alen-1-i];
                            coory[k] = ay[alen-1-i];
                            i++;
                            k++;
                        } else {
                            coorx[k] = bx[j];
                            coory[k] = by[j];
                            j++;
                            k++;
                        }
                    }
                    
                    while (i < alen) {
                        coorx[k] = ax[alen-1-i];
                        coory[k] = ay[alen-1-i];
                        i++;
                        k++;
                    }
                    
                    while (j < blen) {
                        coorx[k] = bx[j];
                        coory[k] = by[j];
                        j++;
                        k++;
                    }
                }
                
                
                for (n = 0; n < len-1; n++) {
                    diffx = coorx[n+1] - coorx[n];
                    diffy = coory[n+1] - coory[n];
                    leng2[n] = diffx * diffx + diffy * diffy;
                    leng[n] = sqrt(leng2[n]);
                    
                    midx = (coorx[n+1] + coorx[n])/2;
                    midy = (coory[n+1] + coory[n])/2;
                    
                    indx_[n] = floor(midx + float(num_grid_)/2);
                    indy_[n] = floor(midy + float(num_grid_)/2);
                }
                
                a2 = 0;
                for (n = 0; n < len-1; n++) {
                    a2 += leng2[n];
                }
                
                for (n = 0; n < len-1; n++) {
                    indi[n] = (indx_[n] + (indy_[n] * num_grid_));
                }
                
                
                
                for (k = 0; k < num_slices_; k++) {
                    
                    io = m + (k * num_pixels_) + q * (num_pixels_ * num_slices_);
                    
                    simdata = 0;
                    for (n = 0; n < len-1; n++) {
                        ii = indi[n] + k * (num_grid_ * num_grid_);
                        simdata += recon[ii] * leng[n];
                        cout << ii << "    " <<  recon[ii] << "    " << leng[n] << endl;
                    }
//                    cout << simdata << endl;
                    for (n = 0; n < len-1; n++) {
                        ii = indi[n] + k * (num_grid_ * num_grid_);
//                        recon[ii] += (data_[io] - simdata) / a2 * leng[n];
                    }
                }
            }
        }
    }
}