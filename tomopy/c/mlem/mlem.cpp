#include "mlem.h"
using namespace std;

extern "C"
{
    Mlem* create(float *data, float *theta, float *center, 
                 int *num_projections, int *num_slices, 
                 int *num_pixels, int *num_grid, int *num_air) {         
        return new Mlem(data, theta, center, 
                        num_projections, num_slices, 
                        num_pixels, num_grid, num_air);
    }
    
    void reconstruct(Mlem *Mlem, float *recon, int *iters, 
                     int *slices_start, int *slices_end) {
        Mlem->reconstruct(recon, iters, slices_start, slices_end);
    }
    
} // extern "C"


Mlem::Mlem(float *data, float *theta, float *center, 
         int *num_projections, int *num_slices, 
         int *num_pixels, int *num_grid, int *num_air) :
data_(data),
theta_(theta),
center_(*center),
num_projections_(*num_projections),
num_slices_(*num_slices),
num_pixels_(*num_pixels),
num_grid_(*num_grid),
num_air_(*num_air) {

    padded_data_size = num_pixels_*sqrt(2)+1;
    padded_data = new float[padded_data_size * num_slices_]();
    pad_size = (padded_data_size - num_pixels_)/2;
    air = new float[num_pixels_];

    gridx = new float[num_grid_+1];
    gridy = new float[num_grid_+1];
    
    for (int m = 0; m <= num_grid_; m++) {
        gridx[m] = -float(num_grid_)/2 + m;
        gridy[m] = -float(num_grid_)/2 + m;
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
    indx = new int[2*num_grid_];
    indy = new int[2*num_grid_];
    indi = new int[2*num_grid_];
}


Mlem::~Mlem() {
    delete [] padded_data; padded_data = NULL;
    delete [] air; air = NULL;
    delete [] gridx; gridx = NULL;
    delete [] gridy; gridy = NULL;
    delete [] coordx; coordx = NULL;
    delete [] coordy; coordy = NULL;
    delete [] ax; ax = NULL;
    delete [] ay; ay = NULL;
    delete [] bx; bx = NULL;
    delete [] by; by = NULL;
    delete [] coorx; coorx = NULL;
    delete [] coory; coory = NULL;
    delete [] leng; leng = NULL;
    delete [] indx; indx = NULL;
    delete [] indy; indy = NULL;
    delete [] indi; indi = NULL;
    }


void Mlem::reconstruct(float *recon, int *iters, 
                       int *slices_start, int *slices_end)
{
    int m, n, k, q, i, j, t;
    float xi, yi;
    float slope, islope;
    int alen, blen, len;
    int ii, io;
    float simdata;
    float srcx, srcy, detx, dety;
    float air_left, air_right, air_slope;
    
    float mov = num_pixels_/2 - center_;
    if (mov-ceil(mov) < 1e-2) {
        mov += 1e-2;
    }
    
    for (t = 0; t < *iters; t++) {
        
        suma = new float[num_grid_*num_grid_]();
        sumay = new float[num_slices_*num_grid_*num_grid_]();
        
        for (q = 0; q < num_projections_; q++) {
            
            for (n = *slices_start; n < *slices_end; n++) {

                if (num_air_ > 0) {
                    i = n * num_pixels_ + q * (num_pixels_ * num_slices_);

                    for (j = 0, air_left = 0, air_right = 0; j < num_air_; j++) {
                        air_left += data_[i+j];
                        air_right += data_[i+num_pixels_-1-j];
                    }
                    air_left /= float(num_air_);
                    air_right /= float(num_air_);
                    if (air_left <= 0.) {
                        air_left = 1.;
                    }
                    if (air_right <= 0.) {
                        air_right = 1.;
                    }
                    air_slope = (air_right - air_left)/(num_pixels_ - 1);

                    for (j = 0; j < num_pixels_; j++) {
                        air[j] = air_left + air_slope*j;
                    }
                }




            for (m = 0; m < num_pixels_; m++) {
                    
                    i = m + (n * num_pixels_) + q * (num_pixels_ * num_slices_);
                    j = pad_size + m + (n * padded_data_size);
                    padded_data[j] = -log(data_[i] / air[m]);
                }
            }
            
            
            
            
            
            for (m = 0; m < padded_data_size; m++) {
                
                xi = -1e6;
                yi = -float(padded_data_size-1)/2 + m + mov;
                srcx = xi * cos(theta_[q]) - yi * sin(theta_[q]);
                srcy = xi * sin(theta_[q]) + yi * cos(theta_[q]);
                detx = -xi * cos(theta_[q]) - yi * sin(theta_[q]);
                dety = -xi * sin(theta_[q]) + yi * cos(theta_[q]);
                
                slope = (srcy - dety) / (srcx - detx);
                islope = 1 / slope;
                
                for (n = 0; n <= num_grid_; n++) {
                    coordx[n] = islope * (gridy[n] - srcy) + srcx;
                    coordy[n] = slope * (gridx[n] - srcx) + srcy;
                }
                
                alen = 0;
                blen = 0;
                for (n = 0; n <= num_grid_; n++) {
                    if (coordx[n] > gridx[0]) {
                        if (coordx[n] < gridx[num_grid_]) {
                            ax[alen] = coordx[n];
                            ay[alen] = gridy[n];
                            alen++;
                        }
                    }
                    if (coordy[n] > gridy[0]) {
                        if (coordy[n] < gridy[num_grid_]) {
                            bx[blen] = gridx[n];
                            by[blen] = coordy[n];
                            blen++;
                        }
                    }
                }
                len = alen+blen;
                
                
                i = 0;
                j = 0;
                k = 0;
                if ((theta_[q] >= 0 && theta_[q] < PI/2) || (theta_[q] >= PI && theta_[q] < 3*PI/2)) {
                    
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
                    leng[n] = sqrt(diffx * diffx + diffy * diffy);
                    
                    midx = (coorx[n+1] + coorx[n])/2;
                    midy = (coory[n+1] + coory[n])/2;
                    
                    indx[n] = floor(midx + float(num_grid_)/2);
                    indy[n] = floor(midy + float(num_grid_)/2);
                }
                
                
                for (n = 0; n < len-1; n++) {
                    indi[n] = (indx[n] + (indy[n] * num_grid_));
                    suma[indi[n]] += leng[n];
                }
                
                for (k = *slices_start; k < *slices_end; k++) {
                    
                    io = m + (k * padded_data_size);
                    
                    simdata = 0;
                    for (n = 0; n < len-1; n++) {
                        ii = indi[n] + k * (num_grid_ * num_grid_);
                        simdata += recon[ii] * leng[n];
//                        cout << recon[ii] << "    " <<  ii << "    " << leng[n] << endl;
                    }
//                    cout << simdata << endl;
                    
                    
//                    cout << ".    " << endl;
                    for (n = 0; n < len-1; n++) {
                        ii = indi[n] + k * (num_grid_ * num_grid_);
                        sumay[ii] += (padded_data[io] / simdata) * leng[n];
                    }
                    
                    
                }
                
            }
            
        }
        i = 0;
        for (k = 0; k < *slices_end-*slices_start; k++) {
            for (n = 0; n < num_grid_*num_grid_; n++) {
                recon[i] = recon[i] * (sumay[i] / suma[n]);
                i++;
            }
        }
    }
}

