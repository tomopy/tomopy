#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

void mlem_transmission(float* data, float* white, float* theta, float center, 
          int num_projections, int num_slices, int num_pixels, 
          int num_grid, int iters, float* recon) {
              
    float* gridx = (float *)malloc((num_grid+1)*sizeof(float));
    float* gridy = (float *)malloc((num_grid+1) * sizeof(float));
    float* coordx = (float *)malloc((num_grid+1) * sizeof(float));
    float* coordy = (float *)malloc((num_grid+1) * sizeof(float));
    float* ax = (float *)malloc((num_grid+1) * sizeof(float));
    float* ay = (float *)malloc((num_grid+1) * sizeof(float));
    float* bx = (float *)malloc((num_grid+1) * sizeof(float));
    float* by = (float *)malloc((num_grid+1) * sizeof(float));
    float* coorx = (float *)malloc((2*num_grid) * sizeof(float));
    float* coory = (float *)malloc((2*num_grid) * sizeof(float));
    float* leng = (float *)malloc((2*num_grid) * sizeof(float));
    int* indx = (int *)malloc((2*num_grid) * sizeof(int));
    int* indy = (int *)malloc((2*num_grid) * sizeof(int));
    int* indi = (int *)malloc((2*num_grid) * sizeof(int));
    
    const double PI = 3.141592653589793238462;
    
    int m, n, k, q, i, j, t, iproj, ext1, ext2;
    bool quadrant;
    float xi, yi;
    float slope, islope;
    float sinq, cosq;
    int alen, blen, len;
    int i1, i2;
    float x1, x2;
    int io;
    float simintensity, simdata;
    float srcx, srcy, detx, dety;
    float midx, midy, diffx, diffy;
    float* sum_num; 
    float* sum_denom;
    float mov;
    
        
    mov = num_pixels/2 - center;
    if (mov-ceil(mov) < 1e-6) {
        mov += 1e-6;
    }
    
    for (m = 0; m <= num_grid; m++) {
        gridx[m] = -num_grid/2. + m;
        gridy[m] = -num_grid/2. + m;
    }
    
    for (t = 0; t < iters; t++) {
        
        printf ("mlem_transmission iteration: %i \n", t);
        
        sum_num = (float *)calloc((num_slices*num_grid*num_grid), sizeof(float));
        sum_denom = (float *)calloc((num_slices*num_grid*num_grid), sizeof(float));

        for (q = 0; q < num_projections; q++) {
            iproj = q * (num_slices * num_pixels);
            
            sinq = sin(theta[q]);
            cosq =  cos(theta[q]);
            
            if ((theta[q] >= 0 && theta[q] < PI/2) || 
                    (theta[q] >= PI && theta[q] < 3*PI/2)) {
                quadrant = true;
            } else {
                quadrant = false;
            }

            for (m = 0; m < num_pixels; m++) {
                
                xi = 1e6;
                yi = -(num_pixels-1)/2. + m + mov;

                srcx = xi * cosq - yi * sinq;
                srcy = xi * sinq + yi * cosq;
                detx = -xi * cosq - yi * sinq;
                dety = -xi * sinq + yi * cosq;
                
                slope = (srcy - dety) / (srcx - detx);
                islope = 1 / slope;
                
                for (n = 0; n <= num_grid; n++) {
                    coordx[n] = islope * (gridy[n] - srcy) + srcx;
                    coordy[n] = slope * (gridx[n] - srcx) + srcy;
                }
                
                alen = 0;
                blen = 0;
                for (n = 0; n <= num_grid; n++) {
                    if (coordx[n] > gridx[0]) {
                        if (coordx[n] < gridx[num_grid]) {
                            ax[alen] = coordx[n];
                            ay[alen] = gridy[n];
                            alen++;
                        }
                    }
                    if (coordy[n] > gridy[0]) {
                        if (coordy[n] < gridy[num_grid]) {
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
                if (quadrant) {
                    
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
                    
                    x1 = midx + num_grid/2.;
                    x2 = midy + num_grid/2.;
                    
                    i1 = (int)(midx + num_grid/2.);
                    i2 = (int)(midy + num_grid/2.);

                    indx[n] = i1 - (i1 > x1);
                    indy[n] = i2 - (i2 > x2);
                    
                    //indx[n] = floor(midx + num_grid/2.);
                    //indy[n] = floor(midy + num_grid/2.);
                }
                
                
                for (n = 0; n < len-1; n++) {
                    indi[n] = indx[n] + (indy[n] * num_grid);
                }
                
                for (k = 0; k < num_slices; k++) {
                    i = k * num_grid * num_grid;
                    io = iproj + m + (k * num_pixels);
                    
                    simdata = 0;
                    for (n = 0; n < len-1; n++) {
                        simdata += recon[indi[n]+i] * leng[n] * 1e-4;
                    }
                    simintensity = white[m + (k * num_pixels)] * exp(-simdata);
                    
                    for (n = 0; n < len-1; n++) {
                        sum_num[indi[n]+i] += (simintensity-data[io])*leng[n];
                        sum_denom[indi[n]+i] += (simdata*simintensity*leng[n]);
                    }
                }
            }
            
            for (n = 0; n < num_slices*num_grid*num_grid; n++) {
                recon[n] = recon[n]+ recon[n]*(sum_num[n] / sum_denom[n]);
            }
            
        }
        free(sum_num);
        free(sum_denom);
        
    }
    
    free(gridx);
    free(gridy);
    free(coordx);
    free(coordy);
    free(ax);
    free(ay);
    free(bx);
    free(by);
    free(coorx);
    free(coory);
    free(leng);
    free(indx);
    free(indy);
    free(indi);
}

