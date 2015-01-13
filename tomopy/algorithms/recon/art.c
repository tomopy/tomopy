#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#ifdef WIN32
#define DLL __declspec(dllexport)
#else
#define DLL 
#endif

DLL void art(float* data, float* theta, float center, 
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
    float* leng2 = (float *)malloc((2*num_grid) * sizeof(float));
    int* indi = (int *)malloc((2*num_grid) * sizeof(int));
    
    const double PI = 3.141592653589793238462;
    
    int m, n, k, q, i, j, t;
    bool quadrant;
    float xi, yi;
    float slope, islope;
    float sinq, cosq;
    int alen, blen, len;
    int i1, i2;
    float x1, x2;
    int indx, indy;
    int io;
    float simdata;
    float srcx, srcy, detx, dety;
    float midx, midy, diffx, diffy;
    float a2;
    float mov;
    float upd;
        
    mov = num_pixels/2 - center;
    if (mov-ceil(mov) < 1e-6) {
        mov += 1e-6;
    }
    
    // Define the reconstruction grid lines.
    for (m = 0; m <= num_grid; m++) {
        gridx[m] = -num_grid/2. + m;
        gridy[m] = -num_grid/2. + m;
    }
    
    // For each iteration
    for (t = 0; t < iters; t++) {
        printf ("art iteration: %i \n", t+1);
        
        // For each projection angle
        for (q = 0; q < num_projections; q++) {
            
            // Calculate the sin and cos values 
            // of the projection angle and find
            // at which quadrant on the cartesian grid.
            sinq = sin(theta[q]);
            cosq =  cos(theta[q]);
            if ((theta[q] >= 0 && theta[q] < PI/2) || 
                    (theta[q] >= PI && theta[q] < 3*PI/2)) {
                quadrant = true;
            } else {
                quadrant = false;
            }

            // For each line trajectory on a slice
            for (m = 0; m < num_pixels; m++) {
                
                // Find the corresponding source and
                // detector locations for a given line
                // trajectory of a projection (Projection
                // is specified by sinq and cosq). 
                xi = -1e6;
                yi = -(num_pixels-1)/2.+m+mov;
                srcx = xi*cosq-yi*sinq;
                srcy = xi*sinq+yi*cosq;
                detx = -xi*cosq-yi*sinq;
                dety = -xi*sinq+yi*cosq;
                
                // Find the intersection points of the 
                // line connecting the source and the detector
                // points with the reconstruction grid. The 
                // intersection points are then defined as: 
                // (coordx, gridy) and (gridx, coordy)
                slope = (srcy-dety)/(srcx-detx);
                islope = 1/slope;
                for (n = 0; n <= num_grid; n++) {
                    coordx[n] = islope*(gridy[n]-srcy)+srcx;
                    coordy[n] = slope*(gridx[n]-srcx)+srcy;
                }
                
                // Merge the (coordx, gridy) and (gridx, coordy)
                // on a single array of points (ax, ay) and trim
                // the coordinates that are outside the
                // reconstruction grid. 
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
                
                // Sort the array of intersection points (ax, ay).
                // The new sorted intersection points are 
                // stored in (coorx, coory).
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
                
                // Calculate the distances (leng) between the 
                // intersection points (coorx, coory). Find 
                // the indices of the pixels on the  
                // reconstruction grid (indi).
                for (n = 0; n < len-1; n++) {
                    diffx = coorx[n+1]-coorx[n];
                    diffy = coory[n+1]-coory[n];
                    leng2[n] = diffx*diffx+diffy*diffy;
                    leng[n] = sqrt(leng2[n]);
                    midx = (coorx[n+1]+coorx[n])/2;
                    midy = (coory[n+1]+coory[n])/2;
                    x1 = midx+num_grid/2.;
                    x2 = midy+num_grid/2.;
                    i1 = (int)(midx+num_grid/2.);
                    i2 = (int)(midy+num_grid/2.);
                    indx = i1-(i1>x1);
                    indy = i2-(i2>x2);
                    indi[n] = indx+indy*num_grid;
                }
                
                // Note: The indices (indi) and the corresponding 
                // weights (leng) are the same for all slices. So,
                // there is no need to calculate them for each slice.
                

                //*******************************************************
                // Below is for updating the reconstruction grid.
                
                a2 = 0;
                for (n = 0; n < len-1; n++) {
                    a2 += leng2[n];
                }
                
                // For each slice.
                for (k = 0; k < num_slices; k++) {
                    i = k*num_grid*num_grid;
                    io = m+k*num_pixels+q*(num_slices*num_pixels);
                    
                    simdata = 0;
                    for (n = 0; n < len-1; n++) {
                        simdata += recon[indi[n]+i]*leng[n];
                    }
                    upd = (data[io]-simdata)/a2;
                    for (n = 0; n < len-1; n++) {
                        recon[indi[n]+i] += upd*leng[n];
                    }   
                    
                //*******************************************************                 
                }               
            }           
        }
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
    free(leng2);
    free(indi);
}



