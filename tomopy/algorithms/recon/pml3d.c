#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#ifdef WIN32
#define DLL __declspec(dllexport)
#else
#define DLL 
#endif

void pml3d(float* data, float* theta, float center, 
          int num_projections, int num_slices, int num_pixels, 
          int num_grid, int iters, float beta, float delta, float beta1, float delta1, float regw1, float regw2, 
          float* recon) {
              
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
    int* indi = (int *)malloc((2*num_grid) * sizeof(int));
    
    
    const double PI = 3.141592653589793238462;

    float totalwg;
    
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
    float mov;
    float upd;
    float* suma; 
    float* E;
    float* F;
    float* G;
    int ind0, indg[26];
    float rg[26];
    float mg[26];
    float wg[26];
    float gammag[26];
        
    mov = num_pixels/2-center;
    if (mov-ceil(mov) < 1e-6) {
        mov += 1e-6;
    }
    
    // Define the reconstruction grid lines.
    for (m = 0; m <= num_grid; m++) {
        gridx[m] = -num_grid/2.+m;
        gridy[m] = -num_grid/2.+m;
    }
    
    // For each iteration
    for (t = 0; t < iters; t++) {
        printf ("pml3d iteration: %i \n", t+1);
        
        suma = (float *)calloc((num_grid*num_grid), sizeof(float));
        E = (float *)calloc((num_slices*num_grid*num_grid), sizeof(float));
        F = (float *)calloc((num_slices*num_grid*num_grid), sizeof(float));
        G = (float *)calloc((num_slices*num_grid*num_grid), sizeof(float));
        
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
                    leng[n] = sqrt(diffx*diffx+diffy*diffy);
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

                for (n = 0; n < len-1; n++) {
                    suma[indi[n]] += leng[n];
                }
                
                for (k = 0; k < num_slices; k++) {
                    i = k*num_grid*num_grid;
                    io = m + k*num_pixels + q*num_slices*num_pixels;
                    
                    simdata = 0;
                    for (n = 0; n < len-1; n++) {
                        simdata += recon[indi[n]+i] * leng[n];
                    }
                    upd = data[io]/simdata;
                    for (n = 0; n < len-1; n++) {
                        E[indi[n]+i] -= recon[indi[n]+i]*upd*leng[n];
                    }
                }
            }
        }













           
        // Weights for inner neighborhoods.
        totalwg = 4+4/sqrt(2)+2*regw1+8*regw2/sqrt(2)+8*regw2/sqrt(3);
        wg[0] = 1/totalwg;
        wg[1] = 1/totalwg;
        wg[2] = 1/totalwg;
        wg[3] = 1/totalwg;
        wg[4] = 1/sqrt(2)/totalwg;
        wg[5] = 1/sqrt(2)/totalwg;
        wg[6] = 1/sqrt(2)/totalwg;
        wg[7] = 1/sqrt(2)/totalwg;

        wg[8] = regw1/totalwg;
        wg[9] = regw2/sqrt(2)/totalwg;
        wg[10] = regw2/sqrt(2)/totalwg;
        wg[11] = regw2/sqrt(2)/totalwg;
        wg[12] = regw2/sqrt(2)/totalwg;
        wg[13] = regw2/sqrt(3)/totalwg;
        wg[14] = regw2/sqrt(3)/totalwg;
        wg[15] = regw2/sqrt(3)/totalwg;
        wg[16] = regw2/sqrt(3)/totalwg;

        wg[17] = regw1/totalwg;
        wg[18] = regw2/sqrt(2)/totalwg;
        wg[19] = regw2/sqrt(2)/totalwg;
        wg[20] = regw2/sqrt(2)/totalwg;
        wg[21] = regw2/sqrt(2)/totalwg;
        wg[22] = regw2/sqrt(3)/totalwg;
        wg[23] = regw2/sqrt(3)/totalwg;
        wg[24] = regw2/sqrt(3)/totalwg;
        wg[25] = regw2/sqrt(3)/totalwg;
        
        // (inner region)
        for (k = 1; k < num_slices-1; k++) {
            for (n = 1; n < num_grid-1; n++) {
                for (m = 1; m < num_grid-1; m++) {
                    ind0 = m + n*num_grid + k*num_grid*num_grid;
                    
                    indg[0] = ind0+1;
                    indg[1] = ind0-1;
                    indg[2] = ind0+num_grid;
                    indg[3] = ind0-num_grid;
                    indg[4] = ind0+num_grid+1; 
                    indg[5] = ind0+num_grid-1;
                    indg[6] = ind0-num_grid+1;
                    indg[7] = ind0-num_grid-1;

                    indg[8] = ind0+num_grid*num_grid;
                    indg[9] = ind0+num_grid*num_grid+1;
                    indg[10] = ind0+num_grid*num_grid-1;
                    indg[11] = ind0+num_grid*num_grid+num_grid;
                    indg[12] = ind0+num_grid*num_grid-num_grid;
                    indg[13] = ind0+num_grid*num_grid+num_grid+1;
                    indg[14] = ind0+num_grid*num_grid+num_grid-1;
                    indg[15] = ind0+num_grid*num_grid-num_grid+1;
                    indg[16] = ind0+num_grid*num_grid-num_grid-1;

                    indg[17] = ind0-num_grid*num_grid;
                    indg[18] = ind0-num_grid*num_grid+1;
                    indg[19] = ind0-num_grid*num_grid-1;
                    indg[20] = ind0-num_grid*num_grid+num_grid;
                    indg[21] = ind0-num_grid*num_grid-num_grid;
                    indg[22] = ind0-num_grid*num_grid+num_grid+1;
                    indg[23] = ind0-num_grid*num_grid+num_grid-1;
                    indg[24] = ind0-num_grid*num_grid-num_grid+1;
                    indg[25] = ind0-num_grid*num_grid-num_grid-1;
                    
                    for (q = 0; q < 8; q++) {
                        mg[q] = recon[ind0]+recon[indg[q]];
                        rg[q] = recon[ind0]-recon[indg[q]];
                        gammag[q] = 1/(1+fabs(rg[q]/delta));
                        F[ind0] += 2*beta*wg[q]*gammag[q];
                        G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
                    }

                    for (q = 8; q < 26; q++) {
                        mg[q] = recon[ind0]+recon[indg[q]];
                        rg[q] = recon[ind0]-recon[indg[q]];
                        gammag[q] = 1/(1+fabs(rg[q]/delta1));
                        F[ind0] += 2*beta1*wg[q]*gammag[q];
                        G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
                    }
                }
            }
        }
        
























           
        // Weights for up/down faces.
        totalwg = 4+4/sqrt(2)+regw1+4*regw2/sqrt(2)+4*regw2/sqrt(3);
        wg[0] = 1/totalwg;
        wg[1] = 1/totalwg;
        wg[2] = 1/totalwg;
        wg[3] = 1/totalwg;
        wg[4] = 1/sqrt(2)/totalwg;
        wg[5] = 1/sqrt(2)/totalwg;
        wg[6] = 1/sqrt(2)/totalwg;
        wg[7] = 1/sqrt(2)/totalwg;

        wg[8] = regw1/totalwg;
        wg[9] = regw2/sqrt(2)/totalwg;
        wg[10] = regw2/sqrt(2)/totalwg;
        wg[11] = regw2/sqrt(2)/totalwg;
        wg[12] = regw2/sqrt(2)/totalwg;
        wg[13] = regw2/sqrt(3)/totalwg;
        wg[14] = regw2/sqrt(3)/totalwg;
        wg[15] = regw2/sqrt(3)/totalwg;
        wg[16] = regw2/sqrt(3)/totalwg;
        
        // FACE 1 (up)
        for (n = 1; n < num_grid-1; n++) {
            for (m = 1; m < num_grid-1; m++) {
                ind0 = m + n*num_grid;
                
                indg[0] = ind0+1;
                indg[1] = ind0-1;
                indg[2] = ind0+num_grid;
                indg[3] = ind0-num_grid;
                indg[4] = ind0+num_grid+1; 
                indg[5] = ind0+num_grid-1;
                indg[6] = ind0-num_grid+1;
                indg[7] = ind0-num_grid-1;

                indg[8] = ind0+num_grid*num_grid;
                indg[9] = ind0+num_grid*num_grid+1;
                indg[10] = ind0+num_grid*num_grid-1;
                indg[11] = ind0+num_grid*num_grid+num_grid;
                indg[12] = ind0+num_grid*num_grid-num_grid;
                indg[13] = ind0+num_grid*num_grid+num_grid+1;
                indg[14] = ind0+num_grid*num_grid+num_grid-1;
                indg[15] = ind0+num_grid*num_grid-num_grid+1;
                indg[16] = ind0+num_grid*num_grid-num_grid-1;
                
                for (q = 0; q < 8; q++) {
                    mg[q] = recon[ind0]+recon[indg[q]];
                    rg[q] = recon[ind0]-recon[indg[q]];
                    gammag[q] = 1/(1+fabs(rg[q]/delta));
                    F[ind0] += 2*beta*wg[q]*gammag[q];
                    G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
                }

                for (q = 8; q < 17; q++) {
                    mg[q] = recon[ind0]+recon[indg[q]];
                    rg[q] = recon[ind0]-recon[indg[q]];
                    gammag[q] = 1/(1+fabs(rg[q]/delta1));
                    F[ind0] += 2*beta1*wg[q]*gammag[q];
                    G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
                }
            }
        }

        // FACE 2 (down)
        for (n = 1; n < num_grid-1; n++) {
            for (m = 1; m < num_grid-1; m++) {
                ind0 = m + n*num_grid + (num_slices-1)*num_grid*num_grid;
                
                indg[0] = ind0+1;
                indg[1] = ind0-1;
                indg[2] = ind0+num_grid;
                indg[3] = ind0-num_grid;
                indg[4] = ind0+num_grid+1; 
                indg[5] = ind0+num_grid-1;
                indg[6] = ind0-num_grid+1;
                indg[7] = ind0-num_grid-1;

                indg[8] = ind0-num_grid*num_grid;
                indg[9] = ind0-num_grid*num_grid+1;
                indg[10] = ind0-num_grid*num_grid-1;
                indg[11] = ind0-num_grid*num_grid+num_grid;
                indg[12] = ind0-num_grid*num_grid-num_grid;
                indg[13] = ind0-num_grid*num_grid+num_grid+1;
                indg[14] = ind0-num_grid*num_grid+num_grid-1;
                indg[15] = ind0-num_grid*num_grid-num_grid+1;
                indg[16] = ind0-num_grid*num_grid-num_grid-1;
                
                for (q = 0; q < 8; q++) {
                    mg[q] = recon[ind0]+recon[indg[q]];
                    rg[q] = recon[ind0]-recon[indg[q]];
                    gammag[q] = 1/(1+fabs(rg[q]/delta));
                    F[ind0] += 2*beta*wg[q]*gammag[q];
                    G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
                }

                for (q = 8; q < 17; q++) {
                    mg[q] = recon[ind0]+recon[indg[q]];
                    rg[q] = recon[ind0]-recon[indg[q]];
                    gammag[q] = 1/(1+fabs(rg[q]/delta1));
                    F[ind0] += 2*beta1*wg[q]*gammag[q];
                    G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
                }
            }
        }

        // weights for other faces
        totalwg = 3+2/sqrt(2)+2*regw1+6*regw2/sqrt(2)+4*regw2/sqrt(3);
        wg[0] = 1/totalwg;
        wg[1] = 1/totalwg;
        wg[2] = 1/totalwg;
        wg[3] = 1/sqrt(2)/totalwg;
        wg[4] = 1/sqrt(2)/totalwg;

        wg[5] = regw1/totalwg;
        wg[6] = regw2/sqrt(2)/totalwg;
        wg[7] = regw2/sqrt(2)/totalwg;
        wg[8] = regw2/sqrt(2)/totalwg;        
        wg[9] = regw2/sqrt(3)/totalwg;
        wg[10] = regw2/sqrt(3)/totalwg;

        wg[11] = regw1/totalwg;
        wg[12] = regw2/sqrt(2)/totalwg;
        wg[13] = regw2/sqrt(2)/totalwg;
        wg[14] = regw2/sqrt(2)/totalwg;
        wg[15] = regw2/sqrt(3)/totalwg;
        wg[16] = regw2/sqrt(3)/totalwg;
        
        // FACE 3 (bottom)
        for (k = 1; k < num_slices-1; k++) {
            for (m = 1; m < num_grid-1; m++) {
                ind0 = m + (num_grid-1)*num_grid + k*num_grid*num_grid;
                
                indg[0] = ind0+1;
                indg[1] = ind0-1;
                indg[2] = ind0-num_grid;
                indg[3] = ind0-num_grid+1;
                indg[4] = ind0-num_grid-1;

                indg[5] = ind0+num_grid*num_grid;
                indg[6] = ind0+num_grid*num_grid+1;
                indg[7] = ind0+num_grid*num_grid-1;
                indg[8] = ind0+num_grid*num_grid-num_grid;
                indg[9] = ind0+num_grid*num_grid-num_grid+1;
                indg[10] = ind0+num_grid*num_grid-num_grid-1;

                indg[11] = ind0-num_grid*num_grid;
                indg[12] = ind0-num_grid*num_grid+1;
                indg[13] = ind0-num_grid*num_grid-1;
                indg[14] = ind0-num_grid*num_grid-num_grid;
                indg[15] = ind0-num_grid*num_grid-num_grid+1;
                indg[16] = ind0-num_grid*num_grid-num_grid-1;
                
                for (q = 0; q < 5; q++) {
                    mg[q] = recon[ind0]+recon[indg[q]];
                    rg[q] = recon[ind0]-recon[indg[q]];
                    gammag[q] = 1/(1+fabs(rg[q]/delta));
                    F[ind0] += 2*beta*wg[q]*gammag[q];
                    G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
                }

                for (q = 5; q < 17; q++) {
                    mg[q] = recon[ind0]+recon[indg[q]];
                    rg[q] = recon[ind0]-recon[indg[q]];
                    gammag[q] = 1/(1+fabs(rg[q]/delta1));
                    F[ind0] += 2*beta1*wg[q]*gammag[q];
                    G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
                }
            }
        }

        // FACE 4 (top)
        for (k = 1; k < num_slices-1; k++) {
            for (m = 1; m < num_grid-1; m++) {
                ind0 = m + k*num_grid*num_grid;
                
                indg[0] = ind0+1;
                indg[1] = ind0-1;
                indg[2] = ind0+num_grid;
                indg[3] = ind0+num_grid+1; 
                indg[4] = ind0+num_grid-1;

                indg[5] = ind0+num_grid*num_grid;
                indg[6] = ind0+num_grid*num_grid+1;
                indg[7] = ind0+num_grid*num_grid-1;
                indg[8] = ind0+num_grid*num_grid+num_grid;
                indg[9] = ind0+num_grid*num_grid+num_grid+1;
                indg[10] = ind0+num_grid*num_grid+num_grid-1;

                indg[11] = ind0-num_grid*num_grid;
                indg[12] = ind0-num_grid*num_grid+1;
                indg[13] = ind0-num_grid*num_grid-1;
                indg[14] = ind0-num_grid*num_grid+num_grid;
                indg[15] = ind0-num_grid*num_grid+num_grid+1;
                indg[16] = ind0-num_grid*num_grid+num_grid-1;
                
                for (q = 0; q < 5; q++) {
                    mg[q] = recon[ind0]+recon[indg[q]];
                    rg[q] = recon[ind0]-recon[indg[q]];
                    gammag[q] = 1/(1+fabs(rg[q]/delta));
                    F[ind0] += 2*beta*wg[q]*gammag[q];
                    G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
                }

                for (q = 5; q < 17; q++) {
                    mg[q] = recon[ind0]+recon[indg[q]];
                    rg[q] = recon[ind0]-recon[indg[q]];
                    gammag[q] = 1/(1+fabs(rg[q]/delta1));
                    F[ind0] += 2*beta1*wg[q]*gammag[q];
                    G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
                }
            }
        }
        
        // FACE 5 (right)
        for (k = 1; k < num_slices-1; k++) {
            for (n = 1; n < num_grid-1; n++) {
                ind0 = (num_grid-1) + n*num_grid + k*num_grid*num_grid;
                
                indg[0] = ind0-1;
                indg[1] = ind0+num_grid;
                indg[2] = ind0-num_grid;
                indg[3] = ind0+num_grid-1;
                indg[4] = ind0-num_grid-1;

                indg[5] = ind0+num_grid*num_grid;
                indg[6] = ind0+num_grid*num_grid-1;
                indg[7] = ind0+num_grid*num_grid+num_grid;
                indg[8] = ind0+num_grid*num_grid-num_grid;
                indg[9] = ind0+num_grid*num_grid+num_grid-1;
                indg[10] = ind0+num_grid*num_grid-num_grid-1;

                indg[11] = ind0-num_grid*num_grid;
                indg[12] = ind0-num_grid*num_grid-1;
                indg[13] = ind0-num_grid*num_grid+num_grid;
                indg[14] = ind0-num_grid*num_grid-num_grid;
                indg[15] = ind0-num_grid*num_grid+num_grid-1;
                indg[16] = ind0-num_grid*num_grid-num_grid-1;
                
                for (q = 0; q < 5; q++) {
                    mg[q] = recon[ind0]+recon[indg[q]];
                    rg[q] = recon[ind0]-recon[indg[q]];
                    gammag[q] = 1/(1+fabs(rg[q]/delta));
                    F[ind0] += 2*beta*wg[q]*gammag[q];
                    G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
                }

                for (q = 5; q < 17; q++) {
                    mg[q] = recon[ind0]+recon[indg[q]];
                    rg[q] = recon[ind0]-recon[indg[q]];
                    gammag[q] = 1/(1+fabs(rg[q]/delta1));
                    F[ind0] += 2*beta1*wg[q]*gammag[q];
                    G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
                }
            }
        }

        // FACE 6 (left)
        for (k = 1; k < num_slices-1; k++) {
            for (n = 1; n < num_grid-1; n++) {
                ind0 = n*num_grid + k*num_grid*num_grid;

                indg[0] = ind0+1;
                indg[1] = ind0+num_grid;
                indg[2] = ind0-num_grid;
                indg[3] = ind0+num_grid+1; 
                indg[4] = ind0-num_grid+1;

                indg[5] = ind0+num_grid*num_grid;
                indg[6] = ind0+num_grid*num_grid+1;
                indg[7] = ind0+num_grid*num_grid+num_grid;
                indg[8] = ind0+num_grid*num_grid-num_grid;
                indg[9] = ind0+num_grid*num_grid+num_grid+1;
                indg[10] = ind0+num_grid*num_grid-num_grid+1;

                indg[11] = ind0-num_grid*num_grid;
                indg[12] = ind0-num_grid*num_grid+1;
                indg[13] = ind0-num_grid*num_grid+num_grid;
                indg[14] = ind0-num_grid*num_grid-num_grid;
                indg[15] = ind0-num_grid*num_grid+num_grid+1;
                indg[16] = ind0-num_grid*num_grid-num_grid+1;
                
                for (q = 0; q < 5; q++) {
                    mg[q] = recon[ind0]+recon[indg[q]];
                    rg[q] = recon[ind0]-recon[indg[q]];
                    gammag[q] = 1/(1+fabs(rg[q]/delta));
                    F[ind0] += 2*beta*wg[q]*gammag[q];
                    G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
                }

                for (q = 5; q < 17; q++) {
                    mg[q] = recon[ind0]+recon[indg[q]];
                    rg[q] = recon[ind0]-recon[indg[q]];
                    gammag[q] = 1/(1+fabs(rg[q]/delta1));
                    F[ind0] += 2*beta1*wg[q]*gammag[q];
                    G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
                }
            }
        }






























        // Weights for edges
        totalwg = 3+2/sqrt(2)+regw1+3*regw2/sqrt(2)+2*regw2/sqrt(3);
        wg[0] = 1/totalwg;
        wg[1] = 1/totalwg;
        wg[2] = 1/totalwg;
        wg[3] = 1/sqrt(2)/totalwg;
        wg[4] = 1/sqrt(2)/totalwg;

        wg[5] = regw1/totalwg;
        wg[6] = regw2/sqrt(2)/totalwg;
        wg[7] = regw2/sqrt(2)/totalwg;
        wg[8] = regw2/sqrt(2)/totalwg;
        wg[9] = regw2/sqrt(3)/totalwg;
        wg[10] = regw2/sqrt(3)/totalwg;
        
        // EDGE 1
        for (m = 1; m < num_grid-1; m++) {
            ind0 = m + (num_grid-1)*num_grid;
            
            indg[0] = ind0+1;
            indg[1] = ind0-1;
            indg[2] = ind0-num_grid;
            indg[3] = ind0-num_grid+1;
            indg[4] = ind0-num_grid-1;

            indg[5] = ind0+num_grid*num_grid;
            indg[6] = ind0+num_grid*num_grid+1;
            indg[7] = ind0+num_grid*num_grid-1;
            indg[8] = ind0+num_grid*num_grid-num_grid;
            indg[9] = ind0+num_grid*num_grid-num_grid+1;
            indg[10] = ind0+num_grid*num_grid-num_grid-1;
            
            for (q = 0; q < 5; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta));
                F[ind0] += 2*beta*wg[q]*gammag[q];
                G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
            }

            for (q = 5; q < 11; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta1));
                F[ind0] += 2*beta1*wg[q]*gammag[q];
                G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
            }
        }
        
        // EDGE 2
        for (m = 1; m < num_grid-1; m++) {
            ind0 = m;
                    
            indg[0] = ind0+1;
            indg[1] = ind0-1;
            indg[2] = ind0+num_grid;
            indg[3] = ind0+num_grid+1; 
            indg[4] = ind0+num_grid-1;

            indg[5] = ind0+num_grid*num_grid;
            indg[6] = ind0+num_grid*num_grid+1;
            indg[7] = ind0+num_grid*num_grid-1;
            indg[8] = ind0+num_grid*num_grid+num_grid;
            indg[9] = ind0+num_grid*num_grid+num_grid+1;
            indg[10] = ind0+num_grid*num_grid+num_grid-1;
            
            for (q = 0; q < 5; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta));
                F[ind0] += 2*beta*wg[q]*gammag[q];
                G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
            }

            for (q = 5; q < 11; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta1));
                F[ind0] += 2*beta1*wg[q]*gammag[q];
                G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
            }
        }
        
        // EDGE 3
        for (n = 1; n < num_grid-1; n++) {
            ind0 = (num_grid-1) + n*num_grid;
                    
            indg[0] = ind0-1;
            indg[1] = ind0+num_grid;
            indg[2] = ind0-num_grid;
            indg[3] = ind0+num_grid-1;
            indg[4] = ind0-num_grid-1;

            indg[5] = ind0+num_grid*num_grid;
            indg[6] = ind0+num_grid*num_grid-1;
            indg[7] = ind0+num_grid*num_grid+num_grid;
            indg[8] = ind0+num_grid*num_grid-num_grid;
            indg[9] = ind0+num_grid*num_grid+num_grid-1;
            indg[10] = ind0+num_grid*num_grid-num_grid-1;
            
            for (q = 0; q < 5; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta));
                F[ind0] += 2*beta*wg[q]*gammag[q];
                G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
            }

            for (q = 5; q < 11; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta1));
                F[ind0] += 2*beta1*wg[q]*gammag[q];
                G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
            }
        }
        
        // EDGE 4
        for (n = 1; n < num_grid-1; n++) {
            ind0 = n*num_grid;
                    
            indg[0] = ind0+1;
            indg[1] = ind0+num_grid;
            indg[2] = ind0-num_grid;
            indg[3] = ind0+num_grid+1; 
            indg[4] = ind0-num_grid+1;

            indg[5] = ind0+num_grid*num_grid;
            indg[6] = ind0+num_grid*num_grid+1;
            indg[7] = ind0+num_grid*num_grid+num_grid;
            indg[8] = ind0+num_grid*num_grid-num_grid;
            indg[9] = ind0+num_grid*num_grid+num_grid+1;
            indg[10] = ind0+num_grid*num_grid-num_grid+1;
            
            for (q = 0; q < 5; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta));
                F[ind0] += 2*beta*wg[q]*gammag[q];
                G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
            }

            for (q = 5; q < 11; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta1));
                F[ind0] += 2*beta1*wg[q]*gammag[q];
                G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
            }
        }

        // EDGE 5
        for (m = 1; m < num_grid-1; m++) {
            ind0 = m + (num_grid-1)*num_grid + (num_slices-1)*num_grid*num_grid;
            
            indg[0] = ind0+1;
            indg[1] = ind0-1;
            indg[2] = ind0-num_grid;
            indg[3] = ind0-num_grid+1;
            indg[4] = ind0-num_grid-1;

            indg[5] = ind0-num_grid*num_grid;
            indg[6] = ind0-num_grid*num_grid+1;
            indg[7] = ind0-num_grid*num_grid-1;
            indg[8] = ind0-num_grid*num_grid-num_grid;
            indg[9] = ind0-num_grid*num_grid-num_grid+1;
            indg[10] = ind0-num_grid*num_grid-num_grid-1;
            
            for (q = 0; q < 5; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta));
                F[ind0] += 2*beta*wg[q]*gammag[q];
                G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
            }

            for (q = 5; q < 11; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta1));
                F[ind0] += 2*beta1*wg[q]*gammag[q];
                G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
            }
        }
        
        // EDGE 6
        for (m = 1; m < num_grid-1; m++) {
            ind0 = m + (num_slices-1)*num_grid*num_grid;
                    
            indg[0] = ind0+1;
            indg[1] = ind0-1;
            indg[2] = ind0+num_grid;
            indg[3] = ind0+num_grid+1; 
            indg[4] = ind0+num_grid-1;

            indg[5] = ind0-num_grid*num_grid;
            indg[6] = ind0-num_grid*num_grid+1;
            indg[7] = ind0-num_grid*num_grid-1;
            indg[8] = ind0-num_grid*num_grid+num_grid;
            indg[9] = ind0-num_grid*num_grid+num_grid+1;
            indg[10] = ind0-num_grid*num_grid+num_grid-1;
            
            for (q = 0; q < 5; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta));
                F[ind0] += 2*beta*wg[q]*gammag[q];
                G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
            }

            for (q = 5; q < 11; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta1));
                F[ind0] += 2*beta1*wg[q]*gammag[q];
                G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
            }
        }
        
        // EDGE 7
        for (n = 1; n < num_grid-1; n++) {
            ind0 = (num_grid-1) + n*num_grid + (num_slices-1)*num_grid*num_grid;
                    
            indg[0] = ind0-1;
            indg[1] = ind0+num_grid;
            indg[2] = ind0-num_grid;
            indg[3] = ind0+num_grid-1;
            indg[4] = ind0-num_grid-1;

            indg[5] = ind0-num_grid*num_grid;
            indg[6] = ind0-num_grid*num_grid-1;
            indg[7] = ind0-num_grid*num_grid+num_grid;
            indg[8] = ind0-num_grid*num_grid-num_grid;
            indg[9] = ind0-num_grid*num_grid+num_grid-1;
            indg[10] = ind0-num_grid*num_grid-num_grid-1;
            
            for (q = 0; q < 5; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta));
                F[ind0] += 2*beta*wg[q]*gammag[q];
                G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
            }

            for (q = 5; q < 11; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta1));
                F[ind0] += 2*beta1*wg[q]*gammag[q];
                G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
            }
        }
        
        // EDGE 8
        for (n = 1; n < num_grid-1; n++) {
            ind0 = n*num_grid + (num_slices-1)*num_grid*num_grid;
                    
            indg[0] = ind0+1;
            indg[1] = ind0+num_grid;
            indg[2] = ind0-num_grid;
            indg[3] = ind0+num_grid+1; 
            indg[4] = ind0-num_grid+1;

            indg[5] = ind0-num_grid*num_grid;
            indg[6] = ind0-num_grid*num_grid+1;
            indg[7] = ind0-num_grid*num_grid+num_grid;
            indg[8] = ind0-num_grid*num_grid-num_grid;
            indg[9] = ind0-num_grid*num_grid+num_grid+1;
            indg[10] = ind0-num_grid*num_grid-num_grid+1;
            
            for (q = 0; q < 5; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta));
                F[ind0] += 2*beta*wg[q]*gammag[q];
                G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
            }

            for (q = 5; q < 11; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta1));
                F[ind0] += 2*beta1*wg[q]*gammag[q];
                G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
            }
        }

           
        // EDGE 9
        totalwg = 2+1/sqrt(2)+2*regw1+4*regw2/sqrt(2)+2*regw2/sqrt(3);
        wg[0] = 1/totalwg;
        wg[1] = 1/totalwg;
        wg[2] = 1/sqrt(2)/totalwg;

        wg[3] = regw1/totalwg;
        wg[4] = regw2/sqrt(2)/totalwg;
        wg[5] = regw2/sqrt(2)/totalwg;
        wg[6] = regw2/sqrt(3)/totalwg;

        wg[7] = regw1/totalwg;
        wg[8] = regw2/sqrt(2)/totalwg;
        wg[9] = regw2/sqrt(2)/totalwg;
        wg[10] = regw2/sqrt(3)/totalwg;
        
        for (k = 1; k < num_slices-1; k++) {
            ind0 = k*num_grid*num_grid;
            
            indg[0] = ind0+1;
            indg[1] = ind0+num_grid;
            indg[2] = ind0+num_grid+1;

            indg[3] = ind0+num_grid*num_grid;
            indg[4] = ind0+num_grid*num_grid+1;
            indg[5] = ind0+num_grid*num_grid+num_grid;
            indg[6] = ind0+num_grid*num_grid+num_grid+1;

            indg[7] = ind0-num_grid*num_grid;
            indg[8] = ind0-num_grid*num_grid+1;
            indg[9] = ind0-num_grid*num_grid+num_grid;
            indg[10] = ind0-num_grid*num_grid+num_grid+1;
            
            for (q = 0; q < 3; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta));
                F[ind0] += 2*beta*wg[q]*gammag[q];
                G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
            }

            for (q = 3; q < 11; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta1));
                F[ind0] += 2*beta1*wg[q]*gammag[q];
                G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
            }
        }

           
        // EDGE 10
        for (k = 1; k < num_slices-1; k++) {
            ind0 = (num_grid-1)*num_grid + k*num_grid*num_grid;
            
            indg[0] = ind0+1;
            indg[1] = ind0-num_grid;
            indg[2] = ind0-num_grid+1;

            indg[3] = ind0+num_grid*num_grid;
            indg[4] = ind0+num_grid*num_grid+1;
            indg[5] = ind0+num_grid*num_grid-num_grid;
            indg[6] = ind0+num_grid*num_grid-num_grid+1;

            indg[7] = ind0-num_grid*num_grid;
            indg[8] = ind0-num_grid*num_grid+1;
            indg[9] = ind0-num_grid*num_grid-num_grid;
            indg[10] = ind0-num_grid*num_grid-num_grid+1;
            
            for (q = 0; q < 3; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta));
                F[ind0] += 2*beta*wg[q]*gammag[q];
                G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
            }

            for (q = 3; q < 11; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta1));
                F[ind0] += 2*beta1*wg[q]*gammag[q];
                G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
            }
        }
           
        // EDGE 11
        for (k = 1; k < num_slices-1; k++) {
            ind0 = (num_grid-1) + (num_grid-1)*num_grid + k*num_grid*num_grid;
            
            indg[0] = ind0-1;
            indg[1] = ind0-num_grid;
            indg[2] = ind0-num_grid-1;

            indg[3] = ind0+num_grid*num_grid;
            indg[4] = ind0+num_grid*num_grid-1;
            indg[5] = ind0+num_grid*num_grid-num_grid;
            indg[6] = ind0+num_grid*num_grid-num_grid-1;

            indg[7] = ind0-num_grid*num_grid;
            indg[8] = ind0-num_grid*num_grid-1;
            indg[9] = ind0-num_grid*num_grid-num_grid;
            indg[10] = ind0-num_grid*num_grid-num_grid-1;
            
            for (q = 0; q < 3; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta));
                F[ind0] += 2*beta*wg[q]*gammag[q];
                G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
            }

            for (q = 3; q < 11; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta1));
                F[ind0] += 2*beta1*wg[q]*gammag[q];
                G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
            }
        }
           
        // EDGE 12
        for (k = 1; k < num_slices-1; k++) {
            ind0 = (num_grid-1) + k*num_grid*num_grid;
            
            indg[0] = ind0-1;
            indg[1] = ind0+num_grid;
            indg[2] = ind0+num_grid-1;

            indg[3] = ind0+num_grid*num_grid;
            indg[4] = ind0+num_grid*num_grid-1;
            indg[5] = ind0+num_grid*num_grid+num_grid;
            indg[6] = ind0+num_grid*num_grid+num_grid-1;

            indg[7] = ind0-num_grid*num_grid;
            indg[8] = ind0-num_grid*num_grid-1;
            indg[9] = ind0-num_grid*num_grid+num_grid;
            indg[10] = ind0-num_grid*num_grid+num_grid-1;
            
            for (q = 0; q < 3; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta));
                F[ind0] += 2*beta*wg[q]*gammag[q];
                G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
            }

            for (q = 3; q < 11; q++) {
                mg[q] = recon[ind0]+recon[indg[q]];
                rg[q] = recon[ind0]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/delta1));
                F[ind0] += 2*beta1*wg[q]*gammag[q];
                G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
            }
        }























           
        // Weights for vertices
        totalwg = 2+1/sqrt(2)+regw1+2*regw2/sqrt(2)+regw2/sqrt(3);
        wg[0] = 1/totalwg;
        wg[1] = 1/totalwg;
        wg[2] = 1/sqrt(2)/totalwg;

        wg[3] = regw1/totalwg;
        wg[4] = regw2/sqrt(2)/totalwg;
        wg[5] = regw2/sqrt(2)/totalwg;
        wg[6] = regw2/sqrt(3)/totalwg;
        
        // VERTEX 1
        ind0 = (num_grid-1) + (num_grid-1)*num_grid;
        
        indg[0] = ind0-1;
        indg[1] = ind0-num_grid;
        indg[2] = ind0-num_grid-1;

        indg[3] = ind0+num_grid*num_grid;
        indg[4] = ind0+num_grid*num_grid-1;
        indg[5] = ind0+num_grid*num_grid-num_grid;
        indg[6] = ind0+num_grid*num_grid-num_grid-1;
        
        for (q = 0; q < 3; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta));
            F[ind0] += 2*beta*wg[q]*gammag[q];
            G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
        }

        for (q = 3; q < 7; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta1));
            F[ind0] += 2*beta1*wg[q]*gammag[q];
            G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
        }

        
        // VERTEX 2
        ind0 = (num_grid-1);
        
        indg[0] = ind0-1;
        indg[1] = ind0+num_grid;
        indg[2] = ind0+num_grid-1;

        indg[3] = ind0+num_grid*num_grid;
        indg[4] = ind0+num_grid*num_grid-1;
        indg[5] = ind0+num_grid*num_grid+num_grid;
        indg[6] = ind0+num_grid*num_grid+num_grid-1;
        
        for (q = 0; q < 3; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta));
            F[ind0] += 2*beta*wg[q]*gammag[q];
            G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
        }

        for (q = 3; q < 7; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta1));
            F[ind0] += 2*beta1*wg[q]*gammag[q];
            G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
        }

        
        // VERTEX 3
        ind0 = 0;
        
        indg[0] = ind0+1;
        indg[1] = ind0+num_grid;
        indg[2] = ind0+num_grid+1;

        indg[3] = ind0+num_grid*num_grid;
        indg[4] = ind0+num_grid*num_grid+1;
        indg[5] = ind0+num_grid*num_grid+num_grid;
        indg[6] = ind0+num_grid*num_grid+num_grid+1;
        
        for (q = 0; q < 3; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta));
            F[ind0] += 2*beta*wg[q]*gammag[q];
            G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
        }

        for (q = 3; q < 7; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta1));
            F[ind0] += 2*beta1*wg[q]*gammag[q];
            G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
        }

        
        // VERTEX 4
        ind0 = (num_grid-1)*num_grid;
        
        indg[0] = ind0+1;
        indg[1] = ind0-num_grid;
        indg[2] = ind0-num_grid+1;

        indg[3] = ind0+num_grid*num_grid;
        indg[4] = ind0+num_grid*num_grid+1;
        indg[5] = ind0+num_grid*num_grid-num_grid;
        indg[6] = ind0+num_grid*num_grid-num_grid+1;
        
        for (q = 0; q < 3; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta));
            F[ind0] += 2*beta*wg[q]*gammag[q];
            G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
        }

        for (q = 3; q < 7; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta1));
            F[ind0] += 2*beta1*wg[q]*gammag[q];
            G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
        }

        
        // VERTEX 5
        ind0 = (num_grid-1) + (num_grid-1)*num_grid + (num_slices-1)*num_grid*num_grid;
        
        indg[0] = ind0-1;
        indg[1] = ind0-num_grid;
        indg[2] = ind0-num_grid-1;

        indg[3] = ind0-num_grid*num_grid;
        indg[4] = ind0-num_grid*num_grid-1;
        indg[5] = ind0-num_grid*num_grid-num_grid;
        indg[6] = ind0-num_grid*num_grid-num_grid-1;
        
        for (q = 0; q < 3; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta));
            F[ind0] += 2*beta*wg[q]*gammag[q];
            G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
        }

        for (q = 3; q < 7; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta1));
            F[ind0] += 2*beta1*wg[q]*gammag[q];
            G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
        }

        
        // VERTEX 6
        ind0 = (num_grid-1) + (num_slices-1)*num_grid*num_grid;
        
        indg[0] = ind0-1;
        indg[1] = ind0+num_grid;
        indg[2] = ind0+num_grid-1;

        indg[3] = ind0-num_grid*num_grid;
        indg[4] = ind0-num_grid*num_grid-1;
        indg[5] = ind0-num_grid*num_grid+num_grid;
        indg[6] = ind0-num_grid*num_grid+num_grid-1;
        
        for (q = 0; q < 3; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta));
            F[ind0] += 2*beta*wg[q]*gammag[q];
            G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
        }

        for (q = 3; q < 7; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta1));
            F[ind0] += 2*beta1*wg[q]*gammag[q];
            G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
        }

        
        // VERTEX 7
        ind0 = (num_slices-1)*num_grid*num_grid;
        
        indg[0] = ind0+1;
        indg[1] = ind0+num_grid;
        indg[2] = ind0+num_grid+1;

        indg[3] = ind0-num_grid*num_grid;
        indg[4] = ind0-num_grid*num_grid+1;
        indg[5] = ind0-num_grid*num_grid+num_grid;
        indg[6] = ind0-num_grid*num_grid+num_grid+1;
        
        for (q = 0; q < 3; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta));
            F[ind0] += 2*beta*wg[q]*gammag[q];
            G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
        }

        for (q = 3; q < 7; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta1));
            F[ind0] += 2*beta1*wg[q]*gammag[q];
            G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
        }


        // VERTEX 8
        ind0 = (num_grid-1)*num_grid + (num_slices-1)*num_grid*num_grid;
        
        indg[0] = ind0+1;
        indg[1] = ind0-num_grid;
        indg[2] = ind0-num_grid+1;

        indg[3] = ind0-num_grid*num_grid;
        indg[4] = ind0-num_grid*num_grid+1;
        indg[5] = ind0-num_grid*num_grid-num_grid;
        indg[6] = ind0-num_grid*num_grid-num_grid+1;
        
        for (q = 0; q < 3; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta));
            F[ind0] += 2*beta*wg[q]*gammag[q];
            G[ind0] -= 2*beta*wg[q]*gammag[q]*mg[q];
        }

        for (q = 3; q < 7; q++) {
            mg[q] = recon[ind0]+recon[indg[q]];
            rg[q] = recon[ind0]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/delta1));
            F[ind0] += 2*beta1*wg[q]*gammag[q];
            G[ind0] -= 2*beta1*wg[q]*gammag[q]*mg[q];
        }

















        i = 0;
        for (k = 0; k < num_slices; k++) {
            for (n = 0; n < num_grid*num_grid; n++) {
                G[i] += suma[n];
                i++;
            }
        }
        
        //i = 0;
        for (k = 0; k < num_slices; k++) {
            for (n = 0; n < num_grid; n++) {
                for (m = 0; m < num_grid; m++) {
                    i = m + n*num_grid + k*num_grid*num_grid;
                    recon[i] = (-G[i]+sqrt(G[i]*G[i]-8*E[i]*F[i]))/(4*F[i]);
                }
            }
        }
        
        free(suma);
        free(E);
        free(F);
        free(G);
        
        //*******************************************************        
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
    free(indi);
}

