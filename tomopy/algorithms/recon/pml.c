#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

void pml(float* data, float* theta, float center, 
         int num_projections, int num_slices, int num_pixels, 
         int num_grid, int iters, float beta, float* recon) {
              
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
    int ind0, indg[8];
    float wg[8];
        
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
        printf ("pml iteration: %i \n", t+1);
        
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
        wg[0] = 0.1464466094;
        wg[1] = 0.1464466094;
        wg[2] = 0.1464466094;
        wg[3] = 0.1464466094;
        wg[4] = 0.10355339059;
        wg[5] = 0.10355339059;
        wg[6] = 0.10355339059;
        wg[7] = 0.10355339059;
        
        // (inner region)
        for (k = 0; k < num_slices; k++) {
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
                    
                    for (q = 0; q < 8; q++) {
                        F[ind0] += 2*beta*wg[q];
                        G[ind0] -= 2*beta*wg[q]*(recon[ind0]+recon[indg[q]]);
                    }
                }
            }
        }
        
        // Weights for edges.
        wg[0] = 0.226540919667;
        wg[1] = 0.226540919667;
        wg[2] = 0.226540919667;
        wg[3] = 0.1601886205;
        wg[4] = 0.1601886205;
        
        // (top)
        for (k = 0; k < num_slices; k++) {
            for (m = 1; m < num_grid-1; m++) {
                ind0 = m + k*num_grid*num_grid;
                
                indg[0] = ind0+1;
                indg[1] = ind0-1;
                indg[2] = ind0+num_grid;
                indg[3] = ind0+num_grid+1; 
                indg[4] = ind0+num_grid-1;
                    
                for (q = 0; q < 5; q++) {
                    F[ind0] += 2*beta*wg[q];
                    G[ind0] -= 2*beta*wg[q]*(recon[ind0]+recon[indg[q]]);
                }
            }
        }

        // (bottom)
        for (k = 0; k < num_slices; k++) {
            for (m = 1; m < num_grid-1; m++) {
                ind0 = m + (num_grid-1)*num_grid + k*num_grid*num_grid;
                
                indg[0] = ind0+1;
                indg[1] = ind0-1;
                indg[2] = ind0-num_grid;
                indg[3] = ind0-num_grid+1;
                indg[4] = ind0-num_grid-1;
                    
                for (q = 0; q < 5; q++) {
                    F[ind0] += 2*beta*wg[q];
                    G[ind0] -= 2*beta*wg[q]*(recon[ind0]+recon[indg[q]]);
                }
            }
        }

        // (left)  
        for (k = 0; k < num_slices; k++) {
            for (n = 1; n < num_grid-1; n++) {
                ind0 = n*num_grid + k*num_grid*num_grid;
                
                indg[0] = ind0+1;
                indg[1] = ind0+num_grid;
                indg[2] = ind0-num_grid;
                indg[3] = ind0+num_grid+1; 
                indg[4] = ind0-num_grid+1;
                    
                for (q = 0; q < 5; q++) {
                    F[ind0] += 2*beta*wg[q];
                    G[ind0] -= 2*beta*wg[q]*(recon[ind0]+recon[indg[q]]);
                }
            }
        }

        // (right)                
        for (k = 0; k < num_slices; k++) {
            for (n = 1; n < num_grid-1; n++) {
                ind0 = (num_grid-1) + n*num_grid + k*num_grid*num_grid;
                
                indg[0] = ind0-1;
                indg[1] = ind0+num_grid;
                indg[2] = ind0-num_grid;
                indg[3] = ind0+num_grid-1;
                indg[4] = ind0-num_grid-1;
                    
                for (q = 0; q < 5; q++) {
                    F[ind0] += 2*beta*wg[q];
                    G[ind0] -= 2*beta*wg[q]*(recon[ind0]+recon[indg[q]]);
                }
            }
        }
        
        // Weights for corners.
        wg[0] = 0.36939806251;
        wg[1] = 0.36939806251;
        wg[2] = 0.26120387496;
        
        // (top-left)
        for (k = 0; k < num_slices; k++) {     
            ind0 = k*num_grid*num_grid;
            
            indg[0] = ind0+1;
            indg[1] = ind0+num_grid;
            indg[2] = ind0+num_grid+1; 
                    
            for (q = 0; q < 3; q++) {
                F[ind0] += 2*beta*wg[q];
                G[ind0] -= 2*beta*wg[q]*(recon[ind0]+recon[indg[q]]);
            }
        }

        // (top-right)
        for (k = 0; k < num_slices; k++) {     
            ind0 = (num_grid-1) + k*num_grid*num_grid;
            
            indg[0] = ind0-1;
            indg[1] = ind0+num_grid;
            indg[2] = ind0+num_grid-1;
                    
            for (q = 0; q < 3; q++) {
                F[ind0] += 2*beta*wg[q];
                G[ind0] -= 2*beta*wg[q]*(recon[ind0]+recon[indg[q]]);
            }
        }

        // (bottom-left)
        for (k = 0; k < num_slices; k++) {     
            ind0 = (num_grid-1)*num_grid + k*num_grid*num_grid;
            
            indg[0] = ind0+1;
            indg[1] = ind0-num_grid;
            indg[2] = ind0-num_grid+1;
                    
            for (q = 0; q < 3; q++) {
                F[ind0] += 2*beta*wg[q];
                G[ind0] -= 2*beta*wg[q]*(recon[ind0]+recon[indg[q]]);
            }
        }

        // (bottom-right)        
        for (k = 0; k < num_slices; k++) {     
            ind0 = (num_grid-1) + (num_grid-1)*num_grid + k*num_grid*num_grid;
            
            indg[0] = ind0-1;
            indg[1] = ind0-num_grid;
            indg[2] = ind0-num_grid-1;
                    
            for (q = 0; q < 3; q++) {
                F[ind0] += 2*beta*wg[q];
                G[ind0] -= 2*beta*wg[q]*(recon[ind0]+recon[indg[q]]);
            }
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

