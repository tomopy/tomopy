// Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.

// Copyright 2015. UChicago Argonne, LLC. This software was produced 
// under U.S. Government contract DE-AC02-06CH11357 for Argonne National 
// Laboratory (ANL), which is operated by UChicago Argonne, LLC for the 
// U.S. Department of Energy. The U.S. Government has rights to use, 
// reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR 
// UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR 
// ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is 
// modified to produce derivative works, such modified software should 
// be clearly marked, so as not to confuse it with the version available 
// from ANL.

// Additionally, redistribution and use in source and binary forms, with 
// or without modification, are permitted provided that the following 
// conditions are met:

//     * Redistributions of source code must retain the above copyright 
//       notice, this list of conditions and the following disclaimer. 

//     * Redistributions in binary form must reproduce the above copyright 
//       notice, this list of conditions and the following disclaimer in 
//       the documentation and/or other materials provided with the 
//       distribution. 

//     * Neither the name of UChicago Argonne, LLC, Argonne National 
//       Laboratory, ANL, the U.S. Government, nor the names of its 
//       contributors may be used to endorse or promote products derived 
//       from this software without specific prior written permission. 

// THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS 
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago 
// Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.

#include "utils.h"


void 
sirt(
    float *data, data_pars *dpars, 
    float *recon, recon_pars *rpars)
{
    int dx, dy, dz, ry, rz;

    dx = dpars->dx;
    dy = dpars->dy;
    dz = dpars->dz;
    ry = rpars->ry;
    rz = rpars->rz;

    float *gridx = (float *)malloc((ry+1)*sizeof(float));
    float *gridy = (float *)malloc((rz+1)*sizeof(float));
    float *coordx = (float *)malloc((rz+1)*sizeof(float));
    float *coordy = (float *)malloc((ry+1)*sizeof(float));
    float *ax = (float *)malloc((ry+rz)*sizeof(float));
    float *ay = (float *)malloc((ry+rz)*sizeof(float));
    float *bx = (float *)malloc((ry+rz)*sizeof(float));
    float *by = (float *)malloc((ry+rz)*sizeof(float));
    float *coorx = (float *)malloc((ry+rz)*sizeof(float));
    float *coory = (float *)malloc((ry+rz)*sizeof(float));
    float *dist = (float *)malloc((ry+rz)*sizeof(float));
    int *indi = (int *)malloc((ry+rz)*sizeof(int));

    assert(coordx != NULL && coordy != NULL &&
        ax != NULL && ay != NULL && by != NULL && bx != NULL &&
        coorx != NULL && coory != NULL && dist != NULL && indi != NULL);

    int s, p, d, i, m, n;
    int quadrant;
    float proj_angle, sin_p, cos_p;
    float mov, xi, yi;
    int asize, bsize, csize;
    float *simdata;
    float upd;
    int ind_data, ind_recon;
    float *sum_dist;
    float sum_dist2;
    float *update;

    preprocessing(ry, rz, dz, dpars->center, 
        &mov, gridx, gridy); // Outputs: mov, gridx, gridy

    for (i=0; i<rpars->num_iter; i++) 
    {
        printf("SIRT iteration : %i\n", i+1);

        simdata = (float *)calloc((dx*dy*dz), sizeof(float));

        // For each slice
        for (s=0; s<dy; s++) 
        {
            sum_dist = (float *)calloc((ry*rz), sizeof(float));
            update = (float *)calloc((ry*rz), sizeof(float));
            
            // For each projection angle 
            for (p=0; p<dx; p++) 
            {
                // Calculate the sin and cos values 
                // of the projection angle and find
                // at which quadrant on the cartesian grid.
                proj_angle = fmod(dpars->proj_angle[p], 2*M_PI);
                quadrant = calc_quadrant(proj_angle);
                sin_p = sinf(proj_angle);
                cos_p = cosf(proj_angle);

                // For each detector pixel 
                for (d=0; d<dz; d++) 
                {
                    // Calculate coordinates
                    xi = -1e6;
                    yi = -(dz-1)/2.0+d+mov;
                    calc_coords(
                        ry, rz, xi, yi, sin_p, cos_p, gridx, gridy, 
                        coordx, coordy);

                    // Merge the (coordx, gridy) and (gridx, coordy)
                    trim_coords(
                        ry, rz, coordx, coordy, gridx, gridy, 
                        &asize, ax, ay, &bsize, bx, by);

                    // Sort the array of intersection points (ax, ay) and
                    // (bx, by). The new sorted intersection points are 
                    // stored in (coorx, coory). Total number of points 
                    // are csize.
                    sort_intersections(
                        quadrant, asize, ax, ay, bsize, bx, by, 
                        &csize, coorx, coory);

                    // Calculate the distances (dist) between the 
                    // intersection points (coorx, coory). Find the 
                    // indices of the pixels on the reconstruction grid.
                    calc_dist(
                        ry, rz, csize, coorx, coory, 
                        indi, dist);

                    // Calculate simdata 
                    calc_simdata(p, s, d, ry, rz, dy, dz,
                        csize, indi, dist, recon,
                        simdata); // Output: simdata


                    // Calculate dist*dist
                    sum_dist2 = 0.0;
                    for (n=0; n<csize-1; n++) 
                    {
                        sum_dist2 += dist[n]*dist[n];
                        sum_dist[indi[n]] += dist[n];
                    }

                    // Update
                    if (sum_dist2 != 0.0) 
                    {
                        ind_data = d+s*dz+p*dy*dz;
                        upd = (data[ind_data]-simdata[ind_data])/sum_dist2;
                        for (n=0; n<csize-1; n++) 
                        {
                            update[indi[n]] += upd*dist[n];
                        }
                    }
                }
            }

            m = 0;
            for (n = 0; n < ry*rz; n++) {
                if (sum_dist[n] != 0.0) {
                    ind_recon = s*ry*rz;
                    recon[m+ind_recon] += update[m]/sum_dist[n];
                }
                m++;
            }

            free(sum_dist);
            free(update);
        }

        free(simdata);
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
    free(dist);
    free(indi);
}