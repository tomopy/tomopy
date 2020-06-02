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
osem(const float* data, int dy, int dt, int dx, const float* center, const float* theta,
     float* recon, int ngridx, int ngridy, int num_iter, int num_block,
     const float* ind_block)
{
    if(dy == 0 || dt == 0 || dx == 0)
        return;

    float* gridx    = (float*) malloc((ngridx + 1) * sizeof(float));
    float* gridy    = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordx   = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordy   = (float*) malloc((ngridx + 1) * sizeof(float));
    float* ax       = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* ay       = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* bx       = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* by       = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coorx    = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coory    = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* dist     = (float*) malloc((ngridx + ngridy) * sizeof(float));
    int*   indi     = (int*) malloc((ngridx + ngridy) * sizeof(int));
    float* simdata  = (float*) malloc((dy * dt * dx) * sizeof(float));
    float* sum_dist = (float*) malloc((ngridx * ngridy) * sizeof(float));
    float* update   = (float*) malloc((ngridx * ngridy) * sizeof(float));

    assert(coordx != NULL && coordy != NULL && ax != NULL && ay != NULL && by != NULL &&
           bx != NULL && coorx != NULL && coory != NULL && dist != NULL && indi != NULL &&
           simdata != NULL && sum_dist != NULL && update != NULL);

    int   s, q, p, d, i, m, n, os;
    int   quadrant;
    float theta_p, sin_p, cos_p;
    float mov, xi, yi;
    int   asize, bsize, csize;
    float upd;
    int   ind_data, ind_recon;
    float sum_dist2;
    int   subset_ind1, subset_ind2;

    for(i = 0; i < num_iter; i++)
    {
        // initialize simdata to zero
        memset(simdata, 0, dy * dt * dx * sizeof(float));

        // For each slice
        for(s = 0; s < dy; s++)
        {
            preprocessing(ngridx, ngridy, dx, center[s], &mov, gridx,
                          gridy);  // Outputs: mov, gridx, gridy

            subset_ind1 = dt / num_block;
            subset_ind2 = subset_ind1;

            // For each ordered-subset num_subset
            for(os = 0; os < num_block + 1; os++)
            {
                if(os == num_block)
                {
                    subset_ind2 = dt % num_block;
                }

                // initialize sum_dist and update to zero
                memset(sum_dist, 0, (ngridx * ngridy) * sizeof(float));
                memset(update, 0, (ngridx * ngridy) * sizeof(float));

                // For each projection angle
                for(q = 0; q < subset_ind2; q++)
                {
                    p = ind_block[q + os * subset_ind1];

                    // Calculate the sin and cos values
                    // of the projection angle and find
                    // at which quadrant on the cartesian grid.
                    theta_p  = fmodf(theta[p], 2.0f * (float) M_PI);
                    quadrant = calc_quadrant(theta_p);
                    sin_p    = sinf(theta_p);
                    cos_p    = cosf(theta_p);

                    // For each detector pixel
                    for(d = 0; d < dx; d++)
                    {
                        // Calculate coordinates
                        xi = -ngridx - ngridy;
                        yi = 0.5f * (1 - dx) + d + mov;
                        calc_coords(ngridx, ngridy, xi, yi, sin_p, cos_p, gridx, gridy,
                                    coordx, coordy);

                        // Merge the (coordx, gridy) and (gridx, coordy)
                        trim_coords(ngridx, ngridy, coordx, coordy, gridx, gridy, &asize,
                                    ax, ay, &bsize, bx, by);

                        // Sort the array of intersection points (ax, ay) and
                        // (bx, by). The new sorted intersection points are
                        // stored in (coorx, coory). Total number of points
                        // are csize.
                        sort_intersections(quadrant, asize, ax, ay, bsize, bx, by, &csize,
                                           coorx, coory);

                        // Calculate the distances (dist) between the
                        // intersection points (coorx, coory). Find the
                        // indices of the pixels on the reconstruction grid.
                        calc_dist(ngridx, ngridy, csize, coorx, coory, indi, dist);

                        // Calculate simdata
                        calc_simdata(s, p, d, ngridx, ngridy, dt, dx, csize, indi, dist,
                                     recon,
                                     simdata);  // Output: simdata

                        // Calculate dist*dist
                        sum_dist2 = 0.0f;
                        for(n = 0; n < csize - 1; n++)
                        {
                            sum_dist2 += dist[n] * dist[n];
                            sum_dist[indi[n]] += dist[n];
                        }

                        // Update
                        if(sum_dist2 != 0.0f)
                        {
                            ind_data = d + p * dx + s * dt * dx;
                            upd      = data[ind_data] / simdata[ind_data];
                            for(n = 0; n < csize - 1; n++)
                            {
                                update[indi[n]] += upd * dist[n];
                            }
                        }
                    }
                }

                m = 0;
                for(n = 0; n < ngridx * ngridy; n++)
                {
                    if(sum_dist[n] != 0.0f)
                    {
                        ind_recon = s * ngridx * ngridy;
                        recon[m + ind_recon] *= update[m] / sum_dist[n];
                    }
                    m++;
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
    free(dist);
    free(indi);
    free(simdata);
    free(sum_dist);
    free(update);
}
