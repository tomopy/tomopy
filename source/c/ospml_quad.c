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
ospml_quad(const float* data, int dy, int dt, int dx, const float* center,
           const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
           const float* reg_pars, int num_block, const float* ind_block)
{
    if(dy == 0 || dt == 0 || dx == 0)
        return;

    float* gridx  = (float*) malloc((ngridx + 1) * sizeof(float));
    float* gridy  = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordx = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordy = (float*) malloc((ngridx + 1) * sizeof(float));
    float* ax     = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* ay     = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* bx     = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* by     = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coorx  = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coory  = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* dist   = (float*) malloc((ngridx + ngridy) * sizeof(float));
    int*   indi   = (int*) malloc((ngridx + ngridy) * sizeof(int));

    assert(coordx != NULL && coordy != NULL && ax != NULL && ay != NULL && by != NULL &&
           bx != NULL && coorx != NULL && coory != NULL && dist != NULL && indi != NULL);

    int    s, q, p, d, i, m, n, os;
    int    quadrant;
    float  theta_p, sin_p, cos_p;
    float  mov, xi, yi;
    int    asize, bsize, csize;
    float* simdata;
    float  upd;
    int    ind_data, ind_recon;
    float* sum_dist;
    float  sum_dist2;
    float *E, *F, *G;
    int    ind0, ind1, indg[8];
    float  totalwg, wg[8], mg[8];
    int    subset_ind1, subset_ind2;

    for(i = 0; i < num_iter; i++)
    {
        simdata = (float*) calloc((dt * dy * dx), sizeof(float));

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

                sum_dist = (float*) calloc((ngridx * ngridy), sizeof(float));
                E        = (float*) calloc((ngridx * ngridy), sizeof(float));
                F        = (float*) calloc((ngridx * ngridy), sizeof(float));
                G        = (float*) calloc((ngridx * ngridy), sizeof(float));

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
                            ind_data  = d + p * dx + s * dt * dx;
                            ind_recon = s * ngridx * ngridy;
                            upd       = data[ind_data] / simdata[ind_data];
                            for(n = 0; n < csize - 1; n++)
                            {
                                E[indi[n]] -= recon[indi[n] + ind_recon] * upd * dist[n];
                            }
                        }
                    }
                }

                // Weights for inner neighborhoods.
                totalwg = 4 + 4 / sqrt(2);
                wg[0]   = 1 / totalwg;
                wg[1]   = 1 / totalwg;
                wg[2]   = 1 / totalwg;
                wg[3]   = 1 / totalwg;
                wg[4]   = 1 / sqrt(2) / totalwg;
                wg[5]   = 1 / sqrt(2) / totalwg;
                wg[6]   = 1 / sqrt(2) / totalwg;
                wg[7]   = 1 / sqrt(2) / totalwg;

                // (inner region)
                for(n = 1; n < ngridx - 1; n++)
                {
                    for(m = 1; m < ngridy - 1; m++)
                    {
                        ind0 = m + n * ngridy;
                        ind1 = ind0 + s * ngridx * ngridy;

                        indg[0] = ind1 + 1;
                        indg[1] = ind1 - 1;
                        indg[2] = ind1 + ngridy;
                        indg[3] = ind1 - ngridy;
                        indg[4] = ind1 + ngridy + 1;
                        indg[5] = ind1 + ngridy - 1;
                        indg[6] = ind1 - ngridy + 1;
                        indg[7] = ind1 - ngridy - 1;

                        for(q = 0; q < 8; q++)
                        {
                            mg[q] = recon[ind1] + recon[indg[q]];
                            F[ind0] += 2 * reg_pars[0] * wg[q];
                            G[ind0] -= 2 * reg_pars[0] * wg[q] * mg[q];
                        }
                    }
                }

                // Weights for edges.
                totalwg = 3 + 2 / sqrt(2);
                wg[0]   = 1 / totalwg;
                wg[1]   = 1 / totalwg;
                wg[2]   = 1 / totalwg;
                wg[3]   = 1 / sqrt(2) / totalwg;
                wg[4]   = 1 / sqrt(2) / totalwg;

                // (top)
                for(m = 1; m < ngridy - 1; m++)
                {
                    ind0 = m;
                    ind1 = ind0 + s * ngridx * ngridy;

                    indg[0] = ind1 + 1;
                    indg[1] = ind1 - 1;
                    indg[2] = ind1 + ngridy;
                    indg[3] = ind1 + ngridy + 1;
                    indg[4] = ind1 + ngridy - 1;

                    for(q = 0; q < 5; q++)
                    {
                        mg[q] = recon[ind1] + recon[indg[q]];
                        F[ind0] += 2 * reg_pars[0] * wg[q];
                        G[ind0] -= 2 * reg_pars[0] * wg[q] * mg[q];
                    }
                }

                // (bottom)
                for(m = 1; m < ngridy - 1; m++)
                {
                    ind0 = m + (ngridx - 1) * ngridy;
                    ind1 = ind0 + s * ngridx * ngridy;

                    indg[0] = ind1 + 1;
                    indg[1] = ind1 - 1;
                    indg[2] = ind1 - ngridy;
                    indg[3] = ind1 - ngridy + 1;
                    indg[4] = ind1 - ngridy - 1;

                    for(q = 0; q < 5; q++)
                    {
                        mg[q] = recon[ind1] + recon[indg[q]];
                        F[ind0] += 2 * reg_pars[0] * wg[q];
                        G[ind0] -= 2 * reg_pars[0] * wg[q] * mg[q];
                    }
                }

                // (left)
                for(n = 1; n < ngridx - 1; n++)
                {
                    ind0 = n * ngridy;
                    ind1 = ind0 + s * ngridx * ngridy;

                    indg[0] = ind1 + 1;
                    indg[1] = ind1 + ngridy;
                    indg[2] = ind1 - ngridy;
                    indg[3] = ind1 + ngridy + 1;
                    indg[4] = ind1 - ngridy + 1;

                    for(q = 0; q < 5; q++)
                    {
                        mg[q] = recon[ind1] + recon[indg[q]];
                        F[ind0] += 2 * reg_pars[0] * wg[q];
                        G[ind0] -= 2 * reg_pars[0] * wg[q] * mg[q];
                    }
                }

                // (right)
                for(n = 1; n < ngridx - 1; n++)
                {
                    ind0 = (ngridy - 1) + n * ngridy;
                    ind1 = ind0 + s * ngridx * ngridy;

                    indg[0] = ind1 - 1;
                    indg[1] = ind1 + ngridy;
                    indg[2] = ind1 - ngridy;
                    indg[3] = ind1 + ngridy - 1;
                    indg[4] = ind1 - ngridy - 1;

                    for(q = 0; q < 5; q++)
                    {
                        mg[q] = recon[ind1] + recon[indg[q]];
                        F[ind0] += 2 * reg_pars[0] * wg[q];
                        G[ind0] -= 2 * reg_pars[0] * wg[q] * mg[q];
                    }
                }

                // Weights for corners.
                totalwg = 2 + 1 / sqrt(2);
                wg[0]   = 1 / totalwg;
                wg[1]   = 1 / totalwg;
                wg[2]   = 1 / sqrt(2) / totalwg;

                // (top-left)
                ind0 = 0;
                ind1 = ind0 + s * ngridx * ngridy;

                indg[0] = ind1 + 1;
                indg[1] = ind1 + ngridy;
                indg[2] = ind1 + ngridy + 1;

                for(q = 0; q < 3; q++)
                {
                    mg[q] = recon[ind1] + recon[indg[q]];
                    F[ind0] += 2 * reg_pars[0] * wg[q];
                    G[ind0] -= 2 * reg_pars[0] * wg[q] * mg[q];
                }

                // (top-right)
                ind0 = (ngridy - 1);
                ind1 = ind0 + s * ngridx * ngridy;

                indg[0] = ind1 - 1;
                indg[1] = ind1 + ngridy;
                indg[2] = ind1 + ngridy - 1;

                for(q = 0; q < 3; q++)
                {
                    mg[q] = recon[ind1] + recon[indg[q]];
                    F[ind0] += 2 * reg_pars[0] * wg[q];
                    G[ind0] -= 2 * reg_pars[0] * wg[q] * mg[q];
                }

                // (bottom-left)
                ind0 = (ngridx - 1) * ngridy;
                ind1 = ind0 + s * ngridx * ngridy;

                indg[0] = ind1 + 1;
                indg[1] = ind1 - ngridy;
                indg[2] = ind1 - ngridy + 1;

                for(q = 0; q < 3; q++)
                {
                    mg[q] = recon[ind1] + recon[indg[q]];
                    F[ind0] += 2 * reg_pars[0] * wg[q];
                    G[ind0] -= 2 * reg_pars[0] * wg[q] * mg[q];
                }

                // (bottom-right)
                ind0 = (ngridy - 1) + (ngridx - 1) * ngridy;
                ind1 = ind0 + s * ngridx * ngridy;

                indg[0] = ind1 - 1;
                indg[1] = ind1 - ngridy;
                indg[2] = ind1 - ngridy - 1;

                for(q = 0; q < 3; q++)
                {
                    mg[q] = recon[ind1] + recon[indg[q]];
                    F[ind0] += 2 * reg_pars[0] * wg[q];
                    G[ind0] -= 2 * reg_pars[0] * wg[q] * mg[q];
                }

                q = 0;
                for(n = 0; n < ngridx * ngridy; n++)
                {
                    G[q] += sum_dist[n];
                    q++;
                }

                for(n = 0; n < ngridx; n++)
                {
                    for(m = 0; m < ngridy; m++)
                    {
                        q = m + n * ngridy;
                        if(F[q] != 0.0)
                        {
                            ind0        = q + s * ngridx * ngridy;
                            recon[ind0] = (-G[q] + sqrt(G[q] * G[q] - 8 * E[q] * F[q])) /
                                          (4 * F[q]);
                        }
                    }
                }

                free(sum_dist);
                free(E);
                free(F);
                free(G);
            }
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
