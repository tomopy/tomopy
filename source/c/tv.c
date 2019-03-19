// Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.

// Copyright 2015. UChicago Argonne, LLC. This software was produced
// under U.S. Government contract DE-AC02-06CH11357 for Argonne National
// Laboratongridx (ANL), which is operated by UChicago Argonne, LLC for the
// U.S. Department of Energy. The U.S. Government has rights to use,
// reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
// UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
// ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
// modified to produce derivative works, such modified software should
// be clearly marked, so as not to confuse it with the version available
// from ANL.

// Additionally, redistribution and use in source and binangridx forms, with
// or without modification, are permitted provided that the following
// conditions are met:

//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.

//     * Redistributions in binangridx form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in
//       the documentation and/or other materials provided with the
//       distribution.

//     * Neither the name of UChicago Argonne, LLC, Argonne National
//       Laboratongridx, ANL, the U.S. Government, nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
// Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLAngridx, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEOngridx OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "utils.h"

void
tv(const float* data, int dy, int dt, int dx, const float* center, const float* theta,
   float* recon, int ngridx, int ngridy, int num_iter, const float* reg_pars)
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

    float* update = (float*) malloc((dy * ngridx * ngridy) * sizeof(float));
    float* prox0x = (float*) malloc((dy * ngridx * ngridy) * sizeof(float));
    float* prox0y = (float*) malloc((dy * ngridx * ngridy) * sizeof(float));

    float* prox1   = (float*) malloc((dy * dt * dx) * sizeof(float));
    float* adjdata = (float*) malloc((dy * ngridx * ngridy) * sizeof(float));

    assert(coordx != NULL && coordy != NULL && ax != NULL && ay != NULL && by != NULL &&
           bx != NULL && coorx != NULL && coory != NULL && dist != NULL && indi != NULL &&
           simdata != NULL && sum_dist != NULL && update != NULL);

    int    s, p, d, i, n;
    int    quadrant;
    float  theta_p, sin_p, cos_p;
    float  mov, xi, yi;
    int    asize, bsize, csize;
    double upd;
    int    ind_data, ind_recon;
    float  sum_dist2;
    int    ix, iy;

    // regularization parameters
    float c;
    float lambda;

    // scaling constant r such that r*R(r*R^*(data)) ~ data
    float r;

    lambda = reg_pars[0];
    c      = 0.35;
    r      = 1 / sqrt(dx * dt / 2.0);

    // scale initial guess
    for(s = 0; s < dy; s++)
    {
        ind_recon = s * ngridx * ngridy;
        for(iy = 0; iy < ngridy; iy++)
            for(ix = 0; ix < ngridx; ix++)
                recon[ind_recon + iy * ngridx + ix] /= r;
    }

    memcpy(update, recon, dy * ngridx * ngridy * sizeof(float));
    memset(prox0x, 0, dy * ngridx * ngridy * sizeof(float));
    memset(prox0y, 0, dy * ngridx * ngridy * sizeof(float));
    memset(prox1, 0, dy * dt * dx * sizeof(float));

    // Iterations
    for(i = 0; i < num_iter; i++)
    {
        // initialize simdata to 0
        memset(simdata, 0, dy * dt * dx * sizeof(float));
        memset(adjdata, 0, dy * ngridx * ngridy * sizeof(float));

        // For each slice
        for(s = 0; s < dy; s++)
        {
            ind_recon = s * ngridx * ngridy;
            // compute proximal of the gradient in x and y directions
            // prox0 = prox0+c*grad(recon);
            // prox0 = prox0/max(1,abs(prox0)/lambda);
            for(iy = 0; iy < ngridy - 1; iy++)
                for(ix = 0; ix < ngridx - 1; ix++)
                {
                    prox0x[ind_recon + iy * ngridx + ix] +=
                        c * (recon[ind_recon + iy * ngridx + ix + 1] -
                             recon[ind_recon + iy * ngridx + ix]);
                    prox0y[ind_recon + iy * ngridx + ix] +=
                        c * (recon[ind_recon + (iy + 1) * ngridx + ix] -
                             recon[ind_recon + iy * ngridx + ix]);
                }
            for(iy = 0; iy < ngridy - 1; iy++)
                for(ix = 0; ix < ngridx - 1; ix++)
                {
                    upd = sqrt(prox0x[ind_recon + iy * ngridx + ix] *
                                   prox0x[ind_recon + iy * ngridx + ix] +
                               prox0y[ind_recon + iy * ngridx + ix] *
                                   prox0y[ind_recon + iy * ngridx + ix]) /
                          lambda;
                    upd = upd < 1 ? 1 : upd;
                    prox0x[ind_recon + iy * ngridx + ix] /= upd;
                    prox0y[ind_recon + iy * ngridx + ix] /= upd;
                }

            // compute proximal of the projections
            // prox1 = 1*(prox1+c*R(recon)-c*data)/(1+c);
            preprocessing(ngridx, ngridy, dx, center[s], &mov, gridx,
                          gridy);  // Outputs: mov, gridx, gridy

            // initialize sum_dist and update to zero
            memset(sum_dist, 0, (ngridx * ngridy) * sizeof(float));

            // For each projection angle
            for(p = 0; p < dt; p++)
            {
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
                    trim_coords(ngridx, ngridy, coordx, coordy, gridx, gridy, &asize, ax,
                                ay, &bsize, bx, by);

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

                    ind_data        = d + p * dx + s * dt * dx;
                    prox1[ind_data] = (prox1[ind_data] + c * simdata[ind_data] * r -
                                       c * data[ind_data]) /
                                      (1 + c);

                    // Calculate dist*dist
                    sum_dist2 = 0.0f;
                    for(n = 0; n < csize - 1; n++)
                    {
                        sum_dist2 += dist[n] * dist[n];
                        sum_dist[indi[n]] += dist[n];
                    }

                    // adjoint Radon of the prox1 for further computations
                    // adjdata = R^*(prox1)
                    if(sum_dist2 != 0.0f)
                        for(n = 0; n < csize - 1; n++)
                            adjdata[ind_recon + indi[n]] += r * prox1[ind_data] * dist[n];
                }
            }

            // copy recon = update
            memcpy(&recon[ind_recon], &update[ind_recon],
                   ngridx * ngridy * sizeof(float));

            // backward step. update with the divergence of prox0 and the
            // adjoint of prox1 update = update-c*R^*(prox1)-c*div(prox0);
            for(iy = 0; iy < ngridy; iy++)
                for(ix = 0; ix < ngridx; ix++)
                {
                    update[ind_recon + iy * ngridx + ix] -=
                        c * adjdata[ind_recon + iy * ngridx + ix];
                    if(ix == 0)
                        update[ind_recon + iy * ngridx + ix] +=
                            c * prox0x[ind_recon + iy * ngridx + ix];
                    else
                        update[ind_recon + iy * ngridx + ix] +=
                            c * (prox0x[ind_recon + iy * ngridx + ix] -
                                 prox0x[ind_recon + iy * ngridx + ix - 1]);
                    if(iy == 0)
                        update[ind_recon + iy * ngridx + ix] +=
                            c * prox0y[ind_recon + iy * ngridx + ix];
                    else
                        update[ind_recon + iy * ngridx + ix] +=
                            c * (prox0y[ind_recon + iy * ngridx + ix] -
                                 prox0y[ind_recon + (iy - 1) * ngridx + ix]);
                }

            // update of recon
            // recon = 2*update - recon
            for(iy = 0; iy < ngridy; iy++)
                for(ix = 0; ix < ngridx; ix++)
                    recon[ind_recon + iy * ngridx + ix] =
                        2 * update[ind_recon + iy * ngridx + ix] -
                        recon[ind_recon + iy * ngridx + ix];
        }
    }

    // scale result
    for(s = 0; s < dy; s++)
    {
        ind_recon = s * ngridx * ngridy;
        for(iy = 0; iy < ngridy; iy++)
            for(ix = 0; ix < ngridx; ix++)
                recon[ind_recon + iy * ngridx + ix] *= r;
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
    free(prox0x);
    free(prox0y);
    free(prox1);
    free(adjdata);
}
