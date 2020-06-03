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

#include "profiler.h"
#include "utils.h"

volatile unsigned long counter;

void
project(const float* obj, int oy, int ox, int oz, float* data, int dy, int dt, int dx,
        const float* center, const float* theta)
{
    if(dy == 0 || dt == 0 || dx == 0)
        return;

    float* gridx  = (float*) malloc((ox + 1) * sizeof(float));
    float* gridy  = (float*) malloc((oz + 1) * sizeof(float));
    float* coordx = (float*) malloc((oz + 1) * sizeof(float));
    float* coordy = (float*) malloc((ox + 1) * sizeof(float));
    float* ax     = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* ay     = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* bx     = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* by     = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* coorx  = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* coory  = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* dist   = (float*) malloc((ox + oz + 1) * sizeof(float));
    int*   indi   = (int*) malloc((ox + oz + 1) * sizeof(int));

    assert(coordx != NULL && coordy != NULL && ax != NULL && ay != NULL && by != NULL &&
           bx != NULL && coorx != NULL && coory != NULL && dist != NULL && indi != NULL);

    int   s, p, d;
    int   quadrant;
    float theta_p, sin_p, cos_p;
    float mov, xi, yi;
    int   asize, bsize, csize;

    preprocessing(ox, oz, dx, center[0], &mov, gridx,
                  gridy);  // Outputs: mov, gridx, gridy

    // For each projection angle
    for(p = 0; p < dt; p++)
    {
        // Calculate the sin and cos values
        // of the projection angle and find
        // at which quadrant on the cartesian grid.
        theta_p  = fmod(theta[p], 2 * M_PI);
        quadrant = calc_quadrant(theta_p);
        sin_p    = sinf(theta_p);
        cos_p    = cosf(theta_p);

        for(d = 0; d < dx; d++)
        {
            // Calculate coordinates
            xi = -ox - oz;
            yi = (1 - dx) / 2.0 + d + mov;
            calc_coords(ox, oz, xi, yi, sin_p, cos_p, gridx, gridy, coordx, coordy);

            // Merge the (coordx, gridy) and (gridx, coordy)
            trim_coords(ox, oz, coordx, coordy, gridx, gridy, &asize, ax, ay, &bsize, bx,
                        by);

            // Sort the array of intersection points (ax, ay) and
            // (bx, by). The new sorted intersection points are
            // stored in (coorx, coory). Total number of points
            // are csize.
            sort_intersections(quadrant, asize, ax, ay, bsize, bx, by, &csize, coorx,
                               coory);

            // Calculate the distances (dist) between the
            // intersection points (coorx, coory). Find the
            // indices of the pixels on the object grid.
            calc_dist(ox, oz, csize, coorx, coory, indi, dist);

            // For each slice
            for(s = 0; s < dy; s++)
            {
                // Calculate simdata
                calc_simdata(s, p, d, ox, oz, dt, dx, csize, indi, dist, obj,
                             data);  // Output: simulated data
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
}

void
project2(const float* objx, const float* objy, int oy, int ox, int oz, float* data,
         int dy, int dt, int dx, const float* center, const float* theta)
{
    if(dy == 0 || dt == 0 || dx == 0)
        return;

    float* gridx  = (float*) malloc((ox + 1) * sizeof(float));
    float* gridy  = (float*) malloc((oz + 1) * sizeof(float));
    float* coordx = (float*) malloc((oz + 1) * sizeof(float));
    float* coordy = (float*) malloc((ox + 1) * sizeof(float));
    float* ax     = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* ay     = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* bx     = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* by     = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* coorx  = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* coory  = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* dist   = (float*) malloc((ox + oz + 1) * sizeof(float));
    int*   indx   = (int*) malloc((ox + oz + 1) * sizeof(int));
    int*   indy   = (int*) malloc((ox + oz + 1) * sizeof(int));
    int*   indi   = (int*) malloc((ox + oz + 1) * sizeof(int));

    assert(coordx != NULL && coordy != NULL && ax != NULL && ay != NULL && by != NULL &&
           bx != NULL && coorx != NULL && coory != NULL && dist != NULL && indi != NULL);

    int   s, p, d;
    int   quadrant;
    float theta_p, sin_p, cos_p;
    float srcx, srcy, detx, dety, dv, vx, vy;
    float mov, xi, yi;
    int   asize, bsize, csize;

    preprocessing(ox, oz, dx, center[0], &mov, gridx,
                  gridy);  // Outputs: mov, gridx, gridy

    // For each projection angle
    for(p = 0; p < dt; p++)
    {
        // Calculate the sin and cos values
        // of the projection angle and find
        // at which quadrant on the cartesian grid.
        theta_p  = fmod(theta[p], 2 * M_PI);
        quadrant = calc_quadrant(theta_p);
        sin_p    = sinf(theta_p);
        cos_p    = cosf(theta_p);

        for(d = 0; d < dx; d++)
        {
            // Calculate coordinates
            xi = -ox - oz;
            yi = (1 - dx) / 2.0 + d + mov;

            srcx = xi * cos_p - yi * sin_p;
            srcy = xi * sin_p + yi * cos_p;
            detx = -xi * cos_p - yi * sin_p;
            dety = -xi * sin_p + yi * cos_p;

            dv = sqrt(pow(srcx - detx, 2) + pow(srcy - dety, 2));
            vx = (srcx - detx) / dv;
            vy = (srcy - dety) / dv;

            calc_coords(ox, oz, xi, yi, sin_p, cos_p, gridx, gridy, coordx, coordy);

            // Merge the (coordx, gridy) and (gridx, coordy)
            trim_coords(ox, oz, coordx, coordy, gridx, gridy, &asize, ax, ay, &bsize, bx,
                        by);

            // Sort the array of intersection points (ax, ay) and
            // (bx, by). The new sorted intersection points are
            // stored in (coorx, coory). Total number of points
            // are csize.
            sort_intersections(quadrant, asize, ax, ay, bsize, bx, by, &csize, coorx,
                               coory);

            // Calculate the distances (dist) between the
            // intersection points (coorx, coory). Find the
            // indices of the pixels on the object grid.
            calc_dist2(ox, oz, csize, coorx, coory, indx, indy, dist);

            // For each slice
            for(s = 0; s < dy; s++)
            {
                // Calculate simdata
                calc_simdata2(s, p, d, ox, oz, dt, dx, csize, indx, indy, dist, vx, vy,
                              objx, objy,
                              data);  // Output: simulated data
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
}

void
project3(const float* objx, const float* objy, const float* objz, int oy, int ox, int oz,
         float* data, int dy, int dt, int dx, const float* center, const float* theta,
         int axis)
{
    if(dy == 0 || dt == 0 || dx == 0)
        return;

    float* gridx  = (float*) malloc((ox + 1) * sizeof(float));
    float* gridy  = (float*) malloc((oz + 1) * sizeof(float));
    float* coordx = (float*) malloc((oz + 1) * sizeof(float));
    float* coordy = (float*) malloc((ox + 1) * sizeof(float));
    float* ax     = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* ay     = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* bx     = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* by     = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* coorx  = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* coory  = (float*) malloc((ox + oz + 2) * sizeof(float));
    float* dist   = (float*) malloc((ox + oz + 1) * sizeof(float));
    int*   indx   = (int*) malloc((ox + oz + 1) * sizeof(int));
    int*   indy   = (int*) malloc((ox + oz + 1) * sizeof(int));
    int*   indi   = (int*) malloc((ox + oz + 1) * sizeof(int));

    assert(coordx != NULL && coordy != NULL && ax != NULL && ay != NULL && by != NULL &&
           bx != NULL && coorx != NULL && coory != NULL && dist != NULL && indi != NULL);

    int   s, p, d;
    int   quadrant;
    float theta_p, sin_p, cos_p;
    float srcx, srcy, detx, dety, dv, vx, vy;
    float mov, xi, yi;
    int   asize, bsize, csize;

    preprocessing(ox, oz, dx, center[0], &mov, gridx,
                  gridy);  // Outputs: mov, gridx, gridy

    // For each projection angle
    for(p = 0; p < dt; p++)
    {
        // Calculate the sin and cos values
        // of the projection angle and find
        // at which quadrant on the cartesian grid.
        theta_p  = fmod(theta[p], 2 * M_PI);
        quadrant = calc_quadrant(theta_p);
        sin_p    = sinf(theta_p);
        cos_p    = cosf(theta_p);

        for(d = 0; d < dx; d++)
        {
            // Calculate coordinates
            xi = -ox - oz;
            yi = (1 - dx) / 2.0 + d + mov;

            srcx = xi * cos_p - yi * sin_p;
            srcy = xi * sin_p + yi * cos_p;
            detx = -xi * cos_p - yi * sin_p;
            dety = -xi * sin_p + yi * cos_p;

            dv = sqrt(pow(srcx - detx, 2) + pow(srcy - dety, 2));
            vx = (srcx - detx) / dv;
            vy = (srcy - dety) / dv;

            calc_coords(ox, oz, xi, yi, sin_p, cos_p, gridx, gridy, coordx, coordy);

            // Merge the (coordx, gridy) and (gridx, coordy)
            trim_coords(ox, oz, coordx, coordy, gridx, gridy, &asize, ax, ay, &bsize, bx,
                        by);

            // Sort the array of intersection points (ax, ay) and
            // (bx, by). The new sorted intersection points are
            // stored in (coorx, coory). Total number of points
            // are csize.
            sort_intersections(quadrant, asize, ax, ay, bsize, bx, by, &csize, coorx,
                               coory);

            // Calculate the distances (dist) between the
            // intersection points (coorx, coory). Find the
            // indices of the pixels on the object grid.
            calc_dist2(ox, oz, csize, coorx, coory, indx, indy, dist);

            // For each slice
            for(s = 0; s < dy; s++)
            {
                // Calculate simdata
                calc_simdata3(s, p, d, ox, oz, dt, dx, csize, indx, indy, dist, vx, vy,
                              objx, objy, objz, axis,
                              data);  // Output: simulated data
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
}
