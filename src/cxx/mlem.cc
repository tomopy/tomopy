//  Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.
//  Copyright 2015. UChicago Argonne, LLC. This software was produced
//  under U.S. Government contract DE-AC02-06CH11357 for Argonne National
//  Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
//  UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
//  ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
//  modified to produce derivative works, such modified software should
//  be clearly marked, so as not to confuse it with the version available
//  from ANL.
//  Additionally, redistribution and use in source and binary forms, with
//  or without modification, are permitted provided that the following
//  conditions are met:
//      * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//      * Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in
//        the documentation andwith the
//        distribution.
//      * Neither the name of UChicago Argonne, LLC, Argonne National
//        Laboratory, ANL, the U.S. Government, nor the names of its
//        contributors may be used to endorse or promote products derived
//        from this software without specific prior written permission.
//  THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
//  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
//  Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
//  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//  ---------------------------------------------------------------
//   TOMOPY class header

#include "PTL/TaskManager.hh"
#include "PTL/TaskRunManager.hh"
#include "gpu.hh"
#include "utils.hh"
#include "utils_cuda.h"

BEGIN_EXTERN_C
#include "mlem.h"
#include "utils.h"
#include "utils_openacc.h"
#include "utils_openmp.h"
END_EXTERN_C

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <numeric>

#define PRAGMA_SIMD _Pragma("omp simd")
#define PRAGMA_SIMD_REDUCTION(var) _Pragma("omp simd reducton(+ : var)")
#define HW_CONCURRENCY std::thread::hardware_concurrency()

#if defined(TOMOPY_USE_CUDA)
extern void
mlem_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter);
#endif

//============================================================================//

int
cxx_mlem(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    // check to see if the C implementation is requested
    bool use_c_algorithm = GetEnv<bool>("TOMOPY_USE_C_MLEM", false);
    // if C implementation is requested, return non-zero (failure)
    if(use_c_algorithm)
        return (int) false;

#if defined(TOMOPY_USE_PTL)
    auto tid = GetThisThreadID();
#else
    static std::atomic<uintmax_t> tcounter;
    static thread_local auto      tid = tcounter++;
#endif
    ConsumeParameters(tid);

#if defined(TOMOPY_USE_TIMEMORY)
    tim::timer t(__FUNCTION__);
    t.format().get()->width(10);
    t.start();
#endif

    TIMEMORY_AUTO_TIMER("");
    printf("\n\t[%lu] %s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           GetThisThreadID(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

#if defined(TOMOPY_USE_GPU)
    // TODO: select based on memory
    bool use_cpu = GetEnv<bool>("TOMOPY_USE_CPU", false);
    if(use_cpu)
        mlem_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
    else
        run_gpu_algorithm(mlem_cpu, mlem_cuda, mlem_openacc, mlem_openmp, data, dy, dt,
                          dx, center, theta, recon, ngridx, ngridy, num_iter);
#else
    mlem_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
#endif

#if defined(TOMOPY_USE_TIMEMORY)
    AutoLock l(TypeMutex<decltype(std::cout)>());
    std::cout << "[" << tid << "]> " << t.stop_and_return() << std::endl;
#endif

    return (int) true;
}

//============================================================================//

void
cpu_mlem_compute_pixel(int ngridx, int ngridy, int dy, int dt, int dx, const float* mov,
                       const float* gridx, const float* gridy, float* sum_dist,
                       float* update, const float* data, const float* recon, int s, int p,
                       int quadrant, float sin_p, float cos_p)
{
    int ry = ngridx;
    int rz = ngridy;
    /*
    float gridx_gt = gridx[0] + 0.01f;
    float gridx_le = gridx[ngridx] - 0.01f;
    float gridy_gt = gridy[0] + 0.01f;
    float gridy_le = gridy[ngridy] - 0.01f;
    */

    float xoff = floorf(0.5f * ngridx) - ((ngridx % 2 == 0) ? 0.5f : 0.0f);
    float yoff = floorf(0.5f * ngridy) - ((ngridy % 2 == 0) ? 0.5f : 0.0f);

    // PRINT_HERE("");
    for(int d = 0; d < dx; ++d)
    {
        // Calculate coordinates
        /*
        float xi     = -ngridx - ngridy;
        float yi     = 0.5f * (1 - dx) + d + *mov;
        float srcx   = xi * cos_p - yi * sin_p;
        float srcy   = xi * sin_p + yi * cos_p;
        float detx   = -xi * cos_p - yi * sin_p;
        float dety   = -xi * sin_p + yi * cos_p;
        float slope  = (srcy - dety) / (srcx - detx);
        float islope = (srcx - detx) / (srcy - dety);*/

        int i = d;
        for(int j = 0; j < ngridy; ++j)
        {
            // centered coordinates
            float cx0 = float(i) - xoff;
            float cy0 = float(j) - yoff;
            // transform coords and then offset to global coords
            float x0 = (cx0 * cos_p - cy0 * sin_p);  // + xoff;
            float y0 = (cx0 * sin_p + cy0 * cos_p);  // + yoff;
            // centered coordinates
            float mx = floorf(x0);
            float my = floorf(y0);
            float px = ceilf(x0);
            float py = ceilf(y0);

            // printf("x,y = [%8.3f, %8.3f], mx,my = [%8.3f, %8.3f], px,py = [%8.3f,
            // %8.3f]\n",
            //       x0, y0, mx, my, px, py);

            //--------------------------------------------------------------------------//
            auto compute_indi = [x0, y0, ry, rz](float x1, float y1) {
                float _midx = 0.5f * (x1 + x0);
                float _x1   = _midx + 0.5f * ry;
                float _i1   = (int) (_midx + 0.5f * ry);
                int   _indx = _i1 - (_i1 > _x1);

                float _midy = 0.5f * (y1 + y0);
                float _x2   = _midy + 0.5f * rz;
                float _i2   = (int) (_midy + 0.5f * rz);
                int   _indy = _i2 - (_i2 > _x2);

                // compute index
                int indi = _indy + (_indx * rz);
                return indi;
            };
            //--------------------------------------------------------------------------//
            auto update_indi = [&](int indi, float simdata, float dist) {
                sum_dist[indi] += dist;
                if(simdata != 0.0f)
                {
                    int   index_data = d + p * dx + s * dt * dx;
                    float upd        = data[index_data] / simdata;
                    update[indi] += upd * dist;
                }
            };
            //--------------------------------------------------------------------------//

            // data offset
            int index_model = s * ry * rz;

            int mindi = compute_indi(mx, my);
            if(mindi >= 0 && mindi < ry * rz)
            {
                // compute distance
                float mdist = sqrtf(powf(x0 - mx, 2.0f) + powf(y0 - my, 2.0f));
                // Calculate simdata
                float msimdata = recon[mindi + index_model] * mdist;
                // Update
                update_indi(mindi, msimdata, mdist);
            }

            int pindi = compute_indi(px, py);
            if(pindi >= 0 && pindi < ry * rz)
            {
                // compute distance
                float pdist = sqrtf(powf(px - x0, 2.0f) + powf(py - y0, 2.0f));
                // Calculate simdata
                float psimdata = recon[pindi + index_model] * pdist;
                // Update
                update_indi(pindi, psimdata, pdist);
            }

            // printf("[%2i, %2i]> [x,y] = [%8.3f, %8.3f], indi = %i, dist = %8.3f,
            // sum_dist = %8.3f, update = %8.3f\n",
            //       i, j, x0, y0, indi, dist, sum_dist[indi], update[indi]);
        }
    }
}

//============================================================================//

void
cpu_mlem_compute_projection(int ngridx, int ngridy, int dy, int dt, int dx,
                            const float* theta, const float* mov, const float* gridx,
                            const float* gridy, const float* data, float* recon, int s,
                            float* sum_dist, float* update)
{
    int   nx           = ngridx;
    int   ny           = ngridy;
    float theta_offset = 0.5f * (float) M_PI;

    for(int p = 0; p < dt; ++p)
    {
        // START_TIMER(p_timer);
        // Calculate the sin and cos values
        // of the projection angle and find
        // at which quadrant on the cartesian grid.
        float theta_p  = fmodf(theta[p] + theta_offset, 2.0f * (float) M_PI);
        float sin_p    = sinf(theta_p);
        float cos_p    = cosf(theta_p);
        int   quadrant = calc_quadrant(theta_p);

        cpu_mlem_compute_pixel(ngridx, ngridy, dy, dt, dx, mov, gridx, gridy, sum_dist,
                               update, data, recon, s, p, quadrant, sin_p, cos_p);

        // REPORT_TIMER(p_timer, "\t\tangle", p, dt);
    }

    int offset = s * ngridx * ngridy;
    for(int i = 0; i < (nx * ny); ++i)
    {
        if(sum_dist[i] != 0.0f)
        {
            recon[i + offset] *= update[i] / sum_dist[i];
        }
    }
}

//============================================================================//

void
mlem_cpu(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    printf("\n\t[%lu] %s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           GetThisThreadID(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    TIMEMORY_AUTO_TIMER("");

    bool use_classic = GetEnv<bool>("TOMOPY_USE_CLASSIC_MLEM", false);

    if(!use_classic)
    {
        // GPU allocated copies
        float* mov      = cpu_malloc<float>(1);
        float* gridx    = cpu_malloc<float>(ngridx + 1);
        float* gridy    = cpu_malloc<float>(ngridy + 1);
        float* update   = cpu_malloc<float>(ngridx * ngridy);
        float* sum_dist = cpu_malloc<float>(ngridx * ngridy);

        for(int i = 0; i < num_iter; i++)
        {
            START_TIMER(t_start);
            // For each slice
            for(int s = 0; s < dy; s++)
            {
                START_TIMER(s_start);

                preprocessing(ngridx, ngridy, dx, center[s], mov, gridx, gridy);
                // Outputs: mov, gridx, gridy

                // initialize sum_dist and update to zero
                memset(sum_dist, 0, (ngridx * ngridy) * sizeof(float));
                memset(update, 0, (ngridx * ngridy) * sizeof(float));

                // For each projection angle
                cpu_mlem_compute_projection(ngridx, ngridy, dy, dt, dx, theta, mov, gridx,
                                            gridy, data, recon, s, sum_dist, update);

                if(dy > 1)
                    REPORT_TIMER(s_start, "    slice", s, dy);
            }
            REPORT_TIMER(t_start, "iteration", i, num_iter);
        }

        delete[] mov;
        delete[] gridx;
        delete[] gridy;
        delete[] update;
        delete[] sum_dist;
    }
    else
    {
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

        assert(coordx != NULL && coordy != NULL && ax != NULL && ay != NULL &&
               by != NULL && bx != NULL && coorx != NULL && coory != NULL &&
               dist != NULL && indi != NULL && simdata != NULL && sum_dist != NULL &&
               update != NULL);

        int   s, p, d, i, m, n;
        int   quadrant;
        float theta_p, sin_p, cos_p;
        float mov, xi, yi;
        int   asize, bsize, csize;
        float upd;
        int   ind_data, ind_recon;
        float sum_dist2;

        for(i = 0; i < num_iter; i++)
        {
            START_TIMER(t_start);
            // initialize simdata to zero
            memset(simdata, 0, dy * dt * dx * sizeof(float));

            // For each slice
            for(s = 0; s < dy; s++)
            {
                START_TIMER(s_start);
                preprocessing(ngridx, ngridy, dx, center[s], &mov, gridx,
                              gridy);  // Outputs: mov, gridx, gridy

                // initialize sum_dist and update to zero
                memset(sum_dist, 0, (ngridx * ngridy) * sizeof(float));
                memset(update, 0, (ngridx * ngridy) * sizeof(float));

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
                    printf("theta = %8.3f\n", theta_p * (180.0 / (float) M_PI));
                    // For each detector pixel
                    for(d = 0; d < dx; d++)
                    {
                        //#if defined(DEBUG)
                        memset(coordx, 0, (ngridy + 1) * sizeof(float));
                        memset(coordy, 0, (ngridx + 1) * sizeof(float));
                        memset(coorx, 0, (ngridx + ngridy) * sizeof(float));
                        memset(coory, 0, (ngridx + ngridy) * sizeof(float));
                        memset(dist, 0, (ngridx + ngridy) * sizeof(float));
                        memset(indi, 0, (ngridx + ngridy) * sizeof(int));
                        //#endif
                        // Calculate coordinates
                        xi = -ngridx - ngridy;
                        yi = 0.5f * (1 - dx) + d + mov;
                        calc_coords(ngridx, ngridy, xi, yi, sin_p, cos_p, gridx, gridy,
                                    coordx, coordy);
                        /*
                    printf("\n");
                    for(int n = 0; n <= ngridx; ++n)
                        printf("\tgridx[%2i] = %8.3f, gridy[%2i] = %8.3f, coordx[%2i] = "
                               "%8.3f, coordy[%2i] = %8.3f\n",
                               n, gridx[n], n, gridy[n], n, coordx[n], n, coordy[n]);
                    */
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
                            // printf("sum_dist[%i] = %f\n", indi[n], sum_dist[indi[n]]);
                        }

                        // Update
                        if(sum_dist2 != 0.0f)
                        {
                            ind_data = d + p * dx + s * dt * dx;
                            upd      = data[ind_data] / simdata[ind_data];
                            for(n = 0; n < csize - 1; n++)
                            {
                                update[indi[n]] += upd * dist[n];
                                // printf("update[%i] = %f\n", indi[n], update[indi[n]]);
                            }
                        }

                        //#if defined(DEBUG)
                        float _size[] = { (float) asize, (float) bsize, (float) csize, xi,
                                          yi };
                        print_cpu_array(5, 1, _size, i, s, p, d, "sizes");
                        // print_cpu_array(ngridx + 1, 1, gridx, i, s, p, d, "gridx");
                        // print_cpu_array(ngridy + 1, 1, gridy, i, s, p, d, "gridy");
                        print_cpu_array(ngridy + 1, 1, coordx, i, s, p, d, "coordx");
                        print_cpu_array(ngridx + 1, 1, coordy, i, s, p, d, "coordy");
                        print_cpu_array(ngridx + ngridy, 1, coorx, i, s, p, d, "coorx");
                        print_cpu_array(ngridx + ngridy, 1, coory, i, s, p, d, "coory");
                        print_cpu_array(ngridx + ngridy, 1, dist, i, s, p, d, "dist");
                        print_cpu_array(ngridx + ngridy, 1, indi, i, s, p, d, "indi");
                        //#endif
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
                if(dy > 1)
                    REPORT_TIMER(s_start, "    slice", s, dy);
            }
            REPORT_TIMER(t_start, "iteration", i, num_iter);
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
}

//============================================================================//

#if !defined(TOMOPY_USE_CUDA)
void
mlem_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    printf("\n\t[%lu] %s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           GetThisThreadID(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);
    mlem_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
}
#endif

//============================================================================//

void
mlem_openacc(const float* data, int dy, int dt, int dx, const float* center,
             const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    printf("\n\t[%lu] %s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           GetThisThreadID(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);
    mlem_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
}

//============================================================================//

void
mlem_openmp(const float* data, int dy, int dt, int dx, const float* center,
            const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    printf("\n\t[%lu] %s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           GetThisThreadID(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);
    mlem_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
}

//============================================================================//
