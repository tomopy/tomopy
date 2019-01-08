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
#include "utils.hh"

BEGIN_EXTERN_C
#include "sirt.h"
#include "utils.h"
#include "utils_cuda.h"
#include "utils_openacc.h"
#include "utils_openmp.h"
END_EXTERN_C

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <numeric>

#define PRAGMA_SIMD _Pragma("omp simd")
#define PRAGMA_SIMD_REDUCTION(var) _Pragma("omp simd reducton(+ : var)")
#define HW_CONCURRENCY std::thread::hardware_concurrency()

//============================================================================//

int
cxx_sirt(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    // check to see if the C implementation is requested
    bool use_c_algorithm = GetEnv<bool>("TOMOPY_USE_C_SIRT", false);
    // if C implementation is requested, return non-zero (failure)
    if(use_c_algorithm)
        return (int) false;

    auto tid = ThreadPool::GetThisThreadID();
    ConsumeParameters(tid);

#if defined(TOMOPY_USE_TIMEMORY)
    tim::timer t(__FUNCTION__);
    t.format().get()->width(10);
    t.start();
#endif

    TIMEMORY_AUTO_TIMER("");
    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

#if defined(TOMOPY_USE_GPU)
    // TODO: select based on memory
    bool use_cpu = GetEnv<bool>("TOMOPY_USE_CPU", false);
    if(use_cpu)
        sirt_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
    else
        run_gpu_algorithm(sirt_cpu, sirt_cuda, sirt_openacc, sirt_cpu, data, dy, dt, dx,
                          center, theta, recon, ngridx, ngridy, num_iter);
#else
    sirt_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
#endif

#if defined(TOMOPY_USE_TIMEMORY)
    AutoLock l(TypeMutex<decltype(std::cout)>());
    std::cout << "[" << tid << "]> " << t.stop_and_return() << std::endl;
#endif

    return (int) true;
}

//============================================================================//

void
compute_projection(int dt, int dx, int ngridx, int ngridy, const float* data,
                   const float* theta, int s, int p, farray_t* simdata, farray_t* update,
                   farray_t* recon_off)
{
    // needed for recon to output at proper orientation
    float pi_offset = 0.5f * (float) M_PI;
    float fngridx   = ngridx;
    float theta_p   = fmodf(theta[p] + pi_offset, 2.0f * (float) M_PI);

    // Rotate object
    auto recon_rot = cxx_rotate(*recon_off, -theta_p, ngridx, ngridy);

    for(int d = 0; d < dx; d++)
    {
        int    pix_offset = d * ngridx;  // pixel offset
        int    idx_data   = d + p * dx + s * dt * dx;
        float* _simdata   = simdata->data() + idx_data;
        float* _recon_rot = recon_rot.data() + pix_offset;
        float  _sim       = 0.0f;

        // Calculate simulated data by summing up along x-axis
        PRAGMA_SIMD_REDUCTION(_sim)
        for(int n = 0; n < ngridx; n++)
            _sim += _recon_rot[n];

        // update shared simdata array
        (*simdata)[idx_data] += _sim;

        // Make update by backprojecting error along x-axis
        float upd = (data[idx_data] - *_simdata) / fngridx;
        PRAGMA_SIMD
        for(int n = 0; n < ngridx; n++)
            _recon_rot[n] += upd;
    }
    // Back-Rotate object
    auto tmp = cxx_rotate(recon_rot, theta_p, ngridx, ngridy);

    // update shared update array
    PRAGMA_SIMD
    for(uint64_t i = 0; i < tmp.size(); ++i)
        (*update)[i] += tmp[i];
}

//============================================================================//

void
sirt_cpu(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    TIMEMORY_AUTO_TIMER("");

    int             nthreads = GetEnv("TOMOPY_NUM_THREADS", HW_CONCURRENCY);
    TaskRunManager* run_man  = cpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();
    ThreadPool*  tp       = task_man->thread_pool();

    for(int i = 0; i < num_iter; i++)
    {
        farray_t simdata(dy * dt * dx, 0.0f);
        // For each slice
        for(int s = 0; s < dy; s++)
        {
            farray_t update(ngridx * ngridy, 0.0f);
            farray_t recon_off(ngridx * ngridy, 0.0f);

            int    slice_offset = s * ngridx * ngridy;
            float* _recon       = recon + slice_offset;

            // recon offset for the slice
            for(int ii = 0; ii < recon_off.size(); ++ii)
                recon_off[ii] = _recon[ii];

            TaskGroup<void> tg;
            // For each projection angle
            for(int p = 0; p < dt; p++)
            {
                task_man->exec(tg, compute_projection, dt, dx, ngridx, ngridy, data,
                               theta, s, p, &simdata, &update, &recon_off);
            }
            tg.join();

            for(int ii = 0; ii < (ngridx * ngridy); ++ii)
                _recon[ii] += update[ii] / static_cast<float>(dt);
        }
    }
}

//============================================================================//

#if !defined(TOMOPY_USE_CUDA)
void
sirt_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    sirt_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
}
#endif

//============================================================================//

void
sirt_openacc(const float* data, int dy, int dt, int dx, const float* center,
             const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    tim::enable_signal_detection();
    TIMEMORY_AUTO_TIMER("[openacc]");

    float* gridx   = (float*) malloc((ngridx + 1) * sizeof(float));
    float* gridy   = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordx  = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordy  = (float*) malloc((ngridx + 1) * sizeof(float));
    float* ax      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* ay      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* bx      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* by      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coorx   = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coory   = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* dist    = (float*) malloc((ngridx + ngridy) * sizeof(float));
    int*   indi    = (int*) malloc((ngridx + ngridy) * sizeof(int));
    float* simdata = (float*) malloc((dy * dt * dx) * sizeof(float));

    assert(coordx != nullptr && coordy != nullptr && ax != nullptr && ay != nullptr &&
           by != nullptr && bx != nullptr && coorx != nullptr && coory != nullptr &&
           dist != nullptr && indi != nullptr && simdata != nullptr);

    int   s, p, d, i, n;
    int   quadrant;
    float theta_p, sin_p, cos_p;
    float mov, xi, yi;
    int   asize, bsize, csize;
    float upd;
    int   ind_data, ind_recon;

    for(i = 0; i < num_iter; i++)
    {
        // initialize simdata to zero
        memset(simdata, 0, dy * dt * dx * sizeof(float));

        openacc_preprocessing(ngridx, ngridy, dx, center[0], &mov, gridx,
                              gridy);  // Outputs: mov, gridx, gridy

        // For each projection angle
        for(p = 0; p < dt; p++)
        {
            // Calculate the sin and cos values
            // of the projection angle and find
            // at which quadrant on the cartesian grid.
            theta_p  = fmod(theta[p], 2.0f * scast<float>(M_PI));
            quadrant = openacc_calc_quadrant(theta_p);
            sin_p    = sinf(theta_p);
            cos_p    = cosf(theta_p);

            // For each detector pixel
            for(d = 0; d < dx; d++)
            {
                // Calculate coordinates
                xi = -ngridx - ngridy;
                yi = (1 - dx) / 2.0f + d + mov;
                openacc_calc_coords(ngridx, ngridy, xi, yi, sin_p, cos_p, gridx, gridy,
                                    coordx, coordy);

                // Merge the (coordx, gridy) and (gridx, coordy)
                openacc_trim_coords(ngridx, ngridy, coordx, coordy, gridx, gridy, &asize,
                                    ax, ay, &bsize, bx, by);

                // Sort the array of intersection points (ax, ay) and
                // (bx, by). The new sorted intersection points are
                // stored in (coorx, coory). Total number of points
                // are csize.
                openacc_sort_intersections(quadrant, asize, ax, ay, bsize, bx, by, &csize,
                                           coorx, coory);

                // Calculate the distances (dist) between the
                // intersection points (coorx, coory). Find the
                // indices of the pixels on the reconstruction grid.
                openacc_calc_dist(ngridx, ngridy, csize, coorx, coory, indi, dist);

                // Calculate dist*dist
                float sum_dist2 = 0.0f;
                for(n = 0; n < csize - 1; n++)
                {
                    sum_dist2 += dist[n] * dist[n];
                }

                if(sum_dist2 != 0.0f)
                {
                    // For each slice
                    for(s = 0; s < dy; s++)
                    {
                        // Calculate simdata
                        openacc_calc_simdata(s, p, d, ngridx, ngridy, dt, dx, csize, indi,
                                             dist, recon,
                                             simdata);  // Output: simdata

                        // Update
                        ind_data  = d + p * dx + s * dt * dx;
                        ind_recon = s * ngridx * ngridy;
                        upd       = (data[ind_data] - simdata[ind_data]) / sum_dist2;
                        for(n = 0; n < csize - 1; n++)
                        {
                            recon[indi[n] + ind_recon] += upd * dist[n];
                        }
                    }
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

    tim::disable_signal_detection();
}

//============================================================================//

void
sirt_openmp(const float* data, int dy, int dt, int dx, const float* center,
            const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    tim::enable_signal_detection();
    TIMEMORY_AUTO_TIMER("[openmp]");

    float* gridx   = (float*) malloc((ngridx + 1) * sizeof(float));
    float* gridy   = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordx  = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordy  = (float*) malloc((ngridx + 1) * sizeof(float));
    float* ax      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* ay      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* bx      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* by      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coorx   = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coory   = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* dist    = (float*) malloc((ngridx + ngridy) * sizeof(float));
    int*   indi    = (int*) malloc((ngridx + ngridy) * sizeof(int));
    float* simdata = (float*) malloc((dy * dt * dx) * sizeof(float));

    assert(coordx != nullptr && coordy != nullptr && ax != nullptr && ay != nullptr &&
           by != nullptr && bx != nullptr && coorx != nullptr && coory != nullptr &&
           dist != nullptr && indi != nullptr && simdata != nullptr);

    int   s, p, d, i, n;
    int   quadrant;
    float theta_p, sin_p, cos_p;
    float mov, xi, yi;
    int   asize, bsize, csize;
    float upd;
    int   ind_data, ind_recon;

    for(i = 0; i < num_iter; i++)
    {
        // initialize simdata to zero
        memset(simdata, 0, dy * dt * dx * sizeof(float));

        openmp_preprocessing(ngridx, ngridy, dx, center[0], &mov, gridx,
                             gridy);  // Outputs: mov, gridx, gridy

        // For each projection angle
        for(p = 0; p < dt; p++)
        {
            // Calculate the sin and cos values
            // of the projection angle and find
            // at which quadrant on the cartesian grid.
            theta_p  = fmod(theta[p], 2.0f * scast<float>(M_PI));
            quadrant = openmp_calc_quadrant(theta_p);
            sin_p    = sinf(theta_p);
            cos_p    = cosf(theta_p);

            // For each detector pixel
            for(d = 0; d < dx; d++)
            {
                // Calculate coordinates
                xi = -ngridx - ngridy;
                yi = (1 - dx) / 2.0f + d + mov;
                openmp_calc_coords(ngridx, ngridy, xi, yi, sin_p, cos_p, gridx, gridy,
                                   coordx, coordy);

                // Merge the (coordx, gridy) and (gridx, coordy)
                openmp_trim_coords(ngridx, ngridy, coordx, coordy, gridx, gridy, &asize,
                                   ax, ay, &bsize, bx, by);

                // Sort the array of intersection points (ax, ay) and
                // (bx, by). The new sorted intersection points are
                // stored in (coorx, coory). Total number of points
                // are csize.
                openmp_sort_intersections(quadrant, asize, ax, ay, bsize, bx, by, &csize,
                                          coorx, coory);

                // Calculate the distances (dist) between the
                // intersection points (coorx, coory). Find the
                // indices of the pixels on the reconstruction grid.
                openmp_calc_dist(ngridx, ngridy, csize, coorx, coory, indi, dist);

                // Calculate dist*dist
                float sum_dist2 = 0.0f;
                for(n = 0; n < csize - 1; n++)
                {
                    sum_dist2 += dist[n] * dist[n];
                }

                if(sum_dist2 != 0.0f)
                {
                    // For each slice
                    for(s = 0; s < dy; s++)
                    {
                        // Calculate simdata
                        openmp_calc_simdata(s, p, d, ngridx, ngridy, dt, dx, csize, indi,
                                            dist, recon,
                                            simdata);  // Output: simdata

                        // Update
                        ind_data  = d + p * dx + s * dt * dx;
                        ind_recon = s * ngridx * ngridy;
                        upd       = (data[ind_data] - simdata[ind_data]) / sum_dist2;
                        for(n = 0; n < csize - 1; n++)
                        {
                            recon[indi[n] + ind_recon] += upd * dist[n];
                        }
                    }
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

    tim::disable_signal_detection();
}

//============================================================================//
