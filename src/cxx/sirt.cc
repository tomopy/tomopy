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
        run_gpu_algorithm(sirt_cpu, sirt_cuda, sirt_openacc, sirt_openmp, data, dy, dt,
                          dx, center, theta, recon, ngridx, ngridy, num_iter);
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
    auto  recon_rot = cxx_rotate(*recon_off, -theta_p, ngridx, ngridy);

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
    auto recon_tmp = cxx_rotate(recon_rot, theta_p, ngridx, ngridy);

    static Mutex _mutex;
    _mutex.lock();
    // update shared update array
    PRAGMA_SIMD
    for(uint64_t i = 0; i < recon_tmp.size(); ++i)
        (*update)[i] += recon_tmp[i];
    _mutex.unlock();
}

//============================================================================//

void
sirt_cpu(const float* data, int dy, int dt, int dx, const float*, const float* theta,
         float* recon, int ngridx, int ngridy, int num_iter)
{
    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    TIMEMORY_AUTO_TIMER("");

    int             nthreads = GetEnv("TOMOPY_NUM_THREADS", HW_CONCURRENCY);
    TaskRunManager* run_man  = cpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();

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
sirt_openacc(const float* data, int dy, int dt, int dx, const float*, const float* theta,
             float* recon, int ngridx, int ngridy, int num_iter)
{
    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    TIMEMORY_AUTO_TIMER("[openacc]");

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

            // For each projection angle
            for(int p = 0; p < dt; p++)
            {
                openacc_compute_projection(dt, dx, ngridx, ngridy, data, theta, s, p,
                                           simdata.data(), update.data(),
                                           recon_off.data());
            }

            for(int ii = 0; ii < (ngridx * ngridy); ++ii)
                _recon[ii] += update[ii] / static_cast<float>(dt);
        }
    }
}

//============================================================================//

void
sirt_openmp(const float* data, int dy, int dt, int dx, const float*, const float* theta,
            float* recon, int ngridx, int ngridy, int num_iter)
{
    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    TIMEMORY_AUTO_TIMER("[openmp]");

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

            // For each projection angle
            for(int p = 0; p < dt; p++)
            {
                openmp_compute_projection(dt, dx, ngridx, ngridy, data, theta, s, p,
                                          simdata.data(), update.data(),
                                          recon_off.data());
            }

            for(int ii = 0; ii < (ngridx * ngridy); ++ii)
                _recon[ii] += update[ii] / static_cast<float>(dt);
        }
    }
}

//============================================================================//
