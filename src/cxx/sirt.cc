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

#include "common.hh"
#include "utils.hh"

BEGIN_EXTERN_C
#include "sirt.h"
#include "utils.h"
#include "utils_cuda.h"
#include "utils_openacc.h"
#include "utils_openmp.h"
END_EXTERN_C

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <numeric>

//======================================================================================//

int
cxx_sirt(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    // check to see if the C implementation is requested
    bool use_c_algorithm = GetEnv<bool>("TOMOPY_USE_C_SIRT", false);
    // if C implementation is requested, return non-zero (failure)
    if(use_c_algorithm)
        return (int) false;

    auto tid = GetThisThreadID();
    ConsumeParameters(tid);

    START_TIMER(cxx_timer);
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

    REPORT_TIMER(cxx_timer, __FUNCTION__, 0, 0);

    return (int) true;
}

//======================================================================================//

void
compute_projection(int dy, int dt, int dx, int nx, int ny, const float* theta, int s,
                   int p, float* m_recon, float* m_simdata, const float* m_data,
                   float* update)
{
    static thread_local cpu_rotate_data* _cache = nullptr;
    static Mutex                         _mutex;
    if(!_cache)
    {
        _cache = new cpu_rotate_data(GetThisThreadID(), dy, dt, dx, nx, ny, nx, ny,
                                     m_recon, m_simdata, m_data);
    }

    // needed for recon to output at proper orientation
    float theta_rad_p = fmodf(theta[p], pi) + halfpi;
    float theta_deg_p = theta_rad_p * degrees;
    // these structures are cached and re-used
    const float* data      = m_data + s * dt * dx;
    float*       simdata   = m_simdata + s * dt * dx;
    float*       recon     = m_recon + s * nx * ny;
    farray_t&    recon_rot = _cache->rot();
    farray_t&    recon_tmp = _cache->tmp();
    int          px        = _cache->px();
    int          py        = _cache->py();
    float        fngridx   = static_cast<float>(nx);

    // Forward-Rotate object
    memset(recon_rot.data(), 0, px * py * sizeof(float));
    cxx_affine_transform(recon_rot, recon, -theta_rad_p, -theta_deg_p, px, py);

    for(int d = 0; d < dx; d++)
    {
        int pix_offset = d * nx + px;  // pixel offset
        int idx_data   = d + p * dx;
        // instead of including all the offsets later in the
        // index lookup, offset the pointer itself
        // this should make it easier for compiler to apply SIMD
        const float* _data      = data + idx_data;
        float*       _simdata   = simdata + idx_data;
        float*       _recon_rot = recon_rot.data() + pix_offset;
        float        _sum       = 0.0f;

        // Calculate simulated data by summing up along x-axis
        for(int n = 0; n < nx; ++n)
            _sum += _recon_rot[n];
        *_simdata += _sum;

        // Make update by backprojecting error along x-axis
        float upd = (*_data - *_simdata) / fngridx;
        for(int n = 0; n < nx; ++n)
            _recon_rot[n] += upd;
    }

    // Back-Rotate object
    memset(recon_tmp.data(), 0, px * py * sizeof(float));
    cxx_affine_transform(recon_tmp, recon_rot.data(), theta_rad_p, theta_deg_p, px, py);

    // update shared update array
    _mutex.lock();
    for(uintmax_t i = 0; i < recon_tmp.size(); ++i)
        update[i] += recon_tmp[i] / static_cast<float>(dx);
    _mutex.unlock();
}

//======================================================================================//

void
sirt_cpu(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

#if defined(TOMOPY_USE_PTL)
    int             nthreads = GetEnv("TOMOPY_NUM_THREADS", HW_CONCURRENCY);
    TaskRunManager* run_man  = cpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();
    init_thread_data(run_man->GetThreadPool());
#endif

    TIMEMORY_AUTO_TIMER("");

    //----------------------------------------------------------------------------------//
    for(int i = 0; i < num_iter; i++)
    {
        START_TIMER(t_start);
        // reset the simulation data
        farray_t simdata(dy * dt * dx, 0.0f);

        // For each slice
        for(int s = 0; s < dy; s++)
        {
            // for each thread, calculate all the offsets of the data
            // for this slice
            farray_t update(ngridx * ngridy, 0.0f);

#if defined(TOMOPY_USE_PTL)
            TaskGroup<void> tg;
            for(int p = 0; p < dt; ++p)
                task_man->exec(tg, compute_projection, dy, dt, dx, ngridx, ngridy, theta,
                               s, p, recon, simdata.data(), data, update.data());
            tg.join();
#else
            for(int p = 0; p < dt; ++p)
                compute_projection(dy, dt, dx, ngridx, ngridy, theta, s, p, recon,
                                   simdata.data(), data, update.data());
#endif
            for(int j = 0; j < (ngridx * ngridy); ++j)
                recon[j + s * ngridx * ngridy] += update[j];
        }
        REPORT_TIMER(t_start, "iteration", i, num_iter);
    }

    printf("\n");
}

//======================================================================================//
#if !defined(TOMOPY_USE_CUDA)
void
sirt_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    sirt_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
}
#endif
//======================================================================================//

void
sirt_openacc(const float* data, int dy, int dt, int dx, const float*, const float* theta,
             float* recon, int ngridx, int ngridy, int num_iter)
{
    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    TIMEMORY_AUTO_TIMER("[openacc]");

    for(int i = 0; i < num_iter; i++)
    {
        auto     t_start = std::chrono::system_clock::now();
        farray_t simdata(dy * dt * dx, 0.0f);
        // For each slice
        for(int s = 0; s < dy; s++)
        {
            farray_t update(ngridx * ngridy, 0.0f);
            farray_t recon_off(ngridx * ngridy, 0.0f);

            int    slice_offset = s * ngridx * ngridy;
            float* _recon       = recon + slice_offset;

            // recon offset for the slice
            for(uintmax_t ii = 0; ii < recon_off.size(); ++ii)
                recon_off[ii] = _recon[ii];

            // For each projection angle
            for(int p = 0; p < dt; p++)
            {
                openacc_compute_projection(dt, dx, ngridx, ngridy, data, theta, s, p,
                                           simdata.data(), update.data(),
                                           recon_off.data());
            }

            for(int ii = 0; ii < (ngridx * ngridy); ++ii)
                _recon[ii] += update[ii] / static_cast<float>(dx);
        }
        auto                          t_end           = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = t_end - t_start;
        printf("[%li]> iteration %3i of %3i... %5.2f seconds\n", GetThisThreadID(), i,
               num_iter, elapsed_seconds.count());
    }
}

//======================================================================================//

void
sirt_openmp(const float* data, int dy, int dt, int dx, const float*, const float* theta,
            float* recon, int ngridx, int ngridy, int num_iter)
{
    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    TIMEMORY_AUTO_TIMER("[openmp]");

    for(int i = 0; i < num_iter; i++)
    {
        auto     t_start = std::chrono::system_clock::now();
        farray_t simdata(dy * dt * dx, 0.0f);
        // For each slice
        for(int s = 0; s < dy; s++)
        {
            farray_t update(ngridx * ngridy, 0.0f);
            farray_t recon_off(ngridx * ngridy, 0.0f);

            int    slice_offset = s * ngridx * ngridy;
            float* _recon       = recon + slice_offset;

            // recon offset for the slice
            for(uintmax_t ii = 0; ii < recon_off.size(); ++ii)
                recon_off[ii] = _recon[ii];

            // For each projection angle
            for(int p = 0; p < dt; p++)
            {
                openmp_compute_projection(dt, dx, ngridx, ngridy, data, theta, s, p,
                                          simdata.data(), update.data(),
                                          recon_off.data());
            }

            for(int ii = 0; ii < (ngridx * ngridy); ++ii)
                _recon[ii] += update[ii] / static_cast<float>(dx);
        }
        auto                          t_end           = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = t_end - t_start;
        printf("[%li]> iteration %3i of %3i... %5.2f seconds\n", GetThisThreadID(), i,
               num_iter, elapsed_seconds.count());
    }
}

//======================================================================================//
