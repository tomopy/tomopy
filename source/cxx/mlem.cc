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
//   TOMOPY implementation

#include "common.hh"
#include "utils.hh"
#include "utils_cuda.hh"

BEGIN_EXTERN_C
#include "cxx_extern.h"
#include "utils.h"
END_EXTERN_C

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <numeric>

//======================================================================================//

typedef cpu_data::int_type     int_type;
typedef cpu_data::init_data_t  init_data_t;
typedef cpu_data::data_array_t data_array_t;

//======================================================================================//

int
cxx_mlem(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    // check to see if the C implementation is requested
    bool use_c_algorithm = GetEnv<bool>("TOMOPY_USE_C_MLEM", false);
    use_c_algorithm      = GetEnv<bool>("TOMOPY_USE_C_ALGORITHMS", use_c_algorithm);
    // if C implementation is requested, return non-zero (failure)
    if(use_c_algorithm)
        return scast<int>(false);

    auto tid = GetThisThreadID();
    ConsumeParameters(tid);
    static std::atomic<int> active;
    int                     count = active++;

    START_TIMER(cxx_timer);
    TIMEMORY_AUTO_TIMER("");

    printf("[%lu]> %s : nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i\n",
           GetThisThreadID(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    {
        TIMEMORY_AUTO_TIMER("");
        run_algorithm(mlem_cpu, mlem_cuda, data, dy, dt, dx, center, theta, recon, ngridx,
                      ngridy, num_iter);
    }

    auto tcount = GetEnv("TOMOPY_PYTHON_THREADS", HW_CONCURRENCY);
    auto remain = --active;
    REPORT_TIMER(cxx_timer, __FUNCTION__, count, tcount);
    if(remain == 0)
    {
        std::stringstream ss;
        PrintEnv(ss);
        printf("[%lu] Reporting environment...\n\n%s\n", GetThisThreadID(),
               ss.str().c_str());
    }
    else
    {
        printf("[%lu] Threads remaining: %i...\n", GetThisThreadID(), remain);
    }

    return scast<int>(true);
}

//======================================================================================//

void
mlem_cpu_compute_projection(data_array_t& _cpu_data, int _s, int p, int dy, int dt,
                            int dx, int nx, int ny, const float* theta,
                            uint32_t* global_sum_dist)
{
    ConsumeParameters(dy);
    auto _cache = _cpu_data[GetThisThreadID() % _cpu_data.size()];

    // calculate some values
    float theta_p   = fmodf(theta[p] + halfpi, twopi);
    auto& use_rot   = _cache->use_rot();
    auto& use_tmp   = _cache->use_tmp();
    auto& recon_rot = _cache->rot();
    auto& recon_tmp = _cache->tmp();

    // Forward-Rotate object
    cxx_rotate_ip<int_type>(use_rot, use_tmp.data(), -theta_p, nx, ny, CPU_NN);

    for(int s = 0; s < dy; ++s)
    {
        const float* data         = _cache->data() + s * dt * dx;
        const float* recon        = _cache->recon() + s * nx * ny;
        float*       update       = _cache->update() + s * nx * ny;
        uint32_t*    sum_dist     = global_sum_dist + s * nx * ny;
        uint16_t*    sum_dist_tmp = _cache->sum_dist();

        // reset intermediate data
        _cache->reset();

        cxx_rotate_ip<float>(recon_rot, recon, -theta_p, nx, ny);
        for(int d = 0; d < dx; ++d)
        {
            float sum = 0.0f;
            for(int i = 0; i < nx; ++i)
                sum += recon_rot[d * nx + i];
            for(int i = 0; i < nx; ++i)
                sum_dist_tmp[d * nx + i] += (use_rot[d * nx + i] > 0) ? 1 : 0;
            if(sum != 0.0f)
            {
                float upd = data[p * dx + d] / sum;
                if(std::isfinite(upd))
                    for(int i = 0; i < nx; ++i)
                        recon_rot[d * nx + i] += upd;
            }
        }
        // Back-Rotate object
        cxx_rotate_ip<float>(recon_tmp, recon_rot.data(), theta_p, nx, ny);

        // update shared update array
        _cache->upd_mutex()->lock();
        for(uintmax_t i = 0; i < scast<uintmax_t>(nx * ny); ++i)
            update[i] += recon_tmp[i];
        _cache->upd_mutex()->unlock();

        // update shared sum_dist array
        _cache->sum_mutex()->lock();
        for(uintmax_t i = 0; i < scast<uintmax_t>(nx * ny); ++i)
            sum_dist[i] += sum_dist_tmp[i];
        _cache->sum_mutex()->unlock();
    }
}

//======================================================================================//

void
mlem_cpu(const float* data, int dy, int dt, int dx, const float* /*center*/,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    typedef decltype(HW_CONCURRENCY) nthread_type;

    printf("[%lu]> %s : nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i\n",
           GetThisThreadID(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    // explicitly set OpenMP number of threads to 1 so OpenCV doesn't try to
    // create (HW_CONCURRENCY * PYTHON_NUM_THREADS * TOMOPY_NUM_THREADS) threads
    setenv("OMP_NUM_THREADS", "1", 1);

    // compute some properties (expected python threads, max threads, device assignment)
    auto min_threads = nthread_type(1);
    auto pythreads   = GetEnv("TOMOPY_PYTHON_THREADS", HW_CONCURRENCY);
    auto max_threads = HW_CONCURRENCY / std::max(pythreads, min_threads);
    auto nthreads    = std::max(GetEnv("TOMOPY_NUM_THREADS", max_threads), min_threads);

#if defined(TOMOPY_USE_PTL)
    typedef TaskManager manager_t;
    TaskRunManager*     run_man = cpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();
#else
    typedef void manager_t;
    void*        task_man = nullptr;
#endif

    TIMEMORY_AUTO_TIMER("");

    Mutex       upd_mutex;
    Mutex       sum_mutex;
    uintmax_t   recon_pixels = scast<uintmax_t>(dy * ngridx * ngridy);
    farray_t    update(recon_pixels, 0.0f);
    uarray_t    sum_dist(recon_pixels, 0);
    init_data_t init_data =
        cpu_data::initialize(nthreads, dy, dt, dx, ngridx, ngridy, recon, data,
                             update.data(), &upd_mutex, &sum_mutex);
    data_array_t _cpu_data = std::get<0>(init_data);

    //----------------------------------------------------------------------------------//
    for(int i = 0; i < num_iter; i++)
    {
        START_TIMER(t_start);
        // reset the simulation data

        // reset global update and sum_dist
        memset(update.data(), 0, recon_pixels * sizeof(float));
        memset(sum_dist.data(), 0, recon_pixels * sizeof(float));

        // sync and reset
        cpu_data::reset(_cpu_data);

        // execute the loop over slices and projection angles
        execute<manager_t, data_array_t>(task_man, 1, dt, std::ref(_cpu_data),
                                         mlem_cpu_compute_projection, dy, dt, dx, ngridx,
                                         ngridy, theta, sum_dist.data());

        // update the global recon with global update and sum_dist
        for(uintmax_t ii = 0; ii < recon_pixels; ++ii)
        {
            if(sum_dist[ii] != 0.0f)
                recon[ii] *= update[ii] / sum_dist[ii] / scast<float>(dx);
        }
        REPORT_TIMER(t_start, "iteration", i, num_iter);
    }

    printf("\n");
}

//======================================================================================//
#if !defined(TOMOPY_USE_CUDA)
void
mlem_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    mlem_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
}
#endif
//======================================================================================//
