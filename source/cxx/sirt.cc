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
#include "data.hh"
#include "utils.hh"

//======================================================================================//

typedef CpuData::init_data_t  init_data_t;
typedef CpuData::data_array_t data_array_t;

//======================================================================================//

int
cxx_sirt(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    // check to see if the C implementation is requested
    bool use_c_algorithm = GetEnv<bool>("TOMOPY_USE_C_SIRT", false);
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
        run_algorithm(sirt_cpu, sirt_cuda, data, dy, dt, dx, center, theta, recon, ngridx,
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
#if defined(TOMOPY_USE_CUDA)
        for(int i = 0; i < cuda_device_count(); ++i)
        {
            // set the device
            cudaSetDevice(i);
            // sync the device
            cudaDeviceSynchronize();
            // reset the device
            cudaDeviceReset();
        }
#endif
    }
    else
    {
        printf("[%lu] Threads remaining: %i...\n", GetThisThreadID(), remain);
    }

    return scast<int>(true);
}

//======================================================================================//

void
sirt_cpu_compute_projection(data_array_t& cpu_data, int p, int dy, int dt, int dx, int nx,
                            int ny, const float* theta)
{
    ConsumeParameters(dy);
    auto cache = cpu_data[GetThisThreadID() % cpu_data.size()];

    // calculate some values
    float    theta_p = fmodf(theta[p] + halfpi, twopi);
    farray_t tmp_update(dy * nx * ny, 0.0);

    for(int s = 0; s < dy; ++s)
    {
        const float* data  = cache->data() + s * dt * dx;
        const float* recon = cache->recon() + s * nx * ny;
        auto&        rot   = cache->rot();
        auto&        tmp   = cache->tmp();

        // reset intermediate data
        cache->reset();

        // forward-rotate
        cxx_rotate_ip<float>(rot, recon, -theta_p, nx, ny);

        // compute simdata
        for(int d = 0; d < dx; ++d)
        {
            float sum = 0.0f;
            for(int i = 0; i < nx; ++i)
                sum += rot[d * nx + i];
            float upd = (data[p * dx + d] - sum);
            for(int i = 0; i < nx; ++i)
                rot[d * nx + i] += upd;
        }

        // back-rotate object
        cxx_rotate_ip<float>(tmp, rot.data(), theta_p, nx, ny);

        // update local update array
        for(uintmax_t i = 0; i < scast<uintmax_t>(nx * ny); ++i)
            tmp_update[(s * nx * ny) + i] += tmp[i];
    }

    cache->upd_mutex()->lock();
    for(int s = 0; s < dy; ++s)
    {
        // update shared update array
        float* update = cache->update() + s * nx * ny;
        float* tmp    = tmp_update.data() + s * nx * ny;
        for(uintmax_t i = 0; i < scast<uintmax_t>(nx * ny); ++i)
            update[i] += tmp[i];
    }
    cache->upd_mutex()->unlock();
}

//======================================================================================//

void
sirt_cpu(const float* data, int dy, int dt, int dx, const float* /*center*/,
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
    init_data_t init_data =
        CpuData::initialize(nthreads, dy, dt, dx, ngridx, ngridy, recon, data,
                            update.data(), &upd_mutex, &sum_mutex);
    data_array_t cpu_data = std::get<0>(init_data);
    iarray_t     sum_dist = cxx_compute_sum_dist(dy, dt, dx, ngridx, ngridy, theta);

    //----------------------------------------------------------------------------------//
    for(int i = 0; i < num_iter; i++)
    {
        START_TIMER(t_start);
        TIMEMORY_AUTO_TIMER();

        // reset global update
        memset(update.data(), 0, recon_pixels * sizeof(float));

        // sync and reset
        CpuData::reset(cpu_data);

        // execute the loop over slices and projection angles
        execute<manager_t, data_array_t>(task_man, dt, std::ref(cpu_data),
                                         sirt_cpu_compute_projection, dy, dt, dx, ngridx,
                                         ngridy, theta);

        // update the global recon with global update and sum_dist
        for(uintmax_t ii = 0; ii < recon_pixels; ++ii)
        {
            if(sum_dist[ii] != 0.0f && dx != 0 && std::isfinite(update[ii]))
                recon[ii] += update[ii] / sum_dist[ii] / scast<float>(dx);
            else if(!std::isfinite(update[ii]))
            {
                std::cout << "update[" << ii << "] is not finite : " << update[ii]
                          << std::endl;
            }
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
