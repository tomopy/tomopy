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

#include <cstdlib>

//======================================================================================//

typedef CpuData::init_data_t  init_data_t;
typedef CpuData::data_array_t data_array_t;

//======================================================================================//

// directly call the CPU version
DLL void
mlem_cpu(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
         RuntimeOptions*);

// directly call the GPU version
DLL void
mlem_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
          RuntimeOptions*);

//======================================================================================//

int
cxx_mlem(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
         int pool_size, const char* interp, const char* device, int* grid_size,
         int* block_size)
{
    auto tid = GetThisThreadID();
    // registration
    static Registration registration;
    // local count for the thread
    int count = registration.initialize();
    // number of threads started at Python level
    auto tcount = GetEnv("TOMOPY_PYTHON_THREADS", HW_CONCURRENCY);

    // configured runtime options
    RuntimeOptions opts(pool_size, interp, device, grid_size, block_size);

    // create the thread-pool
    opts.init();

    START_TIMER(cxx_timer);
    TIMEMORY_AUTO_TIMER("");

    printf("[%lu]> %s : nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i\n", tid,
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    try
    {
        if(opts.device.key == "gpu")
        {
            mlem_cuda(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter,
                      &opts);
        }
        else
        {
            mlem_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter,
                     &opts);
        }
    }
    catch(std::exception& e)
    {
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cerr << "[TID: " << tid << "] " << e.what()
                  << "\nFalling back to CPU algorithm..." << std::endl;
        return EXIT_FAILURE;
    }

    registration.cleanup(&opts);
    REPORT_TIMER(cxx_timer, __FUNCTION__, count, tcount);

    return EXIT_SUCCESS;
}

//======================================================================================//

void
mlem_cpu_compute_projection(data_array_t& cpu_data, int p, int dy, int dt, int dx, int nx,
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
        cxx_rotate_ip<float>(rot, recon, -theta_p, nx, ny, cache->interpolation());

        // compute simdata
        for(int d = 0; d < dx; ++d)
        {
            float sum = 0.0f;
            for(int i = 0; i < nx; ++i)
                sum += rot[d * nx + i];
            if(sum != 0.0f)
            {
                float upd = data[p * dx + d] / sum;
                if(std::isfinite(upd))
                {
                    for(int i = 0; i < nx; ++i)
                        rot[d * nx + i] += upd;
                }
            }
        }

        // back-rotate object
        cxx_rotate_ip<float>(tmp, rot.data(), theta_p, nx, ny, cache->interpolation());

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
mlem_cpu(const float* data, int dy, int dt, int dx, const float*, const float* theta,
         float* recon, int ngridx, int ngridy, int num_iter, RuntimeOptions* opts)
{
    printf("[%lu]> %s : nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i\n",
           GetThisThreadID(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    TIMEMORY_AUTO_TIMER("");

    uintmax_t   recon_pixels = scast<uintmax_t>(dy * ngridx * ngridy);
    farray_t    update(recon_pixels, 0.0f);
    init_data_t init_data =
        CpuData::initialize(opts, dy, dt, dx, ngridx, ngridy, recon, data, update.data());
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

        // execute the loop over projection angles
        execute<data_array_t>(opts, dt, std::ref(cpu_data), mlem_cpu_compute_projection,
                              dy, dt, dx, ngridx, ngridy, theta);

        // update the global recon with global update and sum_dist
        for(uintmax_t ii = 0; ii < recon_pixels; ++ii)
        {
            if(sum_dist[ii] != 0.0f && dx != 0 && update[ii] == update[ii])
                recon[ii] *= update[ii] / sum_dist[ii] / scast<float>(dx);
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
mlem_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
          RuntimeOptions* opts)
{
    mlem_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter, opts);
}
#endif
//======================================================================================//
