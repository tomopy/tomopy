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
//   TOMOPY CUDA implementation

#include "common.hh"
#include "gpu.hh"
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

#if !defined(cast)
#    define cast static_cast
#endif

#if defined(TOMOPY_USE_NVTX)
extern nvtxEventAttributes_t nvtx_update;
#endif

//======================================================================================//

__global__ void
cuda_sirt_sum_kernel(float* dst, const float* src, int size, const float factor)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        dst[i] += factor * src[i];
    }
}

//======================================================================================//

struct gpu_thread_data
{
    typedef gpu_thread_data this_type;

    int          m_device;
    int          m_id;
    int          m_block;
    int          m_dy;
    int          m_dt;
    int          m_dx;
    int          m_nx;
    int          m_ny;
    int          m_size;
    float*       m_rot;
    float*       m_tmp;
    float*       m_update;
    const float* m_recon;
    const float* m_data;

    gpu_thread_data(int device, int id, int dy, int dt, int dx, int nx, int ny)
    : m_device(device)
    , m_id(id)
    , m_block(GetEnv<int>("CUDA_BLOCK_SIZE", 128))
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_size(m_nx * m_ny)
    , m_rot(gpu_malloc<float>(m_size))
    , m_tmp(gpu_malloc<float>(m_size))
    , m_update(gpu_malloc<float>(m_size))
    , m_recon(nullptr)
    , m_data(nullptr)
    {
    }

    ~gpu_thread_data()
    {
        cudaFree(m_rot);
        cudaFree(m_tmp);
        cudaFree(m_update);
    }

    int compute_grid(int size) const { return (size + m_block - 1) / m_block; }

    void initialize(const float* data, const float* recon, int s)
    {
        uintmax_t offset = s * m_dt * m_dx;
        m_data           = data + offset;
        offset           = s * m_size;
        m_recon          = recon + offset;
        cudaMemset(m_update, 0, m_size * sizeof(float));
    }

    void finalize(float* recon, int s)
    {
        int      grid   = compute_grid(m_size);
        int      offset = s * m_size;
        float    factor = 1.0f / static_cast<float>(m_dt);
        AutoLock l(TypeMutex<this_type>());  // lock update
        cuda_sirt_sum_kernel<<<grid, m_block>>>(recon + offset, m_update, m_size, factor);
    }

    void sync()
    {
        cudaStreamSynchronize(0);
        CUDA_CHECK_LAST_ERROR();
    }

    int          block() const { return m_block; }
    float*       rot() const { return m_rot; }
    float*       tmp() const { return m_tmp; }
    float*       update() const { return m_update; }
    const float* recon() const { return m_recon; }
    const float* data() const { return m_data; }
};

//======================================================================================//

__global__ void
cuda_sirt_pixels_kernel(int p, int nx, int dx, float* recon, const float* data)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int d = i0; d < dx; d += istride)
    {
        int   pix_offset = d * nx;      // pixel offset
        int   idx_data   = d + p * dx;  // data offset
        float sum        = 0.0f;
        for(int i = 0; i < nx; ++i)
            sum += recon[i + pix_offset];
        float upd = (data[idx_data] - sum) / static_cast<float>(nx);
        for(int i = 0; i < nx; ++i)
            recon[i + pix_offset] += upd;
    }
}

//======================================================================================//

void
cuda_compute_projection(int dt, int dx, int ngridx, int ngridy, const float* theta, int s,
                        int p, int nthreads, gpu_thread_data** _gpu_thread_data)
{
    auto              thread_number = GetThisThreadID() % nthreads;
    gpu_thread_data*& _cache        = _gpu_thread_data[thread_number];

    nppSetStream(0);

    NVTX_NAME_THREAD(thread_number, __FUNCTION__);

    // needed for recon to output at proper orientation
    float theta_p_rad = fmodf(theta[p] + halfpi, twopi);
    float theta_p_deg = theta_p_rad * (180.0f / pi);
    int   block       = _cache->block();
    int   grid        = _cache->compute_grid(dx);
    int   smem        = 0;

    const float* recon     = _cache->recon();
    const float* data      = _cache->data();
    float*       update    = _cache->update();
    float*       recon_rot = _cache->rot();
    float*       recon_tmp = _cache->tmp();

    // Rotate object
    cuda_rotate_ip(recon_rot, recon, -theta_p_rad, -theta_p_deg, ngridx, ngridy);

    NVTX_RANGE_PUSH(&nvtx_update);
    cuda_sirt_pixels_kernel<<<grid, block, smem>>>(p, ngridx, dx, recon_rot, data);
    NVTX_RANGE_POP(&nvtx_update);

    // Back-Rotate object
    cuda_rotate_ip(recon_tmp, recon_rot, theta_p_rad, theta_p_deg, ngridx, ngridy);

    // update shared update array
    cuda_sirt_sum_kernel<<<grid, block, smem>>>(update, recon_tmp, ngridx * ngridy, 1.0f);
}

//--------------------------------------------------------------------------------------//

void
sirt_cuda(const float* cpu_data, int dy, int dt, int dx, const float* center,
          const float* theta, float* cpu_recon, int ngridx, int ngridy, int num_iter)
{
    if(cuda_device_count() == 0)
        throw std::runtime_error("No CUDA device(s) available");

    cuda_device_query();

    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    auto tid = GetThisThreadID();

    // assign the thread to a device
    set_this_thread_device();

    // get some properties
    int num_devices = cuda_device_count();
    int nthreads    = GetEnv("TOMOPY_NUM_THREADS", HW_CONCURRENCY);
    nthreads        = std::max(nthreads, 1);

#if defined(TOMOPY_USE_PTL)
    TaskRunManager* run_man = cpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();
    ThreadPool*  tp       = task_man->thread_pool();
#endif

    TIMEMORY_AUTO_TIMER("");

    // GPU allocated copies
    float* recon = gpu_malloc<float>(dy * ngridx * ngridy);
    float* data  = gpu_malloc<float>(dy * dt * dx);
    cudaMemcpy(recon, cpu_recon, dy * ngridx * ngridy * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(data, cpu_data, dy * dt * dx * sizeof(float), cudaMemcpyHostToDevice);

    gpu_thread_data** _gpu_thread_data = new gpu_thread_data*[nthreads];

    auto allocate = [&]() {
        int idx = GetThisThreadID() % nthreads;
        _gpu_thread_data[idx] =
            new gpu_thread_data(idx % num_devices, idx, dy, dt, dx, ngridx, ngridy);
    };

    auto initialize = [&](int s) {
        int idx = GetThisThreadID() % nthreads;
        _gpu_thread_data[idx]->initialize(data, recon, s);
    };

    auto finalize = [&](int s) {
        int idx = GetThisThreadID() % nthreads;
        _gpu_thread_data[idx]->sync();
        _gpu_thread_data[idx]->finalize(recon, s);
    };

#if defined(TOMOPY_USE_PTL)
    tp->get_queue()->ExecuteOnAllThreads(tp, allocate);
#else
    allocate();
#endif

    printf("\n");
    for(int i = 0; i < num_iter; i++)
    {
        START_TIMER(t_start);
        // For each slice
        for(int s = 0; s < dy; ++s)
        {
            auto init  = std::bind(initialize, s);
            auto final = std::bind(finalize, s);

#if defined(TOMOPY_USE_PTL)
            tp->get_queue()->ExecuteOnAllThreads(tp, init);
            TaskGroup<void> tg;
            // For each projection angle
            for(int p = 0; p < dt; p++)
            {
                task_man->exec(tg, cuda_compute_projection, dt, dx, ngridx, ngridy, theta,
                               s, p, nthreads, _gpu_thread_data);
            }
            tg.join();
            tp->get_queue()->ExecuteOnAllThreads(tp, final);
#else
            init();
            // For each projection angle
            for(int p = 0; p < dt; p++)
            {
                cuda_compute_projection(dt, dx, ngridx, ngridy, theta, s, p, nthreads,
                                        _gpu_thread_data);
            }
            final();
#endif
        }
        // cudaDeviceSynchronize();
        REPORT_TIMER(t_start, "iteration", i, num_iter);
    }
    printf("\n");

    cudaDeviceSynchronize();
    cudaMemcpy(cpu_recon, recon, dy * ngridx * ngridy * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(recon);

    for(int i = 0; i < nthreads; ++i)
        delete _gpu_thread_data[i];
    delete[] _gpu_thread_data;

    cudaDeviceSynchronize();
}

//======================================================================================//
