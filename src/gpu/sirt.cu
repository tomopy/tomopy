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
extern nvtxEventAttributes_t nvtx_total;
extern nvtxEventAttributes_t nvtx_iteration;
extern nvtxEventAttributes_t nvtx_slice;
extern nvtxEventAttributes_t nvtx_projection;
extern nvtxEventAttributes_t nvtx_update;
extern nvtxEventAttributes_t nvtx_rotate;
#endif

//======================================================================================//

struct gpu_data
{
    typedef gpu_data this_type;

    int           m_device;
    int           m_block;
    int           m_dy;
    int           m_dt;
    int           m_dx;
    int           m_nx;
    int           m_ny;
    float*        m_rot;
    float*        m_tmp;
    float*        m_update;
    float*        m_recon;
    float*        m_data;
    int           m_num_streams = 1;
    cudaStream_t* m_streams     = nullptr;

    gpu_data(int device, int dy, int dt, int dx, int nx, int ny, const float* cpu_data)
    : m_device(device)
    , m_block(GetEnv<int>("CUDA_BLOCK_SIZE", 128))
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_rot(nullptr)
    , m_tmp(nullptr)
    , m_update(nullptr)
    , m_recon(nullptr)
    , m_data(nullptr)
    {
        cuda_set_device(m_device);
        m_streams = create_streams(m_num_streams);
        m_rot     = gpu_malloc<float>(m_nx * m_ny);
        m_tmp     = gpu_malloc<float>(m_nx * m_ny);
        m_update  = gpu_malloc<float>(m_dy * m_nx * m_ny);
        m_recon   = gpu_malloc<float>(m_dy * m_nx * m_ny);
        m_data    = gpu_malloc<float>(m_dy * m_dt * m_dx);
        cudaMemcpy(m_data, cpu_data, m_dy * m_dt * m_dx * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    ~gpu_data()
    {
        cudaFree(m_rot);
        cudaFree(m_tmp);
        cudaFree(m_update);
        cudaFree(m_recon);
        cudaFree(m_data);
        destroy_streams(m_streams, m_num_streams);
    }

    int compute_grid(int size) const { return (size + m_block - 1) / m_block; }

    void sync(int stream_id = -1)
    {
        auto _sync = [&](int _stream_id) {
            cudaStreamSynchronize(m_streams[_stream_id]);
            CUDA_CHECK_LAST_ERROR();
        };

        if(stream_id < 0)
        {
            for(int i = 0; i < m_num_streams; ++i)
                _sync(i);
        }
        else
            _sync(stream_id);
    }

    void reset()
    {
        cudaMemsetAsync(m_update, 0, m_dy * m_nx * m_ny * sizeof(float), *m_streams);
    }

    void copy(const float* recon)
    {
        cudaMemcpyAsync(m_recon, recon, m_dy * m_nx * m_ny * sizeof(float),
                   cudaMemcpyDeviceToDevice, *m_streams);
    }

    int          device() const { return m_device; }
    int          block() const { return m_block; }
    float*       rot() const { return m_rot; }
    float*       tmp() const { return m_tmp; }
    float*       update() const { return m_update; }
    float*       recon() const { return m_recon; }
    float*       data() const { return m_data; }
    cudaStream_t stream(int stream_id = 0)
    {
        return m_streams[stream_id % m_num_streams];
    }
};

//======================================================================================//

__global__ void
cuda_sirt_sum_kernel(float* dst, const float* src, int size, const float factor)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
        dst[i] += factor * src[i];
}

//======================================================================================//

__global__ void
cuda_sirt_atomic_sum_kernel(float* dst, const float* src, int size, const float factor)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
        atomicAdd(&dst[i], factor * src[i]);
}

//======================================================================================//

__global__ void
cuda_sirt_pixels_kernel(int p, int nx, int dx, float* recon, const float* data,
                        float* dst)
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
            dst[i + pix_offset] += upd;
    }
}

//======================================================================================//

void
cuda_compute_projection(int dt, int dx, int nx, int ny, const float* theta, int s, int p,
                        int nthreads, gpu_data** _gpu_data)
{
    auto       thread_number = GetThisThreadID() % nthreads;
    gpu_data*& _cache        = _gpu_data[thread_number];

    cudaStream_t stream = _cache->stream(0);
    cuda_set_device(_cache->device());

#if defined(DEBUG)
    printf("[%lu] Running slice %i, projection %i on device %i...\n", GetThisThreadID(),
           s, p, _cache->device());
#endif

    // needed for recon to output at proper orientation
    float theta_p_rad = fmodf(theta[p] + halfpi, twopi);
    float theta_p_deg = theta_p_rad * degrees;
    int   block       = _cache->block();
    int   grid        = _cache->compute_grid(dx);
    int   smem        = 0;

    const float* recon     = _cache->recon() + s * nx * ny;
    const float* data      = _cache->data() + s * dt * dx;
    float*       update    = _cache->update() + s * nx * ny;
    float*       recon_rot = _cache->rot();
    float*       recon_tmp = _cache->tmp();

    // Rotate object
    cudaMemsetAsync(recon_rot, 0, nx * ny * sizeof(float), stream);
    CUDA_CHECK_LAST_ERROR();

    cuda_rotate_ip(recon_rot, recon, -theta_p_rad, -theta_p_deg, nx, ny, stream);
    CUDA_CHECK_LAST_ERROR();

    NVTX_RANGE_PUSH(&nvtx_update);
    cuda_sirt_pixels_kernel<<<grid, block, smem, stream>>>(p, nx, dx, recon_rot, data,
                                                           recon_rot);
    CUDA_CHECK_LAST_ERROR();
    NVTX_RANGE_POP(stream);

    // Back-Rotate object
    cudaMemsetAsync(recon_tmp, 0, nx * ny * sizeof(float), stream);
    CUDA_CHECK_LAST_ERROR();

    cuda_rotate_ip(recon_tmp, recon_rot, theta_p_rad, theta_p_deg, nx, ny, stream);
    CUDA_CHECK_LAST_ERROR();

    // update shared update array
    float factor = 1.0f / static_cast<float>(dx);
    cuda_sirt_atomic_sum_kernel<<<grid, block, smem, stream>>>(update, recon_tmp, nx * ny,
                                                               factor);
    CUDA_CHECK_LAST_ERROR();

    cudaStreamSynchronize(stream);
    CUDA_CHECK_LAST_ERROR();
}

//--------------------------------------------------------------------------------------//

void
sirt_cuda(const float* cpu_data, int dy, int dt, int dx, const float* center,
          const float* theta, float* cpu_recon, int ngridx, int ngridy, int num_iter)
{
    if(cuda_device_count() == 0)
        throw std::runtime_error("No CUDA device(s) available");

    init_nvtx();
    cuda_device_query();

    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    auto tid = GetThisThreadID();

    // get some properties
    int num_devices       = cuda_device_count();
    int nthreads          = GetEnv("TOMOPY_NUM_THREADS", 1);
    nthreads              = std::max(nthreads, 1);
    cudaStream_t* streams = create_streams(num_devices);

    // assign the thread to a device
    static std::atomic<int> ntid;
    int                     thread_device = (ntid++) % num_devices;

#if defined(TOMOPY_USE_PTL)
    TaskRunManager* run_man = cpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();
    ThreadPool*  tp       = task_man->thread_pool();
#endif

    TIMEMORY_AUTO_TIMER("");

    // GPU allocated copies
    cuda_set_device(thread_device);
    printf("[%lu] Running on device %i...\n", GetThisThreadID(), thread_device);

    float* tmp_recon = gpu_malloc<float>(dy * ngridx * ngridy);
    float* recon     = gpu_malloc<float>(dy * ngridx * ngridy);
    cudaMemcpy(recon, cpu_recon, dy * ngridx * ngridy * sizeof(float),
               cudaMemcpyHostToDevice);
    gpu_data** _gpu_data = new gpu_data*[nthreads];

    for(int ii = 0; ii < nthreads; ++ii)
        _gpu_data[ii] = new gpu_data(thread_device, dy, dt, dx, ngridx, ngridy, cpu_data);

    NVTX_RANGE_PUSH(&nvtx_total);

    for(int i = 0; i < num_iter; i++)
    {
        NVTX_RANGE_PUSH(&nvtx_iteration);
        START_TIMER(t_start);

        // set "update" to zero, copy in "recon"
        for(int ii = 0; ii < nthreads; ++ii)
        {
            _gpu_data[ii]->reset();
            _gpu_data[ii]->copy(recon);
        }

        // Loop over slices
        for(int s = 0; s < dy; ++s)
        {
            NVTX_RANGE_PUSH(&nvtx_slice);

#if defined(TOMOPY_USE_PTL)
            TaskGroup<void> tg;
            // For each projection angle
            for(int p = 0; p < dt; p++)
                task_man->exec(tg, cuda_compute_projection, dt, dx, ngridx, ngridy, theta,
                               s, p, nthreads, _gpu_data);
            tg.join();
#else
            // For each projection angle
            for(int p = 0; p < dt; p++)
                cuda_compute_projection(dt, dx, ngridx, ngridy, theta, s, p, nthreads,
                                        _gpu_data);
#endif
            NVTX_RANGE_POP(0);
        }

        for(int ii = 0; ii < nthreads; ++ii)
        {
            int    block  = _gpu_data[ii]->block();
            int    grid   = _gpu_data[ii]->compute_grid(dy * ngridx * ngridy);
            float* update = _gpu_data[ii]->update();
            cudaStream_t stream = _gpu_data[ii]->stream();
            float  factor = 1.0f;
            cuda_sirt_atomic_sum_kernel<<<grid, block, 0, stream>>>(recon, update, dy * ngridx * ngridy,
                                                  factor);
        }
        REPORT_TIMER(t_start, "iteration", i, num_iter);
        NVTX_RANGE_POP(0);
    }
    printf("\n");

    cudaDeviceSynchronize();
    cudaMemcpy(cpu_recon, recon, dy * ngridx * ngridy * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(recon);
    cudaFree(tmp_recon);

    for(int i = 0; i < nthreads; ++i)
        delete _gpu_data[i];
    delete[] _gpu_data;

    cudaDeviceSynchronize();
    destroy_streams(streams, num_devices);
    NVTX_RANGE_POP(0);
}

//======================================================================================//
