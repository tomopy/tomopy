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

int
GetEnvBlockSize()
{
    static thread_local int _instance = GetEnv<int>("CUDA_BLOCK_SIZE", 128);
    return _instance;
}

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
    float*        m_sum;
    float*        m_update;
    float*        m_recon;
    float*        m_data;
    uintmax_t     m_sync_freq;
    uintmax_t     m_sync_counter;
    int           m_num_streams = 1;
    cudaStream_t* m_streams     = nullptr;

    gpu_data(int device, int dy, int dt, int dx, int nx, int ny, const float* cpu_data,
             const float* cpu_recon, uintmax_t sync_freq)
    : m_device(device)
    , m_block(GetEnvBlockSize())
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_rot(nullptr)
    , m_tmp(nullptr)
    , m_sum(nullptr)
    , m_update(nullptr)
    , m_recon(nullptr)
    , m_data(nullptr)
    , m_sync_freq(sync_freq)
    {
        cuda_set_device(m_device);
        m_streams = create_streams(m_num_streams, cudaStreamNonBlocking);
        m_rot     = gpu_malloc<float>(m_nx * m_ny);
        m_tmp     = gpu_malloc<float>(m_nx * m_ny);
        m_sum     = gpu_malloc<float>(m_dx);
        m_update  = gpu_malloc<float>(m_dy * m_nx * m_ny);
        m_recon   = gpu_malloc<float>(m_dy * m_nx * m_ny);
        m_data    = gpu_malloc<float>(m_dy * m_dt * m_dx);
        cpu2gpu_memcpy<float>(m_data, cpu_data, m_dy * m_dt * m_dx, *m_streams);
        cpu2gpu_memcpy<float>(m_recon, cpu_recon, m_dy * m_nx * m_ny, *m_streams);
    }

    ~gpu_data()
    {
        cudaFree(m_rot);
        cudaFree(m_tmp);
        cudaFree(m_sum);
        cudaFree(m_update);
        cudaFree(m_recon);
        cudaFree(m_data);
        destroy_streams(m_streams, m_num_streams);
    }

    int       compute_grid(int size) const { return (size + m_block - 1) / m_block; }
    uintmax_t operator++() { return ++m_sync_counter; }
    uintmax_t operator++(int) { return m_sync_counter++; }

    void sync(int stream_id = -1)
    {
        auto _sync = [&](cudaStream_t _stream) { stream_sync(_stream); };

        if(stream_id >= 0)
            _sync(m_streams[stream_id % m_num_streams]);
        else
            for(int i = 0; i < m_num_streams; ++i)
                _sync(m_streams[i]);
    }

    void reset() { gpu_memset<float>(m_update, 0, m_dy * m_nx * m_ny, *m_streams); }

    void copy(const float* recon)
    {
        gpu2gpu_memcpy<float>(m_recon, recon, m_dy * m_nx * m_ny, *m_streams);
    }

    int          device() const { return m_device; }
    int          block() const { return m_block; }
    float*       rot() const { return m_rot; }
    float*       tmp() const { return m_tmp; }
    float*       sum() const { return m_sum; }
    float*       update() const { return m_update; }
    float*       recon() const { return m_recon; }
    const float* data() const { return m_data; }
    uintmax_t    sync_freq() const { return m_sync_freq; }
    cudaStream_t stream(int stream_id = 0)
    {
        return m_streams[stream_id % m_num_streams];
    }
};

//======================================================================================//

__global__ void
cuda_sirt_sum_kernel(float* dst, const float* src, int xsize, int ysize,
                     const float factor)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    int j0      = blockIdx.y * blockDim.y + threadIdx.y;
    int jstride = blockDim.y * gridDim.y;

    for(int j = j0; j < ysize; j += jstride)
        for(int i = i0; i < xsize; i += istride)
            dst[j * xsize + i] += factor * src[j * xsize + i];
}

//======================================================================================//

__global__ void
cuda_sirt_atomic_sum_kernel(float* dst, const float* src, int xsize, int ysize,
                            const float factor)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    int j0      = blockIdx.y * blockDim.y + threadIdx.y;
    int jstride = blockDim.y * gridDim.y;

    for(int j = j0; j < ysize; j += jstride)
        for(int i = i0; i < xsize; i += istride)
            atomicAdd(&dst[j * xsize + i], factor * src[j * xsize + i]);
}

//======================================================================================//

__global__ void
cuda_sirt_pixels_kernel(int p, int nx, int dx, float* sum, float* recon,
                        const float* data)
{
    int d0      = blockIdx.x * blockDim.x + threadIdx.x;
    int dstride = blockDim.x * gridDim.x;

    /*
    for(int d = d0; d < dx; d += dstride)
    {
        for(int i = 0; i < nx; ++i)
            sum[d] += recon[d * nx + i];
        sum[d] = (data[p * dx + d] - sum[d]) / static_cast<float>(nx);
        for(int i = 0; i < nx; ++i)
            recon[d * nx + i] += sum[d];
    }*/

    int cacheSize  = 32;
    int remainSize = nx % cacheSize;
    int nnx        = nx - remainSize;

    // cache-blocking
    for(int ii = 0; ii < nnx; ii += cacheSize)
        for(int d = d0; d < dx; d += dstride)
            for(int i = ii; i < ii + cacheSize; ++i)
                sum[d] += recon[d * nx + i];

    // remainder
    if(nnx < nx)
        for(int d = d0; d < nx; d += dstride)
            for(int i = nnx; i < nx; ++i)
                sum[d] += recon[d * nx + i];

    // calculate
    for(int d = d0; d < dx; d += dstride)
        sum[d] = (data[p * dx + d] - sum[d]) / static_cast<float>(nx);

    // cache-blocking
    for(int ii = 0; ii < nnx; ii += cacheSize)
        for(int d = d0; d < dx; d += dstride)
            for(int i = ii; i < ii + cacheSize; ++i)
                recon[d * nx + i] += sum[d];

    // remainder
    if(nnx < nx)
        for(int d = d0; d < dx; d += dstride)
            for(int i = nnx; i < nx; ++i)
                recon[d * nx + i] += sum[d];
}

//======================================================================================//
/*
__global__ void
cuda_sirt_sum_pixels_kernel(int p, int nx, int dx, float* sum, const float* recon,
                            const float* data)
{
    int d0      = blockIdx.x * blockDim.x + threadIdx.x;
    int dstride = blockDim.x * gridDim.x;

    int i0      = blockIdx.y * blockDim.y + threadIdx.y;
    int istride = blockDim.y * gridDim.y;

    for(int d = d0; d < dx; d += dstride)
    {
        int   pix_offset = d * nx;  // pixel offset
        float _sum       = 0.0f;
        for(int i = i0; i < nx; i += istride)
            _sum += recon[i + pix_offset];
        atomicAdd(&sum[d], _sum);
    }
}

//======================================================================================//

__global__ void
cuda_sirt_recon_pixels_kernel(int p, int nx, int dx, const float* sum, float* recon,
                              const float* data)
{
    int d0      = blockIdx.x * blockDim.x + threadIdx.x;
    int dstride = blockDim.x * gridDim.x;

    int i0      = blockIdx.y * blockDim.y + threadIdx.y;
    int istride = blockDim.y * gridDim.y;

    for(int d = d0; d < dx; d += dstride)
    {
        int   pix_offset = d * nx;      // pixel offset
        int   idx_data   = d + p * dx;  // data offset
        float upd        = (data[idx_data] - sum[d]) / static_cast<float>(nx);
        for(int i = i0; i < nx; i += istride)
            atomicAdd(&recon[i + pix_offset], upd);
        // atomicAdd(&recon[i + pix_offset], upd);
    }
}*/

//======================================================================================//

void
cuda_compute_projection(int dy, int dt, int dx, int nx, int ny, const float* theta, int s,
                        int p, int nthreads, gpu_data** _gpu_data)
{
    auto       thread_number = GetThisThreadID() % nthreads;
    gpu_data*& _cache        = _gpu_data[thread_number];
    cuda_set_device(_cache->device());

#if defined(DEBUG)
    printf("[%lu] Running slice %i, projection %i on device %i...\n", GetThisThreadID(),
           s, p, _cache->device());
#endif

    // needed for recon to output at proper orientation
    float        theta_p_rad = fmodf(theta[p] + halfpi, twopi);
    float        theta_p_deg = theta_p_rad * degrees;
    const float* recon       = _cache->recon() + s * nx * ny;
    const float* data        = _cache->data() + s * dt * dx;
    float*       update      = _cache->update() + s * nx * ny;
    float*       rot         = _cache->rot();
    float*       tmp         = _cache->tmp();
    float*       sum         = _cache->sum();
    int          smem        = 0;
    const float  factor      = 1.0f / scast<float>(dx);
    int          iblock      = _cache->block();
    int          igrid       = _cache->compute_grid(dx);
    cudaStream_t stream      = _cache->stream();
    dim3         block       = dim3(_cache->block(), 1);
    dim3         grid        = dim3(_cache->compute_grid(nx * ny), 1);

    gpu_memset<float>(sum, 0, dx, stream);
    gpu_memset<float>(rot, 0, nx * ny, stream);
    gpu_memset<float>(tmp, 0, nx * ny, stream);

    // forward-otate
    cuda_rotate_ip(rot, recon, -theta_p_rad, -theta_p_deg, nx, ny, stream);
    // compute simdata
    cuda_sirt_pixels_kernel<<<igrid, iblock, smem, stream>>>(p, nx, dx, sum, rot, data);
    // back-rotate
    cuda_rotate_ip(tmp, rot, theta_p_rad, theta_p_deg, nx, ny, stream);
    // update shared update array
    cuda_sirt_atomic_sum_kernel<<<grid, block, smem, stream>>>(update, tmp, nx * ny, 1,
                                                               factor);
    // stream_sync(stream);
    if(++(*_cache) % _cache->sync_freq() == 0)
        stream_sync(stream);
}

//--------------------------------------------------------------------------------------//

void
sirt_cuda(const float* cpu_data, int dy, int dt, int dx, const float* center,
          const float* theta, float* cpu_recon, int ngridx, int ngridy, int num_iter)
{
    typedef decltype(HW_CONCURRENCY) nthread_type;

    if(cuda_device_count() == 0)
        throw std::runtime_error("No CUDA device(s) available");

    init_nvtx();
    cuda_device_query();
    static std::atomic<int> ntid;
    auto                    tid = GetThisThreadID();

    printf("\n\t[%lu] %s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           tid, __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    // get some properties
    // default number of threads == 1 to ensure an excess of threads are not created
    // unless desired
    auto min_threads   = nthread_type(1);
    auto pythreads     = GetEnv("TOMOPY_PYTHON_THREADS", HW_CONCURRENCY);
    auto max_threads   = HW_CONCURRENCY / std::max(pythreads, min_threads);
    auto nthreads      = std::max(GetEnv("TOMOPY_NUM_THREADS", max_threads), min_threads);
    int  num_devices   = cuda_device_count();
    int  thread_device = (ntid++) % num_devices;           // assign to device
    auto sync_freq     = GetEnv("TOMOPY_STREAM_SYNC", 4);  // sync freq

#if defined(TOMOPY_USE_PTL)
    TaskRunManager* run_man = gpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();
    ThreadPool*  tp       = task_man->thread_pool();
#endif

    TIMEMORY_AUTO_TIMER("");

    // GPU allocated copies
    cuda_set_device(thread_device);
    printf("[%lu] Running on device %i...\n", GetThisThreadID(), thread_device);

    float*     recon     = gpu_malloc<float>(dy * ngridx * ngridy);
    gpu_data** _gpu_data = new gpu_data*[nthreads];
    for(int ii = 0; ii < nthreads; ++ii)
        _gpu_data[ii] = new gpu_data(thread_device, dy, dt, dx, ngridx, ngridy, cpu_data,
                                     cpu_recon, sync_freq);

    NVTX_RANGE_PUSH(&nvtx_total);

    for(int i = 0; i < num_iter; i++)
    {
        NVTX_RANGE_PUSH(&nvtx_iteration);
        START_TIMER(t_start);

        // set "update" to zero, copy in "recon"
        for(int ii = 0; ii < nthreads; ++ii)
            _gpu_data[ii]->reset();

#if defined(TOMOPY_USE_PTL)
        int  pbatch  = 16;
        auto project = [&](int s, int pbeg) {
            int pend = std::min(pbeg + pbatch, dt);
            for(int p = pbeg; p < pend; ++p)
                cuda_compute_projection(dy, dt, dx, ngridx, ngridy, theta, s, p, nthreads,
                                        _gpu_data);
        };
        TaskGroup<void> tg;
        // Loop over slices and projection angles
        for(int s = 0; s < dy; ++s)
            for(int p = 0; p < dt; p += pbatch)
                task_man->exec(tg, project, s, p);
        tg.join();
#else
        // Loop over slices and projection angles
        for(int s = 0; s < dy; ++s)
            for(int p = 0; p < dt; p++)
                cuda_compute_projection(dy, dt, dx, ngridx, ngridy, theta, s, p, nthreads,
                                        _gpu_data);
#endif

        for(int ii = 0; ii < nthreads; ++ii)
        {
            _gpu_data[ii]->sync();
            int          smem   = 0;
            int          nblock = _gpu_data[ii]->block();
            int          xgrid  = _gpu_data[ii]->compute_grid(dy * ngridx * ngridy);
            int          ygrid  = 1;
            dim3         grid(xgrid, ygrid);
            dim3         block(nblock, 1);
            float*       update = _gpu_data[ii]->update();
            cudaStream_t stream = _gpu_data[ii]->stream();
            cuda_sirt_atomic_sum_kernel<<<grid, block, smem, stream>>>(
                recon, update, dy * ngridx * ngridy, 1, 1.0f);
            // _gpu_data[ii]->copy(recon);
            float* _recon = _gpu_data[ii]->recon();
            cuda_sirt_atomic_sum_kernel<<<grid, block, smem, stream>>>(
                _recon, update, dy * ngridx * ngridy, 1, 1.0f);
            _gpu_data[ii]->sync();
        }

        NVTX_RANGE_POP(0);
        REPORT_TIMER(t_start, "iteration", i, num_iter);
    }

    for(int ii = 0; ii < nthreads; ++ii)
        _gpu_data[ii]->sync();

    printf("\n");

    gpu2cpu_memcpy_and_free<float>(cpu_recon, recon, dy * ngridx * ngridy, 0);

    for(int i = 0; i < nthreads; ++i)
        delete _gpu_data[i];
    delete[] _gpu_data;

    NVTX_RANGE_POP(0);
}

//======================================================================================//
