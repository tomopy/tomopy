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
    float*        m_update;
    float*        m_recon;
    const float*  m_data;
    uintmax_t     m_sync_freq;
    uintmax_t     m_sync_counter;
    int           m_num_streams = 1;
    cudaStream_t* m_streams     = nullptr;

    gpu_data(int device, int dy, int dt, int dx, int nx, int ny, const float* data,
             float* recon, uintmax_t sync_freq)
    : m_device(device)
    , m_block(GetEnvBlockSize())
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_rot(nullptr)
    , m_tmp(nullptr)
    , m_update(nullptr)
    , m_recon(recon)
    , m_data(data)
    , m_sync_freq(sync_freq)
    {
        cuda_set_device(m_device);
        m_streams = create_streams(m_num_streams, cudaStreamNonBlocking);
        m_rot     = gpu_malloc<float>(m_nx * m_ny);
        m_tmp     = gpu_malloc<float>(m_nx * m_ny);
        m_update  = gpu_malloc<float>(m_dy * m_nx * m_ny);
    }

    ~gpu_data()
    {
        cudaFree(m_rot);
        cudaFree(m_tmp);
        cudaFree(m_update);
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

    /*void copy(const float* recon)
    {
        gpu2gpu_memcpy<float>(m_recon, recon, m_dy * m_nx * m_ny, *m_streams);
    }*/

    int          device() const { return m_device; }
    int          block() const { return m_block; }
    float*       rot() const { return m_rot; }
    float*       tmp() const { return m_tmp; }
    float*       update() const { return m_update; }
    const float* recon() const { return m_recon; }
    const float* data() const { return m_data; }
    uintmax_t    sync_freq() const { return m_sync_freq; }
    cudaStream_t stream(int stream_id = 0)
    {
        return m_streams[stream_id % m_num_streams];
    }
};

//======================================================================================//

__global__ void
cuda_sirt_sum_kernel(float* dst, const float* src, int size,
                     const float factor)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
        dst[i] += factor * src[i];
}

//======================================================================================//

__global__ void
cuda_sirt_atomic_sum_kernel(float* dst, const float* src, int size,
                            const float factor)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
        atomicAdd(&dst[i], factor * src[i]);
}

//======================================================================================//

__global__ void
cuda_sirt_pixels_kernel(int p, int nx, int dx, float* recon,
                        const float* data)
{
    int d0      = blockIdx.x * blockDim.x + threadIdx.x;
    int dstride = blockDim.x * gridDim.x;

    for(int d = d0; d < dx; d += dstride)
    {
        float sum = 0.0f;
        for(int i = 0; i < nx; ++i)
            sum += recon[d * nx + i];
        sum = (data[p * dx + d] - sum) / static_cast<float>(nx);
        for(int i = 0; i < nx; ++i)
            recon[d * nx + i] += sum;
    }

    /*
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
    */
}

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
    int          smem        = 0;
    const float  factor      = 1.0f / scast<float>(dx);
    int          block      = _cache->block();
    int          grid       = _cache->compute_grid(dx);
    cudaStream_t stream      = _cache->stream();

    gpu_memset<float>(rot, 0, nx * ny, stream);
    gpu_memset<float>(tmp, 0, nx * ny, stream);

    // forward-otate
    cuda_rotate_ip(rot, recon, -theta_p_rad, -theta_p_deg, nx, ny, stream);
    // compute simdata
    cuda_sirt_pixels_kernel<<<grid, block, smem, stream>>>(p, nx, dx, rot, data);
    // back-rotate
    cuda_rotate_ip(tmp, rot, theta_p_rad, theta_p_deg, nx, ny, stream);
    // update shared update array
    cuda_sirt_atomic_sum_kernel<<<grid, block, smem, stream>>>(update, tmp, nx * ny,
                                                               factor);
    stream_sync(stream);
    // if(++(*_cache) % _cache->sync_freq() == 0)
    //    stream_sync(stream);
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
    float* data = gpu_malloc<float>(dy * dt * dx);
    auto streams = create_streams(2, cudaStreamNonBlocking);
    cpu2gpu_memcpy<float>(recon, cpu_recon, dy * ngridx * ngridy, streams[0]);
    cpu2gpu_memcpy<float>(data, cpu_data, dy * dt * dx, streams[1]);
    stream_sync(streams[0]);
    stream_sync(streams[1]);
    destroy_streams(streams, 2);
    gpu_data** _gpu_data = new gpu_data*[nthreads];
    for(int ii = 0; ii < nthreads; ++ii)
        _gpu_data[ii] = new gpu_data(thread_device, dy, dt, dx, ngridx, ngridy, data, recon,
                                     sync_freq);

    NVTX_RANGE_PUSH(&nvtx_total);

    for(int i = 0; i < num_iter; i++)
    {
        TIMEMORY_AUTO_TIMER("");
        NVTX_RANGE_PUSH(&nvtx_iteration);
        START_TIMER(t_start);

        // set "update" to zero, copy in "recon"
        for(int ii = 0; ii < nthreads; ++ii)
	{
            _gpu_data[ii]->reset();
	}

#if defined(TOMOPY_USE_PTL)
        // Loop over slices and projection angles
        TaskGroup<void> tg;
        for(int s = 0; s < dy; ++s)
            for(int p = 0; p < dt; ++p)
                task_man->exec(tg, cuda_compute_projection, dy, dt, dx, ngridx, ngridy, theta, s, p, nthreads,
                                        _gpu_data);
        tg.join();
#else
        // Loop over slices and projection angles
        for(int s = 0; s < dy; ++s)
            for(int p = 0; p < dt; p++)
                cuda_compute_projection(dy, dt, dx, ngridx, ngridy, theta, s, p, nthreads,
                                        _gpu_data);
#endif
        for(int ii = 0; ii < nthreads; ++ii)
            _gpu_data[ii]->sync();

        for(int ii = 0; ii < nthreads; ++ii)
        {
            int          smem   = 0;
            int          block = _gpu_data[ii]->block();
            int          grid  = _gpu_data[ii]->compute_grid(dy * ngridx * ngridy);
            float*       update = _gpu_data[ii]->update();
            cudaStream_t stream = _gpu_data[ii]->stream();
            cuda_sirt_atomic_sum_kernel<<<grid, block, smem, stream>>>(
                recon, update, dy * ngridx * ngridy, 1.0f);
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
