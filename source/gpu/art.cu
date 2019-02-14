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
#include "utils_cuda.hh"

BEGIN_EXTERN_C
#include "art.h"
#include "utils.h"
#include "utils_openacc.h"
#include "utils_openmp.h"
END_EXTERN_C

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <numeric>

//======================================================================================//

#if defined(TOMOPY_USE_NVTX)
extern nvtxEventAttributes_t nvtx_total;
extern nvtxEventAttributes_t nvtx_iteration;
extern nvtxEventAttributes_t nvtx_slice;
extern nvtxEventAttributes_t nvtx_projection;
extern nvtxEventAttributes_t nvtx_update;
extern nvtxEventAttributes_t nvtx_rotate;
#endif

//======================================================================================//

__global__ void
cuda_art_pixels_kernel(int p, int nx, int dx, float* recon,
                       const float* data,
                       const gpu_data::int_type* recon_use)
{
    int d0      = blockIdx.x * blockDim.x + threadIdx.x;
    int dstride = blockDim.x * gridDim.x;

    for(int d = d0; d < dx; d += dstride)
    {
        int   fnx = 0;
        float sum = 0.0f;
        for(int i = 0; i < nx; ++i)
            sum += recon[d * nx + i];
        for(int i = 0; i < nx; ++i)
            fnx += (recon_use[d * nx + i] != 0) ? 1 : 0;
        if(fnx != 0)
        {
            sum = (data[p * dx + d] - sum) / scast<float>(fnx);
            for(int i = 0; i < nx; ++i)
                recon[d * nx + i] += sum;
        }
    }
}

//======================================================================================//

void
art_gpu_compute_projection(int dy, int dt, int dx, int nx, int ny, const float* theta,
                           int s, int p, int nthreads, gpu_data** _gpu_data)
{
    typedef gpu_data::int_type int_type;

    auto       thread_number = GetThisThreadID() % nthreads;
    gpu_data*& _cache        = _gpu_data[thread_number];

#if defined(DEBUG)
    printf("[%lu] Running slice %i, projection %i on device %i...\n", GetThisThreadID(),
           s, p, _cache->device());
#endif

    // ensure running on proper device
    cuda_set_device(_cache->device());

    // calculate some values
    float        theta_p_rad = fmodf(theta[p], pi);
    float        theta_p_deg = theta_p_rad * degrees;
    float*       recon       = _cache->recon() + s * nx * ny;
    const float* data        = _cache->data() + s * dt * dx;
    float*       update      = _cache->update() + s * nx * ny;
    auto*        use_rot     = _cache->use_rot();
    auto*        use_tmp     = _cache->use_tmp();
    float*       rot         = _cache->rot();
    float*       tmp         = _cache->tmp();
    int          smem        = 0;
    const float  factor      = 1.0f / scast<float>(dx);
    int          block       = _cache->block();
    int          grid        = _cache->compute_grid(nx);
    cudaStream_t stream      = _cache->stream();

    gpu_memset<int_type>(use_rot, 0, nx * ny, stream);
    gpu_memset<float>(rot, 0, nx * ny, stream);
    gpu_memset<float>(tmp, 0, nx * ny, stream);

    stream_sync(stream);
    // forward-rotate
    cuda_rotate_ip(use_rot, use_tmp, -theta_p_rad, -theta_p_deg, nx, ny, stream, GPU_NN);
    cuda_rotate_ip(rot, recon, -theta_p_rad, -theta_p_deg, nx, ny, stream);
    // compute simdata
    cuda_art_pixels_kernel<<<grid, block, smem, stream>>>(p, nx, dx, rot, data, use_rot);
    // back-rotate
    cuda_rotate_ip(tmp, rot, theta_p_rad, theta_p_deg, nx, ny, stream);
    // update shared update array
    cuda_atomic_sum_kernel<<<grid, block, smem, stream>>>(recon, tmp, nx * ny, factor);
    // synchronize the stream (do this frequently to avoid backlog)
    stream_sync(stream);
    // if(++(*_cache) % _cache->sync_freq() == 0)
    //    stream_sync(stream);
}

//--------------------------------------------------------------------------------------//

void
art_cuda(const float* cpu_data, int dy, int dt, int dx, const float* center,
         const float* theta, float* cpu_recon, int ngridx, int ngridy, int num_iter)
{
    typedef decltype(HW_CONCURRENCY) nthread_type;

    if(cuda_device_count() == 0)
        throw std::runtime_error("No CUDA device(s) available");

    init_nvtx();
    cuda_device_query();
    static std::atomic<int> ntid;
    auto                    tid = GetThisThreadID();

    printf("[%lu]> %s : nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i\n",
           GetThisThreadID(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

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

    float* recon = gpu_malloc<float>(dy * ngridx * ngridy);
    memset(cpu_recon, 0, dy * ngridx * ngridy * sizeof(float));
    float* data    = gpu_malloc<float>(dy * dt * dx);
    auto   streams = create_streams(2, cudaStreamNonBlocking);
    cpu2gpu_memcpy<float>(recon, cpu_recon, dy * ngridx * ngridy, streams[0]);
    cpu2gpu_memcpy<float>(data, cpu_data, dy * dt * dx, streams[1]);
    stream_sync(streams[0]);
    stream_sync(streams[1]);
    destroy_streams(streams, 2);
    gpu_data** _gpu_data = new gpu_data*[nthreads];
    for(int ii = 0; ii < nthreads; ++ii)
        _gpu_data[ii] = new gpu_data(thread_device, dy, dt, dx, ngridx, ngridy, data,
                                     recon, sync_freq);
    int block = GetBlockSize();
    int grid  = ComputeGridSize(dy * ngridx * ngridy);
    NVTX_RANGE_PUSH(&nvtx_total);

    for(int i = 0; i < num_iter; i++)
    {
        TIMEMORY_AUTO_TIMER("");
        NVTX_RANGE_PUSH(&nvtx_iteration);
        START_TIMER(t_start);

        // set "update" to zero, copy in "recon"
        for(int ii = 0; ii < nthreads; ++ii)
            _gpu_data[ii]->reset();

#if defined(TOMOPY_USE_PTL)
        // Loop over slices and projection angles
        TaskGroup<void> tg;
        for(int s = 0; s < dy; ++s)
            for(int p = 0; p < dt; ++p)
                task_man->exec(tg, art_gpu_compute_projection, dy, dt, dx, ngridx, ngridy,
                               theta, s, p, nthreads, _gpu_data);
        tg.join();
#else
        // Loop over slices and projection angles
        for(int s = 0; s < dy; ++s)
            for(int p = 0; p < dt; ++p)
                art_gpu_compute_projection(dy, dt, dx, ngridx, ngridy, theta, s, p,
                                           nthreads, _gpu_data);
#endif
        for(int ii = 0; ii < nthreads; ++ii)
            _gpu_data[ii]->sync();

        /*
        stream_sync(0);
        const float factor = 1.0f / scast<float>(dx);
        cuda_mult_kernel<<<grid, block>>>(recon, dy * ngridx * ngridy, factor);
        stream_sync(0);
        */

        NVTX_RANGE_POP(0);
        REPORT_TIMER(t_start, "iteration", i, num_iter);
    }

    for(int ii = 0; ii < nthreads; ++ii)
        _gpu_data[ii]->sync();

    // create a rotated reconstruction
    float* recon_rot = gpu_malloc<float>(dy * ngridx * ngridy);
    streams          = create_streams(dy, cudaStreamNonBlocking);
    // rotate reconstruction
    for(int i = 0; i < dy; ++i)
    {
        auto offset = i * ngridx * ngridy;
        cuda_rotate_ip(recon_rot + offset, recon + offset, halfpi, halfpi * degrees,
                       ngridx, ngridy, streams[i]);
    }

    for(int i = 0; i < dy; ++i)
        stream_sync(streams[i]);
    destroy_streams(streams, dy);
    cudaFree(recon);

    gpu2cpu_memcpy_and_free<float>(cpu_recon, recon_rot, dy * ngridx * ngridy, 0);
    stream_sync(0);

    for(int i = 0; i < nthreads; ++i)
        delete _gpu_data[i];
    delete[] _gpu_data;

    cudaFree(recon_rot);
    NVTX_RANGE_POP(0);

    printf("\n");
}

//======================================================================================//
