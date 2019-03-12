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
#include "cxx_extern.h"
#include "utils.h"
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

typedef gpu_data::int_type     int_type;
typedef gpu_data::init_data_t  init_data_t;
typedef gpu_data::data_array_t data_array_t;

#define TILE_DIM 16
#define FULL_MASK 0xffffffff

//--------------------------------------------------------------------------------------//

__inline__ __device__ float
warp_reduce_sum(float val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

//--------------------------------------------------------------------------------------//

__inline__ __device__ float
block_reduce_sum(float val)
{
    static __shared__ float shared[32];  // Shared mem for 32 partial sums
    int                     lane = threadIdx.x % warpSize;
    int                     wid  = threadIdx.x / warpSize;
    val = warp_reduce_sum(val);  // Each warp performs partial reduction
    if(lane == 0)
        shared[wid] = val;  // Write reduced value to shared memory
    __syncthreads();        // Wait for all partial reductions
    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if(wid == 0)
        val = warp_reduce_sum(val);  // Final reduce within first warp
    return val;
}

//--------------------------------------------------------------------------------------//

__device__ void
device_reduce_warp_atomic_kernel(float* in, float* out, int N)
{
    float sum     = 0.0f;
    int   i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int   istride = blockDim.x * gridDim.x;
    for(int i = i0; i < N; i += istride)
        sum += in[i];
    sum = warp_reduce_sum(sum);
    if((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(out, sum);
}

//--------------------------------------------------------------------------------------//

__device__ void
device_reduce_block_atomic_kernel(float* in, float* out, int N)
{
    float sum     = 0.0f;
    int   i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int   istride = blockDim.x * gridDim.x;
    for(int i = i0; i < N; i += istride)
        sum += in[i];
    sum = block_reduce_sum(sum);
    if(threadIdx.x == 0)
        atomicAdd(out, sum);
}

//======================================================================================//

__global__ void
cuda_sirt_pixels_kernel(int p, int nx, int dx, float* recon, const float* data)
{
    int d0      = blockIdx.x * blockDim.x + threadIdx.x;
    int dstride = blockDim.x * gridDim.x;

    for(int d = d0; d < dx; d += dstride)
    {
        float sum = 0.0f;
        for(int i = 0; i < nx; ++i)
            sum += recon[d * nx + i];
        float upd = data[p * dx + d] - sum;
        for(int i = 0; i < nx; ++i)
            recon[d * nx + i] += upd;
    }
}

//======================================================================================//

__global__ void
cuda_sirt_pixels_kernel_opt(int p, int nx, int dx, float* recon, const float* data)
{
    int d0      = blockIdx.x * blockDim.x + threadIdx.x;
    int dstride = blockDim.x * gridDim.x;

    for(int d = d0; d < dx; d += dstride)
    {
        float sum = 0.0f;
        device_reduce_warp_atomic_kernel(recon + d * nx, &sum, nx);
        float upd = data[p * dx + d] - sum;
        for(int i = 0; i < nx; ++i)
            recon[d * nx + i] += upd;
    }
}

//======================================================================================//

__global__ void
cuda_sirt_update_kernel(float* recon, const float* update, const uint32_t* sum_dist,
                        int dx, int size)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        if(sum_dist[i] != 0 && dx != 0 && update[i] == update[i])
            recon[i] += update[i] / scast<float>(sum_dist[i]) / scast<float>(dx);
    }
}

//======================================================================================//

void
sirt_gpu_compute_projection(data_array_t& _gpu_data, int _s, int p, int dy, int dt,
                            int dx, int nx, int ny, const float* theta)
{
    static bool use_opt = GetEnv<bool>("TOMOPY_USE_OPT_SUM", false);
    auto        _cache  = _gpu_data[GetThisThreadID() % _gpu_data.size()];

#if defined(DEBUG)
    printf("[%lu] Running slice %i, projection %i on device %i...\n", GetThisThreadID(),
           s, p, _cache->device());
#endif

    // ensure running on proper device
    cuda_set_device(_cache->device());

    // calculate some values
    float        theta_p_rad = fmodf(theta[p] + halfpi, twopi);
    float        theta_p_deg = theta_p_rad * degrees;
    int          _block      = _cache->block();
    int          _grid       = _cache->compute_grid(dx);
    cudaStream_t stream      = _cache->stream();

    dim3 block3 = GetBlockDims();
    dim3 grid3  = GetGridDims();
    dim3 block(_block, block3.y);
    dim3 grid(_grid, (grid3.y == 0) ? 1 : grid3.y);

    // synchronize the stream (do this frequently to avoid backlog)
    stream_sync(stream);

    // reset destination arrays (NECESSARY! or will cause NaNs)
    // only do once bc for same theta, same pixels get overwritten
    _cache->reset();

    for(int s = 0; s < dy; ++s)
    {
        const float* recon  = _cache->recon() + s * nx * ny;
        const float* data   = _cache->data() + s * dt * dx;
        float*       update = _cache->update() + s * nx * ny;
        float*       rot    = _cache->rot() + s * nx * ny;
        float*       tmp    = _cache->tmp() + s * nx * ny;

        // forward-rotate
        cuda_rotate_ip(rot, recon, -theta_p_rad, -theta_p_deg, nx, ny, stream);
        // compute simdata
        if(use_opt)
            cuda_sirt_pixels_kernel_opt<<<grid, block, 0, stream>>>(p, nx, dx, rot, data);
        else
            cuda_sirt_pixels_kernel<<<grid, block, 0, stream>>>(p, nx, dx, rot, data);
        // back-rotate
        cuda_rotate_ip(tmp, rot, theta_p_rad, theta_p_deg, nx, ny, stream);
        // update shared update array
        cuda_atomic_sum_kernel<<<grid, block, 0, stream>>>(update, tmp, nx * ny, 1.0f);
        // synchronize the stream (do this frequently to avoid backlog)
        stream_sync(stream);
    }
}

//======================================================================================//

void
sirt_cuda(const float* cpu_data, int dy, int dt, int dx, const float* center,
          const float* theta, float* cpu_recon, int ngridx, int ngridy, int num_iter)
{
    typedef decltype(HW_CONCURRENCY) nthread_type;

    auto num_devices = cuda_device_count();
    if(num_devices == 0)
        throw std::runtime_error("No CUDA device(s) available");

    printf("[%lu]> %s : nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i\n",
           GetThisThreadID(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    // initialize nvtx data
    init_nvtx();
    // print device info
    cuda_device_query();
    // thread counter for device assignment
    static std::atomic<int> ntid;

    // compute some properties (expected python threads, max threads, device
    // assignment)
    auto min_threads  = nthread_type(1);
    auto pythreads    = GetEnv("TOMOPY_PYTHON_THREADS", HW_CONCURRENCY);
    auto max_threads  = HW_CONCURRENCY / std::max(pythreads, min_threads);
    auto nthreads     = std::max(GetEnv("TOMOPY_NUM_THREADS", max_threads), min_threads);
    int  pythread_num = ntid++;
    int  device       = pythread_num % num_devices;  // assign to device

#if defined(TOMOPY_USE_PTL)
    typedef TaskManager manager_t;
    TaskRunManager*     run_man = gpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();
    ThreadPool*  tp       = task_man->thread_pool();
#else
    typedef void manager_t;
    void*        task_man = nullptr;
#endif

    TIMEMORY_AUTO_TIMER("");

    // GPU allocated copies
    cuda_set_device(device);
    printf("[%lu] Running on device %i...\n", GetThisThreadID(), device);

    // if another thread has not already warmed up
    if(pythread_num < num_devices)
    {
        auto* warm = gpu_malloc<uint64_t>(1);
        cuda_warmup_kernel<uint64_t><<<512, 1>>>(warm, 32, 1);
        cudaFree(warm);
    }

    uintmax_t    recon_pixels = scast<uintmax_t>(dy * ngridx * ngridy);
    auto         block        = GetBlockSize();
    auto         grid         = ComputeGridSize(recon_pixels, block);
    auto         main_stream  = create_streams(1);
    float*       update    = gpu_malloc_and_memset<float>(recon_pixels, 0, *main_stream);
    init_data_t  init_data = gpu_data::initialize(device, nthreads, dy, dt, dx, ngridx,
                                                 ngridy, cpu_recon, cpu_data, update);
    data_array_t _gpu_data = std::get<0>(init_data);
    float*       recon     = std::get<1>(init_data);
    float*       data      = std::get<2>(init_data);
    uint32_t*    sum_dist  = cuda_compute_sum_dist(dy, dt, dx, ngridx, ngridy, theta);

    NVTX_RANGE_PUSH(&nvtx_total);

    for(int i = 0; i < num_iter; i++)
    {
        // timing and profiling
        TIMEMORY_AUTO_TIMER("");
        NVTX_RANGE_PUSH(&nvtx_iteration);
        START_TIMER(t_start);

        // sync the main stream
        stream_sync(*main_stream);

        // reset global update and sum_dist
        gpu_memset(update, 0, recon_pixels, *main_stream);

        // sync
        gpu_data::sync(_gpu_data);

        // execute the loop over slices and projection angles
        execute<manager_t, data_array_t>(task_man, 1, dt, std::ref(_gpu_data),
                                         sirt_gpu_compute_projection, dy, dt, dx, ngridx,
                                         ngridy, theta);

        // sync the thread streams
        gpu_data::sync(_gpu_data);

        // sync the main stream
        stream_sync(*main_stream);

        // update the global recon with global update and sum_dist
        cuda_sirt_update_kernel<<<grid, block, 0, *main_stream>>>(recon, update, sum_dist,
                                                                  dx, recon_pixels);

        // stop profile range and report timing
        NVTX_RANGE_POP(0);
        REPORT_TIMER(t_start, "iteration", i, num_iter);
    }

    // copy to cpu
    gpu2cpu_memcpy<float>(cpu_recon, recon, recon_pixels, *main_stream);

    // sync and destroy main stream
    destroy_streams(main_stream, 1);

    // cleanup
    cudaFree(recon);
    cudaFree(data);
    cudaFree(update);
    cudaFree(sum_dist);

    NVTX_RANGE_POP(0);
}

//======================================================================================//
