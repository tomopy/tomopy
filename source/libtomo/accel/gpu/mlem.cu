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
#include "constants.hh"
#include "data.hh"
#include "utils.hh"

#include <algorithm>
#include <cassert>
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

typedef GpuData::init_data_t  init_data_t;
typedef GpuData::data_array_t data_array_t;

//======================================================================================//

__global__ void
cuda_mlem_pixels_kernel(int p, int nx, int dx, float* recon, const float* data)
{
    int d0      = blockIdx.x * blockDim.x + threadIdx.x;
    int dstride = blockDim.x * gridDim.x;

    for(int d = d0; d < dx; d += dstride)
    {
        float sum = 0.0f;
        for(int i = 0; i < nx; ++i)
            sum += recon[d * nx + i];
        if(sum != 0.0f)
        {
            float upd = data[p * dx + d] / sum;
            if(upd == upd)
                for(int i = 0; i < nx; ++i)
                    recon[d * nx + i] += upd;
        }
    }
}

//======================================================================================//

__global__ void
cuda_mlem_update_kernel(float* recon, const float* update, const uint32_t* sum_dist,
                        int dx, int size)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        if(sum_dist[i] != 0 && dx != 0 && update[i] == update[i])
            recon[i] *= update[i] / scast<float>(sum_dist[i]) / scast<float>(dx);
    }
}

//======================================================================================//

void
mlem_gpu_compute_projection(data_array_t& gpu_data, int p, int dy, int dt, int dx, int nx,
                            int ny, const float* theta)
{
    auto cache = gpu_data[GetThisThreadID() % gpu_data.size()];

    // ensure running on proper device
    cuda_set_device(cache->device());

    // calculate some values
    float        theta_p_rad = fmodf(theta[p] + halfpi, twopi);
    float        theta_p_deg = theta_p_rad * degrees;
    int          block       = cache->block();
    int          grid        = cache->compute_grid(dx);
    cudaStream_t stream      = cache->stream();

    // synchronize the stream (do this frequently to avoid backlog)
    stream_sync(stream);

    // reset destination arrays (NECESSARY! or will cause NaNs)
    // only do once bc for same theta, same pixels get overwritten
    cache->reset();

    for(int s = 0; s < dy; ++s)
    {
        const float* recon  = cache->recon() + s * nx * ny;
        const float* data   = cache->data() + s * dt * dx;
        float*       update = cache->update() + s * nx * ny;
        float*       rot    = cache->rot() + s * nx * ny;
        float*       tmp    = cache->tmp() + s * nx * ny;

        // forward-rotate
        cuda_rotate_ip(rot, recon, -theta_p_rad, -theta_p_deg, nx, ny, stream,
                       cache->interpolation());
        CUDA_CHECK_LAST_STREAM_ERROR(stream);

        // compute simdata
        cuda_mlem_pixels_kernel<<<grid, block, 0, stream>>>(p, nx, dx, rot, data);
        CUDA_CHECK_LAST_STREAM_ERROR(stream);

        // back-rotate
        cuda_rotate_ip(tmp, rot, theta_p_rad, theta_p_deg, nx, ny, stream,
                       cache->interpolation());
        CUDA_CHECK_LAST_STREAM_ERROR(stream);

        // update shared update array
        cuda_atomic_sum_kernel<<<grid, block, 0, stream>>>(update, tmp, nx * ny, 1.0f);
        CUDA_CHECK_LAST_STREAM_ERROR(stream);

        // synchronize the stream (do this frequently to avoid backlog)
        stream_sync(stream);
    }
}

//======================================================================================//

void
mlem_cuda(const float* cpu_data, int dy, int dt, int dx, const float*, const float* theta,
          float* cpu_recon, int ngridx, int ngridy, int num_iter, RuntimeOptions* opts)
{
    printf("[%lu]> %s : nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i\n",
           GetThisThreadID(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    // thread counter for device assignment
    static std::atomic<int> ntid;

    // compute some properties (expected python threads, max threads, device assignment)
    int pythread_num = ntid++;
    int device       = pythread_num % cuda_device_count();  // assign to device

    TIMEMORY_AUTO_TIMER("");

    // GPU allocated copies
    cuda_set_device(device);
    printf("[%lu] Running on device %i...\n", GetThisThreadID(), device);

    uintmax_t    recon_pixels = scast<uintmax_t>(dy * ngridx * ngridy);
    auto         block        = GetBlockSize();
    auto         grid         = ComputeGridSize(recon_pixels, block);
    auto         main_stream  = create_streams(1);
    float*       update    = gpu_malloc_and_memset<float>(recon_pixels, 0, *main_stream);
    init_data_t  init_data = GpuData::initialize(opts, device, dy, dt, dx, ngridx, ngridy,
                                                cpu_recon, cpu_data, update);
    data_array_t gpu_data  = std::get<0>(init_data);
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
        GpuData::sync(gpu_data);

        // execute the loop over slices and projection angles
        execute<data_array_t>(opts, dt, std::ref(gpu_data), mlem_gpu_compute_projection,
                              dy, dt, dx, ngridx, ngridy, theta);

        // sync the thread streams
        GpuData::sync(gpu_data);

        // sync the main stream
        stream_sync(*main_stream);

        // update the global recon with global update and sum_dist
        cuda_mlem_update_kernel<<<grid, block, 0, *main_stream>>>(recon, update, sum_dist,
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

    // sync the device
    cudaDeviceSynchronize();
}

//======================================================================================//
