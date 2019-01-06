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

#include "PTL/TaskManager.hh"
#include "PTL/TaskRunManager.hh"
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
#include <cstdlib>
#include <memory>
#include <numeric>

#if !defined(cast)
#    define cast static_cast
#endif

#if defined(TOMOPY_USE_NVTX)
extern nvtxEventAttributes_t nvtx_update;
#endif

#define HW_CONCURRENCY std::thread::hardware_concurrency()

//============================================================================//

__global__ void
cuda_sirt_arrsum_global(float* dst, const float* src, int size, const float factor)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        dst[i] += factor * src[i];
    }
}

//============================================================================//

__global__ void
cuda_sirt_update_global(const float* data, const float* simdata, int ngridx,
                        float* recon_rot, int size)
{
    float upd     = (data[0] - simdata[0]) / static_cast<float>(ngridx);
    int   i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int   istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        recon_rot[i] += upd;
    }
}

//----------------------------------------------------------------------------//

void
cuda_sirt_update(const float* data, const float* simdata, int ngridx,
                 float* recon_rot, int size, cudaStream_t* stream)
{
    NVTX_RANGE_PUSH(&nvtx_update);
    int nb   = cuda_multi_processor_count();
    int nt   = 4;  // cuda_max_threads_per_block();
    int smem = 0;

    CUDA_CHECK_LAST_ERROR();
    cuda_sirt_update_global<<<nb, nt, smem, *stream>>>(data, simdata, ngridx,
                                                       recon_rot, size);
    CUDA_CHECK_LAST_ERROR();
    NVTX_RANGE_POP(&nvtx_update);
}

//============================================================================//

cudaStream_t
get_stream(bool sync)
{
    typedef std::list<cudaStream_t> stream_list_t;

    static thread_local int nthread_streams =
        GetEnv<int>("TOMOPY_NUM_THREAD_STREAMS", 4);
    static thread_local cudaStream_t* thread_streams = create_streams(nthread_streams);
    static thread_local bool          init           = true;
    static thread_local stream_list_t available_streams;
    static thread_local stream_list_t utilized_streams;

    if(init)
    {
        for(int i = 0; i < nthread_streams; ++i)
            available_streams.push_back(thread_streams[i]);
        init = false;
    }

    cudaStream_t _tmp_stream;
    if(!sync)
    {
        // printf("\t%s [line: %i] fetching stream... avail = %lu, used = %lu\n",
        //       __FUNCTION__, __LINE__, available_streams.size(),
        //       utilized_streams.size());
        if(available_streams.size() > 0)
        {
            // PRINT_HERE("");
            _tmp_stream = available_streams.front();
            available_streams.pop_front();
            utilized_streams.push_back(_tmp_stream);
            return _tmp_stream;
        }
        else
        {
            // PRINT_HERE("");
            _tmp_stream = utilized_streams.front();
            utilized_streams.pop_front();
            utilized_streams.push_back(_tmp_stream);
            cudaStreamSynchronize(_tmp_stream);
            CUDA_CHECK_LAST_ERROR();
            return _tmp_stream;
        }
    }
    // synchronize
    while(!utilized_streams.empty())
    {
        // printf("\t%s [line: %i] synchronizing streams... avail = %lu, used "
        //       "= %lu\n",
        //       __FUNCTION__, __LINE__, available_streams.size(),
        //       utilized_streams.size());
        _tmp_stream = utilized_streams.front();
        utilized_streams.pop_front();
        available_streams.push_back(_tmp_stream);
        cudaStreamSynchronize(_tmp_stream);
        CUDA_CHECK_LAST_ERROR();
    }
    return _tmp_stream;
}

//============================================================================//

void
cuda_compute_projection(int dt, int dx, int ngridx, int ngridy, float* gpu_data,
                        const float* theta, int s, int p, float* gpu_simdata,
                        float* _gpu_update, float* _gpu_recon, int nstreams,
                        cudaStream_t* streams)
{
    int recon_size = ngridx * ngridy;
    int float_size = sizeof(float);
    int nt         = 512;
    int nb         = std::min((ngridx * ngridy + nt - 1) / nt, 1024);
    int smem       = 0;

    static std::atomic<int>    thread_counter;
    static thread_local int    thread_stream_offset = (thread_counter++) % nstreams;
    static thread_local float* gpu_recon_rot = gpu_malloc<float>(ngridx * ngridy);
    static thread_local float* tmp_recon_rot = gpu_malloc<float>(ngridx * ngridy);
    static thread_local float* sum_recon_rot = gpu_malloc<float>(ngridx);
    cudaStream_t _stream                     = streams[thread_stream_offset];

    // needed for recon to output at proper orientation
    float pi_offset = 0.5f * (float) M_PI;

    // printf("%s [line: %i] slice = %i, angle = %i\n", __FUNCTION__, __LINE__,
    //       s, p);

    float theta_p = fmodf(theta[p] + pi_offset, 2.0f * (float) M_PI);

    cudaStreamSynchronize(_stream);

    // Rotate object
    cuda_rotate(gpu_recon_rot, _gpu_recon, theta_p, ngridx, ngridy, &_stream);
    CUDA_CHECK_LAST_ERROR();

    cudaStreamSynchronize(_stream);

    for(int d = 0; d < dx; d++)
    {
        int          pix_offset     = d * ngridx;  // pixel offset
        int          idx_data       = d + p * dx + s * dt * dx;
        float*       _gpu_recon_rot = gpu_recon_rot + pix_offset;
        float*       _gpu_simdata   = gpu_simdata + idx_data;
        float*       _gpu_data      = gpu_data + idx_data;

        // Calculate simulated data by summing up along x-axis
        float _gpu_sim =
            thrust::reduce(thrust::system::cuda::par.on(_stream), _gpu_recon_rot,
                           _gpu_recon_rot + ngridx, 0.0f, thrust::plus<float>());
        CUDA_CHECK_LAST_ERROR();

        {
            static Mutex _mutex;
            AutoLock     l(_mutex);

            // update shared simdata array
            cudaMemcpyAsync(_gpu_simdata, &_gpu_sim, float_size, cudaMemcpyHostToDevice,
                            _stream);
            CUDA_CHECK_LAST_ERROR();

            //cudaMemsetAsync(sum_recon_rot, 0, ngridx * float_size, _stream);
            //deviceReduce(_gpu_recon_rot, sum_recon_rot, ngridx, _stream);
            //reduce(_gpu_recon_rot, sum_recon_rot, ngridx, _stream);
            //cudaMemcpyAsync(_gpu_simdata, sum_recon_rot, float_size, cudaMemcpyDeviceToDevice,
            //                _stream);
        }

        // Make update by backprojecting error along x-axis
        cuda_sirt_update(_gpu_data, _gpu_simdata, ngridx, _gpu_recon_rot,
                         ngridx, &_stream);
        CUDA_CHECK_LAST_ERROR();
    }

    // Back-Rotate object
    cuda_rotate(tmp_recon_rot, gpu_recon_rot, theta_p, ngridx, ngridy, &_stream);
    CUDA_CHECK_LAST_ERROR();

    // update shared update array
    {
        static Mutex _mutex;
        AutoLock     l(_mutex);

        // update shared update array
        cuda_sirt_arrsum_global<<<nb, nt, smem, _stream>>>(_gpu_update, tmp_recon_rot,
                                                           recon_size, 1.0f);
        CUDA_CHECK_LAST_ERROR();
    }

    cudaStreamSynchronize(_stream);
}

//----------------------------------------------------------------------------//

void
sirt_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    if(cuda_device_count() == 0)
        throw std::runtime_error("No CUDA device(s) available");

    cuda_device_query();

    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    // assign the thread to a device
    set_this_thread_device();

    int             nthreads = GetEnv("TOMOPY_NUM_THREADS", HW_CONCURRENCY);
    int             nstreams = GetEnv("TOMOPY_NUM_STREAMS", nthreads);
    cudaStream_t*   streams  = create_streams(nstreams);
    TaskRunManager* run_man  = gpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();
    ThreadPool*  tp       = task_man->thread_pool();

    // needed for recon to output at proper orientation
    int    recon_size  = ngridx * ngridy;
    int    float_size  = sizeof(float);
    float* _gpu_recon  = gpu_malloc<float>(recon_size);
    float* _gpu_update = gpu_malloc<float>(recon_size);
    float* gpu_simdata = gpu_malloc<float>(dy * dt * dx);
    float* gpu_data    = gpu_malloc<float>(dy * dt * dx);
    int    nb          = cuda_multi_processor_count();
    int    nt          = 4;  // cuda_max_threads_per_block();

    cudaMemcpy(gpu_data, data, dy * dt * dx * float_size, cudaMemcpyHostToDevice);

    CUDA_CHECK_LAST_ERROR();
    for(int i = 0; i < num_iter; i++)
    {
        cudaMemset(gpu_simdata, 0, dy * dt * dx * float_size);
        CUDA_CHECK_LAST_ERROR();

        cudaDeviceSynchronize();
        CUDA_CHECK_LAST_ERROR();

        // For each slice
        for(int s = 0; s < dy; s++)
        {
            cudaDeviceSynchronize();
            CUDA_CHECK_LAST_ERROR();

            int    slice_offset = s * ngridx * ngridy;
            float* _recon       = recon + slice_offset;
            // stream for this iteration of the slices

            cudaMemcpy(_gpu_recon, _recon, recon_size * float_size,
                       cudaMemcpyHostToDevice);
            CUDA_CHECK_LAST_ERROR();

            cudaMemset(_gpu_update, 0, recon_size * float_size);
            CUDA_CHECK_LAST_ERROR();

            cudaDeviceSynchronize();
            CUDA_CHECK_LAST_ERROR();

            TaskGroup<void> tg;
            // For each projection angle
            for(int p = 0; p < dt; p++)
            {
                task_man->exec(tg, cuda_compute_projection, dt, dx, ngridx, ngridy,
                               gpu_data, theta, s, p, gpu_simdata, _gpu_update,
                               _gpu_recon, nstreams, streams);
            }
            tg.join();

            cudaDeviceSynchronize();
            CUDA_CHECK_LAST_ERROR();

            float factor = 1.0f / static_cast<float>(dt);
            cuda_sirt_arrsum_global<<<nb, nt>>>(_gpu_recon, _gpu_update, recon_size,
                                                factor);
            CUDA_CHECK_LAST_ERROR();

            cudaMemcpy(_recon, _gpu_recon, recon_size * float_size,
                       cudaMemcpyDeviceToHost);
            CUDA_CHECK_LAST_ERROR();
        }
    }

    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();

    cudaFree(_gpu_recon);
    cudaFree(_gpu_update);
    cudaFree(gpu_simdata);
    cudaFree(gpu_data);

    destroy_streams(streams, nstreams);
}

//============================================================================//
