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
cuda_sirt_arrsum_global(float* dst, const float* src, int size,
                        const float factor)
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
cuda_sirt_update_global(const float* data, const float simval, float* simdata,
                        int ngridx, float* recon_rot, int size)
{
    *simdata      = simval;
    float upd     = (*data - *simdata) / static_cast<float>(ngridx);
    int   i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int   istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        recon_rot[i] += upd;
    }
}

//----------------------------------------------------------------------------//

void
cuda_sirt_update(const float* data, const float simval, float* simdata,
                 int ngridx, float* recon_rot, int size, cudaStream_t*)
{
    NVTX_RANGE_PUSH(&nvtx_update);
    int nb   = cuda_multi_processor_count();
    int nt   = 4; //cuda_max_threads_per_block();
    int smem = 0;

    CUDA_CHECK_LAST_ERROR();
    cuda_sirt_update_global<<<nb, nt>>>(data, simval, simdata, ngridx,
                                        recon_rot, size);
    CUDA_CHECK_LAST_ERROR();
    NVTX_RANGE_POP(&nvtx_update);
}

//----------------------------------------------------------------------------//

void
sirt_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy,
          int num_iter)
{
    if(cuda_device_count() == 0)
        throw std::runtime_error("No CUDA device(s) available");

    cuda_device_query();

    printf(
        "\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
        __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    TIMEMORY_AUTO_TIMER("");

    // assign the thread to a device
    set_this_thread_device();

    int             nthreads = GetEnv("TOMOPY_NUM_THREADS", HW_CONCURRENCY);
    /*
    TaskRunManager* run_man  = gpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager*  task_man = run_man->GetTaskManager();
    ThreadPool*   tp       = task_man->thread_pool();
    */
    cudaStream_t* streams  = create_streams(nthreads);

    // needed for recon to output at proper orientation
    float  pi_offset   = 0.5f * (float) M_PI;
    int    recon_size  = ngridx * ngridy;
    int    float_size  = sizeof(float);
    float* _gpu_recon  = gpu_malloc<float>(recon_size);
    float* _gpu_update = gpu_malloc<float>(recon_size);
    float* gpu_simdata = gpu_malloc<float>(dy * dt * dx);
    float* gpu_data    = gpu_malloc<float>(dy * dt * dx);
    float* gpu_recon_rot = gpu_malloc<float>(ngridx * ngridy);
    float* tmp_recon_rot = gpu_malloc<float>(ngridx * ngridy);
    int    nb          = cuda_multi_processor_count();
    int    nt          = 4; //cuda_max_threads_per_block();
    int    smem        = 0;

    cudaMemcpy(gpu_data, data, dy * dt * dx * float_size,
               cudaMemcpyHostToDevice);
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
            int    slice_offset = s * ngridx * ngridy;
            float* _recon       = recon + slice_offset;

            cudaMemcpy(_gpu_recon, _recon, recon_size * float_size,
                       cudaMemcpyHostToDevice);
            CUDA_CHECK_LAST_ERROR();

            cudaMemset(_gpu_update, 0, recon_size * float_size);
            CUDA_CHECK_LAST_ERROR();

            // For each projection angle
            for(int p = 0; p < dt; p++)
            {
                float theta_p =
                    fmodf(theta[p] + pi_offset, 2.0f * (float) M_PI);

                cudaMemset(gpu_recon_rot, 0, recon_size * float_size);
                // Rotate object
                cuda_rotate(gpu_recon_rot, _gpu_recon, theta_p, ngridx, ngridy, streams);
                CUDA_CHECK_LAST_ERROR();

                for(int d = 0; d < dx; d++)
                {
                    int    pix_offset     = d * ngridx;  // pixel offset
                    int    idx_data       = d + p * dx + s * dt * dx;
                    float* _gpu_recon_rot = gpu_recon_rot + pix_offset;
                    float* _gpu_simdata   = gpu_simdata + idx_data;
                    float* _gpu_data      = gpu_data + idx_data;

                    // Calculate simulated data by summing up along x-axis
                    float _gpu_sim =
                        thrust::reduce(thrust::system::cuda::par,
                                       _gpu_recon_rot, _gpu_recon_rot + ngridx,
                                       0.0f, thrust::plus<float>());
                    CUDA_CHECK_LAST_ERROR();

                    // update shared simdata array
                    //cudaMemcpy(_gpu_simdata, &_gpu_sim, float_size,
                    //           cudaMemcpyHostToDevice);
                    //CUDA_CHECK_LAST_ERROR();

                    // Make update by backprojecting error along x-axis
                    cuda_sirt_update(_gpu_data, _gpu_sim, _gpu_simdata, ngridx,
                                     _gpu_recon_rot, ngridx, streams);
                    CUDA_CHECK_LAST_ERROR();
                }

                cudaMemset(tmp_recon_rot, 0, recon_size * float_size);
                // Back-Rotate object
                cuda_rotate(tmp_recon_rot, gpu_recon_rot, theta_p, ngridx, ngridy, streams);
                CUDA_CHECK_LAST_ERROR();

                // update shared update array
                cuda_sirt_arrsum_global<<<nb, nt, smem>>>(_gpu_update, tmp_recon_rot,
                                                          recon_size, 1.0f);
                CUDA_CHECK_LAST_ERROR();
            }

            float factor = 1.0f / static_cast<float>(dt);
            cuda_sirt_arrsum_global<<<nb, nt, smem>>>(_gpu_recon, _gpu_update,
                                                      recon_size, factor);
            CUDA_CHECK_LAST_ERROR();

            cudaMemcpy(_recon, _gpu_recon, recon_size * float_size,
                       cudaMemcpyDeviceToHost);
            CUDA_CHECK_LAST_ERROR();
        }
    }

    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();

    cudaFree(tmp_recon_rot);
    cudaFree(gpu_recon_rot);
    cudaFree(_gpu_recon);
    cudaFree(_gpu_update);
    cudaFree(gpu_simdata);
    cudaFree(gpu_data);
}

//============================================================================//
