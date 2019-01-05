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

#include "gpu.hh"
#include "PTL/TaskManager.hh"
#include "PTL/TaskRunManager.hh"
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
sirt_update_global(int s, int p, int d, int ry, int rz, int dt, int dx,
                   const int* csize, const float* data, const float* simdata,
                   const int* indi, const float* dist, const float* sum_dist,
                   float* model)
{
    if(*sum_dist != 0.0f)
    {
        int size   = (*csize) - 1;
        int i0     = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        int index_model = s * ry * rz;
        int idx         = d + p * dx + s * dt * dx;

        float upd = (data[idx] - simdata[idx]) / (*sum_dist);
        for(int i = i0; i < size; i += stride)
        {
            float value = upd * dist[i];
            model[indi[i] + index_model] += value;
        }
    }
}

//----------------------------------------------------------------------------//

void
cuda_sirt_update(int s, int p, int d, int ry, int rz, int dt, int dx,
                 const int* csize, const float* data, const float* simdata,
                 const int* indi, const float* dist, const float* sum,
                 float* model, cudaStream_t* streams)
{
    NVTX_RANGE_PUSH(&nvtx_update);
    int nb   = cuda_multi_processor_count();
    int nt   = 16;
    int smem = 0;

    CUDA_CHECK_LAST_ERROR();
    sirt_update_global<<<nb, nt, smem, streams[0]>>>(s, p, d, ry, rz, dt, dx,
                                                     csize, data, simdata, indi,
                                                     dist, sum, model);
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

    tim::enable_signal_detection();
    TIMEMORY_AUTO_TIMER("[cuda]");

    // assign the thread to a device
    set_this_thread_device();

    int             nthreads = GetEnv("TOMOPY_NUM_THREADS", HW_CONCURRENCY);
    TaskRunManager* run_man  = gpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager*  task_man = run_man->GetTaskManager();
    ThreadPool*   tp       = task_man->thread_pool();
    cudaStream_t* streams  = create_streams(nthreads);

    // needed for recon to output at proper orientation
    float pi_offset = 0.5f * (float) M_PI;

    for(int i = 0; i < num_iter; i++)
    {
        float*   simdata = malloc_and_memset<float>(dy * dt * dx, 0);
        // For each slice
        for(int s = 0; s < dy; s++)
        {
            int recon_size = ngridx * ngridy;
            int slice_offset = s * ngridx * ngridy;
            float*   recon_off = malloc_and_memcpy<float>(recon + slice_offset, ngridx * ngridy);
            float* update = malloc_and_memset<float>(ngridx * ngridy, 0);

            // For each projection angle
            for(int p = 0; p < dt; p++)
            {
                float theta_p = fmodf(theta[p] + pi_offset, 2.0f * (float) M_PI);

                // Rotate object
                float* recon_rot =
                    cuda_rotate(recon_off, theta_p, ngridx, ngridy, streams);
                cudaStreamSynchronize(streams[0]);

                for(int d = 0; d < dx; d++)
                {
                    int    pix_offset = d * ngridx;  // pixel offset
                    int    idx_data   = d + p * dx + s * dt * dx;
                    float  fngridx    = ngridx;
                    float  _sim       = 0.0f;
                    float* _simdata   = simdata + idx_data;
                    float* _recon_rot = recon_rot + pix_offset;

                    // Calculate simulated data by summing up along x-axis
                    _sim = reduce(_recon_rot, 0.0f, ngridx, streams[0]);
                    cudaStreamSynchronize(streams[0]);

                    // update shared simdata array
                    gpu_memcpy(_simdata, &_sim, 1);

                    // Make update by backprojecting error along x-axis
                    float _value;
                    cpu_memcpy(_simdata, &_value, 1);

                    float upd = (data[idx_data] - _value) / fngridx;
                    cuda_add(_recon_rot, ngridx, upd, streams);
                    cudaStreamSynchronize(streams[0]);
                }
                // Back-Rotate object
                float* tmp =
                    cuda_rotate(recon_rot, theta_p, ngridx, ngridy, streams);
                cudaStreamSynchronize(streams[0]);

                cudaFree(recon_rot);

                // update shared update array
                transform_sum(tmp, recon_size, update, streams[0]);
                cudaStreamSynchronize(streams[0]);

                cudaFree(tmp);
            }
            cudaFree(recon_off);

            float* _recon = recon + slice_offset;
            farray_t _update(recon_size);
            cpu_memcpy(update, _update.data(), recon_size);
            for(int ii = 0; ii < recon_size; ++ii)
                _recon[ii] += _update[ii] / static_cast<float>(dt);
            cudaFree(update);
        }
    }
}

//============================================================================//
