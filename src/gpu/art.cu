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

extern "C"
{
#include "art.h"
#include "utils.h"
}

#if !defined(cast)
#    define cast static_cast
#endif

#if defined(TOMOPY_USE_NVTX)
extern nvtxEventAttributes_t nvtx_update;
#endif

//============================================================================//

__global__ void
art_update_kernel(int s, int p, int d, int ry, int rz, int dt, int dx, const int* csize,
                  const float* data, const float* simdata, const int* indi,
                  const float* dist, const float* sum_dist, float* model)
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
cuda_art_update(int s, int p, int d, int ry, int rz, int dt, int dx, const int* csize,
                const float* data, const float* simdata, const int* indi,
                const float* dist, const float* sum, float* model, cudaStream_t* streams)
{
    NVTX_RANGE_PUSH(&nvtx_update);
    int nb   = cuda_multi_processor_count();
    int nt   = 16;
    int smem = 0;

    CUDA_CHECK_LAST_ERROR();
    art_update_kernel<<<nb, nt, smem, streams[0]>>>(s, p, d, ry, rz, dt, dx, csize, data,
                                                    simdata, indi, dist, sum, model);
    CUDA_CHECK_LAST_ERROR();
    NVTX_RANGE_POP(&nvtx_update);
}

//============================================================================//
