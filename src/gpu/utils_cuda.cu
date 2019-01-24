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
//   TOMOPY implementation

//============================================================================//

#include "PTL/AutoLock.hh"
#include "PTL/ThreadPool.hh"
#include "gpu.hh"
#include <set>

BEGIN_EXTERN_C
#include "gpu.h"
#include "utils_cuda.h"
END_EXTERN_C

//============================================================================//

#if defined(TOMOPY_USE_NVTX)
extern nvtxEventAttributes_t nvtx_calc_coords;
extern nvtxEventAttributes_t nvtx_calc_dist;
extern nvtxEventAttributes_t nvtx_calc_simdata;
extern nvtxEventAttributes_t nvtx_preprocessing;
extern nvtxEventAttributes_t nvtx_sort_intersections;
extern nvtxEventAttributes_t nvtx_sum_dist;
extern nvtxEventAttributes_t nvtx_trim_coords;
extern nvtxEventAttributes_t nvtx_calc_sum_sqr;
extern nvtxEventAttributes_t nvtx_rotate;
#endif

//============================================================================//

//  gridDim:    This variable contains the dimensions of the grid.
//  blockIdx:   This variable contains the block index within the grid.
//  blockDim:   This variable and contains the dimensions of the block.
//  threadIdx:  This variable contains the thread index within the block.

//============================================================================//
//
//  zero out array
//
//============================================================================//

template <typename _Tp>
__global__ void
cuda_global_zero(_Tp* data, int size, int* offset)
{
    int i0     = blockIdx.x * blockDim.x + threadIdx.x + ((offset) ? (*offset) : 0);
    int stride = blockDim.x * gridDim.x;
    for(int i = i0; i < size; i += stride)
        data[i] = _Tp(0);
}

//============================================================================//
//
//  rotate
//
//============================================================================//

void
cuda_rotate_kernel(float* dst, const float* src, const float theta, const int nx,
                   const int ny)
{
    NVTX_RANGE_PUSH(&nvtx_rotate);

    float xoff = floorf(nx / 2.0) + ((nx % 2 == 0) ? 0.5 : 0.0);
    float yoff = floorf(ny / 2.0) + ((ny % 2 == 0) ? 0.5 : 0.0);

    NppiSize siz;
    siz.width = nx;
    siz.height = ny;
    NppiRect roi;
    roi.x = xoff;
    roi.y = yoff;
    roi.width = nx;
    roi.height = ny;
    int step = nx * sizeof(float);
    float cos_p = cosf(theta);
    float sin_p = sinf(theta);
    double rot[2][3] = {{ cos_p, sin_p, 0.0f}, {-sin_p, cos_p, 0.0f}};
    NppStatus ret = nppiWarpAffine_32f_C1R(src, siz, step, roi,
                                           dst, step, roi,
                                           rot, NPPI_INTER_CUBIC);

    /*
    int j0      = blockIdx.y * blockDim.y + threadIdx.y;
    int jstride = blockDim.y * gridDim.y;
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    int src_size = nx * ny;

    for(int j = j0; j < ny; j += jstride)
    {
        for(int i = i0; i < nx; i += istride)
        {
            // indices in 2D
            float rx = float(i) - xoff;
            float ry = float(j) - yoff;
            // transformation
            float tx = rx * cosf(theta) + -ry * sinf(theta);
            float ty = rx * sinf(theta) + ry * cosf(theta);
            // indices in 2D
            float x = (tx + xoff);
            float y = (ty + yoff);
            // index in 1D array
            int rz = j * nx + i;
            if(rz < 0 || rz >= src_size)
                continue;
            // within bounds
            unsigned x1   = floor(tx + xoff);
            unsigned y1   = floor(ty + yoff);
            unsigned x2   = x1 + 1;
            unsigned y2   = y1 + 1;
            float    fxy1 = 0.0f;
            float    fxy2 = 0.0f;
            if(y1 * nx + x1 < src_size)
                fxy1 += (x2 - x) * src[y1 * nx + x1];
            if(y1 * nx + x2 < src_size)
                fxy1 += (x - x1) * src[y1 * nx + x2];
            if(y2 * nx + x1 < src_size)
                fxy2 += (x2 - x) * src[y2 * nx + x1];
            if(y2 * nx + x2 < src_size)
                fxy2 += (x - x1) * src[y2 * nx + x2];
            dst[rz] += (y2 - y) * fxy1 + (y - y1) * fxy2;
        }
    }
    */

    NVTX_RANGE_POP(&nvtx_rotate);

    cudaStreamSynchronize(0);
    CUDA_CHECK_LAST_ERROR();
}

//============================================================================//

float*
cuda_rotate(const float* src, const float theta, const int nx, const int ny)
{
    float* _dst = gpu_malloc<float>(nx * ny);
    cudaMemset(_dst, 0, nx * ny * sizeof(float));
    cuda_rotate_kernel(_dst, src, theta, nx, ny);
    return _dst;
}

//============================================================================//

void
cuda_rotate_ip(float* dst, const float* src, const float theta, const int nx,
               const int ny)
{
    cudaMemset(dst, 0, nx * ny * sizeof(float));
    cuda_rotate_kernel(dst, src, theta, nx, ny);
}

//============================================================================//
