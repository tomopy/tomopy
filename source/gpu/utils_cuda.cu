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

//======================================================================================//

#include "common.hh"
#include "gpu.hh"
#include "utils_cuda.hh"

BEGIN_EXTERN_C
#include "gpu.h"
END_EXTERN_C

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

//  gridDim:    This variable contains the dimensions of the grid.
//  blockIdx:   This variable contains the block index within the grid.
//  blockDim:   This variable and contains the dimensions of the block.
//  threadIdx:  This variable contains the thread index within the block.

typedef gpu_data::int_type int_type;

//======================================================================================//
//
//  compute sum_dist
//
//======================================================================================//

__global__ void
cuda_sum_dist_compute(int dy, int dx, int nx, int ny, const int_type* ones,
                      uint32_t* sum_dist, int p)
{
    int nx0      = blockIdx.x * blockDim.x + threadIdx.x;
    int nxstride = blockDim.x * gridDim.x;
    int dx0      = blockIdx.y * blockDim.y + threadIdx.y;
    int dxstride = blockDim.y * gridDim.y;
    int dy0      = blockIdx.z * blockDim.z + threadIdx.z;
    int dystride = blockDim.z * gridDim.z;

    for(int s = dy0; s < dy; s += dystride)
    {
        for(int d = dx0; d < dx; d += dxstride)
        {
            uint32_t*       _sum_dist = sum_dist + (s * nx * ny) + (d * nx);
            const int_type* _ones     = ones + (d * nx);
            for(int n = nx0; n < nx; n += nxstride)
            {
                atomicAdd(&_sum_dist[n], (_ones[n] > 0) ? 1 : 0);
            }
        }
    }
}

//======================================================================================//

uint32_t*
cuda_compute_sum_dist(int dy, int dt, int dx, int nx, int ny, const float* theta)
{
    auto ns      = GetNumMasterStreams(1);
    auto streams = create_streams(ns, cudaStreamNonBlocking);
    auto block   = GetBlockDims(dim3(32, 32, 1));
    auto grid    = ComputeGridDims(dim3(nx, dt, dy), block);

    //----------------------------------------------------------------------------------//
    auto sync = [&]() {
        for(int i = 0; i < ns; ++i)
            stream_sync(streams[i]);
    };
    //----------------------------------------------------------------------------------//

    int_type* rot = gpu_malloc<int_type>(ns * nx * ny);
    int_type* tmp = gpu_malloc_and_memset<int_type>(nx * ny, 1, streams[0 % ns]);
    uint32_t* sum_dist =
        gpu_malloc_and_memset<uint32_t>(dy * nx * ny, 0, streams[1 % ns]);

    stream_sync(streams[0 % ns]);
    stream_sync(streams[1 % ns]);

    for(int p = 0; p < dt; ++p)
    {
        float theta_p_rad = fmodf(theta[p] + halfpi, twopi);
        float theta_p_deg = theta_p_rad * degrees;

        for(int q = 0; q < ns; ++q)
        {
            auto      stream = streams[q];
            int_type* _rot   = rot + q * nx * ny;
            gpu_memset<int_type>(_rot, 0, nx * nx, stream);
        }

        for(int q = 0; q < ns; ++q)
        {
            auto      stream = streams[q];
            int_type* _rot   = rot + q * nx * ny;
            cuda_rotate_ip(_rot, tmp, -theta_p_rad, -theta_p_deg, nx, ny, stream, GPU_NN);
        }

        for(int q = 0; q < ns; ++q)
        {
            auto      stream = streams[q];
            int_type* _rot   = rot + q * nx * ny;
            cuda_sum_dist_compute<<<grid, block, 0, stream>>>(dy, dx, nx, ny, _rot,
                                                              sum_dist, p);
        }
    }
    sync();
    cudaFree(tmp);
    cudaFree(rot);
    destroy_streams(streams, ns);

    return sum_dist;
}

//======================================================================================//
//
//  rotate
//
//======================================================================================//

template <typename _Tp>
void
print_array(const _Tp* data, int nx, int ny, const std::string& desc)
{
    std::stringstream ss;
    ss << desc << "\n\n";
    ss << std::fixed;
    ss.precision(3);
    for(int j = 0; j < ny; ++j)
    {
        ss << "  ";
        for(int i = 0; i < nx; ++i)
        {
            ss << std::setw(8) << data[j * nx + i] << " ";
        }
        ss << std::endl;
    }
    std::cout << ss.str() << std::endl;
}

//======================================================================================//
//
//  rotate
//
//======================================================================================//

typedef gpu_data::int_type int_type;

void
cuda_rotate_kernel(int_type* dst, const int_type* src, const float theta_rad,
                   const float theta_deg, const int nx, const int ny,
                   int eInterp = GPU_NN, cudaStream_t stream = 0)
{
    // cudaStreamSynchronize(stream);
    nppSetStream(stream);
    NVTX_RANGE_PUSH(&nvtx_rotate);

    auto getRotationMatrix2D = [&](double m[2][3], double scale) {
        double alpha    = scale * cos(theta_rad);
        double beta     = scale * sin(theta_rad);
        double center_x = (0.5 * nx) - 0.5;
        double center_y = (0.5 * ny) - 0.5;

        m[0][0] = alpha;
        m[0][1] = beta;
        m[0][2] = (1.0 - alpha) * center_x - beta * center_y;
        m[1][0] = -beta;
        m[1][1] = alpha;
        m[1][2] = beta * center_x + (1.0 - alpha) * center_y;
    };

    NppiSize siz;
    siz.width  = nx;
    siz.height = ny;

    NppiRect roi;
    roi.x      = 0;
    roi.y      = 0;
    roi.width  = nx;
    roi.height = ny;

    int    step = nx * sizeof(int_type);
    double rot[2][3];
    getRotationMatrix2D(rot, 1.0);

#if defined(DEBUG)
    printf("theta = %5.1f\n", theta_deg);
    print_array((double*) rot, 3, 2, "rot");
#endif
    // #define USE_NPPI_ROTATE

#if defined(USE_NPPI_ROTATE)
    NppStatus ret = nppiRotate_32s_C1R(src, siz, step, roi, dst, step, roi, theta_deg,
                                       rot[0][2], rot[1][2], eInterp);
#else
    NppStatus ret =
        nppiWarpAffine_32s_C1R(src, siz, step, roi, dst, step, roi, rot, eInterp);
#endif

    if(ret != NPP_SUCCESS)
        printf("[%lu] %s returned non-zero NPP status: %i\n", GetThisThreadID(),
               __FUNCTION__, ret);

    NVTX_RANGE_POP(stream);
    CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//
//
//  rotate
//
//======================================================================================//

void
cuda_rotate_kernel(float* dst, const float* src, const float theta_rad,
                   const float theta_deg, const int nx, const int ny,
                   int eInterp = GPU_CUBIC, cudaStream_t stream = 0)
{
    // cudaStreamSynchronize(stream);
    nppSetStream(stream);
    NVTX_RANGE_PUSH(&nvtx_rotate);

    auto getRotationMatrix2D = [&](double m[2][3], double scale) {
        double alpha    = scale * cos(theta_rad);
        double beta     = scale * sin(theta_rad);
        double center_x = (0.5 * nx) - 0.5;
        double center_y = (0.5 * ny) - 0.5;

        m[0][0] = alpha;
        m[0][1] = beta;
        m[0][2] = (1.0 - alpha) * center_x - beta * center_y;
        m[1][0] = -beta;
        m[1][1] = alpha;
        m[1][2] = beta * center_x + (1.0 - alpha) * center_y;
    };

    NppiSize siz;
    siz.width  = nx;
    siz.height = ny;

    NppiRect roi;
    roi.x      = 0;
    roi.y      = 0;
    roi.width  = nx;
    roi.height = ny;

    int    step = nx * sizeof(float);
    double rot[2][3];
    getRotationMatrix2D(rot, 1.0);

#if defined(DEBUG)
    printf("theta = %5.1f\n", theta_deg);
    print_array((double*) rot, 3, 2, "rot");
#endif
    // #define USE_NPPI_ROTATE

#if defined(USE_NPPI_ROTATE)
    NppStatus ret = nppiRotate_32f_C1R(src, siz, step, roi, dst, step, roi, theta_deg,
                                       rot[0][2], rot[1][2], eInterp);
#else
    NppStatus ret =
        nppiWarpAffine_32f_C1R(src, siz, step, roi, dst, step, roi, rot, eInterp);
#endif

    if(ret != NPP_SUCCESS)
        printf("[%lu] %s returned non-zero NPP status: %i\n", GetThisThreadID(),
               __FUNCTION__, ret);

    NVTX_RANGE_POP(stream);
    CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//

__global__ void
cuda_rotate_internal_kernel(float* dst, const float* src, float theta, const int nx,
                            const int ny)
{
    // this is flawed and should not be production
    int   src_size = nx * ny;
    float xoff     = (0.5f * nx) - 0.5f;
    float yoff     = (0.5f * ny) - 0.5f;

    int j0      = blockIdx.x * blockDim.x + threadIdx.x;
    int jstride = blockDim.x * gridDim.x;

    for(int j = j0; j < ny; j += jstride)
    {
        for(int i = 0; i < nx; ++i)
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
            int  rz    = j * nx + i;
            auto index = [&](int _x, int _y) { return _y * nx + _x; };
            // within bounds
            int   x1    = floorf(tx + xoff);
            int   y1    = floorf(ty + yoff);
            int   x2    = x1 + 1;
            int   y2    = y1 + 1;
            float fxy1  = 0.0f;
            float fxy2  = 0.0f;
            int   ixy11 = index(x1, y1);
            int   ixy21 = index(x2, y1);
            int   ixy12 = index(x1, y2);
            int   ixy22 = index(x2, y2);
            if(ixy11 >= 0 && ixy11 < src_size)
                fxy1 += (x2 - x) * src[ixy11];
            if(ixy21 >= 0 && ixy21 < src_size)
                fxy1 += (x - x1) * src[ixy21];
            if(ixy12 >= 0 && ixy12 < src_size)
                fxy2 += (x2 - x) * src[ixy12];
            if(ixy22 >= 0 && ixy22 < src_size)
                fxy2 += (x - x1) * src[ixy22];
            dst[rz] += (y2 - y) * fxy1 + (y - y1) * fxy2;
        }
    }
}

//======================================================================================//

int_type*
cuda_rotate(const int_type* src, const float theta_rad, const float theta_deg,
            const int nx, const int ny, cudaStream_t stream, const int eInterp)
{
    int_type* _dst = gpu_malloc<int_type>(nx * ny);
    cuda_rotate_kernel(_dst, src, theta_rad, theta_deg, nx, ny, eInterp, stream);
    return _dst;
}

//======================================================================================//

void
cuda_rotate_ip(int_type* dst, const int_type* src, const float theta_rad,
               const float theta_deg, const int nx, const int ny, cudaStream_t stream,
               const int eInterp)
{
    cuda_rotate_kernel(dst, src, theta_rad, theta_deg, nx, ny, eInterp, stream);
    CUDA_CHECK_LAST_ERROR();
    /*
    int block = 128;
    int grid  = (ny + block - 1) / block;
    cuda_rotate_internal_kernel<<<grid, block, 0, stream>>>(dst, src, theta_rad, nx, ny);
    CUDA_CHECK_LAST_ERROR();
    */
}

//======================================================================================//

float*
cuda_rotate(const float* src, const float theta_rad, const float theta_deg, const int nx,
            const int ny, cudaStream_t stream, const int eInterp)
{
    float* _dst = gpu_malloc<float>(nx * ny);
    cuda_rotate_kernel(_dst, src, theta_rad, theta_deg, nx, ny, eInterp, stream);
    return _dst;
}

//======================================================================================//

void
cuda_rotate_ip(float* dst, const float* src, const float theta_rad, const float theta_deg,
               const int nx, const int ny, cudaStream_t stream, const int eInterp)
{
    cuda_rotate_kernel(dst, src, theta_rad, theta_deg, nx, ny, eInterp, stream);
    CUDA_CHECK_LAST_ERROR();
    /*
    int block = 128;
    int grid  = (ny + block - 1) / block;
    cuda_rotate_internal_kernel<<<grid, block, 0, stream>>>(dst, src, theta_rad, nx, ny);
    CUDA_CHECK_LAST_ERROR();
    */
}

//======================================================================================//
