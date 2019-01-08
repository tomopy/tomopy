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

#define FULL_MASK 0xffffffff

//============================================================================//

//  gridDim:    This variable contains the dimensions of the grid.
//  blockIdx:   This variable contains the block index within the grid.
//  blockDim:   This variable and contains the dimensions of the block.
//  threadIdx:  This variable contains the thread index within the block.

//----------------------------------------------------------------------------//

__inline__ __device__ float
warpReduceSum(float val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

//----------------------------------------------------------------------------//

__inline__ __device__ float
warpAllReduceSum(float val)
{
    for(int mask = warpSize / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(FULL_MASK, val, mask);
    return val;
}

//----------------------------------------------------------------------------//

__inline__ __device__ float
blockReduceSum(float val)
{
    static __shared__ float shared[32];  // Shared mem for 32 partial sums
    int                     lane = threadIdx.x % warpSize;
    int                     wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val);  // Each warp performs partial reduction

    if(lane == 0)
        shared[wid] = val;  // Write reduced value to shared memory

    __syncthreads();  // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if(wid == 0)
        val = warpReduceSum(val);  // Final reduce within first warp

    return val;
}

//----------------------------------------------------------------------------//

__global__ void
deviceReduceKernel(const float* in, float* out, int N)
{
    int   i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int   istride = blockDim.x * gridDim.x;
    float sum     = 0;

    // reduce multiple elements per thread
    for(int i = i0; i < N; i += istride)
    {
        sum += in[i];
    }
    sum = blockReduceSum(sum);
    if(threadIdx.x == 0)
        out[blockIdx.x] = sum;
}

//----------------------------------------------------------------------------//

float
deviceReduce(const float* in, float* out, int N, cudaStream_t stream)
{
    int threads = 512;
    int blocks  = min((N + threads - 1) / threads, 1024);
    int smem    = 0;

    deviceReduceKernel<<<blocks, threads, smem, stream>>>(in, out, N);
    deviceReduceKernel<<<1, 1024, 0, stream>>>(out, out, blocks);

    float _sum;
    cudaMemcpyAsync(&_sum, out, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return _sum;
}

//============================================================================//
//
//  efficient reduction
//  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//
//============================================================================//

template <unsigned int blockSize, typename _Tp>
__device__ void
warpReduce(volatile _Tp* _data, unsigned int tid)
{
    if(blockSize >= 64)
        _data[tid] += _data[tid + 32];
    if(blockSize >= 32)
        _data[tid] += _data[tid + 16];
    if(blockSize >= 16)
        _data[tid] += _data[tid + 8];
    if(blockSize >= 8)
        _data[tid] += _data[tid + 4];
    if(blockSize >= 4)
        _data[tid] += _data[tid + 2];
    if(blockSize >= 2)
        _data[tid] += _data[tid + 1];
}

//----------------------------------------------------------------------------//

template <unsigned int blockSize, typename _Tp>
__global__ void
reduce(_Tp* _idata, _Tp* _odata, unsigned int n)
{
    extern __shared__ _Tp _data[];
    unsigned int          tid      = threadIdx.x;
    unsigned int          i        = (2 * blockSize) * blockIdx.x + tid;
    unsigned int          gridSize = 2 * blockSize * gridDim.x;
    _data[tid]                     = 0;

    while(i < n)
    {
        _data[tid] += _idata[i] + _idata[i + blockSize];
        i += gridSize;
    }

    __syncthreads();

    if(blockSize >= 512)
    {
        if(tid < 256)
        {
            _data[tid] += _data[tid + 256];
        }
        __syncthreads();
    }

    if(blockSize >= 256)
    {
        if(tid < 128)
        {
            _data[tid] += _data[tid + 128];
        }
        __syncthreads();
    }

    if(blockSize >= 128)
    {
        if(tid < 64)
        {
            _data[tid] += _data[tid + 64];
        }
        __syncthreads();
    }

    if(tid < 32)
        warpReduce<blockSize, _Tp>(_data, tid);

    if(tid == 0)
        _odata[blockIdx.x] = _data[0];
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void
compute_reduction(int threads, _Tp* _idata, _Tp* _odata, int dimGrid, int dimBlock,
                  int smemSize, cudaStream_t stream)
{
    cudaStreamSynchronize(stream);
    CUDA_CHECK_LAST_ERROR();

    switch(threads)
    {
        case 512:
            reduce<512, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 256:
            reduce<256, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 128:
            reduce<128, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 64:
            reduce<64, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 32:
            reduce<32, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 16:
            reduce<16, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 8:
            reduce<8, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 4:
            reduce<4, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 2:
            reduce<2, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 1:
            reduce<1, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
    }
    CUDA_CHECK_LAST_ERROR();

    cudaStreamSynchronize(stream);
    CUDA_CHECK_LAST_ERROR();
}

//============================================================================//

template <typename _Tp>
void
call_compute_reduction(int& _i, int& _offset, int nthreads, _Tp* _idata, _Tp* _odata,
                       int dimGrid, int dimBlock, int smemSize, cudaStream_t stream)
{
    // assumes nthreads < cuda_max_threads_per_block()
    compute_reduction(nthreads, _idata + _offset, _odata + _offset, dimGrid, dimBlock,
                      smemSize, stream);
    _i -= nthreads;
    _offset += nthreads;
}

//============================================================================//

float
reduce(float* _in, float* _out, int size, cudaStream_t stream)
{
    int remain = size;
    int offset = 0;

    int smemSize = cuda_shared_memory_per_block();
    int dimGrid  = cuda_multi_processor_count();
    int dimBlock = cuda_max_threads_per_block();

    while(remain > 0)
    {
        for(const auto& itr : { 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 })
        {
            if(remain >= itr)
            {
                call_compute_reduction(remain, offset, itr, _in, _out, dimGrid, dimBlock,
                                       smemSize, stream);
                break;
            }
        }
    }

    float _sum;
    cudaMemcpyAsync(&_sum, _out, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return _sum;
}

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

__global__ void
cuda_rotate_kernel(float* dst, const float* src, const float theta, const int nx,
                   const int ny)
{
    float xoff = round(nx / 2.0);
    float yoff = round(ny / 2.0);
    float xop  = (nx % 2 == 0) ? 0.5 : 0.0;
    float yop  = (ny % 2 == 0) ? 0.5 : 0.0;

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
            float rx = float(i) - xoff + xop;
            float ry = float(j) - yoff + yop;
            // transformation
            float tx = rx * cosf(theta) + -ry * sinf(theta);
            float ty = rx * sinf(theta) + ry * cosf(theta);
            // indices in 2D
            float x = (tx + xoff - xop);
            float y = (ty + yoff - yop);
            // index in 1D array
            int rz = j * nx + i;
            if(rz < 0 || rz >= src_size)
                continue;
            // within bounds
            unsigned x1   = floor(tx + xoff - xop);
            unsigned y1   = floor(ty + yoff - yop);
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
}

//============================================================================//

float*
cuda_rotate(const float* src, const float theta, const int nx, const int ny,
            cudaStream_t stream)
{
    NVTX_RANGE_PUSH(&nvtx_rotate);

    dim3 block = dim3(256);
    dim3 grid  = dim3((nx + block.x - 1) / block.x, nx);
    int  smem  = 0;

    float* _dst = gpu_malloc<float>(nx * ny);
    cudaMemsetAsync(_dst, 0, nx * ny * sizeof(float), stream);
    cuda_rotate_kernel<<<grid, block, smem, stream>>>(_dst, src, theta, nx, ny);
    CUDA_CHECK_LAST_ERROR();

    NVTX_RANGE_POP(&nvtx_rotate);
    return _dst;
}

//============================================================================//

void
cuda_rotate_ip(float* dst, const float* src, const float theta, const int nx,
               const int ny, cudaStream_t stream)
{
    NVTX_RANGE_PUSH(&nvtx_rotate);

    dim3 block = dim3(256);
    dim3 grid  = dim3((nx + block.x - 1) / block.x, nx);
    int  smem  = 0;

    cudaMemsetAsync(dst, 0, nx * ny * sizeof(float), stream);
    cuda_rotate_kernel<<<grid, block, smem, stream>>>(dst, src, theta, nx, ny);

    CUDA_CHECK_LAST_ERROR();

    NVTX_RANGE_POP(&nvtx_rotate);
}

//============================================================================//
