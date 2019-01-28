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

//======================================================================================//

#include "common.hh"
#include "gpu.hh"

BEGIN_EXTERN_C
#include "gpu.h"
#include "utils_cuda.h"
END_EXTERN_C

namespace cg = cooperative_groups;

//======================================================================================//

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

//======================================================================================//

//  gridDim:    This variable contains the dimensions of the grid.
//  blockIdx:   This variable contains the block index within the grid.
//  blockDim:   This variable and contains the dimensions of the block.
//  threadIdx:  This variable contains the thread index within the block.

//--------------------------------------------------------------------------------------//

__inline__ __device__ float
warpReduceSum(float val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

//--------------------------------------------------------------------------------------//

__inline__ __device__ float
warpAllReduceSum(float val)
{
    for(int mask = warpSize / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(FULL_MASK, val, mask);
    return val;
}

//--------------------------------------------------------------------------------------//

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

//--------------------------------------------------------------------------------------//

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

//--------------------------------------------------------------------------------------//

float
deviceReduce(const float* in, float* out, int N)
{
    int threads = 512;
    int blocks  = min((N + threads - 1) / threads, 1024);
    int smem    = 0;

    deviceReduceKernel<<<blocks, threads, smem>>>(in, out, N);
    deviceReduceKernel<<<1, 1024, 0>>>(out, out, blocks);

    float _sum;
    cudaMemcpy(&_sum, out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);
    return _sum;
}

//======================================================================================//
//
//  efficient reduction
//  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//
//======================================================================================//

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

//--------------------------------------------------------------------------------------//

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

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
compute_reduction(int threads, _Tp* _idata, _Tp* _odata, int dimGrid, int dimBlock,
                  int smemSize)
{
    cudaStreamSynchronize(0);
    CUDA_CHECK_LAST_ERROR();

    switch(threads)
    {
        case 512:
            reduce<512, _Tp><<<dimGrid, dimBlock, smemSize>>>(_idata, _odata, threads);
            break;
        case 256:
            reduce<256, _Tp><<<dimGrid, dimBlock, smemSize>>>(_idata, _odata, threads);
            break;
        case 128:
            reduce<128, _Tp><<<dimGrid, dimBlock, smemSize>>>(_idata, _odata, threads);
            break;
        case 64:
            reduce<64, _Tp><<<dimGrid, dimBlock, smemSize>>>(_idata, _odata, threads);
            break;
        case 32:
            reduce<32, _Tp><<<dimGrid, dimBlock, smemSize>>>(_idata, _odata, threads);
            break;
        case 16:
            reduce<16, _Tp><<<dimGrid, dimBlock, smemSize>>>(_idata, _odata, threads);
            break;
        case 8:
            reduce<8, _Tp><<<dimGrid, dimBlock, smemSize>>>(_idata, _odata, threads);
            break;
        case 4:
            reduce<4, _Tp><<<dimGrid, dimBlock, smemSize>>>(_idata, _odata, threads);
            break;
        case 2:
            reduce<2, _Tp><<<dimGrid, dimBlock, smemSize>>>(_idata, _odata, threads);
            break;
        case 1:
            reduce<1, _Tp><<<dimGrid, dimBlock, smemSize>>>(_idata, _odata, threads);
            break;
    }
    CUDA_CHECK_LAST_ERROR();

    cudaStreamSynchronize(0);
    CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//

template <typename _Tp>
void
call_compute_reduction(int& _i, int& _offset, int nthreads, _Tp* _idata, _Tp* _odata,
                       int dimGrid, int dimBlock, int smemSize)
{
    // assumes nthreads < cuda_max_threads_per_block()
    compute_reduction(nthreads, _idata + _offset, _odata + _offset, dimGrid, dimBlock,
                      smemSize);
    _i -= nthreads;
    _offset += nthreads;
}

//======================================================================================//

float
reduce(float* _in, float* _out, int size)
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
                                       smemSize);
                break;
            }
        }
    }

    float _sum;
    cudaMemcpy(&_sum, _out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);
    return _sum;
}

//======================================================================================//
//
//      Cooperative Groups for sum
//
//======================================================================================//

__device__ float
reduce_sum(cg::thread_group g, float* temp, float val)
{
    int lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for(int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync();  // wait for all threads to store
        if(lane < i)
            val += temp[lane + i];
        g.sync();  // wait for all threads to load
    }
    return val;  // note: only thread 0 will return full sum
}

//======================================================================================//

__device__ float
thread_sum(const float* input, int n)
{
    int   i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int   iN      = n;
    int   istride = blockDim.x * gridDim.x;
    float sum     = 0;

    for(int i = i0; i < iN; i += istride)
    {
        sum += input[i];
        // float4 in = ((float4*) input)[i];
        // sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}

//======================================================================================//

__global__ void
sum_kernel_block(float* sum, const float* input, int n)
{
    float my_sum = thread_sum(input, n);

    extern __shared__ float temp[];
    auto                    g         = cg::this_thread_block();
    float                   block_sum = reduce_sum(g, temp, my_sum);

    if(g.thread_rank() == 0)
        atomicAdd(sum, block_sum);
}

//======================================================================================//

template <int tile_sz>
__device__ float
reduce_sum_tile_shfl(cg::thread_block_tile<tile_sz> g, float val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for(int i = g.size() / 2; i > 0; i /= 2)
    {
        val += g.shfl_down(val, i);
    }

    return val;  // note: only thread 0 will return full sum
}

//======================================================================================//

template <int tile_sz>
__global__ void
sum_kernel_tile_shfl(float* sum, float* input, int n)
{
    float my_sum = thread_sum(input, n);

    auto  tile     = cg::tiled_partition<tile_sz>(cg::this_thread_block());
    float tile_sum = reduce_sum_tile_shfl<tile_sz>(tile, my_sum);

    if(tile.thread_rank() == 0)
        atomicAdd(sum, tile_sum);
}

//======================================================================================//

__device__ float
atomicAggInc(float* ptr)
{
    cg::coalesced_group g = cg::coalesced_threads();
    float               prev;

    // elect the first active thread to perform atomic add
    if(g.thread_rank() == 0)
    {
        prev = atomicAdd(ptr, g.size());
    }

    // broadcast previous value within the warp
    // and add each active thread’s rank to it
    prev = g.thread_rank() + g.shfl(prev, 0);
    return prev;
}

//======================================================================================//
