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

//----------------------------------------------------------------------------//

__inline__ __device__ float
warpReduceSum(float val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

//----------------------------------------------------------------------------//

__inline__ __device__ float
warpAllReduceSum(float val)
{
    for(int mask = warpSize / 2; mask > 0; mask /= 2)
        val += __shfl_xor(val, mask);
    return val;
}

//----------------------------------------------------------------------------//

__inline__ __device__ float
blockReduceSum(float val)
{
    static __shared__ float shared[32];  // Shared mem for 32 partial sums
    int                   lane = threadIdx.x % warpSize;
    int                   wid  = threadIdx.x / warpSize;

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
deviceReduceKernel(float* in, float* out, int N)
{
    float sum = 0;
    // reduce multiple elements per thread
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }
    sum = blockReduceSum(sum);
    if(threadIdx.x == 0)
        out[blockIdx.x] = sum;
}

//----------------------------------------------------------------------------//

void
deviceReduce(float* in, float* out, int N, cudaStream_t stream)
{
    int threads = 512;
    int blocks  = min((N + threads - 1) / threads, 1024);
    int smem    = 0;

    deviceReduceKernel<<<blocks, threads, smem, stream>>>(in, out, N);
    deviceReduceKernel<<<1, 1024, 0, stream>>>(out, out, blocks);
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
compute_reduction(int threads, _Tp* _idata, _Tp* _odata, int dimGrid,
                  int dimBlock, int smemSize, cudaStream_t stream)
{
    cudaStreamSynchronize(stream);
    CUDA_CHECK_LAST_ERROR();

    switch(threads)
    {
        case 512:
            reduce<512, _Tp><<<dimGrid, dimBlock, smemSize, stream>>>(
                _idata, _odata, threads);
            break;
        case 256:
            reduce<256, _Tp><<<dimGrid, dimBlock, smemSize, stream>>>(
                _idata, _odata, threads);
            break;
        case 128:
            reduce<128, _Tp><<<dimGrid, dimBlock, smemSize, stream>>>(
                _idata, _odata, threads);
            break;
        case 64:
            reduce<64, _Tp><<<dimGrid, dimBlock, smemSize, stream>>>(
                _idata, _odata, threads);
            break;
        case 32:
            reduce<32, _Tp><<<dimGrid, dimBlock, smemSize, stream>>>(
                _idata, _odata, threads);
            break;
        case 16:
            reduce<16, _Tp><<<dimGrid, dimBlock, smemSize, stream>>>(
                _idata, _odata, threads);
            break;
        case 8:
            reduce<8, _Tp><<<dimGrid, dimBlock, smemSize, stream>>>(
                _idata, _odata, threads);
            break;
        case 4:
            reduce<4, _Tp><<<dimGrid, dimBlock, smemSize, stream>>>(
                _idata, _odata, threads);
            break;
        case 2:
            reduce<2, _Tp><<<dimGrid, dimBlock, smemSize, stream>>>(
                _idata, _odata, threads);
            break;
        case 1:
            reduce<1, _Tp><<<dimGrid, dimBlock, smemSize, stream>>>(
                _idata, _odata, threads);
            break;
    }
    CUDA_CHECK_LAST_ERROR();

    cudaStreamSynchronize(stream);
    CUDA_CHECK_LAST_ERROR();
}

//============================================================================//

template <typename _Tp>
void
call_compute_reduction(int& _i, int& _offset, int nthreads,
                       _Tp* _idata, _Tp* _odata, int dimGrid, int dimBlock,
                       int smemSize, cudaStream_t stream)
{
    // assumes nthreads < cuda_max_threads_per_block()
    compute_reduction(nthreads, _idata + _offset, _odata + _offset, dimGrid,
                      dimBlock, smemSize, stream);
    _i -= nthreads;
    _offset += nthreads;
}

//============================================================================//

void
reduce(float* _in, float* _out, int size, cudaStream_t stream)
{

    int  remain = size;
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
                call_compute_reduction(remain, offset, itr, _in, _out,
                                       dimGrid, dimBlock, smemSize, stream);
                break;
            }
        }
    }
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
//  preprocessesing
//
//============================================================================//

__global__ void
cuda_preprocessing_global_x(int ry, float* gridx, int size)
{
    int i0     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += stride)
        gridx[i] = (-ry * 0.5f) + i;
}

//----------------------------------------------------------------------------//

__global__ void
cuda_preprocessing_global_y(int rz, float* gridy, int size)
{
    int i0     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += stride)
        gridy[i] = (-rz * 0.5f) + i;
}

//----------------------------------------------------------------------------//

__global__ void
cuda_preprocessing_global_mov(int num_pixels, float center, float* mov)
{
    float fnum_pixels = (float) num_pixels;
    *mov              = (fnum_pixels - 1.0f) * 0.5f - center;
    if(*mov - floor(*mov) < 0.01)
        *mov += 0.01;
    *mov += 0.5;
}

//----------------------------------------------------------------------------//

void
cuda_preprocessing(int ry, int rz, int num_pixels, float center, float* mov, float* gridx,
                   float* gridy, cudaStream_t* streams)
{
    NVTX_RANGE_PUSH(&nvtx_preprocessing);

    int nb   = cuda_multi_processor_count();
    int nt   = cuda_max_threads_per_block();
    int smem = 0;

    CUDA_CHECK_LAST_ERROR();

    cuda_preprocessing_global_x<<<nb, nt, smem, streams[0]>>>(ry, gridx, ry + 1);
    CUDA_CHECK_LAST_ERROR();

    cuda_preprocessing_global_y<<<nb, nt, smem, streams[1]>>>(rz, gridy, rz + 1);
    CUDA_CHECK_LAST_ERROR();

    cuda_preprocessing_global_mov<<<1, 1, smem, streams[2]>>>(num_pixels, center, mov);
    CUDA_CHECK_LAST_ERROR();

    for(auto i : { 0, 1, 2 })
        cudaStreamSynchronize(streams[i]);

    // copy mov to CPU
    // cpu_memcpy(mov, mov_cpu, 1, _dataset->streams[0]);
    // cudaStreamSynchronize(_dataset->streams[0]);

    NVTX_RANGE_POP(&nvtx_preprocessing);
}

//============================================================================//
//
//  calc_coords
//
//============================================================================//

__global__ void
cuda_calc_coords_global(const float* grid, float* coord, float slope, float src_minus,
                        float src_plus, int size)
{
    int i0     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += stride)
    {
        coord[i] = slope * (grid[i] - src_minus) + src_plus;
    }
}

//----------------------------------------------------------------------------//

void
cuda_calc_coords(int ry, int rz, float xi, float yi, float sin_p, float cos_p,
                 const float* gridx, const float* gridy, float* coordx, float* coordy,
                 cudaStream_t* streams)
{
    if(ry <= 0 && rz <= 0)
        return;

    NVTX_RANGE_PUSH(&nvtx_calc_coords);

    int nb   = cuda_multi_processor_count();
    int nt   = cuda_max_threads_per_block();
    int smem = 0;

    float srcx   = (xi * cos_p) - (yi * sin_p);
    float srcy   = (xi * sin_p) + (yi * cos_p);
    float detx   = (-xi * cos_p) - (yi * sin_p);
    float dety   = (-xi * sin_p) + (yi * cos_p);
    float slope  = (srcy - dety) / (srcx - detx);
    float islope = (srcx - detx) / (srcy - dety);

    CUDA_CHECK_LAST_ERROR();

    cuda_calc_coords_global<<<nb, nt, smem, streams[0]>>>(gridx, coordy, slope, srcx,
                                                          srcy, ry + 1);
    CUDA_CHECK_LAST_ERROR();

    cuda_calc_coords_global<<<nb, nt, smem, streams[1]>>>(gridy, coordx, islope, srcy,
                                                          srcx, rz + 1);
    CUDA_CHECK_LAST_ERROR();

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    NVTX_RANGE_POP(&nvtx_calc_coords);
}

//============================================================================//
//
//  trim_coords
//
//============================================================================//

__global__ void
cuda_trim_coords_global_a(int ry, int rz, const float* coordx, const float* coordy,
                          const float* gridx, const float* gridy, int* asize, float* ax,
                          float* ay, int* bsize, float* bx, float* by)
{
    int i0     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float gridx_gt = gridx[0] + 0.01;
    float gridx_le = gridx[ry] - 0.01;

    for(int i = i0; i <= rz; i += stride)
    {
        if(coordx[i] >= gridx_gt && coordx[i] <= gridx_le)
        {
            ax[*asize] = coordx[i];
            ay[*asize] = gridy[i];
            atomicAdd(asize, 1);
        }
    }
}

//----------------------------------------------------------------------------//

__global__ void
cuda_trim_coords_global_b(int ry, int rz, const float* coordx, const float* coordy,
                          const float* gridx, const float* gridy, int* asize, float* ax,
                          float* ay, int* bsize, float* bx, float* by)
{
    int i0     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float gridy_gt = gridy[0] + 0.01;
    float gridy_le = gridy[rz] - 0.01;

    for(int i = i0; i <= ry; i += stride)
    {
        if(coordy[i] >= gridy_gt && coordy[i] <= gridy_le)
        {
            bx[*bsize] = gridx[i];
            by[*bsize] = coordy[i];
            atomicAdd(bsize, 1);
        }
    }
}

//----------------------------------------------------------------------------//

void
cuda_trim_coords(int ry, int rz, const float* coordx, const float* coordy,
                 const float* gridx, const float* gridy, int* asize, float* ax, float* ay,
                 int* bsize, float* bx, float* by, cudaStream_t* streams)
{
    if(ry <= 0 && rz <= 0)
        return;

    NVTX_RANGE_PUSH(&nvtx_trim_coords);

    int nb   = 1;
    int nt   = 1;
    int smem = 0;

    CUDA_CHECK_LAST_ERROR();

    cuda_global_zero<int><<<1, 1, smem, streams[0]>>>(asize, 1, nullptr);
    CUDA_CHECK_LAST_ERROR();

    cuda_global_zero<int><<<1, 1, smem, streams[1]>>>(bsize, 1, nullptr);
    CUDA_CHECK_LAST_ERROR();

    cuda_trim_coords_global_a<<<nb, nt, smem, streams[0]>>>(ry, rz, coordx, coordy, gridx,
                                                            gridy, asize, ax, ay, bsize,
                                                            bx, by);
    CUDA_CHECK_LAST_ERROR();

    cuda_trim_coords_global_b<<<nb, nt, smem, streams[1]>>>(ry, rz, coordx, coordy, gridx,
                                                            gridy, asize, ax, ay, bsize,
                                                            bx, by);
    CUDA_CHECK_LAST_ERROR();

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    NVTX_RANGE_POP(&nvtx_trim_coords);
}

//============================================================================//
//
//  sort intersections
//
//============================================================================//

__global__ void
cuda_sort_intersections_global_csize(const int* asize, const int* bsize, int* csize)
{
    *csize = (*asize) + (*bsize);
}

//----------------------------------------------------------------------------//

__global__ void
cuda_sort_intersections_global_a(int ind_condition, int* ijk, const int* asize,
                                 const float* ax, const float* ay, const int* bsize,
                                 const float* bx, const float* by, int* csize,
                                 float* coorx, float* coory)
{
    int& i = ijk[0];
    int& k = ijk[2];

    int offset = (ind_condition == 0) ? ((*asize) - 1) : 0;
    int opsign = (ind_condition == 0) ? -1 : 1;

    int size   = (*asize) - i + 1;
    int i0     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int n = i0; n < size; n += stride)
    {
        coorx[k + n] = ax[offset + opsign * (i + n)];
        coory[k + n] = ay[offset + opsign * (i + n)];
    }
}

//----------------------------------------------------------------------------//

__global__ void
cuda_sort_intersections_global_b(int ind_condition, int* ijk, const int* asize,
                                 const float* ax, const float* ay, const int* bsize,
                                 const float* bx, const float* by, int* csize,
                                 float* coorx, float* coory)
{
    int& i = ijk[0];
    int& j = ijk[1];
    int& k = ijk[2];
    k += (*asize) - i;

    int size   = (*bsize) - j + 1;
    int i0     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int n = i0; n < size; n += stride)
    {
        coorx[k + n] = bx[j + n];
        coory[k + n] = by[j + n];
    }
}

//----------------------------------------------------------------------------//

__global__ void
cuda_sort_intersections_global_partial(int ind_condition, int* ijk, const int* asize,
                                       const float* ax, const float* ay, const int* bsize,
                                       const float* bx, const float* by, int* csize,
                                       float* coorx, float* coory)
{
    int  offset = (ind_condition == 0) ? ((*asize) - 1) : 0;
    int  opsign = (ind_condition == 0) ? -1 : 1;
    int& i      = ijk[0];
    int& j      = ijk[1];
    int& k      = ijk[2];
    i = j = k = 0;
    while(i < (*asize) && j < (*bsize))
    {
        if(ax[offset + opsign * i] < bx[j])
        {
            coorx[k] = ax[offset + opsign * i];
            coory[k] = ay[offset + opsign * i];
            ++i;
        }
        else
        {
            coorx[k] = bx[j];
            coory[k] = by[j];
            ++j;
        }
        ++k;
    }
}

//----------------------------------------------------------------------------//

__global__ void
cuda_sort_intersections_global(int ind_condition, const int* asize, const float* ax,
                               const float* ay, const int* bsize, const float* bx,
                               const float* by, int* csize, float* coorx, float* coory)
{
    int i = 0, j = 0, k = 0;

    int offset = (ind_condition == 0) ? ((*asize) - 1) : 0;
    int opsign = (ind_condition == 0) ? -1 : 1;
    while(i < (*asize) && j < (*bsize))
    {
        if(ax[offset + opsign * i] < bx[j])
        {
            coorx[k] = ax[offset + opsign * i];
            coory[k] = ay[offset + opsign * i];
            ++i;
        }
        else
        {
            coorx[k] = bx[j];
            coory[k] = by[j];
            ++j;
        }
        ++k;
    }

    while(i < (*asize))
    {
        coorx[k]   = ax[offset + opsign * i];
        coory[k++] = ay[offset + opsign * (i++)];
    }
    while(j < (*bsize))
    {
        coorx[k]   = bx[j];
        coory[k++] = by[j++];
    }
}

//----------------------------------------------------------------------------//

void
cuda_sort_intersections(int ind_condition, const int* asize, const float* ax,
                        const float* ay, const int* bsize, const float* bx,
                        const float* by, int* csize, float* coorx, float* coory,
                        cudaStream_t* streams)
{
    NVTX_RANGE_PUSH(&nvtx_sort_intersections);

    int nb   = cuda_multi_processor_count();
    int nt   = 1;
    int smem = 0;

    CUDA_CHECK_LAST_ERROR();
    /*
    cuda_sort_intersections_global<<< 1, 1, smem, streams[0] >>>(ind_condition,
                                                                asize, ax, ay,
                                                                bsize, bx, by,
                                                                csize, coorx,
    coory);

    cuda_sort_intersections_global_csize<<< 1, 1, smem, streams[1] >>>(asize,
                                                                      bsize,
                                                                      csize);
    */

    static thread_local int* ijk = gpu_malloc<int>(3);

    cuda_sort_intersections_global_partial<<<1, 1, smem, streams[0]>>>(
        ind_condition, ijk, asize, ax, ay, bsize, bx, by, csize, coorx, coory);

    cuda_sort_intersections_global_csize<<<1, 1, smem, streams[1]>>>(asize, bsize, csize);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    cuda_sort_intersections_global_a<<<nb, nt, smem, streams[0]>>>(ind_condition, ijk,
                                                                   asize, ax, ay, bsize,
                                                                   bx, by, csize, coorx,
                                                                   coory);

    cuda_sort_intersections_global_b<<<nb, nt, smem, streams[1]>>>(ind_condition, ijk,
                                                                   asize, ax, ay, bsize,
                                                                   bx, by, csize, coorx,
                                                                   coory);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    CUDA_CHECK_LAST_ERROR();

    NVTX_RANGE_POP(&nvtx_sort_intersections);
}

//============================================================================//
//
//  calc_dist
//
//============================================================================//

__global__ void
cuda_calc_dist_global_dist(int ry, int rz, const int* csize, const float* coorx,
                           const float* coory, int* indi, float* dist)
{
    int size   = (*csize) - 1;
    int i0     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += stride)
    {
        float diffx = coorx[i + 1] - coorx[i];
        float diffy = coory[i + 1] - coory[i];
        dist[i]     = sqrt(diffx * diffx + diffy * diffy);
    }
}

//----------------------------------------------------------------------------//

__global__ void
cuda_calc_dist_global_indi(int ry, int rz, const int* csize, const float* coorx,
                           const float* coory, int* indi, float* dist)
{
    int size   = (*csize) - 1;
    int i0     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += stride)
    {
        float midx = (coorx[i + 1] + coorx[i]) * 0.5;
        float midy = (coory[i + 1] + coory[i]) * 0.5;
        float x1   = midx + 0.5 * ry;
        float x2   = midy + 0.5 * rz;
        int   i1   = (int) (midx + 0.5 * ry);
        int   i2   = (int) (midy + 0.5 * rz);
        int   indx = i1 - (i1 > x1);
        int   indy = i2 - (i2 > x2);
        indi[i]    = indy + (indx * rz);
    }
}

//----------------------------------------------------------------------------//

void
cuda_calc_dist(int ry, int rz, const int* csize, const float* coorx, const float* coory,
               int* indi, float* dist, cudaStream_t* streams)
{
    NVTX_RANGE_PUSH(&nvtx_calc_dist);

    int nb   = cuda_multi_processor_count();
    int nt   = cuda_max_threads_per_block();
    int smem = 0;

    CUDA_CHECK_LAST_ERROR();

    cuda_calc_dist_global_dist<<<nb, nt, smem, streams[0]>>>(ry, rz, csize, coorx, coory,
                                                             indi, dist);
    CUDA_CHECK_LAST_ERROR();

    cuda_calc_dist_global_indi<<<nb, nt, smem, streams[0]>>>(ry, rz, csize, coorx, coory,
                                                             indi, dist);
    CUDA_CHECK_LAST_ERROR();

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    NVTX_RANGE_POP(&nvtx_calc_dist);
}

//============================================================================//
//
//  calc_sum_sqr
//
//============================================================================//

__global__ void
cuda_calc_sum_sqr_global(const int* csize, const float* dist, float* sum_sqr)
{
    int size   = (*csize) - 1;
    int i0     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float _sum = 0.0f;
    for(int i = i0; i < size; i += stride)
    {
        _sum += dist[i] * dist[i];
    }
    atomicAdd(sum_sqr, _sum);
}

//----------------------------------------------------------------------------//

void
cuda_calc_sum_sqr(const int* csize, const float* dist, float* sum_sqr,
                  cudaStream_t* streams)
{
    NVTX_RANGE_PUSH(&nvtx_calc_sum_sqr);

    // int nb = cuda_multi_processor_count();
    // int nt = 1;
    // int smem = 0;

    CUDA_CHECK_LAST_ERROR();

    // cuda_calc_sum_sqr_global<<< nb, nt, smem, streams[0] >>>(csize, dist,
    // sum_sqr); CUDA_CHECK_LAST_ERROR();

    // std::stringstream ss;
    // ss << "csize: " << csize << ", dist: " << dist << ", sum_sqr: " <<
    // sum_sqr; PRINT_HERE(ss.str().c_str());

    static thread_local int    size      = 0;
    static thread_local int    last_size = -1;
    static thread_local float* data      = NULL;

    cudaMemcpyAsync(&size, csize, sizeof(int), cudaMemcpyDeviceToHost, streams[0]);
    CUDA_CHECK_LAST_ERROR();

    cudaStreamSynchronize(streams[0]);
    CUDA_CHECK_LAST_ERROR();
    size -= 1;

    if(last_size < 0 || data == NULL)
    {
        if(data != NULL)
            cudaFree(data);
        cudaMalloc(&data, size * sizeof(float));
        CUDA_CHECK_LAST_ERROR();
    }

    last_size  = size;
    float _sum = 0.0f;

    thrust::transform(thrust::system::cuda::par.on(streams[0]), dist, dist + size, dist,
                      data, thrust::multiplies<float>());
    CUDA_CHECK_LAST_ERROR();

    _sum = thrust::reduce(thrust::system::cuda::par.on(streams[0]), data, data + size,
                          0.0f, thrust::plus<float>());
    CUDA_CHECK_LAST_ERROR();

    cudaMemcpyAsync(sum_sqr, &_sum, sizeof(float), cudaMemcpyHostToDevice, streams[0]);
    CUDA_CHECK_LAST_ERROR();

    NVTX_RANGE_POP(&nvtx_calc_sum_sqr);
}

//============================================================================//
//
//  calc_simdata
//
//============================================================================//

__global__ void
cuda_calc_simdata_global(int s, int p, int d, int ry, int rz, int dt, int dx,
                         const int* csize, const int* indi, const float* dist,
                         const float* model, const float* sum_dist, float* simdata)
{
    if(*sum_dist != 0.0f)
    {
        int size = (*csize) - 1;
        if(size < 2)
            return;

        int i0     = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        int   index_model = s * ry * rz;
        int   index_data  = d + p * dx + s * dt * dx;
        float local_sum   = 0.0f;
        for(int i = i0; i < size; i += stride)
        {
            local_sum += model[indi[i] + index_model] * dist[i];
        }
        atomicAdd(&(simdata[index_data]), local_sum);
    }
}

//----------------------------------------------------------------------------//

void
cuda_calc_simdata(int s, int p, int d, int ry, int rz, int dt, int dx, const int* csize,
                  const int* indi, const float* dist, const float* model,
                  const float* sum_dist, float* simdata, cudaStream_t* streams)
{
    NVTX_RANGE_PUSH(&nvtx_calc_simdata);

    int nb   = cuda_multi_processor_count();
    int nt   = 1;
    int smem = 0;

    CUDA_CHECK_LAST_ERROR();
    cuda_calc_simdata_global<<<nb, nt, smem, streams[0]>>>(s, p, d, ry, rz, dt, dx, csize,
                                                           indi, dist, model, sum_dist,
                                                           simdata);
    CUDA_CHECK_LAST_ERROR();

    NVTX_RANGE_POP(&nvtx_calc_simdata);
}

//============================================================================//
//
//  rotate
//
//============================================================================//

__global__ void
cuda_rotate_global(float* obj, const float theta, const int nx, const int ny, float* rot)
{
    float xoff = round(nx / 2.0);
    float yoff = round(ny / 2.0);
    float xop  = (nx % 2 == 0) ? 0.5 : 0.0;
    float yop  = (ny % 2 == 0) ? 0.5 : 0.0;

    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int j0      = blockIdx.y * blockDim.y + threadIdx.y;
    int istride = blockDim.x * gridDim.x;
    int jstride = blockDim.y * gridDim.y;

    int obj_size = nx * ny;
    // for(int i = i0; i < size; i += stride)

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
            // within bounds
            int   x1   = floor(tx + xoff - xop);
            int   y1   = floor(ty + yoff - yop);
            int   x2   = x1 + 1;
            int   y2   = y1 + 1;
            float fxy1 = 0.0f;
            float fxy2 = 0.0f;
            if(y1 * nx + x1 < obj_size)
                fxy1 += (x2 - x) * obj[y1 * nx + x1];
            if(y1 * nx + x2 < obj_size)
                fxy1 += (x - x1) * obj[y1 * nx + x2];
            if(y2 * nx + x1 < obj_size)
                fxy2 += (x2 - x) * obj[y2 * nx + x1];
            if(y2 * nx + x2 < obj_size)
                fxy2 += (x - x1) * obj[y2 * nx + x2];
            rot[rz] += (y2 - y) * fxy1 + (y - y1) * fxy2;
        }
    }
}

//============================================================================//

void
cuda_rotate(float* dst, float* src, const float theta, const int nx, const int ny,
            cudaStream_t* stream)
{
    NVTX_RANGE_PUSH(&nvtx_rotate);

    dim3 nb(32, 32);
    dim3 nt(4, 4);
    int  smem = 0;

    CUDA_CHECK_LAST_ERROR();
    cuda_rotate_global<<<nb, nt, smem, *stream>>>(src, theta, nx, ny, dst);
    CUDA_CHECK_LAST_ERROR();

    NVTX_RANGE_POP(&nvtx_rotate);
}

//============================================================================//
