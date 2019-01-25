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
//   TOMOPY class header

#pragma once

#include "common.hh"

BEGIN_EXTERN_C
#include "gpu.h"
#include "utils.h"
END_EXTERN_C

//======================================================================================//
//  CUDA only

#if defined(TOMOPY_USE_CUDA)

//--------------------------------------------------------------------------------------//

template <typename _Tp>
_Tp*
gpu_malloc(uintmax_t size)
{
    _Tp* _gpu;
    cudaMalloc((void**) &_gpu, size * sizeof(_Tp));
    CUDA_CHECK_LAST_ERROR();
    return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
gpu_memcpy(_Tp* _gpu, const _Tp* _cpu, uintmax_t size)
{
    cudaMemcpy(_gpu, _cpu, size * sizeof(_Tp), cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
gpu_memcpy(_Tp* _gpu, const _Tp* _cpu, uintmax_t size, cudaStream_t stream)
{
    cudaMemcpyAsync(_gpu, _cpu, size * sizeof(_Tp), cudaMemcpyHostToDevice, stream);
    CUDA_CHECK_LAST_ERROR();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
cpu_memcpy(const _Tp* _gpu, _Tp* _cpu, uintmax_t size)
{
    cudaMemcpy(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
cpu_memcpy(const _Tp* _gpu, _Tp* _cpu, uintmax_t size, cudaStream_t stream)
{
    cudaMemcpyAsync(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//

template <typename _Tp>
_Tp*
malloc_and_memcpy(const _Tp* _cpu, uintmax_t size)
{
    _Tp* _gpu;
    cudaMalloc((void**) &_gpu, size * sizeof(_Tp));
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(_gpu, _cpu, size * sizeof(_Tp), cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
_Tp*
malloc_and_memset(uintmax_t size, int value)
{
    _Tp* _gpu;
    cudaMalloc((void**) &_gpu, size * sizeof(_Tp));
    CUDA_CHECK_LAST_ERROR();
    cudaMemset(_gpu, value, size * sizeof(_Tp));
    CUDA_CHECK_LAST_ERROR();
    return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
_Tp*
malloc_and_async_memset(uintmax_t size, int value, cudaStream_t stream)
{
    _Tp* _gpu;
    cudaMalloc((void**) &_gpu, size * sizeof(_Tp));
    CUDA_CHECK_LAST_ERROR();
    cudaMemsetAsync(_gpu, value, size * sizeof(_Tp), stream);
    CUDA_CHECK_LAST_ERROR();
    return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
memcpy_and_free(_Tp* _cpu, _Tp* _gpu, uintmax_t size)
{
    cudaMemcpy(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
    cudaFree(_gpu);
    CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//

template <typename _Tp>
_Tp*
malloc_and_async_memcpy(const _Tp* _cpu, uintmax_t size, cudaStream_t stream)
{
    _Tp* _gpu;
    cudaMalloc((void**) &_gpu, size * sizeof(_Tp));
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpyAsync(_gpu, _cpu, size * sizeof(_Tp), cudaMemcpyHostToDevice, stream);
    CUDA_CHECK_LAST_ERROR();
    return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
async_memcpy_and_free(_Tp* _cpu, _Tp* _gpu, uintmax_t size, cudaStream_t stream)
{
    cudaMemcpyAsync(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK_LAST_ERROR();
    cudaFree(_gpu);
}

//======================================================================================//

inline cudaStream_t*
create_streams(const int nstreams)
{
    cudaStream_t* streams = new cudaStream_t[nstreams];
    for(int i = 0; i < nstreams; ++i)
    {
        cudaStreamCreate(&streams[i]);
        CUDA_CHECK_LAST_ERROR();
    }
    return streams;
}

//======================================================================================//

inline void
destroy_streams(cudaStream_t* streams, const int nstreams)
{
    for(int i = 0; i < nstreams; ++i)
    {
        cudaStreamDestroy(streams[i]);
        CUDA_CHECK_LAST_ERROR();
    }
    delete[] streams;
}

//======================================================================================//

template <typename _Tp>
_Tp
reduce(_Tp* data, _Tp init, int nitems, cudaStream_t stream)
{
    _Tp* beg = data;
    _Tp* end = data + nitems;
    return thrust::reduce(thrust::system::cuda::par.on(stream), beg, end, init,
                          thrust::plus<_Tp>());
}

//======================================================================================//

template <typename _Tp>
void
transform_sum(_Tp* input_data, int nitems, _Tp* result, cudaStream_t stream)
{
    _Tp* beg = input_data;
    _Tp* end = input_data + nitems;
    thrust::transform(thrust::system::cuda::par.on(stream), beg, end, result, result,
                      thrust::plus<_Tp>());
}

//======================================================================================//

#else  // not defined(TOMOPY_USE_CUDA)

//======================================================================================//

template <typename _Tp>
_Tp*
gpu_malloc(uintmax_t size)
{
    return nullptr;
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
void
gpu_memcpy(_Tp*, const _Tp*, uintmax_t)
{
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
void
gpu_memcpy(_Tp*, const _Tp*, uintmax_t, cudaStream_t)
{
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
void
cpu_memcpy(const _Tp*, _Tp*, uintmax_t)
{
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
void
cpu_memcpy(const _Tp*, _Tp*, uintmax_t, cudaStream_t)
{
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
_Tp*
malloc_and_memcpy(const _Tp*, uintmax_t)
{
    return nullptr;
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
_Tp*
malloc_and_memset(uintmax_t size, int value)
{
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
_Tp*
malloc_and_async_memset(uintmax_t size, int value, cudaStream_t stream)
{
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
void
memcpy_and_free(_Tp*, _Tp*, uintmax_t)
{
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
_Tp*
malloc_and_async_memcpy(const _Tp*, uintmax_t, cudaStream_t)
{
    return nullptr;
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
void
async_memcpy_and_free(_Tp*, _Tp*, uintmax_t, cudaStream_t)
{
}
//--------------------------------------------------------------------------------------//
inline cudaStream_t*
create_streams(const int)
{
    return nullptr;
}
//--------------------------------------------------------------------------------------//
inline void
destroy_streams(cudaStream_t*, const int)
{
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
_Tp
reduce(_Tp*, _Tp, int, cudaStream_t)
{
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
void
transform_sum(_Tp*, int, _Tp*, cudaStream_t)
{
}
//--------------------------------------------------------------------------------------//

#endif  // if defined(TOMOPY_USE_CUDA)
