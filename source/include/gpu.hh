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
//   TOMOPY header

#pragma once

#include "common.hh"

BEGIN_EXTERN_C
#include "gpu.h"
#include "utils.h"
END_EXTERN_C

#include <new>
#include <sstream>
#include <stdexcept>
#include <string>

//======================================================================================//
//  CUDA only

#if defined(__NVCC__) && defined(TOMOPY_USE_CUDA)

//--------------------------------------------------------------------------------------//

#    if !defined(CUDA_CHECK_CALL)
#        define CUDA_CHECK_CALL(err)                                                     \
            {                                                                            \
                if(cudaSuccess != err)                                                   \
                {                                                                        \
                    std::stringstream ss;                                                \
                    ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'"          \
                       << __FILE__ << "':" << __LINE__ << " : "                          \
                       << cudaGetErrorString(err);                                       \
                    fprintf(stderr, "%s\n", ss.str().c_str());                           \
                    printf("%s\n", ss.str().c_str());                                    \
                    throw std::runtime_error(ss.str().c_str());                          \
                }                                                                        \
            }
#    endif

//--------------------------------------------------------------------------------------//

inline void
stream_sync(cudaStream_t _stream)
{
    cudaStreamSynchronize(_stream);
    CUDA_CHECK_LAST_ERROR();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
_Tp*
gpu_malloc(uintmax_t size)
{
    _Tp* _gpu;
    CUDA_CHECK_CALL(cudaMalloc(&_gpu, size * sizeof(_Tp)));
    if(_gpu == nullptr)
    {
        int _device = 0;
        cudaGetDevice(&_device);
        std::stringstream ss;
        ss << "Error allocating memory on GPU " << _device << " of size "
           << (size * sizeof(_Tp)) << " and type " << typeid(_Tp).name()
           << " (type size = " << sizeof(_Tp) << ")";
        throw std::runtime_error(ss.str().c_str());
    }
    return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
cpu2gpu_memcpy(_Tp* _gpu, const _Tp* _cpu, uintmax_t size, cudaStream_t stream)
{
    cudaMemcpyAsync(_gpu, _cpu, size * sizeof(_Tp), cudaMemcpyHostToDevice, stream);
    CUDA_CHECK_LAST_ERROR();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
gpu2cpu_memcpy(_Tp* _cpu, const _Tp* _gpu, uintmax_t size, cudaStream_t stream)
{
    cudaMemcpyAsync(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK_LAST_ERROR();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
gpu2gpu_memcpy(_Tp* _dst, const _Tp* _src, uintmax_t size, cudaStream_t stream)
{
    cudaMemcpyAsync(_dst, _src, size * sizeof(_Tp), cudaMemcpyDeviceToDevice, stream);
    CUDA_CHECK_LAST_ERROR();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
gpu_memset(_Tp* _gpu, int value, uintmax_t size, cudaStream_t stream)
{
    cudaMemsetAsync(_gpu, value, size * sizeof(_Tp), stream);
    CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//

template <typename _Tp>
_Tp*
gpu_malloc_and_memcpy(const _Tp* _cpu, uintmax_t size, cudaStream_t stream)
{
    _Tp* _gpu = gpu_malloc<_Tp>(size);
    cudaMemcpyAsync(_gpu, _cpu, size * sizeof(_Tp), cudaMemcpyHostToDevice, stream);
    CUDA_CHECK_LAST_ERROR();
    return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
_Tp*
gpu_malloc_and_memset(uintmax_t size, int value, cudaStream_t stream)
{
    _Tp* _gpu = gpu_malloc<_Tp>(size);
    cudaMemsetAsync(_gpu, value, size * sizeof(_Tp), stream);
    CUDA_CHECK_LAST_ERROR();
    return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
gpu2cpu_memcpy_and_free(_Tp* _cpu, _Tp* _gpu, uintmax_t size, cudaStream_t stream)
{
    cudaMemcpyAsync(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK_LAST_ERROR();
    cudaFree(_gpu);
    CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//

inline cudaStream_t*
create_streams(const int nstreams, unsigned int flag = cudaStreamDefault)
{
    cudaStream_t* streams = new cudaStream_t[nstreams];
    for(int i = 0; i < nstreams; ++i)
    {
        cudaStreamCreateWithFlags(&streams[i], flag);
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
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
        CUDA_CHECK_LAST_ERROR();
    }
    delete[] streams;
}

//======================================================================================//

#else  // not defined(TOMOPY_USE_CUDA)

//======================================================================================//
inline void stream_sync(cudaStream_t)
{
    std::stringstream ss;
    ss << "Error! function '" << __FUNCTION__ << "' at line " << __LINE__
       << " not available!";
    throw std::runtime_error(ss.str().c_str());
}
//======================================================================================//
template <typename _Tp>
_Tp* gpu_malloc(uintmax_t)
{
    std::stringstream ss;
    ss << "Error! function '" << __FUNCTION__ << "' at line " << __LINE__
       << " not available!";
    throw std::runtime_error(ss.str().c_str());
    return nullptr;
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
void
cpu2gpu_memcpy(_Tp*, const _Tp*, uintmax_t, cudaStream_t)
{
    std::stringstream ss;
    ss << "Error! function '" << __FUNCTION__ << "' at line " << __LINE__
       << " not available!";
    throw std::runtime_error(ss.str().c_str());
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
void
gpu2cpu_memcpy(_Tp*, const _Tp*, uintmax_t, cudaStream_t)
{
    std::stringstream ss;
    ss << "Error! function '" << __FUNCTION__ << "' at line " << __LINE__
       << " not available!";
    throw std::runtime_error(ss.str().c_str());
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
void
gpu2gpu_memcpy(_Tp*, const _Tp*, uintmax_t, cudaStream_t)
{
    std::stringstream ss;
    ss << "Error! function '" << __FUNCTION__ << "' at line " << __LINE__
       << " not available!";
    throw std::runtime_error(ss.str().c_str());
}
//--------------------------------------------------------------------------------------//
inline cudaStream_t*
create_streams(const int)
{
    std::stringstream ss;
    ss << "Error! function '" << __FUNCTION__ << "' at line " << __LINE__
       << " not available!";
    throw std::runtime_error(ss.str().c_str());
    return nullptr;
}
//--------------------------------------------------------------------------------------//
inline void
destroy_streams(cudaStream_t*, const int)
{
    std::stringstream ss;
    ss << "Error! function '" << __FUNCTION__ << "' at line " << __LINE__
       << " not available!";
    throw std::runtime_error(ss.str().c_str());
}
//--------------------------------------------------------------------------------------//

#endif  // if defined(TOMOPY_USE_CUDA)
