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

#ifndef gpu_hh_
#define gpu_hh_

#ifdef __cplusplus
#    ifndef BEGIN_EXTERN_C
#        define BEGIN_EXTERN_C                                                           \
            extern "C"                                                                   \
            {
#    endif
#    ifndef END_EXTERN_C
#        define END_EXTERN_C }
#    endif
#else
#    ifndef BEGIN_EXTERN_C
#        define BEGIN_EXTERN_C
#    endif
#    ifndef END_EXTERN_C
#        define END_EXTERN_C
#    endif
#endif

//============================================================================//
//  C headers

BEGIN_EXTERN_C
#include "gpu.h"
#include "utils.h"
END_EXTERN_C

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <ctime>

//============================================================================//
//  C++ headers

#include <atomic>
#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "PTL/ThreadData.hh"
#include "PTL/ThreadPool.hh"
#include "PTL/Threading.hh"
#include "PTL/Utility.hh"

//============================================================================//

inline uintmax_t
GetThisThreadID()
{
#if defined(TOMOPY_USE_PTL)
    return ThreadPool::GetThisThreadID();
#else
    static std::atomic<uintmax_t> tcounter;
    static thread_local auto      tid = tcounter++;
    return tid;
#endif
}

//============================================================================//

#if !defined(PRINT_HERE)
#    define PRINT_HERE(extra)                                                            \
        printf("[%lu]> %s@'%s':%i %s\n", GetThisThreadID(), __FUNCTION__, __FILE__,      \
               __LINE__, extra)
#endif

//============================================================================//

template <typename _Tp> using cuda_device_info = std::unordered_map<int, _Tp>;

//============================================================================//

inline tomo_dataset*&
TomoDataset()
{
    static thread_local tomo_dataset* _instance = new tomo_dataset();
    return _instance;
}

//============================================================================//

inline int&
this_thread_device()
{
#if defined(TOMOPY_USE_CUDA)
    static std::atomic<int> _ntid(0);
    ThreadLocalStatic int   _instance =
        (cuda_device_count() > 0) ? ((_ntid++) % cuda_device_count()) : 0;
    return _instance;
#else
    static thread_local int _instance = 0;
    return _instance;
#endif
}

//============================================================================//

inline void
set_this_thread_device()
{
#if defined(TOMOPY_USE_CUDA)
    cuda_set_device(this_thread_device());
#endif
}

//============================================================================//
//  CUDA only
#if defined(TOMOPY_USE_CUDA)

//----------------------------------------------------------------------------//
//  CUDA headers
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#    include <vector_types.h>

//----------------------------------------------------------------------------//
//  Thrust headers
#    include <thrust/device_vector.h>
#    include <thrust/execution_policy.h>
#    include <thrust/fill.h>
#    include <thrust/functional.h>
#    include <thrust/host_vector.h>
#    include <thrust/partition.h>
#    include <thrust/reduce.h>
#    include <thrust/sequence.h>
#    include <thrust/system/cpp/execution_policy.h>
#    include <thrust/system/cuda/execution_policy.h>
#    include <thrust/system/omp/execution_policy.h>
#    include <thrust/system/tbb/execution_policy.h>
#    include <thrust/transform.h>

//============================================================================//

template <typename _Tp>
_Tp*
gpu_malloc(uintmax_t size)
{
    _Tp* _gpu;
    cudaMalloc((void**) &_gpu, size * sizeof(_Tp));
    CUDA_CHECK_LAST_ERROR();
    return _gpu;
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void
gpu_memcpy(_Tp* _gpu, const _Tp* _cpu, uintmax_t size)
{
    cudaMemcpy(_gpu, _cpu, size * sizeof(_Tp), cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void
gpu_memcpy(_Tp* _gpu, const _Tp* _cpu, uintmax_t size, cudaStream_t stream)
{
    cudaMemcpyAsync(_gpu, _cpu, size * sizeof(_Tp), cudaMemcpyHostToDevice, stream);
    CUDA_CHECK_LAST_ERROR();
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void
cpu_memcpy(const _Tp* _gpu, _Tp* _cpu, uintmax_t size)
{
    cudaMemcpy(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void
cpu_memcpy(const _Tp* _gpu, _Tp* _cpu, uintmax_t size, cudaStream_t stream)
{
    cudaMemcpyAsync(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK_LAST_ERROR();
}

//============================================================================//

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

//----------------------------------------------------------------------------//

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

//----------------------------------------------------------------------------//

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

//----------------------------------------------------------------------------//

template <typename _Tp>
void
memcpy_and_free(_Tp* _cpu, _Tp* _gpu, uintmax_t size)
{
    cudaMemcpy(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
    cudaFree(_gpu);
    CUDA_CHECK_LAST_ERROR();
}

//============================================================================//

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

//----------------------------------------------------------------------------//

template <typename _Tp>
void
async_memcpy_and_free(_Tp* _cpu, _Tp* _gpu, uintmax_t size, cudaStream_t stream)
{
    cudaMemcpyAsync(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK_LAST_ERROR();
    cudaFree(_gpu);
}

//============================================================================//

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

//============================================================================//

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

//============================================================================//

template <typename _Tp>
_Tp
reduce(_Tp* data, _Tp init, int nitems, cudaStream_t stream)
{
    _Tp* beg = data;
    _Tp* end = data + nitems;
    return thrust::reduce(thrust::system::cuda::par.on(stream), beg, end, init,
                          thrust::plus<_Tp>());
}

//============================================================================//

template <typename _Tp>
void
transform_sum(_Tp* input_data, int nitems, _Tp* result, cudaStream_t stream)
{
    _Tp* beg = input_data;
    _Tp* end = input_data + nitems;
    thrust::transform(thrust::system::cuda::par.on(stream), beg, end, result, result,
                      thrust::plus<_Tp>());
}

//============================================================================//

#else  // not defined(TOMOPY_USE_CUDA)

#    if !defined(cudaStream_t)
#        define cudaStream_t int
#    endif

//============================================================================//
template <typename _Tp>
_Tp*
gpu_malloc(uintmax_t size)
{
    return nullptr;
}
//----------------------------------------------------------------------------//
template <typename _Tp>
void
gpu_memcpy(_Tp*, const _Tp*, uintmax_t)
{
}
//----------------------------------------------------------------------------//
template <typename _Tp>
void
gpu_memcpy(_Tp*, const _Tp*, uintmax_t, cudaStream_t)
{
}
//----------------------------------------------------------------------------//
template <typename _Tp>
void
cpu_memcpy(const _Tp*, _Tp*, uintmax_t)
{
}
//----------------------------------------------------------------------------//
template <typename _Tp>
void
cpu_memcpy(const _Tp*, _Tp*, uintmax_t, cudaStream_t)
{
}
//----------------------------------------------------------------------------//
template <typename _Tp>
_Tp*
malloc_and_memcpy(const _Tp*, uintmax_t)
{
    return nullptr;
}
//----------------------------------------------------------------------------//
template <typename _Tp>
_Tp*
malloc_and_memset(uintmax_t size, int value)
{
}
//----------------------------------------------------------------------------//
template <typename _Tp>
_Tp*
malloc_and_async_memset(uintmax_t size, int value, cudaStream_t stream)
{
}
//----------------------------------------------------------------------------//
template <typename _Tp>
void
memcpy_and_free(_Tp*, _Tp*, uintmax_t)
{
}
//----------------------------------------------------------------------------//
template <typename _Tp>
_Tp*
malloc_and_async_memcpy(const _Tp*, uintmax_t, cudaStream_t)
{
    return nullptr;
}
//----------------------------------------------------------------------------//
template <typename _Tp>
void
async_memcpy_and_free(_Tp*, _Tp*, uintmax_t, cudaStream_t)
{
}
//----------------------------------------------------------------------------//
inline cudaStream_t*
create_streams(const int)
{
    return nullptr;
}
//----------------------------------------------------------------------------//
inline void
destroy_streams(cudaStream_t*, const int)
{
}
//============================================================================//
template <typename _Tp>
_Tp
reduce(_Tp*, _Tp, int, cudaStream_t)
{
}
//============================================================================//
template <typename _Tp>
void
transform_sum(_Tp*, int, _Tp*, cudaStream_t)
{
}
//============================================================================//

#endif  // if defined(TOMOPY_USE_CUDA)

//============================================================================//

template <typename _Tp>
void
print_gpu_array(const uintmax_t& n, const _Tp* gpu_data, const int& itr, const int& slice,
                const int& angle, const int& pixel, const std::string& tag)
{
    std::ofstream     ofs;
    std::stringstream fname;
    fname << "outputs/gpu/" << tag << "_" << itr << "_" << slice << "_" << angle << "_"
          << pixel << ".dat";
    ofs.open(fname.str().c_str());
    std::vector<_Tp> cpu_data(n, _Tp());
    std::cout << "printing to file " << fname.str() << "..." << std::endl;
    cpu_memcpy<_Tp>(gpu_data, cpu_data.data(), n);
    if(!ofs)
        return;
    for(uintmax_t i = 0; i < n; ++i)
        ofs << std::setw(6) << i << " \t " << std::setw(12) << std::setprecision(8)
            << cpu_data[i] << std::endl;
    ofs.close();
}

#endif
