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
//   TOMOPY header file
//
//  Description:
//
//      Here we declare the GPU interface
//

#ifndef gpu_h_
#define gpu_h_

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

#ifndef DLL
#    ifdef WIN32
#        define DLL __declspec(dllexport)
#    else
#        define DLL
#    endif
#endif

#ifdef __cplusplus
#    include <cstdio>
#    include <cstring>
#else
#    include <stdio.h>
#    include <string.h>
#endif

#include "common.h"

//----------------------------------------------------------------------------//

#if defined(TOMOPY_USE_CUDA)
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#    include <vector_types.h>

#    if !defined(CUDA_CHECK_LAST_ERROR)
#        if defined(DEBUG)
#            define CUDA_CHECK_LAST_ERROR()                                              \
                {                                                                        \
                    cudaDeviceSynchronize();                                             \
                    cudaError err = cudaGetLastError();                                  \
                    if(cudaSuccess != err)                                               \
                    {                                                                    \
                        fprintf(stderr, "cudaCheckError() failed at %s@'%s':%i : %s\n",  \
                                __FUNCTION__, __FILE__, __LINE__,                        \
                                cudaGetErrorString(err));                                \
                        printf("cudaCheckError() failed at %s@'%s':%i : %s\n",           \
                               __FUNCTION__, __FILE__, __LINE__,                         \
                               cudaGetErrorString(err));                                 \
                        exit(1);                                                         \
                    }                                                                    \
                }
#        else
#            define CUDA_CHECK_LAST_ERROR()                                              \
                {                                                                        \
                    ;                                                                    \
                }
#        endif
#    endif
#endif

//============================================================================//
//
//      NVTX macros
//
//============================================================================//

#if defined(TOMOPY_USE_NVTX)
#    include <nvToolsExt.h>
#    ifndef NVTX_RANGE_PUSH
#        define NVTX_RANGE_PUSH(obj) nvtxRangePushEx(obj)
#    endif
#    ifndef NVTX_RANGE_POP
#        define NVTX_RANGE_POP(obj) nvtxRangePop()
#    endif
#    ifndef NVTX_NAME_THREAD
#        define NVTX_NAME_THREAD(num, name) nvtxNameOsThread(num, name)
#    endif
#    ifndef NVTX_MARK
#        define NVTX_MARK(msg) nvtxMark(name)
#    endif

void
init_nvtx();

#else
#    ifndef NVTX_RANGE_PUSH
#        define NVTX_RANGE_PUSH(obj)
#    endif
#    ifndef NVTX_RANGE_POP
#        define NVTX_RANGE_POP(obj)
#    endif
#    ifndef NVTX_NAME_THREAD
#        define NVTX_NAME_THREAD(num, name)
#    endif
#    ifndef NVTX_MARK
#        define NVTX_MARK(msg)
#    endif

inline void
init_nvtx()
{
}

#endif

//============================================================================//

BEGIN_EXTERN_C  // begin extern "C"

    //----------------------------------------------------------------------------//
    //  device-specific info
    //----------------------------------------------------------------------------//

    void DLL
         cuda_device_query();
int      DLL
         cuda_device_count();
// this functions sets "thread_device()" value to device number
int DLL
    cuda_set_device(int device);
// the functions below use "thread_device()" function to get device number
int DLL
    cuda_multi_processor_count();
int DLL
    cuda_max_threads_per_block();
int DLL
    cuda_warp_size();
int DLL
    cuda_shared_memory_per_block();

//----------------------------------------------------------------------------//
//  dataset handling for GPU
//----------------------------------------------------------------------------//

void DLL
     init_tomo_dataset();
void DLL
     free_tomo_dataset(bool is_master);

//============================================================================//

END_EXTERN_C  // end extern "C"

//----------------------------------------------------------------------------//

#endif
