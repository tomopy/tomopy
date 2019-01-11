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

#ifndef utils_cuda_h_
#define utils_cuda_h_

#include "common.h"
#include "gpu.h"

#if !defined(TOMOPY_USE_CUDA)
#    if !defined(__global__)
#        define __global__
#    endif
#    if !defined(__device__)
#        define __device__
#    endif
#endif

//============================================================================//
//  reduce
//============================================================================//

__global__ void
deviceReduceKernel(const float* in, float* out, int N);

//----------------------------------------------------------------------------//
#if defined(TOMOPY_USE_CUDA)
/*
__device__ float
reduce_sum(cg::thread_group g, float* temp, float val);

//----------------------------------------------------------------------------//

__device__ float
thread_sum(const float* input, int n);
*/
#endif  // TOMOPY_USE_CUDA
//----------------------------------------------------------------------------//

__global__ void
sum_kernel_block(float* sum, const float* input, int n);

//----------------------------------------------------------------------------//

DLL float
deviceReduce(const float* in, float* out, int N, cudaStream_t stream);

//============================================================================//
//  reduce
//============================================================================//

DLL float
reduce(float* _in, float* _out, int size, cudaStream_t stream);

//============================================================================//
//  rotate
//============================================================================//

DLL float*
cuda_rotate(const float* src, const float theta, const int nx, const int ny,
            cudaStream_t stream);

//----------------------------------------------------------------------------//

DLL void
cuda_rotate_ip(float* dst, const float* src, const float theta, const int nx,
               const int ny, cudaStream_t stream);

//----------------------------------------------------------------------------//

#endif