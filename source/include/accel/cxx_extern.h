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

/** \file cxx_extern.h
 * \headerfile cxx_extern.h "include/cxx_extern.h"
 * C++ functions that are available to C code (available for Python binding)
 */

#pragma once

#include "macros.h"

//======================================================================================//

BEGIN_EXTERN_C

//======================================================================================//
//
//  CUDA
//      - definitions in gpu/common.cu when CUDA enabled
//      - definitions in cxx/common.cc when CUDA not enabled
//
//======================================================================================//
// print info about devices available (only does this once per process)
DLL void
cuda_device_query();

// get the number of devices available
DLL int
cuda_device_count();

// sets the thread to a specific device
DLL int
cuda_set_device(int device);

// get the number of CUDA multiprocessors
DLL int
cuda_multi_processor_count();

// get the maximum number of threads per block
DLL int
cuda_max_threads_per_block();

// get the size of the warps
DLL int
cuda_warp_size();

// get the maximum amount of shared memory per block
DLL int
cuda_shared_memory_per_block();

//======================================================================================//
//
//  MLEM
//
//======================================================================================//

// generic decision of whether to use CPU or GPU version
//     NOTE: if compiled with GPU support but no devices, will call CPU version
DLL int
cxx_mlem(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
         int pool_size, const char* interp, const char* device, int* grid_size,
         int* block_size);

//======================================================================================//
//
//  SIRT
//
//======================================================================================//

// generic decision of whether to use CPU or GPU version
//     NOTE: if compiled with GPU support but no devices, will call CPU version
DLL int
cxx_sirt(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
         int pool_size, const char* interp, const char* device, int* grid_size,
         int* block_size);

//======================================================================================//

END_EXTERN_C

//======================================================================================//
