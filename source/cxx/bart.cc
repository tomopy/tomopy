// Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.

// Copyright 2015. UChicago Argonne, LLC. This software was produced
// under U.S. Government contract DE-AC02-06CH11357 for Argonne National
// Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
// U.S. Department of Energy. The U.S. Government has rights to use,
// reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
// UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
// ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
// modified to produce derivative works, such modified software should
// be clearly marked, so as not to confuse it with the version available
// from ANL.

// Additionally, redistribution and use in source and binary forms, with
// or without modification, are permitted provided that the following
// conditions are met:

//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.

//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in
//       the documentation and/or other materials provided with the
//       distribution.

//     * Neither the name of UChicago Argonne, LLC, Argonne National
//       Laboratory, ANL, the U.S. Government, nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
// Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "common.hh"
#include "utils.hh"

BEGIN_EXTERN_C
#include "bart.h"
#include "utils.h"
END_EXTERN_C

#include <cstdlib>
#include <memory>

//======================================================================================//

void
cxx_bart(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
         int num_block, const float* ind_block)
{
    auto tid = GetThisThreadID();
    ConsumeParameters(tid);

#if defined(TOMOPY_USE_TIMEMORY)
    tim::timer t(__FUNCTION__);
    t.format().get()->width(10);
    t.start();
#endif

    TIMEMORY_AUTO_TIMER("");

#if defined(TOMOPY_USE_GPU)
    // TODO: select based on memory
    bool use_cpu = GetEnv<bool>("TOMOPY_USE_CPU", false);
    if(use_cpu)
        bart_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter,
                 num_block, ind_block);
    else
        run_gpu_algorithm(bart_cpu, bart_cuda, bart_openacc, bart_openmp, data, dy, dt,
                          dx, center, theta, recon, ngridx, ngridy, num_iter, num_block,
                          ind_block);
#else
    bart_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter, num_block,
             ind_block);
#endif

#if defined(TOMOPY_USE_TIMEMORY)
    AutoLock l(TypeMutex<decltype(std::cout)>());
    std::cout << t.stop_and_return() << std::endl;
#endif
}

//======================================================================================//

void
bart_cpu(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
         int num_block, const float* ind_block)
{
    ConsumeParameters(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter,
                      num_block, ind_block);

    TIMEMORY_AUTO_TIMER("[cpu]");

    throw std::runtime_error("BART algorithm has not been implemented for CXX");
}

//======================================================================================//

void
bart_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
          int num_block, const float* ind_block)
{
    ConsumeParameters(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter,
                      num_block, ind_block);

    throw std::runtime_error("BART algorithm has not been implemented for CUDA");

    TIMEMORY_AUTO_TIMER("[cuda]");

    // insert code here
}

//======================================================================================//

void
bart_openacc(const float* data, int dy, int dt, int dx, const float* center,
             const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
             int num_block, const float* ind_block)
{
    ConsumeParameters(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter,
                      num_block, ind_block);

    throw std::runtime_error("BART algorithm has not been implemented for OpenACC");

    TIMEMORY_AUTO_TIMER("[openacc]");
}

//======================================================================================//

void
bart_openmp(const float* data, int dy, int dt, int dx, const float* center,
            const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
            int num_block, const float* ind_block)
{
    ConsumeParameters(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter,
                      num_block, ind_block);

    throw std::runtime_error("BART algorithm has not been implemented for OpenMP");

    TIMEMORY_AUTO_TIMER("[openmp]");
}

//======================================================================================//
