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
#include "project.h"
#include "utils.h"
END_EXTERN_C

//======================================================================================//

int
cxx_project(const float* obj, int oy, int ox, int oz, float* data, int dy, int dt, int dx,
            const float* center, const float* theta)
{
    // check to see if the C implementation is requested
    bool use_c_algorithm = GetEnv<bool>("TOMOPY_USE_C_PROJECT", true);
    use_c_algorithm      = GetEnv<bool>("TOMOPY_USE_C_ALGORITHMS", use_c_algorithm);
    // if C implementation is requested, return non-zero (failure)
    if(use_c_algorithm)
        return scast<int>(false);

    auto tid = GetThisThreadID();
    ConsumeParameters(tid);
    static std::atomic<int> active;
    int                     count = active++;

    START_TIMER(cxx_timer);

    printf("[%lu]> %s : oy = %i, ox = %i, oz = %i, dy = %i, dt = %i, dx = %i\n",
           GetThisThreadID(), __FUNCTION__, oy, oz, oz, dy, dt, dx);

    {
        TIMEMORY_AUTO_TIMER("");
        run_algorithm(project_cpu, project_cuda, project_openacc, project_openmp, obj, oy,
                      ox, oz, data, dy, dt, dx, center, theta);
    }

    auto tcount = GetEnv("TOMOPY_PYTHON_THREADS", HW_CONCURRENCY);
    auto remain = --active;
    REPORT_TIMER(cxx_timer, __FUNCTION__, count, tcount);
    if(remain == 0)
    {
        std::stringstream ss;
        PrintEnv(ss);
        printf("[%lu] Reporting environment...\n\n%s\n", GetThisThreadID(),
               ss.str().c_str());
    }
    else
    {
        printf("[%lu] Threads remaining: %i...\n", GetThisThreadID(), remain);
    }

    return scast<int>(true);
}

//======================================================================================//

void
project_cpu(const float* obj, int oy, int ox, int oz, float* data, int dy, int dt, int dx,
            const float* center, const float* theta)
{
    printf("[%lu]> %s : oy = %i, ox = %i, oz = %i, dy = %i, dt = %i, dx = %i\n",
           GetThisThreadID(), __FUNCTION__, oy, oz, oz, dy, dt, dx);

    TIMEMORY_AUTO_TIMER("");

    // uintmax_t rot_size = static_cast<uintmax_t>(ox * oz);
    int offset = (dx - oz) / 2 - 1;
    for(int s = 0; s < dy; s++)
    {
        // For each projection angle
        for(int p = 0; p < dt; p++)
        {
            // needed for recon to output at proper orientation
            float    theta_p = fmodf(theta[p] + halfpi, twopi);
            farray_t obj_rot(ox * oz, 0.0f);

            // Forward-Rotate object
            cxx_rotate_ip<float>(obj_rot, obj, -theta_p, oz, ox);

            for(int d = 0; d < ox; ++d)
            {
                int idx_data = d + p * dx + s * dt * dx;
                // Calculate simulated data by summing up along x-axis
                for(int n = 0; n < oz; ++n)
                    data[idx_data + offset] += obj_rot[d * ox + n];
            }
        }
    }

    printf("\n");
}

//======================================================================================//

void
project_cuda(const float* obj, int oy, int ox, int oz, float* data, int dy, int dt,
             int dx, const float* center, const float* theta)
{
    printf("[%lu]> %s : oy = %i, ox = %i, oz = %i, dy = %i, dt = %i, dx = %i\n",
           GetThisThreadID(), __FUNCTION__, oy, oz, oz, dy, dt, dx);
    project_cpu(obj, oy, ox, oz, data, dy, dt, dx, center, theta);
}

//======================================================================================//

void
project_openacc(const float* obj, int oy, int ox, int oz, float* data, int dy, int dt,
                int dx, const float* center, const float* theta)
{
    printf("[%lu]> %s : oy = %i, ox = %i, oz = %i, dy = %i, dt = %i, dx = %i\n",
           GetThisThreadID(), __FUNCTION__, oy, oz, oz, dy, dt, dx);
    project_cpu(obj, oy, ox, oz, data, dy, dt, dx, center, theta);
}

//======================================================================================//

void
project_openmp(const float* obj, int oy, int ox, int oz, float* data, int dy, int dt,
               int dx, const float* center, const float* theta)
{
    printf("[%lu]> %s : oy = %i, ox = %i, oz = %i, dy = %i, dt = %i, dx = %i\n",
           GetThisThreadID(), __FUNCTION__, oy, oz, oz, dy, dt, dx);
    project_cpu(obj, oy, ox, oz, data, dy, dt, dx, center, theta);
}

//======================================================================================//
