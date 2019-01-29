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
    // if C implementation is requested, return non-zero (failure)
    if(use_c_algorithm)
        return static_cast<int>(false);

    printf("\n\t%s [oy = %i, ox = %i, oz = %i, dy = %i, dt = %i, dx = %i]\n\n",
           __FUNCTION__, oy, oz, oz, dy, dt, dx);

#if defined(TOMOPY_USE_PTL)
    auto tid = GetThisThreadID();
    ConsumeParameters(tid);
#endif

    START_TIMER(cxx_start);
    TIMEMORY_AUTO_TIMER("");

#if defined(TOMOPY_USE_GPU)
    bool use_cpu = GetEnv<bool>("TOMOPY_USE_CPU", false);
    if(use_cpu)
        project_cpu(obj, oy, ox, oz, data, dy, dt, dx, center, theta);
    else
        run_gpu_algorithm(project_cpu, project_cuda, project_openacc, project_openmp, obj,
                          oy, ox, oz, data, dy, dt, dx, center, theta);
#else
    project_cpu(obj, oy, ox, oz, data, dy, dt, dx, center, theta);
#endif

    REPORT_TIMER(cxx_start, __FUNCTION__, 0, 1);

    return static_cast<int>(true);
}

//======================================================================================//

void
project_cpu(const float* obj, int oy, int ox, int oz, float* data, int dy, int dt, int dx,
            const float* center, const float* theta)
{
    printf("\n\t%s [oy = %i, ox = %i, oz = %i, dy = %i, dt = %i, dx = %i]\n\n",
           __FUNCTION__, oy, oz, oz, dy, dt, dx);

    TIMEMORY_AUTO_TIMER("");

    // uintmax_t rot_size = static_cast<uintmax_t>(ox * oz);
    int offset = (dx - oz) / 2 - 1;
    for(int s = 0; s < dy; s++)
    {
        // For each projection angle
        for(int p = 0; p < dt; p++)
        {
            // needed for recon to output at proper orientation
            float    theta_rad_p = fmodf(theta[p] + halfpi, twopi);
            float    theta_deg_p = theta_rad_p * degrees;
            farray_t obj_rot(ox * oz, 0.0f);

            // Forward-Rotate object
            cxx_affine_transform(obj_rot, obj, -theta_rad_p, -theta_deg_p, oz, ox);

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
}

//======================================================================================//

void
project_openacc(const float* obj, int oy, int ox, int oz, float* data, int dy, int dt,
                int dx, const float* center, const float* theta)
{
}

//======================================================================================//

void
project_openmp(const float* obj, int oy, int ox, int oz, float* data, int dy, int dt,
               int dx, const float* center, const float* theta)
{
}

//======================================================================================//
