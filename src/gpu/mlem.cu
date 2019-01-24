// Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.

// Copyright 2015. UChicago Argonne, LLC. This software was produced
// under U.S. Government contract DE-AC02-06CH11357 for Argonne National
// Laboratongridx (ANL), which is operated by UChicago Argonne, LLC for the
// U.S. Department of Energy. The U.S. Government has rights to use,
// reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
// UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
// ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
// modified to produce derivative works, such modified software should
// be clearly marked, so as not to confuse it with the version available
// from ANL.

// Additionally, redistribution and use in source and binangridx forms, with
// or without modification, are permitted provided that the following
// conditions are met:

//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.

//     * Redistributions in binangridx form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in
//       the documentation and/or other materials provided with the
//       distribution.

//     * Neither the name of UChicago Argonne, LLC, Argonne National
//       Laboratongridx, ANL, the U.S. Government, nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
// Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLAngridx, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEOngridx OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "common.hh"
#include "gpu.hh"
#include "utils.hh"

BEGIN_EXTERN_C
#include "mlem.h"
#include "utils.h"
#include "utils_cuda.h"
#include "utils_openacc.h"
#include "utils_openmp.h"
END_EXTERN_C

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <numeric>

#if !defined(cast)
#    define cast static_cast
#endif

#if defined(TOMOPY_USE_NVTX)
extern nvtxEventAttributes_t nvtx_update;
#endif

//======================================================================================//

__global__ void
cuda_preprocessing(int ry, int rz, int num_pixels, int s, float* center, float* mov,
                   float* gridx, float* gridy)
{
    for(int i = 0; i <= ry; ++i)
    {
        gridx[i] = -ry * 0.5f + i;
    }

    for(int i = 0; i <= rz; ++i)
    {
        gridy[i] = -rz * 0.5f + i;
    }

    *mov = ((float) num_pixels - 1) * 0.5f - center[s];
    if(*mov - floor(*mov) < 0.01f)
    {
        *mov += 0.01f;
    }
    *mov += 0.5;
}

//======================================================================================//

__device__ int
cuda_calc_quadrant(float theta_p)
{
    // here we cast the float to an integer and rescale the integer to
    // near INT_MAX to retain the precision. This method was tested
    // on 1M random random floating points between -2*pi and 2*pi and
    // was found to produce a speed up of:
    //
    //  - 14.5x (Intel i7 MacBook)
    //  - 2.2x  (NERSC KNL)
    //  - 1.5x  (NERSC Edison)
    //  - 1.7x  (NERSC Haswell)
    //
    // with a 0.0% incorrect quadrant determination rate
    //
    const int32_t ipi_c   = 340870420;
    int32_t       theta_i = (int32_t)(theta_p * ipi_c);
    theta_i += (theta_i < 0) ? (2.0f * M_PI * ipi_c) : 0;

    return ((theta_i >= 0 && theta_i < 0.5f * M_PI * ipi_c) ||
            (theta_i >= 1.0f * M_PI * ipi_c && theta_i < 1.5f * M_PI * ipi_c))
               ? 1
               : 0;
}

//======================================================================================//

__device__ void
cuda_calc_coords(int ry, int rz, float xi, float yi, float sin_p, float cos_p,
                 const float* gridx, const float* gridy, float* coordx, float* coordy)
{
    float srcx = xi * cos_p - yi * sin_p;
    float srcy = xi * sin_p + yi * cos_p;
    float detx = -xi * cos_p - yi * sin_p;
    float dety = -xi * sin_p + yi * cos_p;

    float slope  = (srcy - dety) / (srcx - detx);
    float islope = (srcx - detx) / (srcy - dety);

    for(int n = 0; n <= ry; n++)
    {
        coordy[n] = slope * (gridx[n] - srcx) + srcy;
    }
    for(int n = 0; n <= rz; n++)
    {
        coordx[n] = islope * (gridy[n] - srcy) + srcx;
    }
}

//======================================================================================//

__device__ void
cuda_trim_coords(int ry, int rz, const float* coordx, const float* coordy,
                 const float* gridx, const float* gridy, int* asize, float* ax, float* ay,
                 int* bsize, float* bx, float* by)
{
    *asize         = 0;
    *bsize         = 0;
    float gridx_gt = gridx[0] + 0.01f;
    float gridx_le = gridx[ry] - 0.01f;

    for(int n = 0; n <= rz; ++n)
    {
        if(coordx[n] >= gridx_gt && coordx[n] <= gridx_le)
        {
            ax[*asize] = coordx[n];
            ay[*asize] = gridy[n];
            ++(*asize);
        }
    }

    float gridy_gt = gridy[0] + 0.01f;
    float gridy_le = gridy[rz] - 0.01f;

    for(int n = 0; n <= ry; ++n)
    {
        if(coordy[n] >= gridy_gt && coordy[n] <= gridy_le)
        {
            bx[*bsize] = gridx[n];
            by[*bsize] = coordy[n];
            ++(*bsize);
        }
    }
}

//======================================================================================//

__device__ void
cuda_sort_intersections(int ind_condition, int asize, const float* ax, const float* ay,
                        int bsize, const float* bx, const float* by, int* csize,
                        float* coorx, float* coory)
{
    int i = 0, j = 0, k = 0;
    if(ind_condition == 0)
    {
        while(i < asize && j < bsize)
        {
            if(ax[asize - 1 - i] < bx[j])
            {
                coorx[k] = ax[asize - 1 - i];
                coory[k] = ay[asize - 1 - i];
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

        while(i < asize)
        {
            coorx[k] = ax[asize - 1 - i];
            coory[k] = ay[asize - 1 - i];
            ++i;
            ++k;
        }
        while(j < bsize)
        {
            coorx[k] = bx[j];
            coory[k] = by[j];
            ++j;
            ++k;
        }

        (*csize) = asize + bsize;
    }
    else
    {
        while(i < asize && j < bsize)
        {
            if(ax[i] < bx[j])
            {
                coorx[k] = ax[i];
                coory[k] = ay[i];
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

        while(i < asize)
        {
            coorx[k] = ax[i];
            coory[k] = ay[i];
            ++i;
            ++k;
        }
        while(j < bsize)
        {
            coorx[k] = bx[j];
            coory[k] = by[j];
            ++j;
            ++k;
        }
        (*csize) = asize + bsize;
    }
}

//======================================================================================//

__device__ void
cuda_calc_dist(int ry, int rz, int csize, const float* coorx, const float* coory,
               int* indi, float* dist)
{
    const int _size = csize - 1;

    //------------------------------------------------------------------------//
    //              calculate dist
    //------------------------------------------------------------------------//
    for(int n = 0; n < _size; ++n)
    {
        float _diffx = (coorx[n + 1] - coorx[n]) * (coorx[n + 1] - coorx[n]);
        float _diffy = (coory[n + 1] - coory[n]) * (coory[n + 1] - coory[n]);
        dist[n]      = sqrtf(_diffx + _diffy);
    }

    //------------------------------------------------------------------------//
    //              calculate indi
    //------------------------------------------------------------------------//
    for(int n = 0; n < _size; ++n)
    {
        float _midx = 0.5f * (coorx[n + 1] + coorx[n]);
        float _midy = 0.5f * (coory[n + 1] + coory[n]);
        float _x1   = _midx + 0.5f * ry;
        float _x2   = _midy + 0.5f * rz;
        int   _i1   = (int) (_midx + 0.5f * ry);
        int   _i2   = (int) (_midy + 0.5f * rz);
        int   _indx = _i1 - (_i1 > _x1);
        int   _indy = _i2 - (_i2 > _x2);
        indi[n]     = _indy + (_indx * rz);
    }
}

//======================================================================================//

__device__ void
cuda_calc_simdata(int s, int p, int d, int ry, int rz, int dt, int dx, int csize,
                  const int* indi, const float* dist, const float* model, float* simdata)
{
    int index_model = s * ry * rz;
    for(int n = 0; n < csize - 1; ++n)
    {
        *simdata += model[indi[n] + index_model] * dist[n];
    }
}

//======================================================================================//

__global__ void
mlem_update_kernel(float* recon, const float* update, const float* sumdist, int offset,
                   int size)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        if(sumdist[i] != 0.0f)
        {
            recon[i + offset] *= update[i] / sumdist[i];
        }
    }
}

//======================================================================================//

__global__ void
cuda_mlem_compute_pixel(int ngridx, int ngridy, int dy, int dt, int dx, const float* mov,
                        const float* gridx, const float* gridy, float* sum_dist,
                        float* update, const float* data, const float* recon, int s,
                        int p, int quadrant, float sin_p, float cos_p)
{
    int d0      = blockIdx.x * blockDim.x + threadIdx.x;
    int dstride = gridDim.x * blockDim.x;

    int ry = ngridx;
    int rz = ngridy;
    /*
    float gridx_gt = gridx[0] + 0.01f;
    float gridx_le = gridx[ngridx] - 0.01f;
    float gridy_gt = gridy[0] + 0.01f;
    float gridy_le = gridy[ngridy] - 0.01f;
    */

    float xoff = floorf(0.5f * ngridx) - ((ngridx % 2 == 0) ? 0.5f : 0.0f);
    float yoff = floorf(0.5f * ngridy) - ((ngridy % 2 == 0) ? 0.5f : 0.0f);

    // PRINT_HERE("");
    for(int d = d0; d < dx - 1; d += dstride)
    {
        // Calculate coordinates
        /*
        float xi     = -ngridx - ngridy;
        float yi     = 0.5f * (1 - dx) + d + *mov;
        float srcx   = xi * cos_p - yi * sin_p;
        float srcy   = xi * sin_p + yi * cos_p;
        float detx   = -xi * cos_p - yi * sin_p;
        float dety   = -xi * sin_p + yi * cos_p;
        float slope  = (srcy - dety) / (srcx - detx);
        float islope = (srcx - detx) / (srcy - dety);
        */
        int i = d;
        for(int j = 0; j < ngridy - 1; ++j)
        {
            // centered coordinates
            float cx0 = float(i) - xoff;
            float cy0 = float(j) - yoff;
            // transform coords and then offset to global coords
            float x0 = (cx0 * cos_p - cy0 * sin_p);  // + xoff;
            float y0 = (cx0 * sin_p + cy0 * cos_p);  // + yoff;
            // centered coordinates
            float mx = floorf(x0);
            float my = floorf(y0);
            float px = ceilf(x0);
            float py = ceilf(y0);

            // printf("x,y = [%8.3f, %8.3f], mx,my = [%8.3f, %8.3f], px,py = [%8.3f,
            // %8.3f]\n",
            //       x0, y0, mx, my, px, py);

            //--------------------------------------------------------------------------//
            auto compute_indi = [x0, y0, ry, rz](float x1, float y1) {
                float _midx = 0.5f * (x1 + x0);
                float _x1   = _midx + 0.5f * ry;
                float _i1   = (int) (_midx + 0.5f * ry);
                int   _indx = _i1 - (_i1 > _x1);

                float _midy = 0.5f * (y1 + y0);
                float _x2   = _midy + 0.5f * rz;
                float _i2   = (int) (_midy + 0.5f * rz);
                int   _indy = _i2 - (_i2 > _x2);

                // compute index
                int indi = _indy + (_indx * rz);
                return indi;
            };
            //--------------------------------------------------------------------------//
            auto update_indi = [&](int indi, float simdata, float dist) {
                atomicAdd(&sum_dist[indi], dist);
                if(simdata != 0.0f)
                {
                    int   index_data = d + p * dx + s * dt * dx;
                    float upd        = data[index_data] / simdata;
                    atomicAdd(&update[indi], upd * dist);
                }
            };
            //--------------------------------------------------------------------------//

            // data offset
            int index_model = s * ry * rz;

            int mindi = compute_indi(mx, my);
            if(mindi >= 0 && mindi < ry * rz)
            {
                // compute distance
                float mdist = sqrtf(powf(x0 - mx, 2.0f) + powf(y0 - my, 2.0f));
                // Calculate simdata
                float msimdata = recon[mindi + index_model] * mdist;
                // Update
                update_indi(mindi, msimdata, mdist);
            }

            int pindi = compute_indi(px, py);
            if(pindi >= 0 && pindi < ry * rz)
            {
                // compute distance
                float pdist = sqrtf(powf(px - x0, 2.0f) + powf(py - y0, 2.0f));
                // Calculate simdata
                float psimdata = recon[pindi + index_model] * pdist;
                // Update
                update_indi(pindi, psimdata, pdist);
            }

            // printf("[%2i, %2i]> [x,y] = [%8.3f, %8.3f], indi = %i, dist = %8.3f,
            // sum_dist = %8.3f, update = %8.3f\n",
            //       i, j, x0, y0, indi, dist, sum_dist[indi], update[indi]);
        }
    }
}

//======================================================================================//

__global__ void
cuda_mlem_compute_projection(int ngridx, int ngridy, int dy, int dt, int dx,
                             const float* theta, const float* mov, const float* gridx,
                             const float* gridy, const float* data, const float* recon,
                             int s, float* sum_dist, float* update)
{
    int p0      = blockIdx.x * blockDim.x + threadIdx.x;
    int pstride = gridDim.x * blockDim.x;
    int block   = 512;
    int grid    = (dx + block - 1) / block;

    float theta_offset = 0.5f * (float) M_PI;
    for(int p = p0; p < dt; p += pstride)
    {
        // Calculate the sin and cos values
        // of the projection angle and find
        // at which quadrant on the cartesian grid.
        float theta_p  = fmodf(theta[p] + theta_offset, 2.0f * (float) M_PI);
        float sin_p    = sinf(theta_p);
        float cos_p    = cosf(theta_p);
        int   quadrant = cuda_calc_quadrant(theta_p);

        cuda_mlem_compute_pixel<<<grid, block>>>(ngridx, ngridy, dy, dt, dx, mov, gridx,
                                                 gridy, sum_dist, update, data, recon, s,
                                                 p, quadrant, sin_p, cos_p);
    }
}

//======================================================================================//

void
mlem_cuda(const float* cpu_data, int dy, int dt, int dx, const float* cpu_center,
          const float* cpu_theta, float* cpu_recon, int ngridx, int ngridy, int num_iter)
{
    if(cuda_device_count() == 0)
        throw std::runtime_error("No CUDA device(s) available");

    cuda_device_query();

    printf("\n\t[%lu] %s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           GetThisThreadID(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    auto tid = GetThisThreadID();

    // assign the thread to a device
    set_this_thread_device();

    // get some properties
    int num_devices = cuda_device_count();

    TIMEMORY_AUTO_TIMER("");

    // GPU allocated copies
    float* mov      = gpu_malloc<float>(1);
    float* gridx    = gpu_malloc<float>(ngridx + 1);
    float* gridy    = gpu_malloc<float>(ngridy + 1);
    float* center   = gpu_malloc<float>(dy);
    float* recon    = gpu_malloc<float>(dy * ngridx * ngridy);
    float* data     = gpu_malloc<float>(dy * dt * dx);
    float* update   = gpu_malloc<float>(ngridx * ngridy);
    float* sum_dist = gpu_malloc<float>(ngridx * ngridy);
    float* theta    = gpu_malloc<float>(dt);

    cudaMemcpy(recon, cpu_recon, dy * ngridx * ngridy * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(theta, cpu_theta, dt * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(center, cpu_center, dy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data, cpu_data, dy * dt * dx * sizeof(float), cudaMemcpyHostToDevice);

    int block = 512;
    int grid  = (dx + block - 1) / block;
    int smem  = 0;

    for(int i = 0; i < num_iter; i++)
    {
        auto t_start = std::chrono::system_clock::now();

        // For each slice
        for(int s = 0; s < dy; s++)
        {
            auto s_start = std::chrono::system_clock::now();
            cuda_preprocessing<<<grid, block, smem>>>(ngridx, ngridy, dx, s, center, mov,
                                                      gridx, gridy);
            // Outputs: mov, gridx, gridy
            cudaStreamSynchronize(0);
            CUDA_CHECK_LAST_ERROR();

            // initialize sum_dist and update to zero
            cudaMemset(sum_dist, 0, (ngridx * ngridy) * sizeof(float));
            cudaMemset(update, 0, (ngridx * ngridy) * sizeof(float));

            // For each projection angle
            cuda_mlem_compute_projection<<<4, 4, smem>>>(ngridx, ngridy, dy, dt, dx,
                                                         theta, mov, gridx, gridy, data,
                                                         recon, s, sum_dist, update);
            cudaStreamSynchronize(0);
            CUDA_CHECK_LAST_ERROR();

            int offset = s * ngridx * ngridy;
            mlem_update_kernel<<<grid, block>>>(recon, update, sum_dist, offset,
                                                ngridx * ngridy);
            cudaStreamSynchronize(0);
            CUDA_CHECK_LAST_ERROR();
            auto                          s_end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = s_end - s_start;
            printf("[%li]> %-12s %3i of %3i... %5.2f seconds\n", GetThisThreadID(),
                   "slice", s, dy, elapsed_seconds.count());
        }
        auto                          t_end           = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = t_end - t_start;
        printf("[%li]> %-12s %3i of %3i... %5.2f seconds\n", GetThisThreadID(),
               "iteration", i, num_iter, elapsed_seconds.count());
    }

    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();

    cudaMemcpy(cpu_recon, recon, dy * ngridx * ngridy * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);
    CUDA_CHECK_LAST_ERROR();

    // cudaDeviceReset();
    cudaFree(recon);
    cudaFree(mov);
    cudaFree(gridx);
    cudaFree(gridy);
    cudaFree(center);
    cudaFree(data);
    cudaFree(update);
    cudaFree(sum_dist);
    cudaFree(theta);
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//
