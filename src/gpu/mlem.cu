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

#define HW_CONCURRENCY std::thread::hardware_concurrency()

//============================================================================//

__device__ void
cuda_preprocessing(int ry, int rz, int num_pixels, float center, float* mov, float* gridx,
                   float* gridy)
{
    for(int i = 0; i <= ry; ++i)
    {
        gridx[i] = -ry * 0.5f + i;
    }

    for(int i = 0; i <= rz; ++i)
    {
        gridy[i] = -rz * 0.5f + i;
    }

    *mov = ((float) num_pixels - 1) * 0.5f - center;
    if(*mov - floor(*mov) < 0.01f)
    {
        *mov += 0.01f;
    }
    *mov += 0.5;
}

//============================================================================//

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

//============================================================================//

__device__ void
cuda_calc_coords(int ry, int rz, float xi, float yi, float sin_p, float cos_p,
                 const float* gridx, const float* gridy, float* coordx, float* coordy)
{
    float srcx, srcy, detx, dety;
    float slope, islope;
    int   n;

    srcx = xi * cos_p - yi * sin_p;
    srcy = xi * sin_p + yi * cos_p;
    detx = -xi * cos_p - yi * sin_p;
    dety = -xi * sin_p + yi * cos_p;

    slope  = (srcy - dety) / (srcx - detx);
    islope = (srcx - detx) / (srcy - dety);

    for(n = 0; n <= ry; n++)
    {
        coordy[n] = slope * (gridx[n] - srcx) + srcy;
    }
    for(n = 0; n <= rz; n++)
    {
        coordx[n] = islope * (gridy[n] - srcy) + srcx;
    }
}

//============================================================================//

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

//============================================================================//

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

//============================================================================//

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

//============================================================================//

__device__ void
cuda_calc_simdata(int s, int p, int d, int ry, int rz, int dt, int dx, int csize,
                  const int* indi, const float* dist, const float* model, float* simdata)
{
    // int index_model = s * ry * rz;
    int index_data = d + p * dx;
    for(int n = 0; n < csize - 1; ++n)
    {
        simdata[index_data] += model[indi[n]] * dist[n];
    }
}

//============================================================================//

__global__ void
mlem_update_kernel(float* recon, const float* update, const float* sumdist, int size)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        if(sumdist[i] != 0.0f)
            recon[i] *= update[i] / sumdist[i];
    }
}

//============================================================================//

struct cuda_data
{
    typedef cuda_data this_type;

    int          m_device;
    int          m_id;
    int          m_dy;
    int          m_dt;
    int          m_dx;
    int          m_nx;
    int          m_ny;
    float        m_mov;
    float*       m_gridx;
    float*       m_gridy;
    float*       m_coordx;
    float*       m_coordy;
    float*       m_ax;
    float*       m_ay;
    float*       m_bx;
    float*       m_by;
    float*       m_coorx;
    float*       m_coory;
    float*       m_dist;
    int*         m_indi;
    float*       m_simdata;
    float*       m_sumdist;
    float*       m_update;
    const float* m_recon;
    const float* m_data;

    cuda_data(int device, int id, int dy, int dt, int dx, int nx, int ny, float* simdata)
    : m_device(device)
    , m_id(id)
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_gridx(gpu_malloc<float>(nx + 1))
    , m_gridy(gpu_malloc<float>(ny + 1))
    , m_coordx(gpu_malloc<float>(ny + 1))
    , m_coordy(gpu_malloc<float>(nx + 1))
    , m_ax(gpu_malloc<float>(nx + ny))
    , m_ay(gpu_malloc<float>(nx + ny))
    , m_bx(gpu_malloc<float>(nx + ny))
    , m_by(gpu_malloc<float>(nx + ny))
    , m_coorx(gpu_malloc<float>(nx + ny))
    , m_coory(gpu_malloc<float>(nx + ny))
    , m_dist(gpu_malloc<float>(nx + ny))
    , m_indi(gpu_malloc<int>(nx + ny))
    , m_simdata(simdata)
    , m_sumdist(gpu_malloc<float>(nx * ny))
    , m_update(gpu_malloc<float>(nx * ny))
    , m_recon(nullptr)
    , m_data(nullptr)
    {
    }

    ~cuda_data()
    {
        cudaFree(m_gridx);
        cudaFree(m_gridy);
        cudaFree(m_coordx);
        cudaFree(m_coordy);
        cudaFree(m_ax);
        cudaFree(m_ay);
        cudaFree(m_bx);
        cudaFree(m_by);
        cudaFree(m_coorx);
        cudaFree(m_coory);
        cudaFree(m_dist);
        cudaFree(m_indi);
        cudaFree(m_sumdist);
        cudaFree(m_update);
    }

    void initialize(const float* data, const float* recon, int s, float mov, float* gridx,
                    float* gridy)
    {
        uintmax_t offset = s * m_dt * m_dx;
        cudaMemcpy(m_gridx, gridx, (m_nx + 1) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(m_gridy, gridy, (m_ny + 1) * sizeof(float), cudaMemcpyHostToDevice);
        m_mov   = mov;
        m_data  = data + offset;
        offset  = s * m_nx * m_ny;
        m_recon = recon + offset;
        cudaMemset(m_update, 0, m_nx * m_ny * sizeof(float));
        cudaMemset(m_sumdist, 0, m_nx * m_ny * sizeof(float));
    }

    void finalize(float* recon, int s)
    {
        int      block  = 512;
        int      grid   = (m_nx * m_ny + block - 1) / block;
        int      offset = s * m_nx * m_ny;
        AutoLock l(TypeMutex<this_type>());  // lock update
        mlem_update_kernel<<<grid, block>>>(recon + offset, m_update, m_sumdist,
                                            m_nx * m_ny);
    }

    void sync()
    {
        cudaStreamSynchronize(0);
        CUDA_CHECK_LAST_ERROR();
    }

    // clang-format off
#define DEFINE_GETTER(type, obj, m_obj)                                                  \
    __device__ __host__ type obj () const { return m_obj ; }
    // clang-format on

    DEFINE_GETTER(int, nx, m_nx)
    DEFINE_GETTER(int, ny, m_ny)
    DEFINE_GETTER(int, dy, m_dy)
    DEFINE_GETTER(int, dt, m_dt)
    DEFINE_GETTER(int, dx, m_dx)
    DEFINE_GETTER(float, mov, m_mov)
    DEFINE_GETTER(float*, gridx, m_gridx)
    DEFINE_GETTER(float*, gridy, m_gridy)
    DEFINE_GETTER(float*, coordx, m_coordx)
    DEFINE_GETTER(float*, coordy, m_coordy)
    DEFINE_GETTER(float*, ax, m_ax)
    DEFINE_GETTER(float*, ay, m_ay)
    DEFINE_GETTER(float*, bx, m_bx)
    DEFINE_GETTER(float*, by, m_by)
    DEFINE_GETTER(float*, coorx, m_coorx)
    DEFINE_GETTER(float*, coory, m_coory)
    DEFINE_GETTER(float*, dist, m_dist)
    DEFINE_GETTER(int*, indi, m_indi)
    DEFINE_GETTER(float*, simdata, m_simdata)
    DEFINE_GETTER(float*, sumdist, m_sumdist)
    DEFINE_GETTER(float*, update, m_update)
    DEFINE_GETTER(const float*, recon, m_recon)
    DEFINE_GETTER(const float*, data, m_data)

#undef DEFINE_GETTER
};

//============================================================================//

__global__ void
cuda_mlem_compute_pixel(cuda_data* _cache, int s, int p, int quadrant, float sin_p,
                        float cos_p, int size)
{
    int ngridx = _cache->nx();
    int ngridy = _cache->ny();
    int dy     = _cache->dy();
    int dt     = _cache->dt();
    int dx     = _cache->dx();
    int asize  = 0;
    int bsize  = 0;
    int csize  = 0;

    float        mov      = _cache->mov();
    float*       coordx   = _cache->coordx();
    float*       coordy   = _cache->coordy();
    float*       gridx    = _cache->gridx();
    float*       gridy    = _cache->gridy();
    float*       ax       = _cache->ax();
    float*       ay       = _cache->ay();
    float*       bx       = _cache->bx();
    float*       by       = _cache->by();
    float*       coorx    = _cache->coorx();
    float*       coory    = _cache->coory();
    float*       dist     = _cache->dist();
    int*         indi     = _cache->indi();
    float*       simdata  = _cache->simdata();
    float*       sum_dist = _cache->sumdist();
    float*       update   = _cache->update();
    const float* data     = _cache->data();
    const float* recon    = _cache->recon();

    int d0      = blockIdx.x * blockDim.x + threadIdx.x;
    int dstride = blockDim.x * gridDim.x;

    for(int d = d0; d < size; d += dstride)
    {
        // Calculate coordinates
        float xi = -ngridx - ngridy;
        float yi = 0.5f * (1 - dx) + d + mov;
        cuda_calc_coords(ngridx, ngridy, xi, yi, sin_p, cos_p, gridx, gridy, coordx,
                         coordy);

        // Merge the (coordx, gridy) and (gridx, coordy)
        cuda_trim_coords(ngridx, ngridy, coordx, coordy, gridx, gridy, &asize, ax, ay,
                         &bsize, bx, by);

        // Sort the array of intersection points (ax, ay) and
        // (bx, by). The new sorted intersection points are
        // stored in (coorx, coory). Total number of points
        // are csize.
        cuda_sort_intersections(quadrant, asize, ax, ay, bsize, bx, by, &csize, coorx,
                                coory);

        // Calculate the distances (dist) between the
        // intersection points (coorx, coory). Find the
        // indices of the pixels on the reconstruction grid.
        cuda_calc_dist(ngridx, ngridy, csize, coorx, coory, indi, dist);

        // Calculate simdata
        cuda_calc_simdata(s, p, d, ngridx, ngridy, dt, dx, csize, indi, dist, recon,
                          simdata);  // Output: simdata

        // Calculate dist*dist
        float sum_dist2 = 0.0f;
        for(int n = 0; n < csize - 1; n++)
        {
            sum_dist2 += dist[n] * dist[n];
            sum_dist[indi[n]] += dist[n];
        }

        // Update
        if(sum_dist2 != 0.0f)
        {
            int   ind_data = d + p * dx;
            float upd      = data[ind_data] / simdata[ind_data];
            for(int n = 0; n < csize - 1; n++)
            {
                update[indi[n]] += upd * dist[n];
            }
        }
    }
}

//============================================================================//

void
cuda_mlem_compute_projection(const float* theta, int s, int p, int nthreads,
                             cuda_data** _cuda_data)
{
    auto       thread_number = GetThisThreadID() % nthreads;
    cuda_data* _cache        = _cuda_data[thread_number];

    NVTX_NAME_THREAD(thread_number, __FUNCTION__);

    int dx = _cache->dx();
    // Calculate the sin and cos values
    // of the projection angle and find
    // at which quadrant on the cartesian grid.
    float theta_p  = fmodf(theta[p], 2.0f * (float) M_PI);
    float sin_p    = sinf(theta_p);
    float cos_p    = cosf(theta_p);
    int   quadrant = calc_quadrant(theta_p);
    int   block    = 512;
    int   grid     = (dx + block - 1) / block;
    int   smem     = block * sizeof(float);

    // NVTX_RANGE_PUSH(&nvtx_update);

    cuda_mlem_compute_pixel<<<grid, block, smem>>>(_cache, s, p, quadrant, sin_p, cos_p,
                                                   dx);

    // NVTX_RANGE_POP(&nvtx_update);
}

//============================================================================//

void
mlem_cuda(const float* cpu_data, int dy, int dt, int dx, const float* center,
          const float* theta, float* cpu_recon, int ngridx, int ngridy, int num_iter)
{
    if(cuda_device_count() == 0)
        throw std::runtime_error("No CUDA device(s) available");

    cuda_device_query();

    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    auto tid = GetThisThreadID();

    // assign the thread to a device
    set_this_thread_device();

    // get some properties
    int num_devices = cuda_device_count();
    int nthreads    = GetEnv("TOMOPY_NUM_THREADS", 4);
    nthreads        = std::max(nthreads, 1);
    if(nthreads > 4)
    {
        printf("INFO: Current version allows no more than 4 threads...\n");
        nthreads = 4;
    }

    TIMEMORY_AUTO_TIMER("");

    // GPU allocated copies
    float* gridx   = cpu_malloc<float>(ngridx + 1);
    float* gridy   = cpu_malloc<float>(ngridy + 1);
    float* recon   = gpu_malloc<float>(dy * ngridx * ngridy);
    float* data    = gpu_malloc<float>(dy * dt * dx);
    float* simdata = gpu_malloc<float>(dy * dt * dx);

    cudaMemcpy(recon, cpu_recon, dy * ngridx * ngridy * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(data, cpu_data, dy * dt * dx * sizeof(float), cudaMemcpyHostToDevice);

    cuda_data** _cuda_data = new cuda_data*[nthreads];

    auto index    = [&]() { return GetThisThreadID() % nthreads; };
    auto finalize = [&](int s) { _cuda_data[index()]->finalize(recon, s); };
    auto _sync    = [&]() { _cuda_data[index()]->sync(); };
    auto allocate = [&]() {
        _cuda_data[index()] = new cuda_data(index() % num_devices, index(), dy, dt, dx,
                                            ngridx, ngridy, simdata);
    };
    auto initialize = [&](int s, float mov, float* gridx, float* gridy) {
        _cuda_data[index()]->initialize(data, recon, s, mov, gridx, gridy);
    };

    allocate();

    for(int i = 0; i < num_iter; i++)
    {
        // initialize simdata to zero
        cudaMemset(simdata, 0, dy * dt * dx * sizeof(float));

        // For each slice
        for(int s = 0; s < dy; s++)
        {
            float mov = 0.0f;
            preprocessing(ngridx, ngridy, dx, center[s], &mov, gridx, gridy);
            // Outputs: mov, gridx, gridy

            auto _init  = std::bind(initialize, s, mov, gridx, gridy);
            auto _final = std::bind(finalize, s);

            _init();
            _sync();
            // For each projection angle
            for(int p = 0; p < dt; p++)
            {
                cuda_mlem_compute_projection(theta, s, p, nthreads, _cuda_data);
            }
            _final();
            _sync();
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(cpu_recon, recon, dy * ngridx * ngridy * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(recon);

    for(int i = 0; i < nthreads; ++i)
        delete _cuda_data[i];
    delete[] _cuda_data;

    cudaDeviceSynchronize();
}

//============================================================================//
