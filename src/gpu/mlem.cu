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
    int index_model = s * ry * rz;
    for(int n = 0; n < csize - 1; ++n)
    {
        *simdata += model[indi[n] + index_model] * dist[n];
    }
}

//============================================================================//

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
    float*       m_mov;
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

    cuda_data(int device, int id, int dy, int dt, int dx, int nx, int ny, float* simdata,
              const float* recon, const float* data)
    : m_device(device)
    , m_id(id)
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_mov(nullptr)
    , m_gridx(gpu_malloc<float>(nx + 1))
    , m_gridy(gpu_malloc<float>(ny + 1))
    //, m_coordx(gpu_malloc<float>(ny + 1))
    //, m_coordy(gpu_malloc<float>(nx + 1))
    //, m_ax(gpu_malloc<float>(nx + ny))
    //, m_ay(gpu_malloc<float>(nx + ny))
    //, m_bx(gpu_malloc<float>(nx + ny))
    //, m_by(gpu_malloc<float>(nx + ny))
    //, m_coorx(gpu_malloc<float>(nx + ny))
    //, m_coory(gpu_malloc<float>(nx + ny))
    //, m_dist(gpu_malloc<float>(nx + ny))
    //, m_indi(gpu_malloc<int>(nx + ny))
    //, m_simdata(simdata)
    , m_sumdist(gpu_malloc<float>(nx * ny))
    , m_update(gpu_malloc<float>(nx * ny))
    , m_recon(recon)
    , m_data(data)
    {
    }

    ~cuda_data()
    {
        cudaFree(m_gridx);
        cudaFree(m_gridy);
        // cudaFree(m_coordx);
        // cudaFree(m_coordy);
        // cudaFree(m_ax);
        // cudaFree(m_ay);
        // cudaFree(m_bx);
        // cudaFree(m_by);
        // cudaFree(m_coorx);
        // cudaFree(m_coory);
        // cudaFree(m_dist);
        // cudaFree(m_indi);
        cudaFree(m_sumdist);
        cudaFree(m_update);
    }

    void initialize(int s, float* mov, float* gridx, float* gridy)
    {
        m_gridx = gridx;
        m_gridy = gridy;
        m_mov   = mov;
        cudaMemset(m_update, 0, m_nx * m_ny * sizeof(float));
        cudaMemset(m_sumdist, 0, m_nx * m_ny * sizeof(float));
        cudaStreamSynchronize(0);
        CUDA_CHECK_LAST_ERROR();
    }

    void finalize(float* recon, int s)
    {
        int      block  = 512;
        int      grid   = (m_nx * m_ny + block - 1) / block;
        int      offset = s * m_nx * m_ny;
        AutoLock l(TypeMutex<this_type>());  // lock update
        mlem_update_kernel<<<grid, block>>>(recon, m_update, m_sumdist, offset,
                                            m_nx * m_ny);
        cudaStreamSynchronize(0);
        CUDA_CHECK_LAST_ERROR();
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
    DEFINE_GETTER(float*, mov, m_mov)
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

template <typename _Tp, typename _Up = _Tp>
__device__ _Tp*
           global_malloc(uintmax_t size, int& offset, _Up* pool = nullptr)
{
    _Tp* ptr = nullptr;
    if(pool)
    {
        _Tp* _pool  = (_Tp*) pool;
        ptr         = _pool + offset;
        float ratio = sizeof(_Up) / float(sizeof(_Tp));
        offset += (int) (ratio * size);
    }
    else
    {
        ptr    = new _Tp[size];
        offset = 0;
    }
    return ptr;
}

//============================================================================//
#if defined(DEBUG)
#    define assert_valid_pointer(ptr) assert(ptr != nullptr)
#else
#    define assert_valid_pointer(ptr)
#endif

__global__ void
cuda_mlem_compute_pixel(int ngridx, int ngridy, int dy, int dt, int dx, const float* mov,
                        const float* gridx, const float* gridy, float* sum_dist,
                        float* update, const float* data, const float* recon,
                        float* coordx, float* coordy, float* ax, float* ay, float* bx,
                        float* by, int* indi, float* dist, int s, int p,
                        int quadrant, float sin_p, float cos_p)
{
    int d0      = blockIdx.x * blockDim.x + threadIdx.x;
    int dstride = gridDim.x * blockDim.x;

    for(int d = d0; d < dx; d += dstride)
    {
        int asize = 0;
        int bsize = 0;
        int csize = 0;
        // Calculate coordinates
        float xi = -ngridx - ngridy;
        float yi = 0.5f * (1 - dx) + d + *mov;
        cuda_calc_coords(ngridx, ngridy, xi, yi, sin_p, cos_p, gridx, gridy, coordx, coordy);

        // Merge the (coordx, gridy) and (gridx, coordy)
        cuda_trim_coords(ngridx, ngridy, coordx, coordy, gridx, gridy, &asize, ax, ay, &bsize,
                         bx, by);

        // Sort the array of intersection points (ax, ay) and
        // (bx, by). The new sorted intersection points are
        // stored in (coorx, coory). Total number of points
        // are csize.
        cuda_sort_intersections(quadrant, asize, ax, ay, bsize, bx, by, &csize, coordx,
                                coordy);

        // Calculate the distances (dist) between the
        // intersection points (coorx, coory). Find the
        // indices of the pixels on the reconstruction grid.
        cuda_calc_dist(ngridx, ngridy, csize, coordx, coordy, indi, dist);

        float _simdata = 0.0f;
        // Calculate simdata
        cuda_calc_simdata(s, p, d, ngridx, ngridy, dt, dx, csize, indi, dist, recon,
                          &_simdata);  // Output: simdata

        int   index_data = d + p * dx + s * dt * dx;
        float upd        = data[index_data] / _simdata;

        for(int n = 0; n < csize - 1; n++)
        {
            float* sd = sum_dist + indi[n];
            atomicAdd(sd, dist[n]);
        }
        for(int n = 0; n < csize - 1; n++)
        {
            float* up = update + indi[n];
            atomicAdd(up, upd * dist[n]);
        }
    }
}

//============================================================================//

__global__ void
cuda_mlem_compute_projection(int ngridx, int ngridy, int dy, int dt, int dx,
                             const float* theta, const float* mov, const float* gridx,
                             const float* gridy, const float* data, const float* recon,
                             int s, float* sum_dist, float* update)
{
    int p0      = blockIdx.x * blockDim.x + threadIdx.x;
    int pstride = gridDim.x * blockDim.x;

    int    nx          = ngridx;
    int    ny          = ngridy;
    int    fb          = sizeof(float);
    int    ib          = sizeof(int);
    int    ratio       = (fb == ib) ? 1 : ((fb > ib) ? (fb / ib) : (ib / fb));
    int    total_alloc = (7 + ratio) * (nx + ny);
    int    offset      = 0;
    float* mem_pool    = global_malloc<float>(total_alloc, offset);
    float* coordx      = global_malloc<float>(ny + nx, offset, mem_pool);
    float* coordy      = global_malloc<float>(nx + ny, offset, mem_pool);
    float* ax          = global_malloc<float>(nx + ny, offset, mem_pool);
    float* ay          = global_malloc<float>(nx + ny, offset, mem_pool);
    float* bx          = global_malloc<float>(nx + ny, offset, mem_pool);
    float* by          = global_malloc<float>(nx + ny, offset, mem_pool);
    float* dist        = global_malloc<float>(nx + ny, offset, mem_pool);
    int*   indi        = global_malloc<int>(nx + ny, offset, mem_pool);
    assert_valid_pointer(coordx);
    assert_valid_pointer(coordy);
    assert_valid_pointer(ax);
    assert_valid_pointer(ay);
    assert_valid_pointer(bx);
    assert_valid_pointer(by);
    assert_valid_pointer(dist);
    assert_valid_pointer(indi);
    int size = dt;

    for(int p = p0; p < size; p += pstride)
    {
        // Calculate the sin and cos values
        // of the projection angle and find
        // at which quadrant on the cartesian grid.
        float theta_p  = fmodf(theta[p], 2.0f * (float) M_PI);
        float sin_p    = sinf(theta_p);
        float cos_p    = cosf(theta_p);
        int   quadrant = cuda_calc_quadrant(theta_p);

        cuda_mlem_compute_pixel<<<4, 1>>>(ngridx, ngridy, dy, dt, dx, mov, gridx, gridy,
                                sum_dist, update, data, recon, coordx, coordy, ax, ay,
                                bx, by, indi, dist, s, p, quadrant, sin_p, cos_p);
    }

    delete[] mem_pool;
}

//============================================================================//

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
            printf("[%li]> %-12s %3i of %3i... %5.2f seconds\n", GetThisThreadID(), "slice", s, dy,
                   elapsed_seconds.count());
        }
        auto                          t_end           = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = t_end - t_start;
        printf("[%li]> %-12s %3i of %3i... %5.2f seconds\n", GetThisThreadID(), "iteration", i,
               num_iter, elapsed_seconds.count());
    }

    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();

    cudaMemcpy(cpu_recon, recon, dy * ngridx * ngridy * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);
    CUDA_CHECK_LAST_ERROR();

    //cudaDeviceReset();
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

//============================================================================//
