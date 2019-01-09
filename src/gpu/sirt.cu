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
//   TOMOPY CUDA implementation

#include "gpu.hh"
#include "utils.hh"

BEGIN_EXTERN_C
#include "sirt.h"
#include "utils.h"
#include "utils_cuda.h"
#include "utils_openacc.h"
#include "utils_openmp.h"
END_EXTERN_C

#include <algorithm>
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

class cuda_streamer
{
public:
    typedef std::list<cudaStream_t> stream_list_t;

    cuda_streamer(int nstreams)
    : m_num_streams(nstreams)
    , m_streams(create_streams(m_num_streams))
    {
        for(int i = 0; i < m_num_streams; ++i)
            m_available.push_back(m_streams[i]);
    }

    ~cuda_streamer() { destroy_streams(m_streams, m_num_streams); }

    cudaStream_t get()
    {
        if(m_available.size() > 0)
        {
            cudaStream_t _tmp_stream = m_available.front();
            m_available.pop_front();
            m_utilized.push_back(_tmp_stream);
            return _tmp_stream;
        }
        cudaStream_t _tmp_stream = m_utilized.front();
        m_utilized.pop_front();
        m_utilized.push_back(_tmp_stream);
        cudaStreamSynchronize(_tmp_stream);
        CUDA_CHECK_LAST_ERROR();
        return _tmp_stream;
    }

    void sync()
    {
        // synchronize
        while(!m_utilized.empty())
        {
            cudaStream_t _tmp_stream = m_utilized.front();
            m_utilized.pop_front();
            m_available.push_back(_tmp_stream);
            cudaStreamSynchronize(_tmp_stream);
            CUDA_CHECK_LAST_ERROR();
        }
    }

private:
    uintmax_t     m_num_streams;
    cudaStream_t* m_streams;
    stream_list_t m_available;
    stream_list_t m_utilized;
};

//============================================================================//

__global__ void
cuda_sirt_arrsum_kernel(float* dst, const float* src, int size, const float factor)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        dst[i] += factor * src[i];
    }
}

//============================================================================//

__global__ void
cuda_sirt_arrsum_kernel(float* dst, const float factor, int size)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        dst[i] += factor;
    }
}

//============================================================================//

__global__ void
cuda_sirt_update_kernel(const float* data, const float* simdata, int ngridx,
                        float* recon_rot, int size)
{
    float upd     = (data[0] - simdata[0]) / static_cast<float>(ngridx);
    int   i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int   istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        recon_rot[i] += upd;
    }
}

//============================================================================//

__global__ void
cuda_sirt_copy_kernel(float* dst, const float* src, int size)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        dst[i] = src[i];
    }
}

//============================================================================//

__global__ void
cuda_sirt_update_kernel(float* recon, const float* data, const float* sum, const int nx,
                        int size)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    float upd = (*data - *sum) / static_cast<float>(nx);
    for(int i = i0; i < size; i += istride)
    {
        recon[i] += upd;
    }
}

//----------------------------------------------------------------------------//

void
cuda_sirt_update(int nx, float* recon, const float* data, float* buffer,
                 cudaStream_t stream)
{
    NVTX_RANGE_PUSH(&nvtx_update);
    int block = 512;
    int grid  = (nx + block - 1) / block;
    int smem  = 0;

    // Calculate simulated data by summing up along x-axis
    int _grid = min(grid, 1024);
    deviceReduceKernel<<<_grid, block, smem, stream>>>(recon, buffer, nx);
    deviceReduceKernel<<<1, 1024, 0, stream>>>(buffer, buffer, block);

    // Make update by backprojecting error along x-axis
    cuda_sirt_update_kernel<<<grid, block, smem, stream>>>(recon, data, buffer, nx, nx);

    NVTX_RANGE_POP(&nvtx_update);
}

//============================================================================//

struct thread_data
{
    int          m_device;
    int          m_id;
    int          m_dy;
    int          m_dt;
    int          m_dx;
    int          m_nx;
    int          m_ny;
    int          m_size;
    float*       m_sum;
    float*       m_rot;
    float*       m_tmp;
    float*       m_buffer;
    float*       m_update;
    const float* m_recon;
    const float* m_data;
    cudaStream_t m_stream;

    thread_data(int device, int id, int dy, int dt, int dx, int nx, int ny,
                const float* data)
    : m_device(device)
    , m_id(id)
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_size(m_nx * m_ny)
    , m_sum(gpu_malloc<float>(m_nx))
    , m_rot(gpu_malloc<float>(m_size))
    , m_tmp(gpu_malloc<float>(m_size))
    , m_recon(nullptr)
    , m_buffer(gpu_malloc<float>(m_nx * m_dt))
    , m_update(gpu_malloc<float>(m_size))
    {
        cudaStreamCreate(&m_stream);
    }

    ~thread_data()
    {
        cudaFree(m_sum);
        cudaFree(m_rot);
        cudaFree(m_tmp);
        cudaFree(m_buffer);
        cudaFree(m_update);
        cudaStreamDestroy(m_stream);
    }

    void initialize(const float* data, const float* recon, int s)
    {
        uintmax_t offset = s * m_dt * m_dx;
        m_data           = data + offset;
        offset           = s * m_size;
        m_recon          = recon + offset;
        cudaMemsetAsync(m_update, 0, m_size * sizeof(float), m_stream);
    }

    void finalize(float* recon, int s)
    {
        int   block  = 512;
        int   grid   = (m_size + block - 1) / block;
        int   offset = s * m_size;
        float factor = 1.0f / static_cast<float>(m_dt);
        cuda_sirt_arrsum_kernel<<<grid, block, 0, m_stream>>>(recon + offset, m_update,
                                                              m_size, factor);
    }

    void sync()
    {
        cudaStreamSynchronize(m_stream);
        CUDA_CHECK_LAST_ERROR();
    }

    cudaStream_t stream() const { return m_stream; }

    float*       rot() const { return m_rot; }
    float*       tmp() const { return m_tmp; }
    float*       update() const { return m_update; }
    const float* recon() const { return m_recon; }
    const float* data() const { return m_data; }

    float* buffer() const
    {
        cudaMemsetAsync(m_buffer, 0, m_nx * m_dt * sizeof(float), m_stream);
        return m_buffer;
    }

    float* sum()
    {
        cudaMemsetAsync(m_sum, 0, m_nx * sizeof(float), m_stream);
        return m_sum;
    }
};

//============================================================================//

void
cuda_compute_projection(int dt, int dx, int ngridx, int ngridy, const float* theta, int s,
                        int p, int nthreads, thread_data** _thread_data)
{
    auto         thread_number = GetThisThreadID() % nthreads;
    thread_data* _cache        = _thread_data[thread_number];
    cudaStream_t stream        = _cache->stream();

    // needed for recon to output at proper orientation
    float pi_offset  = 0.5f * (float) M_PI;
    float fngridx    = ngridx;
    float theta_p    = fmodf(theta[p] + pi_offset, 2.0f * (float) M_PI);
    int   recon_size = ngridx * ngridy;
    int   float_size = sizeof(float);
    int   block      = 512;
    int   grid       = (ngridx + block - 1) / block;
    int   smem       = 0;

    const float* recon     = _cache->recon();
    const float* data      = _cache->data();
    float*       update    = _cache->update();
    float*       recon_rot = _cache->rot();
    float*       recon_tmp = _cache->tmp();
    float*       buffer    = _cache->buffer();

    // Rotate object
    cuda_rotate_ip(recon_rot, recon, -theta_p, ngridx, ngridy, stream);

    // static thread_local cuda_streamer* _streamer = new cuda_streamer(dx);
    for(int d = 0; d < dx; ++d)
    {
        int pix_offset = d * ngridx;  // pixel offset
        int idx_data   = d + p * dx;  // data offset
        // cudaStream_t _stream    = _streamer->get();
        cuda_sirt_update(ngridx, recon_rot + pix_offset, data + idx_data, _cache->sum(),
                         stream);
    }
    //_streamer->sync();

    // Back-Rotate object
    cuda_rotate_ip(recon_tmp, recon_rot, theta_p, ngridx, ngridy, stream);

    // update shared update array
    cuda_sirt_arrsum_kernel<<<grid, block, smem, stream>>>(update, recon_tmp,
                                                           ngridx * ngridy, 1.0f);
}

//----------------------------------------------------------------------------//

void
sirt_cuda(const float* cpu_data, int dy, int dt, int dx, const float* center,
          const float* theta, float* cpu_recon, int ngridx, int ngridy, int num_iter)
{
    if(cuda_device_count() == 0)
        throw std::runtime_error("No CUDA device(s) available");

    cuda_device_query();

    printf("\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    TIMEMORY_AUTO_TIMER("");

    // assign the thread to a device
    set_this_thread_device();
    int num_devices = cuda_device_count();

    // needed for recon to output at proper orientation
    float* recon = gpu_malloc<float>(dy * ngridx * ngridy);
    float* data  = gpu_malloc<float>(dy * dt * dx);

    cudaMemcpy(recon, cpu_recon, dy * ngridx * ngridy * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(data, cpu_data, dy * dt * dx * sizeof(float), cudaMemcpyHostToDevice);

    int nstreams = GetEnv("TOMOPY_NUM_STREAMS", 1);
    int nthreads = GetEnv("TOMOPY_NUM_THREADS", nstreams);
    thread_data** _thread_data = new thread_data*[nthreads];
    for(int i = 0; i < nthreads; ++i)
        _thread_data[i] =
            new thread_data(i % num_devices, i, dy, dt, dx, ngridx, ngridy, data);

    for(int i = 0; i < num_iter; i++)
    {
        printf("[%li]> iteration %3i of %3i...\n", GetThisThreadID(), i, num_iter);
        // For each slice
        for(int s = 0; s < dy; ++s)
        {
            int slice_offset = s * ngridx * ngridy;
            for(int i = 0; i < nthreads; ++i)
                _thread_data[i]->initialize(data, recon, s);
            for(int i = 0; i < nthreads; ++i)
                _thread_data[i]->sync();

            // For each projection angle
            for(int p = 0; p < dt; ++p)
            {
                cuda_compute_projection(dt, dx, ngridx, ngridy, theta, s, p, nthreads,
                                        _thread_data);
            }
            // cudaDeviceSynchronize();

            for(int i = 0; i < nthreads; ++i)
                _thread_data[i]->finalize(recon, s);
            // for(int i = 0; i < nthreads; ++i)
            //    _thread_data[i]->sync();
        }
    }
    cudaDeviceSynchronize();
    cudaMemcpy(cpu_recon, recon, dy * ngridx * ngridy * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(recon);
    cudaFree(data);

    for(int i = 0; i < nthreads; ++i)
        delete _thread_data[i];
    delete[] _thread_data;

    cudaDeviceSynchronize();
}

//============================================================================//
