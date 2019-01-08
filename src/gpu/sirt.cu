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

#include "PTL/TBBTaskGroup.hh"
#include "PTL/TaskGroup.hh"
#include "PTL/TaskManager.hh"
#include "PTL/TaskRunManager.hh"
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

cudaStream_t
get_stream(bool sync)
{
    typedef std::list<cudaStream_t> stream_list_t;

    static thread_local int nthread_streams = GetEnv<int>("TOMOPY_NUM_THREAD_STREAMS", 4);
    static thread_local cudaStream_t* thread_streams = create_streams(nthread_streams);
    static thread_local bool          init           = true;
    static thread_local stream_list_t available_streams;
    static thread_local stream_list_t utilized_streams;

    if(init)
    {
        for(int i = 0; i < nthread_streams; ++i)
            available_streams.push_back(thread_streams[i]);
        init = false;
    }

    cudaStream_t _tmp_stream;
    if(!sync)
    {
        // printf("\t%s [line: %i] fetching stream... avail = %lu, used =
        // %lu\n",
        //       __FUNCTION__, __LINE__, available_streams.size(),
        //       utilized_streams.size());
        if(available_streams.size() > 0)
        {
            // PRINT_HERE("");
            _tmp_stream = available_streams.front();
            available_streams.pop_front();
            utilized_streams.push_back(_tmp_stream);
            return _tmp_stream;
        }
        else
        {
            // PRINT_HERE("");
            _tmp_stream = utilized_streams.front();
            utilized_streams.pop_front();
            utilized_streams.push_back(_tmp_stream);
            cudaStreamSynchronize(_tmp_stream);
            CUDA_CHECK_LAST_ERROR();
            return _tmp_stream;
        }
    }
    // synchronize
    while(!utilized_streams.empty())
    {
        // printf("\t%s [line: %i] synchronizing streams... avail = %lu, used "
        //       "= %lu\n",
        //       __FUNCTION__, __LINE__, available_streams.size(),
        //       utilized_streams.size());
        _tmp_stream = utilized_streams.front();
        utilized_streams.pop_front();
        available_streams.push_back(_tmp_stream);
        cudaStreamSynchronize(_tmp_stream);
        CUDA_CHECK_LAST_ERROR();
    }
    return _tmp_stream;
}

//============================================================================//

__global__ void
cuda_sirt_arrsum_global(float* dst, const float* src, int size, const float factor)
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
cuda_sirt_arrsum_global(float* dst, const float factor, int size)
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
cuda_sirt_update_global(const float* data, const float* simdata, int ngridx,
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
cuda_sirt_copy_global(float* dst, const float* src, int size)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < size; i += istride)
    {
        dst[i] = src[i];
    }
}

//----------------------------------------------------------------------------//

void
cuda_sirt_update(const float* data, const float* simdata, int ngridx, float* recon_rot,
                 int size, cudaStream_t* stream)
{
    NVTX_RANGE_PUSH(&nvtx_update);
    int nb   = cuda_multi_processor_count();
    int nt   = 4;  // cuda_max_threads_per_block();
    int smem = 0;

    CUDA_CHECK_LAST_ERROR();
    cuda_sirt_update_global<<<nb, nt, smem>>>(data, simdata, ngridx, recon_rot, size);
    CUDA_CHECK_LAST_ERROR();
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
    float*       m_recon;
    float*       m_update;
    float*       m_simdata;
    float*       m_simdata_cpu;
    float*       m_gpu_recon;
    cudaStream_t m_stream;

    thread_data(int device, int id, int dy, int dt, int dx, int nx, int ny,
                const float* gpu_recon)
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
    , m_recon(gpu_malloc<float>(m_size))
    , m_update(gpu_malloc<float>(m_size))
    , m_simdata(gpu_malloc<float>(m_dy * m_dt * m_dx))
    , m_simdata_cpu(new float[dy * dt * dx])
    {
        cudaStreamCreate(&m_stream);
        cudaMemset(m_simdata, 0, m_dy * m_dt * m_dx * sizeof(float));
        memset(m_simdata_cpu, 0, m_dy * m_dt * m_dx * sizeof(float));
    }

    ~thread_data()
    {
        cudaFree(m_sum);
        cudaFree(m_rot);
        cudaFree(m_tmp);
        cudaFree(m_recon);
        cudaFree(m_update);
        cudaFree(m_simdata);
        delete[] m_simdata_cpu;
        cudaStreamDestroy(m_stream);
    }

    void initialize(int nb, int nt, int smem, const float* gpu_recon, int s)
    {
        int          offset  = s * m_size;
        const float* g_recon = gpu_recon + offset;
        cudaMemset(m_update, 0, m_size * sizeof(float));
        cudaMemcpy(m_recon, g_recon, m_size * sizeof(float), cudaMemcpyDeviceToDevice);

        /*float* _buffer = gpu_malloc<float>(m_size);
        cudaMemset(_buffer, 0, m_size * sizeof(float));
        float _this_sum = deviceReduce(m_recon, _buffer, m_size, m_stream);
        float _real_sum = deviceReduce(g_recon, _buffer, m_size, m_stream);
        printf("[TID: %i] gpu recon = %p, real sum = %f, local recon = %p, "
               "local sum = %f\n",
               m_id, g_recon, _real_sum, m_recon, _this_sum);
        cudaFree(_buffer);*/
    }

    void finalize(int nb, int nt, int smem, float* gpu_recon, int s)
    {
        int   offset = s * m_size;
        float factor = 1.0f / static_cast<float>(m_dt);
        cuda_sirt_arrsum_global<<<nb, nt, smem, m_stream>>>(gpu_recon + offset, m_update,
                                                            m_size, factor);
    }

    void sync() { cudaStreamSynchronize(m_stream); }

    cudaStream_t stream() const { return m_stream; }

    void reset_simdata()
    {
        cudaMemset(m_simdata, 0, m_dy * m_dt * m_dx * sizeof(float));
        memset(m_simdata_cpu, 0, m_dy * m_dt * m_dx * sizeof(float));
    }

    float* cpu_simdata() const { return m_simdata_cpu; }
    float* simdata() const { return m_simdata; }
    float* update() const { return m_update; }
    float* recon() const { return m_recon; }
    float* rot() const { return m_rot; }
    float* tmp() const { return m_tmp; }

    float* sum()
    {
        cudaMemsetAsync(m_sum, 0, m_nx * sizeof(float), m_stream);
        CUDA_CHECK_LAST_ERROR();
        return m_sum;
    }
};

//============================================================================//

void
cuda_compute_projection(int dt, int dx, int ngridx, int ngridy, const float* cpu_data,
                        const float* theta, int s, int p, int nthreads,
                        thread_data** _thread_data)
{
    auto         thread_number = ThreadPool::GetThisThreadID() % nthreads;
    thread_data* _cache        = _thread_data[thread_number];
    cudaStream_t stream        = _cache->stream();

    // needed for recon to output at proper orientation
    float pi_offset  = 0.5f * (float) M_PI;
    float fngridx    = ngridx;
    float theta_p    = fmodf(theta[p] + pi_offset, 2.0f * (float) M_PI);
    int   recon_size = ngridx * ngridy;
    int   float_size = sizeof(float);
    int   nb         = 4 * cuda_multi_processor_count();
    int   nt         = cuda_max_threads_per_block();
    int   smem       = 0;

    float* simdata     = _cache->simdata();
    float* recon       = _cache->recon();
    float* update      = _cache->update();
    float* cpu_simdata = _cache->cpu_simdata();
    float* recon_rot   = _cache->rot();
    float* recon_tmp   = _cache->tmp();

    // Rotate object
    cuda_rotate_ip(recon_rot, recon, -theta_p, ngridx, ngridy, stream);
    CUDA_CHECK_LAST_ERROR();

    for(int d = 0; d < dx; ++d)
    {
        int          pix_offset   = d * ngridx;  // pixel offset
        int          idx_data     = d + p * dx + s * dt * dx;
        float*       _cpu_simdata = cpu_simdata + idx_data;
        const float* _cpu_data    = cpu_data + idx_data;
        float        _sim         = 0.0f;

        // Calculate simulated data by summing up along x-axis
        //_sim = reduce(_recon_rot, _cache->sum(), ngridx, stream);
        _sim = deviceReduce(recon_rot + pix_offset, _cache->sum(), ngridx, stream);

        // update shared simdata array
        *_cpu_simdata += _sim;

        cuda_sirt_arrsum_global<<<1, 1, smem, stream>>>(simdata + idx_data, _sim, 1);

        // Make update by backprojecting error along x-axis
        float upd = (*_cpu_data - *_cpu_simdata) / fngridx;
        cuda_sirt_arrsum_global<<<nb, nt, smem, stream>>>(recon_rot + pix_offset, upd,
                                                          ngridx);
    }

    // Back-Rotate object
    cuda_rotate_ip(recon_tmp, recon_rot, theta_p, ngridx, ngridy, stream);
    CUDA_CHECK_LAST_ERROR();

    /*{
        cudaStreamSynchronize(stream);
        CUDA_CHECK_LAST_ERROR();
        float* g_recon = recon_tmp;
        float* m_recon = recon_rot;
        int    m_size  = ngridx * ngridy;
        float* _buffer = gpu_malloc<float>(m_size);
        cudaMemset(_buffer, 0, m_size * sizeof(float));
        float _this_sum = deviceReduce(m_recon, _buffer, m_size, stream);
        float _real_sum = deviceReduce(g_recon, _buffer, m_size, stream);
        printf("[TID: %i] recon (tmp) = %p, tmpary sum = %f, recon (rot) = %p, "
               "rotate sum = %f\n",
               thread_number, g_recon, _real_sum, m_recon, _this_sum);
        cudaFree(_buffer);
        cudaStreamSynchronize(stream);
        CUDA_CHECK_LAST_ERROR();
    }*/

    // update shared update array
    cuda_sirt_arrsum_global<<<nb, nt, smem, stream>>>(update, recon_tmp, ngridx * ngridy,
                                                      1.0f);
    CUDA_CHECK_LAST_ERROR();
}

//----------------------------------------------------------------------------//

void
sirt_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
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

    int nthreads = GetEnv("TOMOPY_NUM_THREADS", HW_CONCURRENCY);
    // int             nstreams = GetEnv("TOMOPY_NUM_STREAMS", nthreads);
    // cudaStream_t*   streams  = create_streams(nstreams);
    TaskRunManager* run_man = gpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();

    // needed for recon to output at proper orientation
    int    recon_size = ngridx * ngridy;
    int    float_size = sizeof(float);
    int    nb         = 4 * cuda_multi_processor_count();
    int    nt         = cuda_max_threads_per_block();
    int    smem       = 0;
    float* gpu_recon  = gpu_malloc<float>(dy * recon_size);

    cudaMemcpy(gpu_recon, recon, dy * recon_size * float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();

    thread_data** _thread_data = new thread_data*[nthreads];
    for(int i = 0; i < nthreads; ++i)
        _thread_data[i] =
            new thread_data(i % num_devices, i, dy, dt, dx, ngridx, ngridy, gpu_recon);

    for(int i = 0; i < num_iter; i++)
    {
        for(int i = 0; i < nthreads; ++i)
            _thread_data[i]->reset_simdata();

        // For each slice
        for(int s = 0; s < dy; ++s)
        {
            int slice_offset = s * ngridx * ngridy;
            for(int i = 0; i < nthreads; ++i)
                _thread_data[i]->initialize(nb, nt, smem, gpu_recon, s);
            for(int i = 0; i < nthreads; ++i)
                _thread_data[i]->sync();

            TaskGroup<void> tg;
            // For each projection angle
            for(int p = 0; p < dt; ++p)
            {
                task_man->exec(tg, cuda_compute_projection, dt, dx, ngridx, ngridy, data,
                               theta, s, p, nthreads, _thread_data);
                tg.join();
            }
            tg.join();
            cudaDeviceSynchronize();

            for(int i = 0; i < nthreads; ++i)
                _thread_data[i]->finalize(nb, nt, smem, gpu_recon, s);
            for(int i = 0; i < nthreads; ++i)
                _thread_data[i]->sync();
        }
    }
    cudaDeviceSynchronize();

    cudaMemcpy(recon, gpu_recon, dy * recon_size * float_size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < nthreads; ++i)
        delete _thread_data[i];
    delete[] _thread_data;

    cudaFree(gpu_recon);

    // destroy_streams(streams, nstreams);

    cudaDeviceSynchronize();
}

//============================================================================//
