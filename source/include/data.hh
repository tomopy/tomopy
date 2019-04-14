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

/** \file data.hh
 * \headerfile data.hh "include/data.hh"
 * C++ class for storing thread-specific data when doing rotation-based reconstructions
 * CpuData == rotation-based reconstruction with OpenCV
 * GpuData == rotation-based reconstruction with NPP
 */

#pragma once

#include "common.hh"
#include "constants.hh"
#include "macros.hh"
#include "typedefs.hh"
#include "utils.hh"

//======================================================================================//

class CpuData
{
public:
    typedef std::shared_ptr<CpuData>                       data_ptr_t;
    typedef std::vector<data_ptr_t>                        data_array_t;
    typedef std::tuple<data_array_t, float*, const float*> init_data_t;

public:
    CpuData(unsigned id, int dy, int dt, int dx, int nx, int ny, const float* data,
            float* recon, float* update)
    : m_id(id)
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_rot(farray_t(scast<uintmax_t>(m_nx * m_ny), 0.0f))
    , m_tmp(farray_t(scast<uintmax_t>(m_nx * m_ny), 0.0f))
    , m_update(update)
    , m_recon(recon)
    , m_data(data)
    {
    }

    ~CpuData() {}

public:
    farray_t&       rot() { return m_rot; }
    farray_t&       tmp() { return m_tmp; }
    const farray_t& rot() const { return m_rot; }
    const farray_t& tmp() const { return m_tmp; }

    float*       update() const { return m_update; }
    float*       recon() { return m_recon; }
    const float* recon() const { return m_recon; }
    const float* data() const { return m_data; }

    Mutex* upd_mutex() const
    {
        static Mutex mtx;
        return &mtx;
    }

    void reset()
    {
        // reset temporaries to zero (NECESSARY!)
        // -- note: the OpenCV effectively ensures that we overwrite all values
        //          because we use cv::Mat::zeros and copy that to destination
        memset(m_rot.data(), 0, scast<uintmax_t>(m_nx * m_ny) * sizeof(float));
        memset(m_tmp.data(), 0, scast<uintmax_t>(m_nx * m_ny) * sizeof(float));
    }

public:
    // static functions
    static init_data_t initialize(int dy, int dt, int dx, int ngridx, int ngridy,
                                  float* recon, const float* data, float* update)
    {
        auto         nthreads = GetNumThreads();
        data_array_t cpu_data(nthreads);
        for(unsigned ii = 0; ii < nthreads; ++ii)
        {
            cpu_data[ii] = data_ptr_t(
                new CpuData(ii, dy, dt, dx, ngridx, ngridy, data, recon, update));
        }
        return init_data_t(cpu_data, recon, data);
    }

    static void reset(data_array_t& data)
    {
        // reset "update" to zero
        for(auto& itr : data)
            itr->reset();
    }

protected:
    unsigned     m_id;
    int          m_dy;
    int          m_dt;
    int          m_dx;
    int          m_nx;
    int          m_ny;
    farray_t     m_rot;
    farray_t     m_tmp;
    float*       m_update;
    float*       m_recon;
    const float* m_data;
};

//======================================================================================//

#if defined(__NVCC__) && defined(TOMOPY_USE_CUDA)

//======================================================================================//

class GpuData
{
public:
    // typedefs
    typedef GpuData                                  this_type;
    typedef std::shared_ptr<GpuData>                 data_ptr_t;
    typedef std::vector<data_ptr_t>                  data_array_t;
    typedef std::tuple<data_array_t, float*, float*> init_data_t;

public:
    // ctors, dtors, assignment
    GpuData(int device, int dy, int dt, int dx, int nx, int ny, const float* data,
            float* recon, float* update)
    : m_device(device)
    , m_grid(GetGridSize())
    , m_block(GetBlockSize())
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_rot(nullptr)
    , m_tmp(nullptr)
    , m_update(update)
    , m_recon(recon)
    , m_data(data)
    , m_num_streams(GetEnv<int>("TOMOPY_STREAMS_PER_THREAD", 1))
    {
        cuda_set_device(m_device);
        m_streams = create_streams(m_num_streams, cudaStreamNonBlocking);
        m_rot     = gpu_malloc<float>(m_dy * m_nx * m_ny);
        m_tmp     = gpu_malloc<float>(m_dy * m_nx * m_ny);
        CUDA_CHECK_LAST_ERROR();
    }

    ~GpuData()
    {
        cudaFree(m_rot);
        cudaFree(m_tmp);
        destroy_streams(m_streams, m_num_streams);
    }

    GpuData(const this_type&) = delete;
    GpuData(this_type&&)      = default;

    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&&) = default;

public:
    // access functions
    int          device() const { return m_device; }
    int          grid() const { return compute_grid(m_dx); }
    int          block() const { return m_block; }
    float*       rot() const { return m_rot; }
    float*       tmp() const { return m_tmp; }
    float*       update() const { return m_update; }
    float*       recon() { return m_recon; }
    const float* recon() const { return m_recon; }
    const float* data() const { return m_data; }
    cudaStream_t stream(int n = -1)
    {
        if(n < 0)
        {
            // increment to next stream
            n = m_num_stream_requests++;
        }
        return m_streams[n % m_num_streams];
    }

public:
    // assistant functions
    int compute_grid(int size) const
    {
        return (m_grid < 1) ? ((size + m_block - 1) / m_block) : m_grid;
    }

    void sync(int stream_id = -1)
    {
        auto _sync = [&](cudaStream_t _stream) { stream_sync(_stream); };

        if(stream_id < 0)
            for(int i = 0; i < m_num_streams; ++i)
                _sync(m_streams[i]);
        else
            _sync(m_streams[stream_id % m_num_streams]);
    }

    void reset()
    {
        // reset destination arrays (NECESSARY!)
        gpu_memset<float>(m_rot, 0, m_dy * m_nx * m_ny, *m_streams);
        gpu_memset<float>(m_tmp, 0, m_dy * m_nx * m_ny, *m_streams);
    }

public:
    // static functions
    static init_data_t initialize(int device, int dy, int dt, int dx, int ngridx,
                                  int ngridy, float* cpu_recon, const float* cpu_data,
                                  float* update)
    {
        auto      nthreads = GetNumThreads();
        uintmax_t nstreams = 2;
        auto      streams  = create_streams(nstreams, cudaStreamNonBlocking);
        float*    recon =
            gpu_malloc_and_memcpy<float>(cpu_recon, dy * ngridx * ngridy, streams[0]);
        float* data = gpu_malloc_and_memcpy<float>(cpu_data, dy * dt * dx, streams[1]);
        data_array_t gpu_data(nthreads);
        for(int ii = 0; ii < nthreads; ++ii)
        {
            gpu_data[ii] = data_ptr_t(
                new GpuData(device, dy, dt, dx, ngridx, ngridy, data, recon, update));
        }

        // synchronize and destroy
        destroy_streams(streams, nstreams);

        return init_data_t(gpu_data, recon, data);
    }

    static void reset(data_array_t& data)
    {
        // reset "update" to zero
        for(auto& itr : data)
            itr->reset();
    }

    static void sync(data_array_t& data)
    {
        // sync all the streams
        for(auto& itr : data)
            itr->sync();
    }

protected:
    // data
    int           m_device;
    int           m_grid;
    int           m_block;
    int           m_dy;
    int           m_dt;
    int           m_dx;
    int           m_nx;
    int           m_ny;
    float*        m_rot         = nullptr;
    float*        m_tmp         = nullptr;
    float*        m_update      = nullptr;
    float*        m_recon       = nullptr;
    const float*  m_data        = nullptr;
    int           m_num_streams = 0;
    int           m_num_stream_requests;
    cudaStream_t* m_streams = nullptr;
};

#endif  // NVCC and TOMOPY_USE_CUDA

//======================================================================================//
