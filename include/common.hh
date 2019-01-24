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
//   TOMOPY class header

#pragma once

#include "gpu.hh"
#include "utils.hh"

//======================================================================================//

namespace
{
constexpr float pi       = static_cast<float>(M_PI);
constexpr float halfpi   = 0.5f * pi;
constexpr float twopi    = 2.0f * pi;
constexpr float epsilonf = 2.0f * std::numeric_limits<float>::epsilon();
}

//======================================================================================//

#define PRAGMA_SIMD _Pragma("omp simd")
#define PRAGMA_SIMD_REDUCTION(var) _Pragma("omp simd reducton(+ : var)")
#define HW_CONCURRENCY std::thread::hardware_concurrency()

//======================================================================================//

struct cpu_rotate_data
{
    int          m_id;
    int          m_dy;
    int          m_dt;
    int          m_dx;
    int          m_nx;
    int          m_ny;
    uintmax_t    m_size;
    farray_t     m_rot;
    farray_t     m_tmp;
    float*       m_recon;
    float*       m_update;
    float*       m_simdata;
    const float* m_data;

    cpu_rotate_data(int id, int dy, int dt, int dx, int nx, int ny)
    : m_id(id)
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_size(m_nx * m_ny)
    , m_rot(farray_t(m_size, 0.0f))
    , m_tmp(farray_t(m_size, 0.0f))
    , m_recon(nullptr)
    , m_update(new float[m_size])
    , m_simdata(nullptr)
    , m_data(nullptr)
    {
    }

    ~cpu_rotate_data() { delete[] m_update; }

    cpu_rotate_data(const cpu_rotate_data& rhs)
    : m_id(rhs.m_id)
    , m_dy(rhs.m_dy)
    , m_dt(rhs.m_dt)
    , m_dx(rhs.m_dx)
    , m_nx(rhs.m_nx)
    , m_ny(rhs.m_ny)
    , m_size(rhs.m_size)
    , m_rot(rhs.m_rot)
    , m_tmp(rhs.m_tmp)
    , m_recon(rhs.m_recon)
    , m_update(new float[m_size])
    , m_simdata(rhs.m_simdata)
    , m_data(rhs.m_data)
    {
        memcpy(m_update, rhs.m_update, m_size * sizeof(float));
    }

    float*&         simdata() { return m_simdata; }
    float*&         update() { return m_update; }
    float*&         recon() { return m_recon; }
    const float*&   data() { return m_data; }
    farray_t&       rot() { return m_rot; }
    farray_t&       tmp() { return m_tmp; }
    const farray_t& rot() const { return m_rot; }
    const farray_t& tmp() const { return m_tmp; }
};

//--------------------------------------------------------------------------------------//

#ifndef DLL
#    ifdef WIN32
#        define DLL __declspec(dllexport)
#    else
#        define DLL
#    endif
#endif

//--------------------------------------------------------------------------------------//

#ifdef __cplusplus
#    include <cstdio>
#    include <cstring>
#else
#    include <stdio.h>
#    include <string.h>
#endif

//--------------------------------------------------------------------------------------//

#if defined(TOMOPY_USE_CUDA)
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#    include <vector_types.h>
#else
#    if !defined(cudaStream_t)
#        define cudaStream_t int
#    endif
#endif

//--------------------------------------------------------------------------------------//
