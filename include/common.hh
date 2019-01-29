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

//======================================================================================//

#ifdef __cplusplus
#    ifndef BEGIN_EXTERN_C
#        define BEGIN_EXTERN_C                                                           \
            extern "C"                                                                   \
            {
#    endif
#    ifndef END_EXTERN_C
#        define END_EXTERN_C }
#    endif
#else
#    ifndef BEGIN_EXTERN_C
#        define BEGIN_EXTERN_C
#    endif
#    ifndef END_EXTERN_C
#        define END_EXTERN_C
#    endif
#endif

//======================================================================================//
//  C headers

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <ctime>

BEGIN_EXTERN_C
#include "common.h"
END_EXTERN_C

//======================================================================================//
//  C++ headers

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef TOMOPY_USE_TIMEMORY
#    include <timemory/timemory.hpp>
#else
#    include "profiler.hh"
#endif

#include "PTL/AutoLock.hh"
#include "PTL/Types.hh"
#include "PTL/Utility.hh"

#if defined(TOMOPY_USE_PTL)
#    include "PTL/TBBTask.hh"
#    include "PTL/TBBTaskGroup.hh"
#    include "PTL/Task.hh"
#    include "PTL/TaskGroup.hh"
#    include "PTL/TaskManager.hh"
#    include "PTL/TaskRunManager.hh"
#    include "PTL/ThreadData.hh"
#    include "PTL/ThreadPool.hh"
#    include "PTL/Threading.hh"
#endif

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
#    include <cooperative_groups.h>
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#    include <npp.h>
#    include <nppi.h>
#    include <thrust/device_vector.h>
#    include <thrust/execution_policy.h>
#    include <thrust/fill.h>
#    include <thrust/functional.h>
#    include <thrust/host_vector.h>
#    include <thrust/partition.h>
#    include <thrust/reduce.h>
#    include <thrust/sequence.h>
#    include <thrust/system/cpp/execution_policy.h>
#    include <thrust/system/cuda/execution_policy.h>
#    include <thrust/system/omp/execution_policy.h>
#    include <thrust/system/tbb/execution_policy.h>
#    include <thrust/transform.h>
#    include <vector_types.h>
#else
#    if !defined(cudaStream_t)
#        define cudaStream_t int
#    endif
#endif

#if defined(TOMOPY_USE_IPP)
#    include <ipp.h>
#    include <ippdefs.h>
#    include <ippi.h>
#endif

#if defined(TOMOPY_USE_OPENCV)
#    include <opencv2/highgui/highgui.hpp>
#    include <opencv2/imgcodecs.hpp>
#    include <opencv2/imgproc/imgproc.hpp>
#endif

//======================================================================================//

#if !defined(scast)
#    define scast static_cast
#endif

#define PRAGMA_SIMD _Pragma("omp simd")
#define PRAGMA_SIMD_REDUCTION(var) _Pragma("omp simd reducton(+ : var)")
#define HW_CONCURRENCY std::thread::hardware_concurrency()
#if !defined(_forward_args_t)
#    define _forward_args_t(_Args, _args) std::forward<_Args>(_args)...
#endif

#if defined(DEBUG)
#    define PRINT_MAX_ITER 1
#    define PRINT_MAX_SLICE 1
#    define PRINT_MAX_ANGLE 1
#    define PRINT_MAX_PIXEL 5
#else
#    define PRINT_MAX_ITER 0
#    define PRINT_MAX_SLICE 0
#    define PRINT_MAX_ANGLE 0
#    define PRINT_MAX_PIXEL 0
#endif

//======================================================================================//

namespace
{
constexpr float pi       = static_cast<float>(M_PI);
constexpr float halfpi   = 0.5f * pi;
constexpr float twopi    = 2.0f * pi;
constexpr float epsilonf = 2.0f * std::numeric_limits<float>::epsilon();
constexpr float degrees  = 180.0f / pi;
}

//======================================================================================//

typedef std::vector<float> farray_t;
typedef std::vector<int>   iarray_t;
template <typename _Tp>
using cuda_device_info = std::unordered_map<int, _Tp>;

//======================================================================================//

struct GpuOption
{
    int         index;
    std::string key;
    std::string description;

    static void spacer(std::ostream& os, const char c = '-')
    {
        std::stringstream ss;
        ss.fill(c);
        ss << std::setw(90) << ""
           << "\n";
        os << ss.str();
    }

    static void header(std::ostream& os)
    {
        std::stringstream ss;
        ss << "\n";
        spacer(ss, '=');
        ss << "Available GPU options:\n";
        ss << "\t" << std::left << std::setw(5) << "INDEX"
           << "  \t" << std::left << std::setw(12) << "KEY"
           << "  " << std::left << std::setw(40) << "DESCRIPTION"
           << "\n";
        os << ss.str();
    }

    static void footer(std::ostream& os)
    {
        std::stringstream ss;
        ss << "\nTo select an option for runtime, set TOMOPY_GPU_TYPE "
           << "environment variable\n  to an INDEX or KEY above\n";
        spacer(ss, '=');
        os << ss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const GpuOption& opt)
    {
        std::stringstream ss;
        ss << "\t" << std::right << std::setw(5) << opt.index << "  \t" << std::left
           << std::setw(12) << opt.key << "  " << std::left << std::setw(40)
           << opt.description;
        os << ss.str();
        return os;
    }
};

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

    cpu_rotate_data(int id, int dy, int dt, int dx, int nx, int ny, float* recon,
                    float* simdata, const float* data)
    : m_id(id)
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_size(m_nx * m_ny)
    , m_rot(farray_t(m_size, 0.0f))
    , m_tmp(farray_t(m_size, 0.0f))
    , m_recon(recon)
    , m_update(nullptr)
    , m_simdata(simdata)
    , m_data(data)
    {
    }

    ~cpu_rotate_data() {}

    farray_t&       rot() { return m_rot; }
    farray_t&       tmp() { return m_tmp; }
    const farray_t& rot() const { return m_rot; }
    const farray_t& tmp() const { return m_tmp; }
};

//======================================================================================//

inline uintmax_t
GetThisThreadID()
{
#if defined(TOMOPY_USE_PTL)
    return ThreadPool::GetThisThreadID();
#else
    static std::atomic<uintmax_t> tcounter;
    static thread_local auto      tid = tcounter++;
    return tid;
#endif
}

//======================================================================================//

#if !defined(PRINT_HERE)
#    define PRINT_HERE(extra)                                                            \
        printf("[%lu]> %s@'%s':%i %s\n", GetThisThreadID(), __FUNCTION__, __FILE__,      \
               __LINE__, extra)
#endif

//======================================================================================//

#if !defined(GPU_PRINT_HERE)
#    define GPU_PRINT_HERE(extra)                                                        \
        printf("[GPU]> %s@'%s':%i %s\n", __FUNCTION__, __FILE__, __LINE__, extra)
#endif

//======================================================================================//

#if !defined(START_TIMER)
#    define START_TIMER(var) auto var = std::chrono::system_clock::now()
#endif

//======================================================================================//

#if !defined(REPORT_TIMER)
#    define REPORT_TIMER(start_time, note, counter, total_count)                         \
        {                                                                                \
            auto                          end_time = std::chrono::system_clock::now();   \
            std::chrono::duration<double> elapsed_seconds = end_time - start_time;       \
            printf("[%li]> %-16s :: %3i of %3i... %5.2f seconds\n", GetThisThreadID(),   \
                   note, counter, total_count, elapsed_seconds.count());                 \
        }
#endif

//======================================================================================//

inline int&
this_thread_device()
{
#if defined(TOMOPY_USE_CUDA)
    static std::atomic<int> _ntid(0);
    ThreadLocalStatic int   _instance =
        (cuda_device_count() > 0) ? ((_ntid++) % cuda_device_count()) : 0;
    return _instance;
#else
    static thread_local int _instance = 0;
    return _instance;
#endif
}

//======================================================================================//

inline void
set_this_thread_device()
{
#if defined(TOMOPY_USE_CUDA)
    cuda_set_device(this_thread_device());
#endif
}

//======================================================================================//

template <typename _Tp>
_Tp*
cpu_malloc(uintmax_t size)
{
    _Tp* _cpu = (_Tp*) malloc(size * sizeof(_Tp));
    return _cpu;
}

//======================================================================================//

template <typename _Tp>
_Tp
from_string(const std::string& val)
{
    std::stringstream ss;
    _Tp               ret;
    ss << val;
    ss >> ret;
    return ret;
}

//======================================================================================//

inline std::string
tolower(std::string val)
{
    for(uintmax_t i = 0; i < val.size(); ++i)
        val[i] = tolower(val[i]);
    return val;
}

//======================================================================================//
#if defined(TOMOPY_USE_PTL)
inline void
init_thread_data(ThreadPool* tp)
{
    ThreadData*& thread_data = ThreadData::GetInstance();
    if(!thread_data)
        thread_data = new ThreadData(tp);
    thread_data->is_master   = false;
    thread_data->within_task = false;
}
#endif
//======================================================================================//

inline Mutex&
update_mutex()
{
    static Mutex _instance;
    return _instance;
}

//======================================================================================//
#if defined(TOMOPY_USE_PTL)
inline TaskRunManager*&
cpu_run_manager()
{
    AutoLock               l(TypeMutex<TaskRunManager>());
    static TaskRunManager* _instance =
        new TaskRunManager(GetEnv<bool>("TOMOPY_USE_TBB", false, "Enable TBB backend"));
    return _instance;
}

//======================================================================================//

inline TaskRunManager*&
gpu_run_manager()
{
    AutoLock               l(TypeMutex<TaskRunManager>());
    static TaskRunManager* _instance =
        new TaskRunManager(GetEnv<bool>("TOMOPY_USE_TBB", false, "Enable TBB backend"));
    return _instance;
}

//======================================================================================//

inline void
init_run_manager(TaskRunManager*& run_man, uintmax_t nthreads)
{
    auto tid = GetThisThreadID();
    ConsumeParameters(tid);

    {
        AutoLock l(TypeMutex<TaskRunManager>());
        if(!run_man->IsInitialized())
        {
            std::cout << "\n"
                      << "[" << tid << "] Initializing tasking run manager with "
                      << nthreads << " threads..." << std::endl;
            run_man->Initialize(nthreads);
        }
    }
    TaskManager* task_man = run_man->GetTaskManager();
    ThreadPool*  tp       = task_man->thread_pool();
    init_thread_data(tp);

    if(GetEnv<int>("TASKING_VERBOSE", 0) > 0)
    {
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << "> " << __FUNCTION__ << "@" << __LINE__ << " -- "
                  << "run manager = " << run_man << ", "
                  << "task manager = " << task_man << ", "
                  << "thread pool = " << tp << ", "
                  << "..." << std::endl;
    }
}
#endif
//======================================================================================//

template <typename _Func, typename... _Args>
void
run_gpu_algorithm(_Func cpu_func, _Func cuda_func, _Func acc_func, _Func omp_func,
                  _Args... args)
{
    std::deque<GpuOption> options;
    int                   default_idx = 0;
    std::string           default_key = "cpu";

#if defined(TOMOPY_USE_CUDA)
    options.push_back(GpuOption({ 1, "cuda", "Run with CUDA" }));
#endif

#if defined(TOMOPY_USE_OPENACC)
    options.push_back(GpuOption({ 2, "openacc", "Run with OpenACC" }));
#endif

#if defined(TOMOPY_USE_OPENMP)
    options.push_back(GpuOption({ 3, "openmp", "Run with OpenMP" }));
#endif

    //------------------------------------------------------------------------//
    auto print_options = [&]() {
        static bool first = true;
        if(!first)
            return;
        else
            first = false;

        std::stringstream ss;
        GpuOption::header(ss);
        for(const auto& itr : options)
            ss << itr << "\n";
        GpuOption::footer(ss);

        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << "\n" << ss.str() << std::endl;
    };
    //------------------------------------------------------------------------//
    auto print_selection = [&](GpuOption& selected_opt) {
        static bool first = true;
        if(!first)
            return;
        else
            first = false;

        std::stringstream ss;
        GpuOption::spacer(ss, '-');
        ss << "Selected device: " << selected_opt << "\n";
        GpuOption::spacer(ss, '-');

        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << ss.str() << std::endl;
    };
    //------------------------------------------------------------------------//

    // Run on CPU if nothing available
    if(options.size() == 0)
    {
        cpu_func(_forward_args_t(_Args, args));
        return;
    }

    // print the GPU execution type options
    print_options();

    default_idx = options.front().index;
    default_key = options.front().key;
    auto key    = GetEnv("TOMOPY_GPU_TYPE", default_key);

    int selection = default_idx;
    for(auto itr : options)
    {
        if(key == tolower(itr.key) || from_string<int>(key) == itr.index)
        {
            selection = itr.index;
            print_selection(itr);
        }
    }

    try
    {
        if(selection == 1)
        {
            cuda_func(_forward_args_t(_Args, args));
        }
        else if(selection == 2)
        {
            acc_func(_forward_args_t(_Args, args));
        }
        else if(selection == 3)
        {
            omp_func(_forward_args_t(_Args, args));
        }
    }
    catch(std::exception& e)
    {
        {
            AutoLock l(TypeMutex<decltype(std::cout)>());
            std::cerr << "[TID: " << GetThisThreadID() << "] " << e.what() << std::endl;
            std::cerr << "[TID: " << GetThisThreadID() << "] "
                      << "Falling back to CPU algorithm..." << std::endl;
        }
        cpu_func(_forward_args_t(_Args, args));
    }
}

//======================================================================================//
