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
#    include <opencv2/core.hpp>
#    include <opencv2/imgproc.hpp>
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

typedef std::vector<int16_t>  sarray_t;
typedef std::vector<uint16_t> usarray_t;
typedef std::vector<uint32_t> uarray_t;
typedef std::vector<int32_t>  iarray_t;
typedef std::vector<float>    farray_t;
typedef std::vector<double>   darray_t;

template <typename _Tp>
using array_t = std::vector<_Tp>;
template <typename _Tp>
using cuda_device_info = std::unordered_map<int, _Tp>;

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
    static thread_local int _instance =
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
    auto devid = this_thread_device();
    auto thrid = GetThisThreadID();
    cuda_set_device(devid);
    printf("[%lu] Running on GPU %i\n", thrid, devid);
#endif
}

//======================================================================================//

template <typename _Tp>
_Tp*
cpu_malloc(uintmax_t size)
{
    void* _cpu = malloc(size * sizeof(_Tp));
    return scast<_Tp*>(_cpu);
}

//======================================================================================//

inline bool
is_numeric(const std::string& val)
{
    if(val.length() > 0)
    {
        auto f = val.find_first_of("0123456789");
        if(f == std::string::npos)  // no numbers
            return false;
        auto l = val.find_last_of("0123456789");
        if(val.length() <= 2)  // 1, 2., etc.
            return true;
        else
            return (f != l);  // 1.0, 1e3, 23, etc.
    }
    return false;
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
    for(auto& itr : val)
        itr = scast<char>(tolower(itr));
    return val;
}

//======================================================================================//

inline Mutex&
update_mutex()
{
    static Mutex _instance;
    return _instance;
}

//======================================================================================//
//======================================================================================//
#if defined(TOMOPY_USE_PTL)
//======================================================================================//
//======================================================================================//

inline void
init_thread_data(ThreadPool* tp)
{
    ThreadData*& thread_data = ThreadData::GetInstance();
    if(!thread_data)
        thread_data = new ThreadData(tp);
    thread_data->is_master   = false;
    thread_data->within_task = false;
}

//======================================================================================//

inline TaskRunManager*
cpu_run_manager()
{
    AutoLock l(TypeMutex<TaskRunManager>());
    // typedef std::shared_ptr<TaskRunManager> run_man_ptr;
    static TaskRunManager* _instance =
        new TaskRunManager(GetEnv<bool>("TOMOPY_USE_TBB", false, "Enable TBB backend"));
    return _instance;
}

//======================================================================================//

inline TaskRunManager*
gpu_run_manager()
{
    AutoLock                                l(TypeMutex<TaskRunManager>());
    typedef std::shared_ptr<TaskRunManager> pointer;
    static thread_local pointer             _instance = pointer(
        new TaskRunManager(GetEnv<bool>("TOMOPY_USE_TBB", false, "Enable TBB backend")));
    return _instance.get();
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
//======================================================================================//
//======================================================================================//
#endif  // TOMOPY_USE_PTL
//======================================================================================//
//======================================================================================//

struct cpu_rotate_data
{
    typedef iarray_t                         iarray_type;
    typedef typename iarray_type::value_type int_type;

    uintmax_t    m_id;
    int          m_dy;
    int          m_dt;
    int          m_dx;
    int          m_nx;
    int          m_ny;
    iarray_type  m_use_rot;
    iarray_type  m_use_tmp;
    farray_t     m_rot;
    farray_t     m_tmp;
    float*       m_recon;
    float*       m_update;
    float*       m_simdata;
    const float* m_data;

    cpu_rotate_data(uintmax_t id, int dy, int dt, int dx, int nx, int ny,
                    const float* data, float* recon, float* simdata)
    : m_id(id)
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_use_rot(iarray_type(scast<uintmax_t>(m_nx * m_ny), 0))
    , m_use_tmp(iarray_type(scast<uintmax_t>(m_nx * m_ny), 1))
    , m_rot(farray_t(scast<uintmax_t>(m_nx * m_ny), 0.0f))
    , m_tmp(farray_t(scast<uintmax_t>(m_nx * m_ny), 0.0f))
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

    iarray_type&       use_rot() { return m_use_rot; }
    iarray_type&       use_tmp() { return m_use_tmp; }
    const iarray_type& use_rot() const { return m_use_rot; }
    const iarray_type& use_tmp() const { return m_use_tmp; }

    void reset()
    {
        // reset temporaries to zero
        memset(m_use_rot.data(), 0, scast<uintmax_t>(m_nx * m_ny) * sizeof(int_type));
        memset(m_rot.data(), 0, scast<uintmax_t>(m_nx * m_ny) * sizeof(float));
        memset(m_tmp.data(), 0, scast<uintmax_t>(m_nx * m_ny) * sizeof(float));
    }
};

//======================================================================================//

struct DeviceOption
{
    typedef std::string     string_t;
    typedef const string_t& crstring_t;

    int      index;
    string_t key;
    string_t description;

    DeviceOption(const int& _idx, crstring_t _key, crstring_t _desc)
    : index(_idx)
    , key(_key)
    , description(_desc)
    {
    }

    static void spacer(std::ostream& os, const char c = '-')
    {
        std::stringstream ss;
        ss.fill(c);
        ss << std::setw(90) << ""
           << "\n";
        os << ss.str();
    }

    friend bool operator==(const DeviceOption& lhs, const DeviceOption& rhs)
    {
        return (lhs.key == rhs.key && lhs.index == rhs.index);
    }

    friend bool operator==(const DeviceOption& itr, crstring_t cmp)
    {
        return (!is_numeric(cmp)) ? (itr.key == tolower(cmp))
                                  : (itr.index == from_string<int>(cmp));
    }

    friend bool operator!=(const DeviceOption& lhs, const DeviceOption& rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator!=(const DeviceOption& itr, crstring_t cmp)
    {
        return !(itr == cmp);
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
        ss << "\nTo select an option for runtime, set TOMOPY_DEVICE_TYPE "
           << "environment variable\n  to an INDEX or KEY above\n";
        spacer(ss, '=');
        os << ss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const DeviceOption& opt)
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

template <typename _Func, typename... _Args>
void
run_algorithm(_Func cpu_func, _Func cuda_func, _Func acc_func, _Func omp_func,
              _Args... args)
{
    bool use_cpu = GetEnv<bool>("TOMOPY_USE_CPU", false);
    if(use_cpu)
    {
        try
        {
            cpu_func(_forward_args_t(_Args, args));
        }
        catch(const std::exception& e)
        {
            AutoLock l(TypeMutex<decltype(std::cout)>());
            std::cerr << e.what() << '\n';
        }
        return;
    }

    std::deque<DeviceOption> options;
    options.push_back(DeviceOption(0, "cpu", "Run on CPU"));

#if defined(TOMOPY_USE_GPU)
#    if defined(TOMOPY_USE_CUDA)
    options.push_back(DeviceOption(1, "cuda", "Run on GPU with CUDA"));
#    endif

#    if defined(TOMOPY_USE_OPENACC)
    options.push_back(DeviceOption(2, "openacc", "Run on GPU with OpenACC"));
#    endif

#    if defined(TOMOPY_USE_OPENMP)
    options.push_back(DeviceOption(3, "openmp", "Run on GPU with OpenMP"));
#    endif
#endif

#if defined(TOMOPY_USE_GPU) && defined(TOMOPY_USE_CUDA)
    std::string default_key = "cuda";
#else
    std::string default_key = "cpu";
#endif

    auto default_itr =
        std::find_if(options.begin(), options.end(),
                     [&](const DeviceOption& itr) { return (itr == default_key); });

    //------------------------------------------------------------------------//
    auto print_options = [&]() {
        static bool first = true;
        if(!first)
            return;
        else
            first = false;

        std::stringstream ss;
        DeviceOption::header(ss);
        for(const auto& itr : options)
        {
            ss << itr;
            if(itr == *default_itr)
                ss << "\t(default)";
            ss << "\n";
        }
        DeviceOption::footer(ss);

        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << "\n" << ss.str() << std::endl;
    };
    //------------------------------------------------------------------------//
    auto print_selection = [&](DeviceOption& selected_opt) {
        static bool first = true;
        if(!first)
            return;
        else
            first = false;

        std::stringstream ss;
        DeviceOption::spacer(ss, '-');
        ss << "Selected device: " << selected_opt << "\n";
        DeviceOption::spacer(ss, '-');

        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << ss.str() << std::endl;
    };
    //------------------------------------------------------------------------//

    // Run on CPU if nothing available
    if(options.size() <= 1)
    {
        cpu_func(_forward_args_t(_Args, args));
        return;
    }

    // print the GPU execution type options
    print_options();

    default_key = default_itr->key;
    auto key    = GetEnv("TOMOPY_DEVICE_TYPE", default_key);

    auto selection = std::find_if(options.begin(), options.end(),
                                  [&](const DeviceOption& itr) { return (itr == key); });

    if(selection == options.end())
        selection =
            std::find_if(options.begin(), options.end(),
                         [&](const DeviceOption& itr) { return itr == default_key; });

    print_selection(*selection);

    try
    {
        switch(selection->index)
        {
            case 0: cpu_func(_forward_args_t(_Args, args)); break;
            case 1: cuda_func(_forward_args_t(_Args, args)); break;
            case 2: acc_func(_forward_args_t(_Args, args)); break;
            case 3: omp_func(_forward_args_t(_Args, args)); break;
            default: cpu_func(_forward_args_t(_Args, args)); break;
        }
    }
    catch(std::exception& e)
    {
        if(selection != options.end() && selection->index != 0)
        {
            {
                AutoLock l(TypeMutex<decltype(std::cout)>());
                std::cerr << "[TID: " << GetThisThreadID() << "] " << e.what()
                          << std::endl;
                std::cerr << "[TID: " << GetThisThreadID() << "] "
                          << "Falling back to CPU algorithm..." << std::endl;
            }
            cpu_func(_forward_args_t(_Args, args));
        }
    }
}

//======================================================================================//

template <typename _Executor, typename _DataArray, typename _Func, typename... _Args>
void
execute(_Executor* man, int dy, int dt, _DataArray& data, _Func& func, _Args... args)
{
    // does nothing except make sure there is no warning
    ConsumeParameters(man);

    // Loop over slices and projection angles
    auto serial_exec = [&]() {
        uintmax_t idx = 0;
        // Loop over slices and projection angles
        for(int p = 0; p < dt; ++p)
            for(int s = 0; s < dy; ++s)
            {
                auto cache = data[(idx++) % data.size()];
                func(cache, s, p, args...);
            }
    };

    auto parallel_exec = [&]() {
#if defined(TOMOPY_USE_PTL)
        if(!man)
            return false;
        uintmax_t       idx = 0;
        TaskGroup<void> tg;
        for(int p = 0; p < dt; ++p)
            for(int s = 0; s < dy; ++s)
            {
                auto cache = data[(idx++) % data.size()];
                man->exec(tg, func, cache, s, p, args...);
            }
        tg.join();
        return true;
#else
        return false;
#endif
    };

    try
    {
        // if parallel execution fails, run serial
        if(!parallel_exec())
            serial_exec();
    }
    catch(const std::exception& e)
    {
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cerr << e.what() << '\n';
    }
}

//======================================================================================//
