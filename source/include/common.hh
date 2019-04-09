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

#pragma once

#include "constants.hh"
#include "macros.hh"
#include "typedefs.hh"

BEGIN_EXTERN_C
#include "cxx_extern.h"
END_EXTERN_C

//======================================================================================//
//
//      NVTX macros
//
//======================================================================================//

#if defined(TOMOPY_USE_NVTX)
#    include <nvToolsExt.h>

#    ifndef NVTX_RANGE_PUSH
#        define NVTX_RANGE_PUSH(obj) nvtxRangePushEx(obj)
#    endif
#    ifndef NVTX_RANGE_POP
#        define NVTX_RANGE_POP(obj)                                                      \
            cudaStreamSynchronize(obj);                                                  \
            nvtxRangePop()
#    endif
#    ifndef NVTX_NAME_THREAD
#        define NVTX_NAME_THREAD(num, name) nvtxNameOsThread(num, name)
#    endif
#    ifndef NVTX_MARK
#        define NVTX_MARK(msg) nvtxMark(name)
#    endif

static void
init_nvtx();

#else
#    ifndef NVTX_RANGE_PUSH
#        define NVTX_RANGE_PUSH(obj)
#    endif
#    ifndef NVTX_RANGE_POP
#        define NVTX_RANGE_POP(obj)
#    endif
#    ifndef NVTX_NAME_THREAD
#        define NVTX_NAME_THREAD(num, name)
#    endif
#    ifndef NVTX_MARK
#        define NVTX_MARK(msg)
#    endif

inline void
init_nvtx()
{
}

#endif

//======================================================================================//
//
//  The following section provides functions for the initialization of the tasking library
//
//======================================================================================//

inline num_threads_t
GetNumThreads()
{
#if defined(TOMOPY_USE_PTL)
    // compute some properties (expected python threads, max threads, device assignment)
    static auto min_threads = num_threads_t(1);
    static auto pythreads   = GetEnv("TOMOPY_PYTHON_THREADS", HW_CONCURRENCY);
    static auto max_threads = HW_CONCURRENCY / std::max(pythreads, min_threads);
    static auto nthreads =
        std::max(GetEnv("TOMOPY_NUM_THREADS", max_threads), min_threads);
    return nthreads;
#else
    return 1;
#endif
}

//======================================================================================//

inline ThreadPool*
GetThreadPool(intmax_t num_threads = GetNumThreads())
{
#if defined(TOMOPY_USE_PTL)
    // use shared pointer so manager gets deleted when thread gets deleted
    typedef std::unique_ptr<ThreadPool> pointer;
    // first argument ensures we do not use TBB backend to PTL
    static thread_local pointer _instance = nullptr;
    if(!_instance)
    {
        // ensure this thread is assigned id, assign variable so no unused result warning
        auto tid = GetThisThreadID();
        // create the thread-pool
        _instance.reset(new ThreadPool(num_threads));
        // initialize the thread-local data information
        // ThreadData*& thread_data = ThreadData::GetInstance();
        // if(!thread_data)
        //    thread_data = new ThreadData(tp);
        // tell thread that initialized thread-pool to process tasks
        // (typically master thread will only wait for other threads)
        // thread_data->is_master = true;
        // tell thread that it is not currently within task
        // thread_data->within_task = false;
        // notify
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << "\n"
                  << "[" << tid << "] Initialized tasking run manager with "
                  << _instance->size() << " threads..." << std::endl;
    }
    // return pointer
    return _instance.get();
#else
    return nullptr;
#endif
}

//======================================================================================//

struct DeviceOption
{
    //
    //  This class enables the selection of a device at runtime
    //
public:
    using string_t = std::string;

    int      index;
    string_t key;
    string_t description;

    DeviceOption(const int& _idx, const string_t& _key, const string_t& _desc)
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

    friend bool operator==(const DeviceOption& itr, const string_t& cmp)
    {
        return (!is_numeric(cmp)) ? (itr.key == tolower(cmp))
                                  : (itr.index == from_string<int>(cmp));
    }

    friend bool operator!=(const DeviceOption& lhs, const DeviceOption& rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator!=(const DeviceOption& itr, const string_t& cmp)
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

    // helper function for converting to lower-case
    inline static std::string tolower(std::string val)
    {
        for(auto& itr : val)
            itr = scast<char>(::tolower(itr));
        return val;
    }

    // helper function to convert string to another type
    template <typename _Tp>
    static _Tp from_string(const std::string& val)
    {
        std::stringstream ss;
        _Tp               ret;
        ss << val;
        ss >> ret;
        return ret;
    }

    // helper function to determine if numeric represented as string
    inline static bool is_numeric(const std::string& val)
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
};

//======================================================================================//

template <typename _Func, typename... _Args>
void
run_algorithm(_Func&& cpu_func, _Func&& cuda_func, _Args&&... args)
{
    bool use_cpu = GetEnv<bool>("TOMOPY_USE_CPU", false);

    // explicitly set number of threads to 1 so OpenCV doesn't try to create threads
    cv::setNumThreads(1);

    // compute number of threads
    auto nthreads = GetNumThreads();
    // initialize thread-pool
    auto* tp = GetThreadPool(nthreads);
    // no warning about unused variable
    ConsumeParameters(tp);

    if(use_cpu)
    {
        try
        {
            cpu_func(std::forward<_Args>(args)...);
        }
        catch(const std::exception& e)
        {
            AutoLock l(TypeMutex<decltype(std::cout)>());
            std::cerr << e.what() << '\n';
        }
        return;
    }

    std::deque<DeviceOption> options;
    options.push_back(DeviceOption(0, "cpu", "Run on CPU (OpenCV)"));
    std::string default_key = "cpu";

#if defined(TOMOPY_USE_CUDA)
    auto num_devices = cuda_device_count();
    if(num_devices > 0)
    {
        options.push_back(DeviceOption(1, "gpu", "Run on GPU (CUDA NPP)"));
        default_key = "gpu";
        // initialize nvtx data
        init_nvtx();
        // print device info
        cuda_device_query();
    }
    else
    {
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cerr << "##### No CUDA device(s) available #####" << std::endl;
    }
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
        cpu_func(std::forward<_Args>(args)...);
        return;
    }

    // print the GPU execution type options
    print_options();

    default_key = default_itr->key;
    auto key    = GetEnv("TOMOPY_DEVICE", default_key);

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
            case 0: cpu_func(std::forward<_Args>(args)...); break;
            case 1: cuda_func(std::forward<_Args>(args)...); break;
            default: cpu_func(std::forward<_Args>(args)...); break;
        }
    }
    catch(std::exception& e)
    {
        // typically reached here if no GPU devices available
        if(selection != options.end() && selection->index != 0)
        {
            {
                AutoLock l(TypeMutex<decltype(std::cout)>());
                std::cerr << "[TID: " << GetThisThreadID() << "] " << e.what()
                          << std::endl;
                std::cerr << "[TID: " << GetThisThreadID() << "] "
                          << "Falling back to CPU algorithm..." << std::endl;
            }
            try
            {
                cpu_func(std::forward<_Args>(args)...);
            }
            catch(std::exception& _e)
            {
                std::stringstream ss;
                ss << "\n\nError executing :: " << _e.what() << "\n\n";
                {
                    AutoLock l(TypeMutex<decltype(std::cout)>());
                    std::cerr << _e.what() << std::endl;
                }
                throw std::runtime_error(ss.str().c_str());
            }
        }
    }
}

//======================================================================================//

template <typename DataArray, typename Func, typename... Args>
void
execute(int dt, DataArray& data, Func&& func, Args&&... args)
{
    // get the thread pool
    auto* tp = GetThreadPool();
    // does nothing except make sure there is no warning
    ConsumeParameters(tp);

    // Loop over slices and projection angles
    auto serial_exec = [&]() {
        // Loop over slices and projection angles
        for(int p = 0; p < dt; ++p)
        {
            auto _func = std::bind(std::forward<Func>(func), std::ref(data),
                                   std::forward<int>(p), std::forward<Args>(args)...);
            _func();
        }
    };

    auto parallel_exec = [&]() {
#if defined(TOMOPY_USE_PTL)
        if(!tp)
            return false;
        TaskGroup<void> tg(tp);
        for(int p = 0; p < dt; ++p)
        {
            auto _func = std::bind(std::forward<Func>(func), std::ref(data),
                                   std::forward<int>(p), std::forward<Args>(args)...);
            tg.run(_func);
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
        std::stringstream ss;
        ss << "\n\nError executing :: " << e.what() << "\n\n";
        {
            AutoLock l(TypeMutex<decltype(std::cout)>());
            std::cerr << e.what() << std::endl;
        }
        throw std::runtime_error(ss.str().c_str());
    }
}

//======================================================================================//
