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
//  The following section provides functions for the initialization of the tasking library
//
//======================================================================================//

// get a unique pointer to a thread-pool
//
inline void
CreateThreadPool(unique_thread_pool_t& tp, num_threads_t& pool_size)
{
    auto min_threads = num_threads_t(1);
    if(pool_size <= 0)
    {
#if defined(TOMOPY_USE_PTL)
        // compute some properties (expected python threads, max threads)
        auto pythreads = GetEnv("TOMOPY_PYTHON_THREADS", HW_CONCURRENCY);
#    if defined(TOMOPY_USE_CUDA)
        // general oversubscription when CUDA is enabled
        auto max_threads =
            (HW_CONCURRENCY + HW_CONCURRENCY) / std::max(pythreads, min_threads);
#    else
        // if known that CPU only, just try to use all cores
        auto max_threads = HW_CONCURRENCY / std::max(pythreads, min_threads);
#    endif
        auto nthreads = std::max(GetEnv("TOMOPY_NUM_THREADS", max_threads), min_threads);
        pool_size     = nthreads;
#else
        pool_size = 1;
#endif
    }
    // always specify at least one thread even if not creating threads
    pool_size = std::max(pool_size, min_threads);

    // explicitly set number of threads to 0 so OpenCV doesn't try to create threads
    cv::setNumThreads(0);

    // use unique pointer per-thread so manager gets deleted when thread gets deleted
    // create the thread-pool instance
    tp = unique_thread_pool_t(new tomopy::ThreadPool(pool_size));

#if defined(TOMOPY_USE_PTL)
    // ensure this thread is assigned id, assign variable so no unused result warning
    auto tid = GetThisThreadID();

    // initialize the thread-local data information
    auto& thread_data = ThreadData::GetInstance();
    if(!thread_data)
        thread_data.reset(new ThreadData(tp.get()));

    // tell thread that initialized thread-pool to process tasks
    // (typically master thread will only wait for other threads)
    thread_data->is_master = true;

    // tell thread that it is not currently within task
    thread_data->within_task = false;

    // notify
    AutoLock l(TypeMutex<decltype(std::cout)>());
    std::cout << "\n"
              << "[" << tid << "] Initialized tasking run manager with " << tp->size()
              << " threads..." << std::endl;
#endif
}

//======================================================================================//
//  This class enables the selection of a device at runtime
//
struct DeviceOption
{
public:
    using string_t       = std::string;
    int      index       = -1;
    string_t key         = "";
    string_t description = "";

    DeviceOption() {}

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
        ss << "\nTo select an option for runtime, set 'device' parameter to an "
           << "INDEX or KEY above\n";
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
// this function selects the device to run the reconstruction on
//

inline DeviceOption
GetDevice(const std::string& preferred)
{
    auto pythreads               = GetEnv("TOMOPY_PYTHON_THREADS", HW_CONCURRENCY);
    using DeviceOptionList       = std::deque<DeviceOption>;
    DeviceOptionList options     = { DeviceOption(0, "cpu", "Run on CPU (OpenCV)") };
    std::string      default_key = "cpu";

#if defined(TOMOPY_USE_CUDA)
    auto num_devices = cuda_device_count();
    if(num_devices > 0)
    {
        options.push_back(DeviceOption(1, "gpu", "Run on GPU (CUDA NPP)"));
        default_key = "gpu";
#    if defined(TOMOPY_USE_NVTX)
        // initialize nvtx data
        init_nvtx();
#    endif
        // print device info
        cuda_device_query();
    }
    else
    {
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cerr << "\n##### No CUDA device(s) available #####\n" << std::endl;
    }
#endif

    // find the default entry
    auto default_itr =
        std::find_if(options.begin(), options.end(),
                     [&](const DeviceOption& itr) { return (itr == default_key); });

    //------------------------------------------------------------------------//
    // print the options the first time it is encountered
    auto print_options = [&]() {
        static std::atomic_uint _once;
        auto                    _count = _once++;
        if(_count % pythreads > 0)
        {
            if(_count + 1 == pythreads)
            {
                _once.store(0);
            }
            return;
        }

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
    // print the option selection first time it is encountered
    auto print_selection = [&](DeviceOption& selected_opt) {
        static std::atomic_uint _once;
        auto                    _count = _once++;
        if(_count % pythreads > 0)
        {
            if(_count + 1 == pythreads)
            {
                _once.store(0);
            }
            return;
        }

        std::stringstream ss;
        DeviceOption::spacer(ss, '-');
        ss << "Selected device: " << selected_opt << "\n";
        DeviceOption::spacer(ss, '-');

        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << ss.str() << std::endl;
    };
    //------------------------------------------------------------------------//

    // print the GPU execution type options
    print_options();

    default_key = default_itr->key;
    auto key    = preferred;

    auto selection = std::find_if(options.begin(), options.end(),
                                  [&](const DeviceOption& itr) { return (itr == key); });

    if(selection == options.end())
        selection =
            std::find_if(options.begin(), options.end(),
                         [&](const DeviceOption& itr) { return itr == default_key; });

    print_selection(*selection);

    return *selection;
}

//======================================================================================//
// synchronize a CUDA stream
inline void
stream_sync(cudaStream_t _stream)
{
#if defined(__NVCC__) && defined(TOMOPY_USE_CUDA)
    cudaStreamSynchronize(_stream);
    CUDA_CHECK_LAST_STREAM_ERROR(_stream);
#else
    ConsumeParameters(_stream);
#endif
}

//======================================================================================//

// function for printing an array
//
template <typename _Tp, std::size_t _N>
std::ostream&
operator<<(std::ostream& os, const std::array<_Tp, _N>& arr)
{
    std::stringstream ss;
    ss.setf(os.flags());
    for(std::size_t i = 0; i < _N; ++i)
    {
        ss << arr[i];
        if(i + 1 < _N)
        {
            ss << ", ";
        }
    }
    os << ss.str();
    return os;
}

//======================================================================================//
// for generic printing operations in a clean fashion
//
namespace internal
{
//
//----------------------------------------------------------------------------------//
/// Alias template for enable_if
template <bool B, typename T>
using enable_if_t = typename std::enable_if<B, T>::type;

/// Alias template for decay
template <class T>
using decay_t = typename std::decay<T>::type;

struct apply_impl
{
    //----------------------------------------------------------------------------------//
    //  end of recursive expansion
    //
    template <std::size_t _N, std::size_t _Nt, typename _Operator, typename _TupleA,
              typename _TupleB, typename... _Args, enable_if_t<(_N == _Nt), int> = 0>
    static void unroll(_TupleA&& _tupA, _TupleB&& _tupB, _Args&&... _args)
    {
        // call constructor
        using TypeA        = decltype(std::get<_N>(_tupA));
        using TypeB        = decltype(std::get<_N>(_tupB));
        using OperatorType = typename std::tuple_element<_N, _Operator>::type;
        OperatorType(std::forward<TypeA>(std::get<_N>(_tupA)),
                     std::forward<TypeB>(std::get<_N>(_tupB)),
                     std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  recursive expansion until _N == _Nt
    //
    template <std::size_t _N, std::size_t _Nt, typename _Operator, typename _TupleA,
              typename _TupleB, typename... _Args, enable_if_t<(_N < _Nt), int> = 0>
    static void unroll(_TupleA&& _tupA, _TupleB&& _tupB, _Args&&... _args)
    {
        // call constructor
        using TypeA        = decltype(std::get<_N>(_tupA));
        using TypeB        = decltype(std::get<_N>(_tupB));
        using OperatorType = typename std::tuple_element<_N, _Operator>::type;
        OperatorType(std::forward<TypeA>(std::get<_N>(_tupA)),
                     std::forward<TypeB>(std::get<_N>(_tupB)),
                     std::forward<_Args>(_args)...);
        // recursive call
        unroll<_N + 1, _Nt, _Operator, _TupleA, _TupleB, _Args...>(
            std::forward<_TupleA>(_tupA), std::forward<_TupleB>(_tupB),
            std::forward<_Args>(_args)...);
    }
};

//======================================================================================//

struct apply
{
    //----------------------------------------------------------------------------------//
    // invoke the recursive expansion
    template <typename _Operator, typename _TupleA, typename _TupleB, typename... _Args,
              std::size_t _N  = std::tuple_size<decay_t<_TupleA>>::value,
              std::size_t _Nb = std::tuple_size<decay_t<_TupleB>>::value>
    static void unroll(_TupleA&& _tupA, _TupleB&& _tupB, _Args&&... _args)
    {
        static_assert(_N == _Nb, "tuple_size A must match tuple_size B");
        apply_impl::template unroll<0, _N - 1, _Operator, _TupleA, _TupleB, _Args...>(
            std::forward<_TupleA>(_tupA), std::forward<_TupleB>(_tupB),
            std::forward<_Args>(_args)...);
    }
};

//--------------------------------------------------------------------------------------//
// generic operator for printing
//
template <typename Type>
struct GenericPrinter
{
    GenericPrinter(const std::string& _prefix, const Type& obj, std::ostream& os,
                   intmax_t _prefix_width, intmax_t _obj_width,
                   std::ios_base::fmtflags format_flags, bool endline)
    {
        std::stringstream ss;
        ss.setf(format_flags);
        ss << std::setw(_prefix_width) << std::right << _prefix << " = "
           << std::setw(_obj_width) << obj;
        if(endline)
            ss << std::endl;
        os << ss.str();
    }
};

}  // end namespace internal
