// Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.

// Copyright 2015. UChicago Argonne, LLC. This software was produced
// under U.S. Government contract DE-AC02-06CH11357 for Argonne National
// Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
// U.S. Department of Energy. The U.S. Government has rights to use,
// reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
// UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
// ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
// modified to produce derivative works, such modified software should
// be clearly marked, so as not to confuse it with the version available
// from ANL.

// Additionally, redistribution and use in source and binary forms, with
// or without modification, are permitted provided that the following
// conditions are met:

//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.

//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in
//       the documentation and/or other materials provided with the
//       distribution.

//     * Neither the name of UChicago Argonne, LLC, Argonne National
//       Laboratory, ANL, the U.S. Government, nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
// Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//
// ---------------------------------------------------------------
//  TOMOPY class header
//

#ifndef libtomopy_hpp_
#define libtomopy_hpp_

//============================================================================//
// base operating system

#if defined(_WIN32) || defined(_WIN64)
#    if !defined(_WINDOWS)
#        define _WINDOWS
#    endif
//----------------------------------------------------------------------------//

#elif defined(__APPLE__) || defined(__MACH__)
#    if !defined(_MACOS)
#        define _MACOS
#    endif
#    if !defined(_UNIX)
#        define _UNIX
#    endif
//----------------------------------------------------------------------------//

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#    if !defined(_LINUX)
#        define _LINUX
#    endif
#    if !defined(_UNIX)
#        define _UNIX
#    endif
//----------------------------------------------------------------------------//

#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(_)
#    if !defined(_UNIX)
#        define _UNIX
#    endif
#endif

#if !defined(DEFAULT_UMASK)
#    define DEFAULT_UMASK 0777
#endif

//============================================================================//

// C library
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <stdlib.h>
// I/O
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
// general
#include <chrono>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
// container
#include <deque>
#include <map>
#include <set>
#include <vector>
// threading
#include <atomic>
#include <future>
#include <mutex>
#include <thread>

#if defined(_UNIX)
#    include <errno.h>
#    include <stdio.h>
#    include <string.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#elif defined(_WINDOWS)
#    include <direct.h>
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#endif

#ifndef EXTERN_C_
#    define EXTERN_C_                                                                    \
        extern "C"                                                                       \
        {
#endif

#ifndef _EXTERN_C
#    define _EXTERN_C }
#endif

//============================================================================//
//  pybind11 includes
//
#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

//============================================================================//
//  tomopy includes
//
EXTERN_C_
#include "morph.h"
#include "prep.h"
#include "remove_ring.h"
#include "stripe.h"
#include "utils.h"
_EXTERN_C
#include "utils.hh"

//============================================================================//
//  tasking includes
//
#include "PTL/TBBTask.hh"
#include "PTL/TBBTaskGroup.hh"
#include "PTL/Task.hh"
#include "PTL/TaskGroup.hh"
#include "PTL/TaskManager.hh"
#include "PTL/TaskRunManager.hh"
#include "PTL/ThreadData.hh"
#include "PTL/ThreadPool.hh"
#include "PTL/Threading.hh"
#include "PTL/Types.hh"
#include "profiler.hh"

//============================================================================//

namespace py = pybind11;
using namespace py::literals;
using namespace std::placeholders;  // for _1, _2, _3...

//============================================================================//

// define helper macros
#define pyobj_cast(_var, _type, _pyobject) _type* _var = _pyobject.cast<_type*>()

//============================================================================//

class TaskRunManagerWrapper
{
public:
    TaskRunManagerWrapper()
    : _manager(TaskRunManager::GetInstance())
    {
    }

    ~TaskRunManagerWrapper() {}

    inline TaskRunManager* get() const { return _manager; }
    inline void            Initialize(uint64_t n = 0) { _manager->Initialize(n); }
    inline void            Terminate() { _manager->Terminate(); }
    inline void            Wait() { _manager->Wait(); }
    inline bool            IsInitialized() const { return _manager->IsInitialized(); }
    inline size_t GetNumberOfThreads() const { return _manager->GetNumberOfThreads(); }
    inline ThreadPool*  GetThreadPool() const { return _manager->GetThreadPool(); }
    inline TaskManager* GetTaskManager() const { return _manager->GetTaskManager(); }
    inline void         TiMemoryReport(std::string fname, bool echo = true) const
    {
        _manager->TiMemoryReport(fname, echo);
    }

private:
    TaskRunManager* _manager;
};

//============================================================================//

// undefine helper macros
#undef pyobj_cast

//============================================================================//

#endif
