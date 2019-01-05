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
//  TOMOPY python binding implementation
//

#include "tomocxx.hpp"
#include "PTL/AutoLock.hh"
#include "PTL/Threading.hh"

#ifdef USE_TIMEMORY
#    include <timemory/signal_detection.hpp>
#    include <timemory/timemory.hpp>
#endif

EXTERN_C_
#include "gridrec.h"
_EXTERN_C

//============================================================================//

typedef py::array_t<float, py::array::c_style | py::array::forcecast>
    pyfarray_t;

//============================================================================//

int
fibonacci(int n)
{
    return (n < 2) ? 1 : fibonacci(n - 1) + fibonacci(n - 2);
}

//============================================================================//

template <typename _Tp, typename _Arg = _Tp> class TaskGroupWrapper
{
public:
    typedef TaskGroup<_Tp, _Arg> task_group_type;

    TaskGroupWrapper()
    : m_task_group(task_group_type())
    {
    }

    ~TaskGroupWrapper() {}

    task_group_type* get() { return &m_task_group; }

    void join() { m_task_group.join(); }

private:
    task_group_type m_task_group;
};

//============================================================================//

PYBIND11_MODULE(tomocxx, tomo)
{
    // py::add_ostream_redirect(tomo, "ostream_redirect");

    auto run_fib = [=](int n, int nitr) {
        TaskRunManager* rm = TaskRunManager::GetInstance();
        if(!rm->IsInitialized())
            rm->Initialize();
        std::cout << "getting task manager..." << std::endl;
        TaskManager* tm = rm->GetTaskManager();
        std::cout << "creating task group..." << std::endl;
        TaskGroup<int> tg([](int& ref, int i) {
            ref += i;
            return ref;
        });
        std::cout << "executing..." << std::endl;
        for(auto i = 0; i < nitr; ++i)
            tm->exec(tg, fibonacci, n);
        std::cout << "joining..." << std::endl;
        int result = tg.join();
        return result;
    };

    tomo.def("run", run_fib, "Run fibonacci");
    tomo.def("fibonacci", &fibonacci, "Run fibonacci");
    // tomo.def("normalize_bg", &normalize_bg, "Normalize background");
    // tomo.def("remove_stripe_sf", &remove_stripe_sf, "Remove stripe");
    // tomo.def("project", &project, "Project 1D");
    // tomo.def("project2", &project2, "Project 2D");
    // tomo.def("project3", &project3, "Project 3D");
    // tomo.def("sample", &sample, "Sample");
    // tomo.def("art", &art, "Art reconstruction algorithm");
    // tomo.def("bart", &bart, "Bart reconstruction algorithm");
    // tomo.def("fbp", &fbp, "Filtered back projection reconstruction
    // algorithm"); tomo.def("gridrec", &gridrec, "Gridrec reconstruction
    // algorithm"); tomo.def("mlem", &mlem, "mlem reconstruction algorithm");
    // tomo.def("osem", &osem, "osem reconstruction algorithm");
    // tomo.def("ospml_hybrid", &ospml_hybrid,
    //         "ospml hybrid reconstruction algorithm");
    // tomo.def("ospml_quad", &ospml_quad, "ospml quad reconstruction
    // algorithm"); tomo.def("pml_hybrid", &pml_hybrid, "pml hybrid
    // reconstruction algorithm"); tomo.def("pml_quad", &pml_quad, "pml quad
    // reconstruction algorithm"); tomo.def("sirt", &sirt, "sirt reconstruction
    // algorithm"); tomo.def("vector", &vector, "vector 1D");
    // tomo.def("vector2", &vector2, "vector 2D"); tomo.def("vector3", &vector3,
    // "vector 3D"); tomo.def("remove_ring", &remove_ring, "remove ring");
    // tomo.def("art_task", art_task, "Art reconstruction algorithm");

    // py::class_<TaskGroupWrapper<void>> task_group(tomo, "task_group");
    // task_group.def(py::init([] { return new TaskGroupWrapper<void>(); }),
    //               "Create TaskGroup<void>()");
    // task_group.def("join", &TaskGroupWrapper<void>::join,
    //               "Join the task group");

    auto _rotate = [=](pyfarray_t arr, float theta, int nx, int ny) {
        farray_t        cxx_arr(arr.size(), 0.0f);
        py::buffer_info inbuf = arr.request();
        memcpy(cxx_arr.data(), inbuf.ptr, cxx_arr.size() * sizeof(float));
        // for(uintmax_t i = 0; i < arr.size(); ++i)
        //    std::cout << "cxx_arr[" << i << "] = " << cxx_arr.at(i) <<
        //    std::endl;
        cxx_arr = cxx_rotate(cxx_arr, theta, nx, ny);
        pyfarray_t      _arr(cxx_arr.size());
        py::buffer_info outbuf = _arr.request();
        memcpy(outbuf.ptr, cxx_arr.data(), cxx_arr.size() * sizeof(float));
        return _arr;
    };
    tomo.def("rotate", _rotate, "rotate array");
}

//============================================================================//
