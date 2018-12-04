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

#include "utils.hh"

BEGIN_EXTERN_C
#include "bart.h"
#include "utils.h"
END_EXTERN_C

#include <cstdlib>
#include <memory>

//============================================================================//

void
cxx_bart(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
         int num_block, const float* ind_block)
{
    auto tid = ThreadPool::GetThisThreadID();
    ConsumeParameters(tid);

#if defined(TOMOPY_USE_TIMEMORY)
    tim::timer t(__FUNCTION__);
    t.format().get()->width(10);
    t.start();
#endif

    TIMEMORY_AUTO_TIMER("");

#if defined(TOMOPY_USE_GPU)
    // TODO: select based on memory
    bool use_cpu = GetEnv<bool>("TOMOPY_USE_CPU", false);
    if(use_cpu)
        bart_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy,
                 num_iter, num_block, ind_block);
    else
        bart_gpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy,
                 num_iter, num_block, ind_block);
#else
    bart_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter,
             num_block, ind_block);
#endif

#if defined(TOMOPY_USE_TIMEMORY)
    AutoLock l(TypeMutex<decltype(std::cout)>());
    std::cout << t.stop_and_return() << std::endl;
#endif
}

//============================================================================//

void
bart_cpu(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
         int num_block, const float* ind_block)
{
    if(dy == 0 || dt == 0 || dx == 0)
        return;

    tim::enable_signal_detection();
    TIMEMORY_AUTO_TIMER("[cpu]");

    uintmax_t _nx = cast<uintmax_t>(ngridx);
    uintmax_t _ny = cast<uintmax_t>(ngridy);
    uintmax_t _dy = cast<uintmax_t>(dy);
    uintmax_t _dt = cast<uintmax_t>(dt);
    uintmax_t _dx = cast<uintmax_t>(dx);
    uintmax_t _nd = _dy * _dt * _dx;  // number of total entries
    uintmax_t _ng = _nx + _ny;        // number of grid points

    farray_t simdata = farray_t(_nd, 0.0f);

    //------------------------------------------------------------------------//

    uintmax_t nthreads =
        GetEnv<uintmax_t>("TOMOPY_NUM_THREADS", NUM_TASK_THREADS);
    nthreads = (nthreads > uintmax_t(dy * num_block))
                   ? uintmax_t(dy * num_block)
                   : nthreads;

    //------------------------------------------------------------------------//

    TaskRunManager* run_man = cpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();
    ThreadPool*  tp       = task_man->thread_pool();

    {
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << "> bart::" << __FUNCTION__ << "@" << __LINE__ << " -- "
                  << "dy = " << dy << ", "
                  << "dt = " << dt << ", "
                  << "dx = " << dx << ", "
                  << "..." << std::endl;
    }

    auto compute_subset = [&](int i, int s, int os, float mov,
                              const farray_t& gridx, const farray_t& gridy,
                              int subset_ind1, int subset_ind2) {
        ConsumeParameters(i);

        farray_t sum_dist(_nx * _ny);
        farray_t update(_nx * _ny);

        if(os + 1 == num_block)
            subset_ind2 += dt % num_block;

        // For each projection angle
        for(int q = 0; q < subset_ind2; q++)
        {
            int p = (int) ind_block[q + os * subset_ind1];
            int asize, bsize, csize;
            // (float) * (size of ngrid{x,y} + 1)
            farray_t coordx(_ny + 1);
            farray_t coordy(_nx + 1);
            // (float) * (size of ngridx + ngridy)
            farray_t ax(_ng);
            farray_t ay(_ng);
            farray_t bx(_ng);
            farray_t by(_ng);
            farray_t coorx(_ng);
            farray_t coory(_ng);
            iarray_t indi(_nx + _ny);
            farray_t dist(_nx + _ny);

            // Calculate the sin and cos values
            // of the projection angle and find
            // at which quadrant on the cartesian grid.
            float theta_p  = fmodf(theta[p], 2.0f * cast<float>(M_PI));
            float sin_p    = sinf(theta_p);
            float cos_p    = cosf(theta_p);
            int   quadrant = calc_quadrant(theta_p);

            // For each detector pixel
            for(int d = 0; d < dx; d++)
            {
                // Calculate coordinates
                float xi = -ngridx - ngridy;
                float yi = (1 - dx) / 2.0f + d + mov;
                calc_coords(ngridx, ngridy, xi, yi, sin_p, cos_p, gridx.data(),
                            gridy.data(), coordx.data(), coordy.data());

                // Merge the (coordx, gridy) and (gridx, coordy)
                trim_coords(ngridx, ngridy, coordx.data(), coordy.data(),
                            gridx.data(), gridy.data(), &asize, ax.data(),
                            ay.data(), &bsize, bx.data(), by.data());

                // Sort the array of intersection points (ax, ay) and
                // (bx, by). The new sorted intersection points are
                // stored in (coorx, coory). Total number of points
                // are csize.
                sort_intersections(quadrant, asize, ax.data(), ay.data(), bsize,
                                   bx.data(), by.data(), &csize, coorx.data(),
                                   coory.data());

                // Calculate the distances (dist) between the
                // intersection points (coorx, coory). Find the
                // indices of the pixels on the reconstruction grid.
                calc_dist(ngridx, ngridy, csize, coorx.data(), coory.data(),
                          indi.data(), dist.data());

                // Calculate simdata
                calc_simdata(s, p, d, ngridx, ngridy, dt, dx, csize,
                             indi.data(), dist.data(), recon, simdata.data());
                // Output: simdata

                // Calculate dist*dist
                float sum_dist2 = 0.0f;
                for(int n = 0; n < csize - 1; n++)
                {
                    sum_dist2 += dist[n] * dist[n];
                    sum_dist[indi[n]] += dist[n];
                }

                // Update
                if(sum_dist2 != 0.0f)
                {
                    int   ind_data = d + p * dx + s * dt * dx;
                    float upd =
                        (data[ind_data] - simdata[ind_data]) / sum_dist2;
                    for(int n = 0; n < csize - 1; n++)
                    {
                        update[indi[n]] += upd * dist[n];
                    }
                }
            }
        }

        // static Mutex mutex;
        // mutex.lock();
        for(int n = 0; n < ngridx * ngridy; n++)
        {
            if(sum_dist[n] != 0.0f)
            {
                int ind_recon = s * ngridx * ngridy;
                recon[n + ind_recon] += update[n] / sum_dist[n];
            }
        }
        // mutex.unlock();
    };
    //------------------------------------------------------------------------//
    auto compute_slice = [&](int i, int s) {
        float    mov;
        farray_t gridx(_nx + 1);
        farray_t gridy(_ny + 1);

        preprocessing(ngridx, ngridy, dx, center[s], &mov, gridx.data(),
                      gridy.data());
        // Outputs: mov, gridx, gridy

        int subset_ind1 = dt / num_block;
        int subset_ind2 = subset_ind1;

        // create task group
        TaskGroup<void> tg(tp);
        // For each slice
        for(int os = 0; os < num_block; os++)
            task_man->exec(tg, compute_subset, i, s, os, mov, gridx, gridy,
                           subset_ind1, subset_ind2);
        // join task group
        tg.join();

        // For each ordered-subset num_subset
        // for(int os = 0; os < num_block; os++)
        //    compute_subset(i, s, os, mov, gridx, gridy,
        //                   subset_ind1, subset_ind2);
    };

    //------------------------------------------------------------------------//

    for(int i = 0; i < num_iter; i++)
    {
        // initialize simdata to zero
        memset(simdata.data(), 0, _dy * _dt * _dx * sizeof(float));

        // create task group
        TaskGroup<void> tg(tp);
        // For each slice
        for(int s = 0; s < dy; ++s) task_man->exec(tg, compute_slice, i, s);
        // join task group
        tg.join();

        // For each slice
        // for(int s = 0; s < dy; s++)
        //    compute_slice(i, s);
    }

    tim::disable_signal_detection();
}

//============================================================================//

void
bart_gpu(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
         int num_block, const float* ind_block)
{
    if(dy == 0 || dt == 0 || dx == 0)
        return;

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
        for(const auto& itr : options) ss << itr << "\n";
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
        bart_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy,
                 num_iter, num_block, ind_block);
        return;
    }

    // print the GPU execution type options
    print_options();

    default_idx = options.front().index;
    default_key = options.front().key;
    auto key =
        GetEnv("TOMOPY_GPU_TYPE", default_key, "Tomopy GPU execution type");

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
            bart_cuda(data, dy, dt, dx, center, theta, recon, ngridx, ngridy,
                      num_iter, num_block, ind_block);
        }
        else if(selection == 2)
        {
            bart_openacc(data, dy, dt, dx, center, theta, recon, ngridx, ngridy,
                         num_iter, num_block, ind_block);
        }
        else if(selection == 3)
        {
            bart_openmp(data, dy, dt, dx, center, theta, recon, ngridx, ngridy,
                        num_iter, num_block, ind_block);
        }
    }
    catch(std::exception& e)
    {
        {
            AutoLock l(TypeMutex<decltype(std::cout)>());
            std::cerr << "[TID: " << ThreadPool::GetThisThreadID() << "] "
                      << e.what() << std::endl;
        }
        bart_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy,
                 num_iter, num_block, ind_block);
    }
}

//============================================================================//

void
bart_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy,
          int num_iter, int num_block, const float* ind_block)
{
    ConsumeParameters(data, dy, dt, dx, center, theta, recon, ngridx, ngridy,
                      num_iter, num_block, ind_block);
    throw std::runtime_error(
        "BART algorithm has not been implemented for CUDA");

    tim::enable_signal_detection();
    TIMEMORY_AUTO_TIMER("[cuda]");

    // insert code here

    tim::disable_signal_detection();
}

//============================================================================//

void
bart_openacc(const float* data, int dy, int dt, int dx, const float* center,
             const float* theta, float* recon, int ngridx, int ngridy,
             int num_iter, int num_block, const float* ind_block)
{
    ConsumeParameters(data, dy, dt, dx, center, theta, recon, ngridx, ngridy,
                      num_iter, num_block, ind_block);
    throw std::runtime_error(
        "BART algorithm has not been implemented for OpenACC");

    tim::enable_signal_detection();
    TIMEMORY_AUTO_TIMER("[openacc]");

    // insert code here

    tim::disable_signal_detection();
}

//============================================================================//

void
bart_openmp(const float* data, int dy, int dt, int dx, const float* center,
            const float* theta, float* recon, int ngridx, int ngridy,
            int num_iter, int num_block, const float* ind_block)
{
    ConsumeParameters(data, dy, dt, dx, center, theta, recon, ngridx, ngridy,
                      num_iter, num_block, ind_block);
    throw std::runtime_error(
        "BART algorithm has not been implemented for OpenMP");

    tim::enable_signal_detection();
    TIMEMORY_AUTO_TIMER("[openmp]");

    // insert code here

    tim::disable_signal_detection();
}

//============================================================================//
