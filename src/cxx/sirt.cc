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

//============================================================================//

int
cxx_sirt(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    // check to see if the C implementation is requested
    bool use_c_algorithm = GetEnv<bool>("TOMOPY_USE_C_SIRT", false);
    // if C implementation is requested, return non-zero (failure)
    if(use_c_algorithm)
        return (int) false;

    auto tid = ThreadPool::GetThisThreadID();
    ConsumeParameters(tid);

#if defined(TOMOPY_USE_TIMEMORY)
    tim::timer t(__FUNCTION__);
    t.format().get()->width(10);
    t.start();
#endif

    TIMEMORY_AUTO_TIMER("");
    printf(
        "\n\t%s [nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i]\n\n",
        __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

#if defined(TOMOPY_USE_GPU)
    // TODO: select based on memory
    bool use_cpu = GetEnv<bool>("TOMOPY_USE_CPU", false);
    if(use_cpu)
        sirt_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy,
                 num_iter);
    else
        run_gpu_algorithm(sirt_cpu, sirt_cuda, sirt_openacc, sirt_cpu, data, dy,
                          dt, dx, center, theta, recon, ngridx, ngridy,
                          num_iter);
#else
    sirt_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
#endif

#if defined(TOMOPY_USE_TIMEMORY)
    AutoLock l(TypeMutex<decltype(std::cout)>());
    std::cout << "[" << tid << "]> " << t.stop_and_return() << std::endl;
#endif

    return (int) true;
}

//============================================================================//

void
sirt_cpu(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)

{
    for(int i = 0; i < num_iter; i++)
    {
        farray_t simdata(dy * dt * dx, 0.0f);
        // For each slice
        for(int s = 0; s < dy; s++)
        {
            farray_t update(ngridx * ngridx, 0.0f);
            farray_t recon_off(ngridx * ngridy, 0.0f);
            farray_t recon_update(recon_off.size(), 0.0f);

            // recon offset for the slice
            for(int ii = 0; ii < recon_off.size(); ++ii)
            {
                recon_off[ii] = recon[ii + (s * ngridx * ngridy)];
            }
            // For each projection angle
            for(int p = 0; p < dt; p++)
            {
                float theta_p = fmodf(theta[p] + (float) (0.5f * M_PI),
                                      2.0f * (float) M_PI);
                // Rotate object - 2D slices
                auto recon_rot =
                    cxx_rotate(recon_off, -theta_p, ngridx, ngridy);
                // Calculate simulated data by summing up along x-axis
                for(int d = 0; d < dx; d++)
                {
                    int ind_data = d + p * dx + s * dt * dx;
                    for(int n = 0; n < ngridx; n++)
                    {
                        simdata[ind_data] += recon_rot[n + d * ngridx];
                    }
                }
                // Make update by backprojecting error along x-axis
                for(int d = 0; d < dx; d++)
                {
                    float sum_dist2 = ngridx;
                    int   ind_data  = d + p * dx + s * dt * dx;
                    float upd =
                        (data[ind_data] - simdata[ind_data]) / sum_dist2;
                    for(int n = 0; n < ngridx; n++)
                    {
                        recon_rot[n + d * ngridx] += upd / ngridx;
                    }
                }
                // Back-Rotate object
                auto tmp = cxx_rotate(recon_rot, theta_p, ngridx, ngridy);
                for(uint64_t i = 0; i < tmp.size(); ++i)
                    recon_update[i] += tmp[i];
            }
            for(int ii = 0; ii < (ngridx * ngridy); ++ii)
            {
                recon[ii + (s * ngridx * ngridy)] +=
                    recon_update[ii] / static_cast<float>(dt);
            }
        }
    }
}

//============================================================================//

void
sirt_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy,
          int num_iter)
{
    if(dy == 0 || dt == 0 || dx == 0)
        return;

    ConsumeParameters(data, center, theta, recon, ngridx, ngridy, num_iter);

    ThreadPool::GetThisThreadID();

#if defined(TOMOPY_USE_CUDA)

    if(cuda_device_count() == 0)
        throw std::runtime_error("No CUDA device(s) available");

    cuda_device_query();

    tim::enable_signal_detection();
    TIMEMORY_AUTO_TIMER("[cuda]");

    // assign the thread to a device
    set_this_thread_device();

    //------------------------------------------------------------------------//
    // int nthreads = GetEnv<int>("TOMOPY_NUM_THREADS", 0);
    int nthreads = 0;
    //------------------------------------------------------------------------//

    TaskRunManager* run_man = gpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();
    ThreadPool*  tp       = task_man->thread_pool();

    //------------------------------------------------------------------------//

    uintmax_t _nx = scast<uintmax_t>(ngridx);
    uintmax_t _ny = scast<uintmax_t>(ngridy);
    uintmax_t _dy = scast<uintmax_t>(dy);
    uintmax_t _dt = scast<uintmax_t>(dt);
    uintmax_t _dx = scast<uintmax_t>(dx);
    uintmax_t _nd = _dy * _dt * _dx;  // number of total entries
    uintmax_t _ng = _nx + _ny;        // number of grid points

    //------------------------------------------------------------------------//

    tomo_data* master_gpu_data = new tomo_data({
        dy, dt, dx, ngridx, ngridy,
        // pointers
        gpu_malloc<int>(1),                         // asize
        gpu_malloc<int>(1),                         // bsize
        gpu_malloc<int>(1),                         // csize
        gpu_malloc<float>(_nx + 1),                 // gridx
        gpu_malloc<float>(_ny + 1),                 // gridy
        gpu_malloc<float>(_ny + 1),                 // coordx
        gpu_malloc<float>(_nx + 1),                 // coordy
        gpu_malloc<float>(_ng),                     // ax
        gpu_malloc<float>(_ng),                     // ay
        gpu_malloc<float>(_ng),                     // bx
        gpu_malloc<float>(_ng),                     // by
        gpu_malloc<float>(_ng),                     // coorx
        gpu_malloc<float>(_ng),                     // coory
        gpu_malloc<float>(_ng),                     // dist
        gpu_malloc<int>(_ng),                       // indi
        gpu_malloc<float>(_nd),                     // simdata
        malloc_and_memcpy(recon, _dy * _nx * _ny),  // model (recon)
        malloc_and_memcpy(center, _dy),             // center
        malloc_and_memcpy(theta, _dt),              // theta
        gpu_malloc<float>(1),                       // sum
        gpu_malloc<float>(1),                       // mov
        malloc_and_memcpy(data, _nd)                // data
    });

    tomo_dataset* _master_dataset = TomoDataset();
    _master_dataset->gpu          = master_gpu_data;
    init_tomo_dataset();
    cudaStream_t* streams  = _master_dataset->streams;
    auto          nstreams = _master_dataset->nstreams;

    init_nvtx();

    float* mov = new float[1];
    *mov       = 0.0f;

    farray_t gridx(_nx);
    farray_t gridy(_ny);
    farray_t simdata(_nd);

    // Outputs: mov, gridx, gridy
    cuda_preprocessing(ngridx, ngridy, dx, center[0], master_gpu_data->mov,
                       master_gpu_data->gridx, master_gpu_data->gridy, streams);

    cpu_memcpy(master_gpu_data->mov, mov, 1, streams[0]);
    cudaStreamSynchronize(streams[0]);

    if(0 < PRINT_MAX_ITER && 0 < PRINT_MAX_SLICE && 0 < PRINT_MAX_ANGLE &&
       0 < PRINT_MAX_PIXEL)
    {
        print_gpu_array(1, master_gpu_data->mov, 0, 0, 0, 0, "mov");
        print_gpu_array(ngridx, master_gpu_data->gridx, 0, 0, 0, 0, "gridx");
        print_gpu_array(ngridy, master_gpu_data->gridy, 0, 0, 0, 0, "gridy");
    }

    auto compute_slice = [&](int i, int s) {
        tomo_data*    gpu_data  = master_gpu_data;
        tomo_dataset* _dataset  = TomoDataset();
        bool          is_master = (_dataset == _master_dataset);
        // if new thread
        if(!is_master)
        {
            gpu_data = new tomo_data({
                dy, dt, dx, ngridx, ngridy,
                // pointers
                gpu_malloc<int>(1),          // asize
                gpu_malloc<int>(1),          // bsize
                gpu_malloc<int>(1),          // csize
                master_gpu_data->gridx,      // gridx
                master_gpu_data->gridy,      // gridy
                gpu_malloc<float>(_ny + 1),  // coordx
                gpu_malloc<float>(_nx + 1),  // coordy
                gpu_malloc<float>(_ng),      // ax
                gpu_malloc<float>(_ng),      // ay
                gpu_malloc<float>(_ng),      // bx
                gpu_malloc<float>(_ng),      // by
                gpu_malloc<float>(_ng),      // coorx
                gpu_malloc<float>(_ng),      // coory
                gpu_malloc<float>(_ng),      // dist
                gpu_malloc<int>(_ng),        // indi
                master_gpu_data->simdata,    // simdata
                master_gpu_data->model,      // model (recon)
                master_gpu_data->center,     // center
                master_gpu_data->theta,      // theta
                gpu_malloc<float>(1),        // sum
                master_gpu_data->mov,        // mov
                master_gpu_data->data        // data
            });

            _dataset->gpu = gpu_data;
            init_tomo_dataset();
        }
        cudaStream_t* streams  = _dataset->streams;
        auto          nstreams = _dataset->nstreams;

        //#if defined(DEBUG)
        PRINT_HERE(std::string(std::string("|_slice: ") + std::to_string(s) +
                               std::string(" of ") + std::to_string(dy))
                       .c_str());
        //#endif

        uintmax_t recon_offset = s * (ngridx * ngridy);
        float*    _recon       = recon + recon_offset;

        // For each projection angle
        for(int p = 0; p < dt; ++p)
        {
            //#if defined(DEBUG)
            PRINT_HERE(std::string(std::string("  |_angle: ") +
                                   std::to_string(p) + std::string(" of ") +
                                   std::to_string(dt))
                           .c_str());
            //#endif

            // Calculate the sin and cos values
            // of the projection angle and find
            // at which quadrant on the cartesian grid.
            float theta_p  = fmodf(theta[p], 2.0f * scast<float>(M_PI));
            int   quadrant = calc_quadrant(theta_p);
            float sin_p    = sinf(theta_p);
            float cos_p    = cosf(theta_p);

            // For each detector pixel
            for(int d = 0; d < dx; ++d)
            {
                // PRINT_HERE(std::string(std::string("    |_pixel: ") +
                //                       std::to_string(d) +
                //                       std::string(" of ") +
                //                       std::to_string(dx)).c_str());

                int stream_offset = (d % (nstreams - 2)) + (d % 2);

                if(stream_offset + 1 >= nstreams)
                {
                    PRINT_HERE(std::string(std::string("bad stream offset: ") +
                                           std::to_string(stream_offset) +
                                           std::string(" of ") +
                                           std::to_string(nstreams))
                                   .c_str());
                }

                // Calculate coordinates
                float xi = -ngridx - ngridy;
                float yi = 0.5f * (1 - dx) + d + (*mov);

                // calculate the coordinates
                //
                // inputs: gridx, gridy
                // outputs: coordx, coordy
                cuda_calc_coords(ngridx, ngridy, xi, yi, sin_p, cos_p,
                                 gpu_data->gridx, gpu_data->gridy,
                                 gpu_data->coordx, gpu_data->coordy,
                                 streams + stream_offset);

                static Mutex m;
                m.lock();

                // Merge the (coordx, gridy) and (gridx, coordy)
                //
                // inputs: gridx, gridy, coordx, coordy
                // outputs: asize, ax, ay, bsize, bx, by
                cuda_trim_coords(ngridx, ngridy, gpu_data->coordx,
                                 gpu_data->coordy, gpu_data->gridx,
                                 gpu_data->gridy, gpu_data->asize, gpu_data->ax,
                                 gpu_data->ay, gpu_data->bsize, gpu_data->bx,
                                 gpu_data->by, streams + stream_offset);

                // Sort the array of intersection points (ax, ay) and
                // (bx, by). The new sorted intersection points are
                // stored in (coorx, coory). Total number of points
                // are csize.
                //
                // inputs: asize, ax, ay, bsize, bx, by
                // outputs: csize, coorx, coory
                cuda_sort_intersections(quadrant, gpu_data->asize, gpu_data->ax,
                                        gpu_data->ay, gpu_data->bsize,
                                        gpu_data->bx, gpu_data->by,
                                        gpu_data->csize, gpu_data->coorx,
                                        gpu_data->coory,
                                        streams + stream_offset);

                // Calculate the distances (dist) between the
                // intersection points (coorx, coory). Find the
                // indices of the pixels on the reconstruction grid.
                //
                // inputs: csize, coorx, coory
                // outputs: indi, dist
                cuda_calc_dist(ngridx, ngridy, gpu_data->csize, gpu_data->coorx,
                               gpu_data->coory, gpu_data->indi, gpu_data->dist,
                               streams + stream_offset);

                // Calculate dist*dist
                //
                // inputs: csize, dist
                // outputs: sum
                cuda_calc_sum_sqr(gpu_data->csize, gpu_data->dist,
                                  gpu_data->sum, streams + stream_offset);

                if(i < PRINT_MAX_ITER && s < PRINT_MAX_SLICE &&
                   p < PRINT_MAX_ANGLE && d < PRINT_MAX_PIXEL)
                {
                    print_gpu_array(_ny, gpu_data->coordx, i, s, p, d,
                                    "coordx");
                    print_gpu_array(_nx, gpu_data->coordy, i, s, p, d,
                                    "coordy");
                    print_gpu_array(1, gpu_data->asize, i, s, p, d, "asize");
                    print_gpu_array(_ng, gpu_data->ax, i, s, p, d, "ax");
                    print_gpu_array(_ng, gpu_data->ay, i, s, p, d, "ay");
                    print_gpu_array(1, gpu_data->bsize, i, s, p, d, "bsize");
                    print_gpu_array(_ng, gpu_data->bx, i, s, p, d, "bx");
                    print_gpu_array(_ng, gpu_data->by, i, s, p, d, "by");
                    print_gpu_array(1, gpu_data->csize, i, s, p, d, "csize");
                    print_gpu_array(_ng, gpu_data->coorx, i, s, p, d, "coorx");
                    print_gpu_array(_ng, gpu_data->coory, i, s, p, d, "coory");
                    print_gpu_array(_ng, gpu_data->indi, i, s, p, d, "indi");
                    print_gpu_array(_ng, gpu_data->dist, i, s, p, d, "dist");
                    print_gpu_array(1, gpu_data->sum, i, s, p, d, "sum");
                    print_gpu_array(_dy * _dt * _dt, gpu_data->data, i, s, p, d,
                                    "data");
                }

                // calc_simdata_gpu and sirt_update_gpu do this check to
                // avoid the memory synchronization
                // if(sum_cpu != 0.0f)
                // {

                cudaStreamSynchronize(streams[stream_offset]);
                cudaStreamSynchronize(streams[stream_offset + 1]);

                // Calculate simdata
                // PRINT_HERE(std::to_string(d).c_str());
                cuda_calc_simdata(s, p, d, ngridx, ngridy, dt, dx,
                                  gpu_data->csize, gpu_data->indi,
                                  gpu_data->dist, gpu_data->model,
                                  gpu_data->sum, gpu_data->simdata,
                                  streams + stream_offset);

                // PRINT_HERE(std::to_string(d).c_str());
                cuda_sirt_update(s, p, d, ngridx, ngridy, dt, dx,
                                 gpu_data->csize, gpu_data->data,
                                 gpu_data->simdata, gpu_data->indi,
                                 gpu_data->dist, gpu_data->sum, gpu_data->model,
                                 streams + stream_offset);
                // }

                if(i < PRINT_MAX_ITER && s < PRINT_MAX_SLICE &&
                   p < PRINT_MAX_ANGLE && d < PRINT_MAX_PIXEL)
                {
                    print_gpu_array(_dy * _dt * _dx, gpu_data->simdata, i, s, p,
                                    d, "simdata");
                    print_gpu_array(_dy * (ngridx * ngridy), gpu_data->model, i,
                                    s, p, d, "model");
                }
                m.unlock();
            }
        }

        for(auto i = 0; i < nstreams; ++i)
            cudaStreamSynchronize(streams[i]);

        static Mutex m;
        m.lock();
        cudaMemcpyAsync(_recon, gpu_data->model,
                        (ngridx * ngridy) * sizeof(float),
                        cudaMemcpyDeviceToHost, _dataset->streams[0]);
        CUDA_CHECK_LAST_ERROR();
        cudaStreamSynchronize(streams[0]);
        CUDA_CHECK_LAST_ERROR();
        m.unlock();

        if(!is_master)
            free_tomo_dataset(is_master);
    };

    for(int i = 0; i < num_iter; ++i)
    {
        PRINT_HERE(std::string(std::string("iteration: ") + std::to_string(i) +
                               std::string(" of ") + std::to_string(num_iter))
                       .c_str());

        // initialize simdata to zero
        cudaMemset(master_gpu_data->simdata, 0, _nd * sizeof(float));

        // For each slice
        // for(int s = 0; s < dy; ++s)
        //    compute_slice(i, s);

        // create task group
        TaskGroup<void> tg(tp);
        // For each slice
        for(int s = 0; s < dy; ++s)
            task_man->exec(tg, compute_slice, i, s);
        // join task group
        tg.join();
    }

    free_tomo_dataset(true);

#endif

    tim::disable_signal_detection();
}

//============================================================================//

void
sirt_openacc(const float* data, int dy, int dt, int dx, const float* center,
             const float* theta, float* recon, int ngridx, int ngridy,
             int num_iter)
{
    tim::enable_signal_detection();
    TIMEMORY_AUTO_TIMER("[openacc]");

    float* gridx   = (float*) malloc((ngridx + 1) * sizeof(float));
    float* gridy   = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordx  = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordy  = (float*) malloc((ngridx + 1) * sizeof(float));
    float* ax      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* ay      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* bx      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* by      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coorx   = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coory   = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* dist    = (float*) malloc((ngridx + ngridy) * sizeof(float));
    int*   indi    = (int*) malloc((ngridx + ngridy) * sizeof(int));
    float* simdata = (float*) malloc((dy * dt * dx) * sizeof(float));

    assert(coordx != nullptr && coordy != nullptr && ax != nullptr &&
           ay != nullptr && by != nullptr && bx != nullptr &&
           coorx != nullptr && coory != nullptr && dist != nullptr &&
           indi != nullptr && simdata != nullptr);

    int   s, p, d, i, n;
    int   quadrant;
    float theta_p, sin_p, cos_p;
    float mov, xi, yi;
    int   asize, bsize, csize;
    float upd;
    int   ind_data, ind_recon;

    for(i = 0; i < num_iter; i++)
    {
        // initialize simdata to zero
        memset(simdata, 0, dy * dt * dx * sizeof(float));

        openacc_preprocessing(ngridx, ngridy, dx, center[0], &mov, gridx,
                              gridy);  // Outputs: mov, gridx, gridy

        // For each projection angle
        for(p = 0; p < dt; p++)
        {
            // Calculate the sin and cos values
            // of the projection angle and find
            // at which quadrant on the cartesian grid.
            theta_p  = fmod(theta[p], 2.0f * scast<float>(M_PI));
            quadrant = openacc_calc_quadrant(theta_p);
            sin_p    = sinf(theta_p);
            cos_p    = cosf(theta_p);

            // For each detector pixel
            for(d = 0; d < dx; d++)
            {
                // Calculate coordinates
                xi = -ngridx - ngridy;
                yi = (1 - dx) / 2.0f + d + mov;
                openacc_calc_coords(ngridx, ngridy, xi, yi, sin_p, cos_p, gridx,
                                    gridy, coordx, coordy);

                // Merge the (coordx, gridy) and (gridx, coordy)
                openacc_trim_coords(ngridx, ngridy, coordx, coordy, gridx,
                                    gridy, &asize, ax, ay, &bsize, bx, by);

                // Sort the array of intersection points (ax, ay) and
                // (bx, by). The new sorted intersection points are
                // stored in (coorx, coory). Total number of points
                // are csize.
                openacc_sort_intersections(quadrant, asize, ax, ay, bsize, bx,
                                           by, &csize, coorx, coory);

                // Calculate the distances (dist) between the
                // intersection points (coorx, coory). Find the
                // indices of the pixels on the reconstruction grid.
                openacc_calc_dist(ngridx, ngridy, csize, coorx, coory, indi,
                                  dist);

                // Calculate dist*dist
                float sum_dist2 = 0.0f;
                for(n = 0; n < csize - 1; n++)
                {
                    sum_dist2 += dist[n] * dist[n];
                }

                if(sum_dist2 != 0.0f)
                {
                    // For each slice
                    for(s = 0; s < dy; s++)
                    {
                        // Calculate simdata
                        openacc_calc_simdata(s, p, d, ngridx, ngridy, dt, dx,
                                             csize, indi, dist, recon,
                                             simdata);  // Output: simdata

                        // Update
                        ind_data  = d + p * dx + s * dt * dx;
                        ind_recon = s * ngridx * ngridy;
                        upd = (data[ind_data] - simdata[ind_data]) / sum_dist2;
                        for(n = 0; n < csize - 1; n++)
                        {
                            recon[indi[n] + ind_recon] += upd * dist[n];
                        }
                    }
                }
            }
        }
    }
    free(gridx);
    free(gridy);
    free(coordx);
    free(coordy);
    free(ax);
    free(ay);
    free(bx);
    free(by);
    free(coorx);
    free(coory);
    free(dist);
    free(indi);
    free(simdata);

    tim::disable_signal_detection();
}

//============================================================================//

void
sirt_openmp(const float* data, int dy, int dt, int dx, const float* center,
            const float* theta, float* recon, int ngridx, int ngridy,
            int num_iter)
{
    tim::enable_signal_detection();
    TIMEMORY_AUTO_TIMER("[openmp]");

    float* gridx   = (float*) malloc((ngridx + 1) * sizeof(float));
    float* gridy   = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordx  = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordy  = (float*) malloc((ngridx + 1) * sizeof(float));
    float* ax      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* ay      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* bx      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* by      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coorx   = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coory   = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* dist    = (float*) malloc((ngridx + ngridy) * sizeof(float));
    int*   indi    = (int*) malloc((ngridx + ngridy) * sizeof(int));
    float* simdata = (float*) malloc((dy * dt * dx) * sizeof(float));

    assert(coordx != nullptr && coordy != nullptr && ax != nullptr &&
           ay != nullptr && by != nullptr && bx != nullptr &&
           coorx != nullptr && coory != nullptr && dist != nullptr &&
           indi != nullptr && simdata != nullptr);

    int   s, p, d, i, n;
    int   quadrant;
    float theta_p, sin_p, cos_p;
    float mov, xi, yi;
    int   asize, bsize, csize;
    float upd;
    int   ind_data, ind_recon;

    for(i = 0; i < num_iter; i++)
    {
        // initialize simdata to zero
        memset(simdata, 0, dy * dt * dx * sizeof(float));

        openmp_preprocessing(ngridx, ngridy, dx, center[0], &mov, gridx,
                             gridy);  // Outputs: mov, gridx, gridy

        // For each projection angle
        for(p = 0; p < dt; p++)
        {
            // Calculate the sin and cos values
            // of the projection angle and find
            // at which quadrant on the cartesian grid.
            theta_p  = fmod(theta[p], 2.0f * scast<float>(M_PI));
            quadrant = openmp_calc_quadrant(theta_p);
            sin_p    = sinf(theta_p);
            cos_p    = cosf(theta_p);

            // For each detector pixel
            for(d = 0; d < dx; d++)
            {
                // Calculate coordinates
                xi = -ngridx - ngridy;
                yi = (1 - dx) / 2.0f + d + mov;
                openmp_calc_coords(ngridx, ngridy, xi, yi, sin_p, cos_p, gridx,
                                   gridy, coordx, coordy);

                // Merge the (coordx, gridy) and (gridx, coordy)
                openmp_trim_coords(ngridx, ngridy, coordx, coordy, gridx, gridy,
                                   &asize, ax, ay, &bsize, bx, by);

                // Sort the array of intersection points (ax, ay) and
                // (bx, by). The new sorted intersection points are
                // stored in (coorx, coory). Total number of points
                // are csize.
                openmp_sort_intersections(quadrant, asize, ax, ay, bsize, bx,
                                          by, &csize, coorx, coory);

                // Calculate the distances (dist) between the
                // intersection points (coorx, coory). Find the
                // indices of the pixels on the reconstruction grid.
                openmp_calc_dist(ngridx, ngridy, csize, coorx, coory, indi,
                                 dist);

                // Calculate dist*dist
                float sum_dist2 = 0.0f;
                for(n = 0; n < csize - 1; n++)
                {
                    sum_dist2 += dist[n] * dist[n];
                }

                if(sum_dist2 != 0.0f)
                {
                    // For each slice
                    for(s = 0; s < dy; s++)
                    {
                        // Calculate simdata
                        openmp_calc_simdata(s, p, d, ngridx, ngridy, dt, dx,
                                            csize, indi, dist, recon,
                                            simdata);  // Output: simdata

                        // Update
                        ind_data  = d + p * dx + s * dt * dx;
                        ind_recon = s * ngridx * ngridy;
                        upd = (data[ind_data] - simdata[ind_data]) / sum_dist2;
                        for(n = 0; n < csize - 1; n++)
                        {
                            recon[indi[n] + ind_recon] += upd * dist[n];
                        }
                    }
                }
            }
        }
    }
    free(gridx);
    free(gridy);
    free(coordx);
    free(coordy);
    free(ax);
    free(ay);
    free(bx);
    free(by);
    free(coorx);
    free(coory);
    free(dist);
    free(indi);
    free(simdata);

    tim::disable_signal_detection();
}

//============================================================================//
