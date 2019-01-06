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
//   TOMOPY implementation file
//
//  Description:
//
//      Here we copy the memory to GPU and call the GPU kernels
//

extern "C"
{
#include "gpu.h"
}

//============================================================================//
// C++

// includes all C, CUDA, and C++ header files
#include "PTL/ThreadPool.hh"
#include "gpu.hh"

//============================================================================//

#if defined(TOMOPY_USE_NVTX)

nvtxEventAttributes_t nvtx_calc_coords;
nvtxEventAttributes_t nvtx_calc_dist;
nvtxEventAttributes_t nvtx_calc_simdata;
nvtxEventAttributes_t nvtx_preprocessing;
nvtxEventAttributes_t nvtx_sort_intersections;
nvtxEventAttributes_t nvtx_sum_dist;
nvtxEventAttributes_t nvtx_trim_coords;
nvtxEventAttributes_t nvtx_calc_sum_sqr;
nvtxEventAttributes_t nvtx_update;
nvtxEventAttributes_t nvtx_rotate;

//----------------------------------------------------------------------------//

void
init_nvtx()
{
    static bool first = true;
    if(!first)
        return;
    first = false;

    nvtx_calc_coords.version       = NVTX_VERSION;
    nvtx_calc_coords.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_calc_coords.colorType     = NVTX_COLOR_ARGB;
    nvtx_calc_coords.color         = 0xff0000ff; /* blue? */
    nvtx_calc_coords.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_calc_coords.message.ascii = "calc_coords";

    nvtx_calc_dist.version       = NVTX_VERSION;
    nvtx_calc_dist.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_calc_dist.colorType     = NVTX_COLOR_ARGB;
    nvtx_calc_dist.color         = 0xffffff00; /* yellow */
    nvtx_calc_dist.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_calc_dist.message.ascii = "calc_dist";

    nvtx_calc_simdata.version       = NVTX_VERSION;
    nvtx_calc_simdata.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_calc_simdata.colorType     = NVTX_COLOR_ARGB;
    nvtx_calc_simdata.color         = 0xff00ffff; /* cyan */
    nvtx_calc_simdata.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_calc_simdata.message.ascii = "calc_simdata";

    nvtx_preprocessing.version       = NVTX_VERSION;
    nvtx_preprocessing.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_preprocessing.colorType     = NVTX_COLOR_ARGB;
    nvtx_preprocessing.color         = 0xff00ffff; /* pink */
    nvtx_preprocessing.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_preprocessing.message.ascii = "preprocessing";

    nvtx_sort_intersections.version       = NVTX_VERSION;
    nvtx_sort_intersections.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_sort_intersections.colorType     = NVTX_COLOR_ARGB;
    nvtx_sort_intersections.color         = 0xffff0000; /* red */
    nvtx_sort_intersections.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_sort_intersections.message.ascii = "sort_intersections";

    nvtx_sum_dist.version       = NVTX_VERSION;
    nvtx_sum_dist.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_sum_dist.colorType     = NVTX_COLOR_ARGB;
    nvtx_sum_dist.color         = 0xffff0000; /* light purple */
    nvtx_sum_dist.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_sum_dist.message.ascii = "sum_dist2";

    nvtx_trim_coords.version       = NVTX_VERSION;
    nvtx_trim_coords.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_trim_coords.colorType     = NVTX_COLOR_ARGB;
    nvtx_trim_coords.color         = 0xff00ff00; /* green */
    nvtx_trim_coords.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_trim_coords.message.ascii = "trim_coords";

    nvtx_calc_sum_sqr.version       = NVTX_VERSION;
    nvtx_calc_sum_sqr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_calc_sum_sqr.colorType     = NVTX_COLOR_ARGB;
    nvtx_calc_sum_sqr.color         = 0xffffa500; /* orange */
    nvtx_calc_sum_sqr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_calc_sum_sqr.message.ascii = "calc_sum_sqr";

    nvtx_update.version       = NVTX_VERSION;
    nvtx_update.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_update.colorType     = NVTX_COLOR_ARGB;
    nvtx_update.color         = 0xff99ff99; /* light green */
    nvtx_update.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_update.message.ascii = "update";

    nvtx_rotate.version       = NVTX_VERSION;
    nvtx_rotate.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_rotate.colorType     = NVTX_COLOR_ARGB;
    nvtx_rotate.color         = 0xff0000ff; /* blue? */
    nvtx_rotate.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_rotate.message.ascii = "rotate";
}

#endif

//============================================================================//

int
cuda_set_device(int device)
{
    int deviceCount = cuda_device_count();
    if(deviceCount == 0)
        return -1;

    // don't set to higher than number of devices
    device = device % deviceCount;
    // update thread-static variable
    this_thread_device() = device;
    // actually set the device
    cudaSetDevice(device);
    // return the modulus
    return device;
}

//============================================================================//

int
cuda_multi_processor_count()
{
    if(cuda_device_count() == 0)
        return 0;

    // keep from querying device
    static thread_local cuda_device_info<int>* _instance = new cuda_device_info<int>();
    // use the thread assigned devices
    int device = this_thread_device();

    if(_instance->find(device) != _instance->end())
        return _instance->find(device)->second;

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    return ((*_instance)[device] = deviceProp.multiProcessorCount);
}

//============================================================================//

int
cuda_max_threads_per_block()
{
    if(cuda_device_count() == 0)
        return 0;

    // keep from querying device
    static thread_local cuda_device_info<int>* _instance = new cuda_device_info<int>();
    // use the thread assigned devices
    int device = this_thread_device();

    if(_instance->find(device) != _instance->end())
        return _instance->find(device)->second;

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    return ((*_instance)[device] = deviceProp.maxThreadsPerBlock);
}

//============================================================================//

int
cuda_warp_size()
{
    if(cuda_device_count() == 0)
        return 0;

    // keep from querying device
    static thread_local cuda_device_info<int>* _instance = new cuda_device_info<int>();
    // use the thread assigned devices
    int device = this_thread_device();

    if(_instance->find(device) != _instance->end())
        return _instance->find(device)->second;

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    return ((*_instance)[device] = deviceProp.warpSize);
}

//============================================================================//

int
cuda_shared_memory_per_block()
{
    if(cuda_device_count() == 0)
        return 0;

    // keep from querying device
    static thread_local cuda_device_info<int>* _instance = new cuda_device_info<int>();
    // use the thread assigned devices
    int device = this_thread_device();

    if(_instance->find(device) != _instance->end())
        return _instance->find(device)->second;

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    return ((*_instance)[device] = deviceProp.sharedMemPerBlock);
}

//============================================================================//

void
init_tomo_dataset()
{
    if(cuda_device_count() > 0)
    {
        tomo_dataset* _dataset = TomoDataset();
        _dataset->nstreams     = 32;
        _dataset->streams      = create_streams(_dataset->nstreams);
    }
}

//============================================================================//

void
free_tomo_dataset(bool is_master)
{
    if(cuda_device_count() > 0)
    {
        tomo_dataset*& _dataset = TomoDataset();
        cudaFree(_dataset->gpu->asize);
        cudaFree(_dataset->gpu->bsize);
        cudaFree(_dataset->gpu->csize);
        cudaFree(_dataset->gpu->coordx);
        cudaFree(_dataset->gpu->coordy);
        cudaFree(_dataset->gpu->ax);
        cudaFree(_dataset->gpu->ay);
        cudaFree(_dataset->gpu->bx);
        cudaFree(_dataset->gpu->by);
        cudaFree(_dataset->gpu->coorx);
        cudaFree(_dataset->gpu->coory);
        cudaFree(_dataset->gpu->dist);
        cudaFree(_dataset->gpu->indi);
        cudaFree(_dataset->gpu->sum);
        if(is_master)
        {
            cudaFree(_dataset->gpu->gridx);
            cudaFree(_dataset->gpu->gridy);
            cudaFree(_dataset->gpu->simdata);
            cudaFree(_dataset->gpu->model);
            cudaFree((void*) _dataset->gpu->center);
            cudaFree((void*) _dataset->gpu->theta);
            cudaFree(_dataset->gpu->mov);
            cudaFree(_dataset->gpu->data);
        }
        destroy_streams(_dataset->streams, _dataset->nstreams);
        delete _dataset->gpu;
        _dataset->gpu = nullptr;
    }
}

//============================================================================//

int
cuda_device_count()
{
    int         deviceCount = 0;
    cudaError_t error_id    = cudaGetDeviceCount(&deviceCount);

    if(error_id != cudaSuccess)
        return 0;

    return deviceCount;
}

//============================================================================//

void
cuda_device_query()
{
    static bool first = true;
    if(first)
        first = false;
    else
        return;

    int         deviceCount    = 0;
    int         driverVersion  = 0;
    int         runtimeVersion = 0;
    cudaError_t error_id       = cudaGetDeviceCount(&deviceCount);

    if(error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned error code %d\n--> %s\n",
               static_cast<int>(error_id), cudaGetErrorString(error_id));

        if(deviceCount > 0)
        {
            cudaSetDevice(0);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
            printf("\nDevice %d: \"%s\"\n", 0, deviceProp.name);

            // Console log
            cudaDriverGetVersion(&driverVersion);
            cudaRuntimeGetVersion(&runtimeVersion);
            printf("  CUDA Driver Version / Runtime Version          %d.%d / "
                   "%d.%d\n",
                   driverVersion / 1000, (driverVersion % 100) / 10,
                   runtimeVersion / 1000, (runtimeVersion % 100) / 10);
            printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
                   deviceProp.major, deviceProp.minor);
        }

        return;
    }

    if(deviceCount == 0)
        printf("No available CUDA device(s) detected\n");
    else
        printf("Detected %d CUDA capable devices\n", deviceCount);

    for(int dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        // This only available in CUDA 4.0-4.2 (but these were only exposed in
        // the CUDA Driver API)
        int memoryClock;
        int memBusWidth;
        int L2CacheSize;

        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000,
               (runtimeVersion % 100) / 10);

        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
               deviceProp.major, deviceProp.minor);

        char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(msg, sizeof(msg),
                  "  Total amount of global memory:                 %.0f MBytes "
                  "(%llu bytes)\n",
                  static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                  (unsigned long long) deviceProp.totalGlobalMem);
#else
        snprintf(msg, sizeof(msg),
                 "  Total amount of global memory:                 %.0f MBytes "
                 "(%llu bytes)\n",
                 static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                 (unsigned long long) deviceProp.totalGlobalMem);
#endif
        printf("%s", msg);

        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f "
               "GHz)\n",
               deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n",
               deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",
               deviceProp.memoryBusWidth);

        if(deviceProp.l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n",
                   deviceProp.l2CacheSize);
        }

#else
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n",
               memoryClock * 1e-3f);
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                              dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if(L2CacheSize)
            printf("  L2 Cache Size:                                 %d bytes\n",
                   L2CacheSize);
#endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
               "%d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
               deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
               deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d "
               "layers\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
               "layers\n",
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
               deviceProp.maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %lu bytes\n",
               deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n",
               deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
               deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n",
               deviceProp.memPitch);
        printf("  Texture alignment:                             %lu bytes\n",
               deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy "
               "engine(s)\n",
               (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n",
               deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n",
               deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n",
               deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n",
               deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n",
               deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
               deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                    : "WDDM (Windows Display Driver Model)");
#endif
        printf("  Device supports Unified Addressing (UVA):      %s\n",
               deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Device supports Compute Preemption:            %s\n",
               deviceProp.computePreemptionSupported ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n",
               deviceProp.cooperativeLaunch ? "Yes" : "No");
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
               deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
               deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

        const char* sComputeMode[] =
            { "Default (multiple host threads can use ::cudaSetDevice() with "
              "device "
              "simultaneously)",
              "Exclusive (only one host thread in one process is able to use "
              "::cudaSetDevice() with this device)",
              "Prohibited (no host thread can use ::cudaSetDevice() with this "
              "device)",
              "Exclusive Process (many threads in one process is able to use "
              "::cudaSetDevice() with this device)",
              "Unknown",
              NULL };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    printf("\n\n");
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();
}

//============================================================================//
