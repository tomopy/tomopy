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
//   TOMOPY CUDA implementation

#include "common.hh"
#include "macros.hh"
#include "utils.hh"

//======================================================================================//

#if defined(TOMOPY_USE_NVTX)

nvtxEventAttributes_t nvtx_total;
nvtxEventAttributes_t nvtx_iteration;
nvtxEventAttributes_t nvtx_slice;
nvtxEventAttributes_t nvtx_projection;
nvtxEventAttributes_t nvtx_update;
nvtxEventAttributes_t nvtx_rotate;

//--------------------------------------------------------------------------------------//

void
init_nvtx()
{
    static bool first = true;
    if(!first)
        return;
    first = false;

    nvtx_total.version       = NVTX_VERSION;
    nvtx_total.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_total.colorType     = NVTX_COLOR_ARGB;
    nvtx_total.color         = 0xff0000ff; /* blue? */
    nvtx_total.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_total.message.ascii = "total time for all iterations";

    nvtx_iteration.version       = NVTX_VERSION;
    nvtx_iteration.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_iteration.colorType     = NVTX_COLOR_ARGB;
    nvtx_iteration.color         = 0xffffff00; /* yellow */
    nvtx_iteration.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_iteration.message.ascii = "time per iteration";

    nvtx_slice.version       = NVTX_VERSION;
    nvtx_slice.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_slice.colorType     = NVTX_COLOR_ARGB;
    nvtx_slice.color         = 0xff00ffff; /* cyan */
    nvtx_slice.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_slice.message.ascii = "time per slice";

    nvtx_projection.version       = NVTX_VERSION;
    nvtx_projection.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_projection.colorType     = NVTX_COLOR_ARGB;
    nvtx_projection.color         = 0xff00ffff; /* pink */
    nvtx_projection.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_projection.message.ascii = "time per projection";

    nvtx_update.version       = NVTX_VERSION;
    nvtx_update.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_update.colorType     = NVTX_COLOR_ARGB;
    nvtx_update.color         = 0xff99ff99; /* light green */
    nvtx_update.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_update.message.ascii = "time updating";

    nvtx_rotate.version       = NVTX_VERSION;
    nvtx_rotate.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_rotate.colorType     = NVTX_COLOR_ARGB;
    nvtx_rotate.color         = 0xff0000ff; /* blue? */
    nvtx_rotate.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_rotate.message.ascii = "time rotating";
}

#endif

//======================================================================================//

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

//======================================================================================//

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

//======================================================================================//

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

//======================================================================================//

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

//======================================================================================//

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

//======================================================================================//

int
cuda_device_count()
{
    int         deviceCount = 0;
    cudaError_t error_id    = cudaGetDeviceCount(&deviceCount);

    if(error_id != cudaSuccess)
        return 0;

    return deviceCount;
}

//======================================================================================//

void
cuda_device_query()
{
    auto pythreads = GetEnv("TOMOPY_PYTHON_THREADS", HW_CONCURRENCY);
    static std::atomic<int16_t> _once;
    auto                        _count = _once++;
    if(_count + 1 == pythreads)
        _once.store(0);
    if(_count > 0)
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

    AutoLock l(TypeMutex<decltype(std::cout)>());

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
        // This only available in CUDA 4.0-4.2 (but these were only exposed in
        // the CUDA Driver API)
        int memoryClock;
        int memBusWidth;
        int L2CacheSize;
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
        printf("  Multiprocessor count:                          %d\n",
               deviceProp.multiProcessorCount);
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

        const char* sComputeMode[] = {
            "Default (multiple host threads can use ::cudaSetDevice() with "
            "device "
            "simultaneously)",
            "Exclusive (only one host thread in one process is able to use "
            "::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this "
            "device)",
            "Exclusive Process (many threads in one process is able to use "
            "::cudaSetDevice() with this device)",
            "Unknown",
            nullptr
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    printf("\n\n");
    CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//
