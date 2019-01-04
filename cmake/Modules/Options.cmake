
################################################################################
#
#        TiMemory Options
#
################################################################################

include(MacroUtilities)
include(Compilers)

if(CMAKE_C_COMPILER_IS_PGI)
    set(OpenMP_C_IMPL "=nonuma" CACHE STRING "OpenMP C library setting")
endif()

if(CMAKE_CXX_COMPILER_IS_PGI)
    set(OpenMP_CXX_IMPL "=nonuma" CACHE STRING "OpenMP C++ library setting")
endif()

# features
add_feature(CMAKE_BUILD_TYPE "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)")
add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
add_feature(${PROJECT_NAME}_C_FLAGS "C compiler flags")
add_feature(${PROJECT_NAME}_CXX_FLAGS "C++ compiler flags")
add_feature(CMAKE_C_STANDARD "C languae standard")
add_feature(CMAKE_CXX_STANDARD "C++11 STL standard")
add_feature(OpenMP_IMPL "OpenMP implementation (e.g. '=libomp', '=libiomp', '=nonuma', etc.)")
add_feature(PYTHON_INCLUDE_DIR "Python include directory")
add_feature(PYTHON_LIBRARY "Python library")

# options (always available)
add_option(TOMOPY_USE_MKL "Enable MKL" ON)
add_option(TOMOPY_USE_TBB "Enable TBB" OFF)
add_option(TOMOPY_USE_GPU "Enable GPU preprocessor" ON)
add_option(TOMOPY_USE_GPERF "Enable Google perftools profiler" OFF)
add_option(TOMOPY_USE_TIMEMORY "Enable TiMemory for timing+memory analysis" OFF)
add_option(TOMOPY_USE_OPENMP "Enable OpenMP option for GPU execution" ${TOMOPY_USE_GPU})
add_option(TOMOPY_USE_ARCH "Enable architecture specific flags" OFF)
add_option(TOMOPY_USE_PYBIND11 "Enable pybind11 binding" ON)
add_option(TOMOPY_USE_SANTITIZER "Enable sanitizer" OFF)
add_option(TOMOPY_CXX_GRIDREC "Enable gridrec with C++ std::complex" OFF)

if(TOMOPY_USE_SANTITIZER)
    set(SANITIZER_TYPE leak)
    add_feature(SANITIZER_TYPE "Type of sanitizer (-fsanitize=${SANITIZER_TYPE})")
endif()

if(TOMOPY_CXX_GRIDREC)
    add_definitions(-DTOMOPY_CXX_GRIDREC)
endif()

if(TOMOPY_USE_ARCH)
    add_option(TOMOPY_USE_AVX512 "Enable AVX-512 flags (if available)" OFF)
endif()

set(PTL_USE_TBB OFF CACHE BOOL "Enable TBB backend for PTL")

foreach(_OPT ARCH AVX512 GPERF)
    if(TOMOPY_USE_${_OPT})
        set(PTL_USE_${_OPT} ON CACHE BOOL "Enable similar PTL option to TOMOPY_USE_${_OPT}" FORCE)
    endif()
endforeach()

# default settings
set(OpenACC_FOUND OFF)
set(CUDA_FOUND OFF)

# possible options (sometimes available)
if(TOMOPY_USE_GPU)
    find_package(OpenACC QUIET)
    if(NOT COMPILER_IS_PGI)
        find_package(CUDA QUIET)
    endif()
endif()

if(OpenACC_FOUND)
    set(_USE_OPENACC ON)
else()
    set(_USE_OPENACC OFF)
endif()

if(CUDA_FOUND)
    set(_USE_CUDA ON)
else()
    set(_USE_CUDA OFF)
endif()

add_option(TOMOPY_USE_OPENACC "Enable OpenACC option for GPU execution" ${_USE_OPENACC})
add_option(TOMOPY_USE_CUDA "Enable CUDA option for GPU execution" ${_USE_CUDA})
add_option(TOMOPY_USE_NVTX "Enable NVTX for Nsight" ${_USE_CUDA})

if(TOMOPY_USE_CUDA)
    # find the cuda compiler
    find_program(CMAKE_CUDA_COMPILER nvcc
        PATHS /usr/local/cuda
        HINTS /usr/local/cuda
        PATH_SUFFIXES bin)
    if(CMAKE_CUDA_COMPILER)
        include(CudaConfig)
    endif(CMAKE_CUDA_COMPILER)
endif(TOMOPY_USE_CUDA)

if(TOMOPY_USE_GPU)
    add_definitions(-DTOMOPY_USE_GPU)
endif(TOMOPY_USE_GPU)

unset(COMPILER_IS_PGI)

if(APPLE)
    add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH
        "Hardcode installation rpath based on link path" ON NO_FEATURE)
endif()
