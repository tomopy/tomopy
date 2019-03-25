
################################################################################
#
#        TiMemory Options
#
################################################################################

include(MacroUtilities)
include(Compilers)

set(_USE_PTL ON)
set(_USE_PYBIND ON)
set(_USE_OMP ON)

if(CMAKE_C_COMPILER_IS_GNU)
    # GNU compiler will enable OpenMP SIMD with -fopenmp-simd
    set(_USE_OMP OFF)
endif()

if(CMAKE_C_COMPILER_IS_PGI)
    set(OpenMP_C_IMPL "=nonuma" CACHE STRING "OpenMP C library setting")
endif()

if(CMAKE_CXX_COMPILER_IS_PGI)
    set(OpenMP_CXX_IMPL "=nonuma" CACHE STRING "OpenMP C++ library setting")
    set(_USE_PTL OFF)
    set(_USE_PYBIND OFF)
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
add_option(TOMOPY_USE_GPERF "Enable Google perftools profiler" OFF)
add_option(TOMOPY_USE_TIMEMORY "Enable TiMemory for timing+memory analysis" OFF)
add_option(TOMOPY_USE_OPENMP "Enable OpenMP (for SIMD -- GNU will enable without this setting)" ${_USE_OMP})
add_option(TOMOPY_USE_OPENCV "Enable OpenCV for image processing" ON)
add_option(TOMOPY_USE_ARCH "Enable architecture specific flags" OFF)
add_option(TOMOPY_USE_SANITIZER "Enable sanitizer" OFF)
add_option(TOMOPY_CXX_GRIDREC "Enable gridrec with C++ std::complex" OFF)
add_option(TOMOPY_USE_COVERAGE "Enable code coverage" OFF)
add_option(TOMOPY_USE_PTL "Enable Parallel Tasking Library (PTL)" ${_USE_PTL})

if(TOMOPY_USE_SANITIZER)
    set(SANITIZER_TYPE leak CACHE STRING "Type of sanitizer")
    add_feature(SANITIZER_TYPE "Type of sanitizer (-fsanitize=${SANITIZER_TYPE})")
endif()

if(TOMOPY_CXX_GRIDREC)
    list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_CXX_GRIDREC)
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
set(_USE_CUDA ON)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(STATUS "No CUDA support")
        set(_USE_CUDA OFF)
    endif()
else()
    set(_USE_CUDA OFF)
endif()

add_option(TOMOPY_USE_CUDA "Enable CUDA option for GPU execution" ${_USE_CUDA})
add_option(TOMOPY_USE_NVTX "Enable NVTX for Nsight" OFF)

if(TOMOPY_USE_CUDA)
    add_feature(CMAKE_CUDA_STANDARD "CUDA STL standard")
endif(TOMOPY_USE_CUDA)

if(APPLE)
    add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH
        "Hardcode installation rpath based on link path" ON NO_FEATURE)
endif()

unset(COMPILER_IS_PGI)
