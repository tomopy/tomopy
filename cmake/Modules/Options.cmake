
################################################################################
#
#        TiMemory Options
#
################################################################################

include(MacroUtilities)
include(Compilers)

set(_USE_OMP ON)
set(_USE_CXX_GRIDREC OFF)

# if Windows MSVC compiler, use C++ version of gridrec
if(WIN32)
    set(_USE_CXX_GRIDREC ON)
endif()

# GNU compiler will enable OpenMP SIMD with -fopenmp-simd
if(CMAKE_C_COMPILER_IS_GNU)
    set(_USE_OMP OFF)
endif()

# Check if CUDA can be enabled
find_package(CUDA)
if(CUDA_FOUND)
    set(_USE_CUDA ON)
else()
    set(_USE_CUDA OFF)
endif()

# features
add_feature(CMAKE_BUILD_TYPE "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)")
add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
add_feature(${PROJECT_NAME}_C_FLAGS "C compiler flags")
add_feature(${PROJECT_NAME}_CXX_FLAGS "C++ compiler flags")
add_feature(CMAKE_C_STANDARD "C language standard")
add_feature(CMAKE_CXX_STANDARD "C++11 STL standard")
add_feature(PYTHON_EXECUTABLE "Python executable (base path is used to locate MKL, OpenCV, etc.)")
add_feature(TOMOPY_USER_LIBRARIES "Explicit list of libraries to link")
if(SKBUILD)
    add_feature(PYTHON_INCLUDE_DIR "Python include directory")
    add_feature(PYTHON_LIBRARY "Python library")
endif()

# options (always available)
add_option(TOMOPY_USE_GPERF "Enable using Google perftools profiler" OFF)
add_option(TOMOPY_USE_TIMEMORY "Enable TiMemory for timing+memory analysis" OFF)
add_option(TOMOPY_USE_OPENMP "Enable OpenMP (for SIMD -- GNU will enable without this setting)" ${_USE_OMP})
add_option(TOMOPY_USE_OPENCV "Enable OpenCV for image processing" ON)
add_option(TOMOPY_USE_ARCH "Enable architecture specific flags" OFF)
add_option(TOMOPY_USE_SANITIZER "Enable sanitizer" OFF)
add_option(TOMOPY_CXX_GRIDREC "Enable gridrec with C++ std::complex" ${_USE_CXX_GRIDREC})
add_option(TOMOPY_USE_COVERAGE "Enable code coverage for C/C++" OFF)
add_option(TOMOPY_USE_PTL "Enable Parallel Tasking Library (PTL)" ON)
add_option(TOMOPY_USE_CLANG_TIDY "Enable clang-tidy (C++ linter)" OFF)
add_option(TOMOPY_USE_CUDA "Enable CUDA option for GPU execution" ${_USE_CUDA})
add_option(TOMOPY_USER_FLAGS "Insert CFLAGS and CXXFLAGS regardless of whether pass check" OFF)

if(TOMOPY_USE_CUDA)
    add_option(TOMOPY_USE_NVTX "Enable NVTX for Nsight" OFF)
    add_feature(CMAKE_CUDA_STANDARD "CUDA STL standard")
endif(TOMOPY_USE_CUDA)

if(TOMOPY_USE_SANITIZER)
    set(SANITIZER_TYPE leak CACHE STRING "Type of sanitizer")
    add_feature(SANITIZER_TYPE "Type of sanitizer (-fsanitize=${SANITIZER_TYPE})" ${SANITIZER_TYPE})
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

# RPATH settings
set(_RPATH_LINK OFF)
if(APPLE)
    set(_RPATH_LINK ON)
endif()
add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Hardcode installation rpath based on link path" ${_RPATH_LINK} ${_FEATURE})
unset(_RPATH_LINK)

# clang-tidy
if(TOMOPY_USE_CLANG_TIDY)
    find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
    add_feature(CLANG_TIDY_COMMAND "Path to clang-tidy command")
    if(NOT CLANG_TIDY_COMMAND)
        message(WARNING "TOMOPY_USE_CLANG_TIDY is ON but clang-tidy is not found!")
        set(TOMOPY_USE_CLANG_TIDY OFF)
    else()
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
        # Create a preprocessor definition that depends on .clang-tidy content so
        # the compile command will change when .clang-tidy changes.  This ensures
        # that a subsequent build re-runs clang-tidy on all sources even if they
        # do not otherwise need to be recompiled.  Nothing actually uses this
        # definition.  We add it to targets on which we run clang-tidy just to
        # get the build dependency on the .clang-tidy file.
        file(SHA1 ${PROJECT_SOURCE_DIR}/.clang-tidy clang_tidy_sha1)
        set(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
        unset(clang_tidy_sha1)
    endif()
    configure_file(${PROJECT_SOURCE_DIR}/.clang-tidy ${PROJECT_SOURCE_DIR}/.clang-tidy COPYONLY)
endif()
