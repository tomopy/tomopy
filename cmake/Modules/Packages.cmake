#
#   Find packages
#

include(FindPackageHandleStandardArgs)


################################################################################
#
#                               Threading
#
################################################################################

if(CMAKE_C_COMPILER_IS_INTEL OR CMAKE_CXX_COMPILER_IS_INTEL)
    if(NOT WIN32)
        set(THREADS_PREFER_PTHREAD_FLAG OFF CACHE BOOL "Use -pthread vs. -lpthread" FORCE)
    endif()

    find_package(Threads)
    if(Threads_FOUND)
        list(APPEND EXTERNAL_PRIVATE_LIBRARIES Threads::Threads)
    endif()
endif()


################################################################################
#
#        Prefix path to Anaconda installation
#
################################################################################
#
find_package(PythonInterp)
if(PYTHON_EXECUTABLE)
    get_filename_component(PYTHON_ROOT_DIR ${PYTHON_EXECUTABLE} DIRECTORY)
    get_filename_component(PYTHON_ROOT_DIR ${PYTHON_ROOT_DIR} DIRECTORY)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
        ${PYTHON_ROOT_DIR}
        ${PYTHON_ROOT_DIR}/bin
        ${PYTHON_ROOT_DIR}/lib
        ${PYTHON_ROOT_DIR}/include)
endif()


################################################################################
#
#        MKL (required)
#
################################################################################

find_package(MKL REQUIRED)

if(MKL_FOUND)
    list(APPEND EXTERNAL_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
    list(APPEND EXTERNAL_LIBRARIES ${MKL_LIBRARIES})
endif()


################################################################################
#
#        OpenCV (required)
#
################################################################################

set(OpenCV_COMPONENTS opencv_core opencv_imgproc)
find_package(OpenCV REQUIRED COMPONENTS ${OpenCV_COMPONENTS})
list(APPEND EXTERNAL_LIBRARIES ${OpenCV_LIBRARIES})


################################################################################
#
#        GCov
#
################################################################################

if(TOMOPY_USE_COVERAGE)
    find_library(GCOV_LIBRARY gcov)
    if(GCOV_LIBRARY)
        list(APPEND EXTERNAL_LIBRARIES ${GCOV_LIBRARY})
    else()
        list(APPEND EXTERNAL_LIBRARIES gcov)
    endif()
    add(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov")
endif()


################################################################################
#
#                               TiMemory
#
################################################################################

if(TOMOPY_USE_TIMEMORY)
    find_package(TiMemory)

    if(TiMemory_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${TiMemory_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES
            ${TiMemory_LIBRARIES} ${TiMemory_C_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_TIMEMORY)
    endif()

endif()


################################################################################
#
#        Google PerfTools
#
################################################################################

if(TOMOPY_USE_GPERF)
    find_package(GPerfTools COMPONENTS profiler)

    if(GPerfTools_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${GPerfTools_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${GPerfTools_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_GPERF)
    endif()

endif()


################################################################################
#
#        OpenMP
#
################################################################################

if(TOMOPY_USE_OPENMP)

    if(NOT c_fopenmp_simd AND NOT WIN32)
        find_package(OpenMP)

        if(OpenMP_FOUND)
            if(CMAKE_C_COMPILER_IS_PGI)
                string(REPLACE "-mp" "-mp${OpenMP_C_IMPL}" OpenMP_C_FLAGS "${OpenMP_C_FLAGS}")
            endif()

            if(CMAKE_CXX_COMPILER_IS_PGI)
                string(REPLACE "-mp" "-mp${OpenMP_C_IMPL}" OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
            endif()

            # C
            if(OpenMP_C_FOUND)
                list(APPEND ${PROJECT_NAME}_C_FLAGS ${OpenMP_C_FLAGS})
            endif()

            # C++
            if(OpenMP_CXX_FOUND)
                list(APPEND ${PROJECT_NAME}_CXX_FLAGS ${OpenMP_CXX_FLAGS})
            endif()
        else()
            message(WARNING "OpenMP not found")
            set(TOMOPY_USE_OPENMP OFF)
        endif()
    elseif(WIN32)
        message(STATUS "Ignoring TOMOPY_USE_OPENMP=ON because Windows + omp simd is supported")
        set(TOMOPY_USE_OPENMP OFF)
    else()
        message(STATUS "Ignoring TOMOPY_USE_OPENMP=ON because '-fopenmp-simd' is supported")
        set(TOMOPY_USE_OPENMP OFF)
    endif()

endif()


################################################################################
#
#        CUDA
#
################################################################################

if(TOMOPY_USE_CUDA)

    if(NOT CMAKE_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    endif()

    if(NOT CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
    endif()

    enable_language(CUDA)

    list(APPEND EXTERNAL_LIBRARIES ${CUDA_npp_LIBRARY})
    list(APPEND EXTERNAL_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

    if("CUDA" IN_LIST LANGUAGES)
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_CUDA)
        add_feature(${PROJECT_NAME}_CUDA_FLAGS "CUDA NVCC compiler flags")
        add_feature(CUDA_ARCH "CUDA architecture (e.g. '35' means '-arch=sm_35')")

        #   30, 32      + Kepler support
        #               + Unified memory programming
        #   35          + Dynamic parallelism support
        #   50, 52, 53  + Maxwell support
        #   60, 61, 62  + Pascal support
        #   70, 72      + Volta support
        #   75          + Turing support
        if(NOT DEFINED CUDA_ARCH)
            set(CUDA_ARCH "35")
        endif()

        if(TOMOPY_USE_NVTX)
            find_library(NVTX_LIBRARY
                NAMES nvToolsExt
                PATHS /usr/local/cuda
                HINTS /usr/local/cuda
                PATH_SUFFIXES lib lib64)
        else()
            unset(NVTX_LIBRARY CACHE)
        endif()

        if(NVTX_LIBRARY)
            list(APPEND EXTERNAL_LIBRARIES ${NVTX_LIBRARY})
            list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_NVTX)
        else()
            if(TOMOPY_USE_NVTX)
                set(TOMOPY_USE_NVTX OFF)
            endif()
        endif()

        list(APPEND ${PROJECT_NAME}_CUDA_FLAGS
            -arch=sm_${CUDA_ARCH}
            --default-stream per-thread)

        if(NOT WIN32)
            list(APPEND ${PROJECT_NAME}_CUDA_FLAGS}
                --compiler-bindir=${CMAKE_CXX_COMPILER})
        endif()

        add_option(TOMOPY_USE_CUDA_MAX_REGISTER_COUNT "Enable setting maximum register count" OFF)
        if(TOMOPY_USE_CUDA_MAX_REGISTER_COUNT)
            add_feature(CUDA_MAX_REGISTER_COUNT "CUDA maximum register count")
            set(CUDA_MAX_REGISTER_COUNT "24" CACHE STRING "CUDA maximum register count")
            list(APPEND ${PROJECT_NAME}_CUDA_FLAGS
            --maxrregcount=${CUDA_MAX_REGISTER_COUNT})
        endif()

    endif()

endif()


################################################################################
#
#        External variables
#
################################################################################

# user customization to force link libs
to_list(_LINKLIBS "${TOMOPY_USER_LIBRARIES};$ENV{TOMOPY_USER_LIBRARIES}")
foreach(_LIB ${_LINKLIBS})
    list(APPEND EXTERNAL_LIBRARIES ${_LIB})
endforeach()

# including the directories
safe_remove_duplicates(EXTERNAL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS})
safe_remove_duplicates(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES})
foreach(_DIR ${EXTERNAL_INCLUDE_DIRS})
    include_directories(SYSTEM ${_DIR})
endforeach()

# include dirs
set(TARGET_INCLUDE_DIRECTORIES
    ${PROJECT_SOURCE_DIR}/source/include
    ${PROJECT_SOURCE_DIR}/source/PTL/source
    ${EXTERNAL_INCLUDE_DIRS})
