#
#   Find packages
#

include(FindPackageHandleStandardArgs)


################################################################################
#
#                               Threading
#
################################################################################

if(NOT WIN32)
    set(CMAKE_THREAD_PREFER_PTHREAD ON)
endif()

find_package(Threads)
if(Threads_FOUND)
    list(APPEND EXTERNAL_LIBRARIES Threads::Threads)
endif()


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
    find_package(GPerfTools COMPONENTS profiler tcmalloc)

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

        # only define if GPU enabled
        if(TOMOPY_USE_GPU)
            list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_OPENMP)
        endif()
    endif()

endif()


################################################################################
#
#        OpenACC
#
################################################################################

if(TOMOPY_USE_OPENACC AND TOMOPY_USE_GPU)
    find_package(OpenACC)

    foreach(LANG C CXX)
        if(OpenACC_${LANG}_FOUND)
            list(APPEND ${PROJECT_NAME}_${LANG}_FLAGS ${OpenACC_${LANG}_FLAGS})
            list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_OPENACC)
        endif()
    endforeach()

endif()


################################################################################
#
#        TBB
#
################################################################################

if(TOMOPY_USE_TBB)
    find_package(TBB COMPONENTS malloc malloc_proxy)

    if(TBB_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${TBB_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${TBB_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_TBB)
    endif()

endif()


################################################################################
#
#        MKL
#
################################################################################

find_package(PythonInterp)

# anaconda should have installed MKL under this prefix
if(PYTHON_EXECUTABLE)
    get_filename_component(_MKL_PREFIX ${PYTHON_EXECUTABLE} DIRECTORY)
    get_filename_component(_MKL_PREFIX ${_MKL_PREFIX} DIRECTORY)
    list(APPEND CMAKE_PREFIX_PATH ${_MKL_PREFIX} ${_MKL_PREFIX}/lib ${_MKL_PREFIX}/include)
endif()

find_package(MKL REQUIRED)

if(MKL_FOUND)
    list(APPEND EXTERNAL_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
    list(APPEND EXTERNAL_LIBRARIES ${MKL_LIBRARIES})
    list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_MKL)
    list(APPEND ${PROJECT_NAME}_DEFINITIONS USE_MKL)
endif()


################################################################################
#
#        CUDA
#
################################################################################

if(TOMOPY_USE_CUDA AND TOMOPY_USE_GPU)

    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

    if("CUDA" IN_LIST LANGUAGES)
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_CUDA)
        add_feature(${PROJECT_NAME}_CUDA_FLAGS "CUDA NVCC compiler flags")
        add_feature(CUDA_ARCH "CUDA architecture (e.g. sm_35)")
        set(CUDA_ARCH "sm_35" CACHE STRING "CUDA architecture flag")

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
            list(APPEND EXTERNAL_CUDA_LIBRARIES ${NVTX_LIBRARY})
            list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_NVTX)
        endif()

        list(APPEND ${PROJECT_NAME}_CUDA_FLAGS
            -arch=${CUDA_ARCH}
            --default-stream per-thread)
    endif()

    #find_package(CUDA)

    #if(CUDA_FOUND)
        #foreach(_DIR ${CUDA_INCLUDE_DIRS})
        #    include_directories(SYSTEM ${_DIR})
        #endforeach()
        #list(APPEND EXTERNAL_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})

        #find_library(CUDA_LIBRARY
        #    NAMES cuda
        #    PATHS /usr/local/cuda
        #    HINTS /usr/local/cuda
        #    PATH_SUFFIXES lib lib64)

        #find_library(CUDART_LIBRARY
        #    NAMES cudart
        #    PATHS /usr/local/cuda
        #    HINTS /usr/local/cuda
        #    PATH_SUFFIXES lib lib64)

        #find_library(CUDART_STATIC_LIBRARY
        #    NAMES cudart_static
        #    PATHS /usr/local/cuda
        #    HINTS /usr/local/cuda
        #    PATH_SUFFIXES lib lib64)

        #find_library(RT_LIBRARY
        #    NAMES rt
        #    PATHS /usr /usr/local /opt/local
        #    HINTS /usr /usr/local /opt/local
        #    PATH_SUFFIXES lib lib64)

        #find_library(DL_LIBRARY
        #    NAMES dl
        #    PATHS /usr /usr/local /opt/local
        #    HINTS /usr /usr/local /opt/local
        #    PATH_SUFFIXES lib lib64)

        #foreach(NAME CUDA CUDART CUDART_STATIC NVTX RT DL)
        #    if(${NAME}_LIBRARY)
        #        list(APPEND EXTERNAL_CUDA_LIBRARIES ${${NAME}_LIBRARY})
        #    endif()
        #endforeach()

        #set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_BINARY_DIR}/BuildProducts/bin)
        #set(CUDA_SEPARABLE_COMPILATION OFF)

        #set(CUDA_ARCH "sm_35" CACHE STRING "CUDA architecture flag")
        #add(${PROJECT_NAME}_CUDA_FLAGS "-arch=${CUDA_ARCH} --default-stream per-thread")
        #list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_CUDA)
        #add_feature(CUDA_ARCH "CUDA architecture (e.g. sm_35)")
        #add_feature(CUDA_NVCC_FLAGS "CUDA NVCC compiler flags")
    #endif()

endif()


################################################################################
#
#        OpenCV
#
################################################################################
if(TOMOPY_USE_OPENCV)
    set(OpenCV_COMPONENTS opencv_core opencv_imgproc opencv_highgui)
    find_package(OpenCV COMPONENTS ${OpenCV_COMPONENTS})

    foreach(_COMPONENT ${OpenCV_COMPONENTS})
        if(OpenCV_${_COMPONENT}_FOUND)
            list(APPEND _COMPONENT_LIBS ${_COMPONENT})
        endif()
    endforeach()

    LIST(LENGTH OpenCV_COMPONENTS _NUM_TOTAL_COMPONENT_LIBS)
    LIST(LENGTH _COMPONENT_LIBS _NUM_FOUND_COMPONENT_LIBS)

    if(OpenCV_FOUND AND ${_NUM_FOUND_COMPONENT_LIBS} EQUAL ${_NUM_TOTAL_COMPONENT_LIBS})
        list(APPEND EXTERNAL_LIBRARIES ${OpenCV_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_OPENCV)
    else()
        set(msg "     OpenCV found: ${_COMPONENT_LIBS} (${_NUM_FOUND_COMPONENT_LIBS})\n")
        set(msg "${msg}    OpenCV needed: ${OpenCV_COMPONENTS} (${_NUM_FOUND_COMPONENT_LIBS})\n")
        message(WARNING "${msg}")
        unset(msg)
        set(TOMOPY_USE_OPENCV OFF)
    endif()

    unset(_NUM_FOUND_COMPONENT_LIBS)
    unset(_NUM_TOTAL_COMPONENT_LIBS)
endif()


################################################################################
#
#        ITTNOTIFY (for VTune)
#
################################################################################
if(TOMOPY_USE_ITTNOTIFY)
    find_package(ittnotify)

    if(ittnotify_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${ITTNOTIFY_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${ITTNOTIFY_LIBRARIES})
    else()
        message(WARNING "ittnotify not found. Set \"VTUNE_AMPLIFIER_201{7,8,9}_DIR\" or \"VTUNE_AMPLIFIER_XE_201{7,8,9}_DIR\" in environment")
    endif()
endif()


################################################################################
#
#        External variables
#
################################################################################

# including the directories
safe_remove_duplicates(EXTERNAL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS})
safe_remove_duplicates(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES})
foreach(_DIR ${EXTERNAL_INCLUDE_DIRS})
    include_directories(SYSTEM ${_DIR})
endforeach()

# configure package files
configure_file(${PROJECT_SOURCE_DIR}/tomopy/allocator/__init__.py.in
    ${PROJECT_SOURCE_DIR}/tomopy/allocator/__init__.py
    @ONLY)

configure_file(${PROJECT_SOURCE_DIR}/tomopy/allocator/__init__.py.in
    ${PROJECT_BINARY_DIR}/tomopy/allocator/__init__.py
    @ONLY)
