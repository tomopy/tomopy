#
#   Find packages
#

include(FindPackageHandleStandardArgs)

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
        add_definitions(-DTOMOPY_USE_TIMEMORY)
    endif(TiMemory_FOUND)

endif(TOMOPY_USE_TIMEMORY)


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
        add_definitions(-DTOMOPY_USE_GPERF)
    endif(GPerfTools_FOUND)

endif(TOMOPY_USE_GPERF)


################################################################################
#
#        MKL
#
################################################################################

find_package(PythonInterp REQUIRED)

# anaconda should have installed MKL under this prefix
if(PYTHON_EXECUTABLE)
    get_filename_component(_MKL_PREFIX ${PYTHON_EXECUTABLE} DIRECTORY)
    if(UNIX)
        get_filename_component(_MKL_PREFIX ${_MKL_PREFIX} DIRECTORY)
    endif()
    list(APPEND CMAKE_PREFIX_PATH
        ${_MKL_PREFIX} # common path for UNIX
        ${_MKL_PREFIX}/Library # common path for Windows
        $ENV{CONDA_PREFIX} # fallback if set
    )
endif()

find_package(MKL REQUIRED)

if(MKL_FOUND)
    list(APPEND EXTERNAL_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
    list(APPEND EXTERNAL_LIBRARIES ${MKL_LIBRARIES})
    add_definitions(-DTOMOPY_USE_MKL)
endif(MKL_FOUND)


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
#        TBB
#
################################################################################
if(TOMOPY_USE_TBB)

    find_package(TBB COMPONENTS malloc)

    if(TBB_MALLOC_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${TBB_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${TBB_MALLOC_LIBRARIES})
    endif()

endif()


################################################################################
#
#        OpenMP
#
################################################################################
if(TOMOPY_USE_OPENMP)
    if(NOT c_fopenmp_simd) # no need if -fopenmp-simd is available
        find_package(OpenMP)

        if(OpenMP_FOUND)
            add(${PROJECT_NAME}_C_FLAGS "${OpenMP_C_FLAGS}")
        endif()
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
endforeach(_DIR ${EXTERNAL_INCLUDE_DIRS})


