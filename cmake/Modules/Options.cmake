
################################################################################
#
#        TiMemory Options
#
################################################################################

include(MacroUtilities)

# features
add_feature(CMAKE_BUILD_TYPE "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)")
add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
add_feature(${PROJECT_NAME}_C_FLAGS "C compiler flags")
add_feature(CMAKE_C_STANDARD "C languae standard")

# options (always available)
add_option(TOMOPY_USE_GPERF "Enable Google perftools profiler" OFF)
add_option(TOMOPY_USE_TIMEMORY "Enable TiMemory for timing+memory analysis" OFF)
add_option(TOMOPY_USE_ARCH "Enable architecture specific flags" OFF)
add_option(TOMOPY_USE_TBB "Enable TBB malloc" OFF)
add_option(TOMOPY_USE_ITTNOTIFY "Enable VTune API" OFF)
add_option(TOMOPY_USE_OPENMP "Enable OpenMP for SIMD" ON)

if(TOMOPY_USE_ARCH)
    add_option(TOMOPY_USE_AVX512 "Enable AVX-512 flags (if available)" OFF)
endif()

if(APPLE)
    add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH
        "Hardcode installation rpath based on link path" ON NO_FEATURE)
endif()
