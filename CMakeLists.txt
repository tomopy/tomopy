# need CUDA as a first-class language
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(WARNING "In-source build")
endif()

if(SKBUILD)
    message("Using scikit-build. cmake: ${CMAKE_COMMAND}...")
else()
    message("Using cmake: ${CMAKE_COMMAND}...")
endif()

################################################################################
#
#   CMake settings
#
################################################################################
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0053 NEW)

project(tomopy LANGUAGES C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME python)

################################################################################

include(MacroUtilities)
include(Compilers)
include(Options)
include(GNUInstallDirs)
include(ProjectSettings)
include(BuildSettings)
include(Packages)
include(ClangFormat)

################################################################################
#   tomopy source
################################################################################

add_subdirectory(source)

################################################################################
#   Information
################################################################################

print_features()
