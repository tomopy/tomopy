cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(WARNING "In-source build")
endif()

if(SKBUILD)
    message("Using scikit-build. cmake: ${CMAKE_COMMAND}...")
endif()

################################################################################
#
#   CMake settings
#
################################################################################

file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" VERSION_STRING "${VERSION_STRING}")
string(REGEX REPLACE "[A-Za-z].*" "" VERSION_STRING "${VERSION_STRING}")

set(tomopy_VERSION "${VERSION_STRING}" CACHE STRING "Version of tomopy project" FORCE)
message(STATUS "tomopy version ${tomopy_VERSION}")

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0053 NEW)
project(tomopy LANGUAGES C CXX VERSION ${tomopy_VERSION})

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

foreach(_TYPE MAJOR MINOR PATCH)
    set(tomopy_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}}
        CACHE STRING "tomopy ${_TYPE} version" FORCE)
    mark_as_advanced(tomopy_VERSION_${_TYPE})
endforeach(_TYPE MAJOR MINOR PATCH)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME development)
mark_as_advanced(tomopy_VERSION)

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

add_subdirectory(src)
add_subdirectory(test)

################################################################################
#   Information
################################################################################

print_features()
