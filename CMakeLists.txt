# need CUDA as a first-class language; use newer FindCUDAToolkit
# need CMAKE_CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  set(MSG "")
  message(
    STATUS "Warning! Building from the source directory is not recommended")
  message(
    STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
  message(STATUS "and build from a separate directory")
  message(WARNING "In-source build")
endif()

# ##############################################################################
#
# CMake settings
#
# ##############################################################################
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0053 NEW)
cmake_policy(SET CMP0104 NEW)

# get version from git if available; fall back to .gitarchival.txt
execute_process(
  COMMAND git describe --tags
  OUTPUT_VARIABLE GIT_TAG
  ERROR_QUIET)
if("${GIT_TAG}" STREQUAL "")
  file(READ ".git_archival.txt" GIT_TAG)
endif()
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" GIT_TAG ${GIT_TAG})
if("${GIT_TAG}" STREQUAL "")
  set(GIT_TAG "0.0.0")
endif()

project(
  tomopy
  VERSION ${GIT_TAG}
  LANGUAGES C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules
                      ${CMAKE_MODULE_PATH})

# ##############################################################################

include(MacroUtilities)
include(Compilers)
include(Options)
include(GNUInstallDirs)
include(ProjectSettings)
include(BuildSettings)
include(Packages)
include(ClangFormat)

# ##############################################################################
# tomopy source
# ##############################################################################

add_subdirectory(source)

get_property(libtomo_TARGETS GLOBAL PROPERTY libtomo_TARGETS)

# ##############################################################################
# Information
# ##############################################################################

print_features()

install(
  EXPORT libtomoTargets
  FILE libtomoTargets.cmake
  NAMESPACE libtomo::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libtomo)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/libtomoConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/libtomoConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libtomo)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libtomoConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/libtomoConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libtomo)

export(
  EXPORT libtomoTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/libtomoTargets.cmake"
  NAMESPACE libtomo::)
