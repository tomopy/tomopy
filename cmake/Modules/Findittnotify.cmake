# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

#------------------------------------------------------------------------------#

set(_ITTNOTIFY_PATH_HINTS)
foreach(_YEAR 2019 2018 2017)
    foreach(_SIGNATURE VTUNE_AMPLIFIER VTUNE_AMPLIFIER_XE)
        set(_VAR ${_SIGNATURE}_${_YEAR}_DIR)
        if(NOT "$ENV{${_VAR}}" STREQUAL "")
            list(APPEND _ITTNOTIFY_PATH_HINTS "$ENV{${_VAR}}")
        endif()
        if(NOT "${${_VAR}}" STREQUAL "")
            list(APPEND _ITTNOTIFY_PATH_HINTS "${${_VAR}}")
        endif()
        unset(_VAR)
endforeach()

#------------------------------------------------------------------------------#

find_path(ITTNOTIFY_INCLUDE_DIR
    NAMES ittnotify.h
    PATH_SUFFIXES include
    HINTS ${_ITTNOTIFY_PATH_HINTS}
    PATHS ${_ITTNOTIFY_PATH_HINTS}
)

#------------------------------------------------------------------------------#

find_library(ITTNOTIFY_LIBRARY
    NAMES ittnotify
    PATH_SUFFIXES lib lib64 lib32
    HINTS ${_ITTNOTIFY_PATH_HINTS}
    PATHS ${_ITTNOTIFY_PATH_HINTS}
)

#------------------------------------------------------------------------------#

if(ITTNOTIFY_INCLUDE_DIR)
    set(ITTNOTIFY_INCLUDE_DIRS ${ITTNOTIFY_INCLUDE_DIR})
endif()

#------------------------------------------------------------------------------#

if(ITTNOTIFY_LIBRARY)
    set(ITTNOTIFY_LIBRARIES ${ITTNOTIFY_LIBRARY})
endif()

#------------------------------------------------------------------------------#

mark_as_advanced(ITTNOTIFY_INCLUDE_DIR ITTNOTIFY_LIBRARY)
find_package_handle_standard_args(ittnotify REQUIRED_VARS
    ITTNOTIFY_INCLUDE_DIR ITTNOTIFY_LIBRARY)

#------------------------------------------------------------------------------#

unset(_ITTNOTIFY_PATH_HINTS)
#------------------------------------------------------------------------------#
