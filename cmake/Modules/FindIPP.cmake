#
# Module for locating Intel's Math Kernel Library (IPP).
#
# Customizable variables:
#   IPP_ROOT_DIR
#       Specifies IPP's root directory.
#
#   IPP_64BIT_INTEGER
#       Enable IPP with 64-bit integer
#
#   IPP_THREADING
#       IPP threading model (options: sequential, OpenMP, TBB)
#
# Read-only variables:
#   IPP_FOUND
#       Indicates whether the library has been found.
#
#   IPP_INCLUDE_DIRS
#       Specifies IPP's include directory.
#
#   IPP_LIBRARIES
#       Specifies IPP libraries that should be passed to target_link_libararies.
#
#   IPP_<COMPONENT>_LIBRARIES
#       Specifies the libraries of a specific <COMPONENT>.
#
#   IPP_<COMPONENT>_FOUND
#       Indicates whether the specified <COMPONENT> was found.
#
#   IPP_CXX_LINK_FLAGS
#       C++ linker compile flags
#
# Copyright (c) 2017 Jonathan Madsen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTIPPLAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

INCLUDE (FindPackageHandleStandardArgs)

IF (CMAKE_VERSION VERSION_GREATER 2.8.7)
  SET (_IPP_CHECK_COMPONENTS FALSE)
ELSE (CMAKE_VERSION VERSION_GREATER 2.8.7)
  SET (_IPP_CHECK_COMPONENTS TRUE)
ENDIF (CMAKE_VERSION VERSION_GREATER 2.8.7)


#----- IPP installation root
FIND_PATH (IPP_ROOT_DIR
  NAMES include/ipp.h
  PATHS ${IPP_ROOT_DIR}
        ENV IPP_ROOT_DIR
        ENV IPPROOT
  DOC "IPP root directory")


#----- IPP include directory
FIND_PATH (IPP_INCLUDE_DIR
  NAMES ipp.h
  HINTS ${IPP_ROOT_DIR}
  PATH_SUFFIXES include
  DOC "IPP include directory")
SET (IPP_INCLUDE_DIRS ${IPP_INCLUDE_DIR})


#----- Library suffix
IF (CMAKE_SIZEOF_VOID_P EQUAL 8)
  SET (_IPP_POSSIBLE_LIB_SUFFIXES lib/intel64/${_IPP_COMPILER})
  SET (_IPP_POSSIBLE_BIN_SUFFIXES bin/intel64/${_IPP_COMPILER})
ELSE (CMAKE_SIZEOF_VOID_P EQUAL 8)
  SET (_IPP_POSSIBLE_LIB_SUFFIXES lib/ia32/${_IPP_COMPILER})
  SET (_IPP_POSSIBLE_BIN_SUFFIXES bin/ia32/${_IPP_COMPILER})
ENDIF (CMAKE_SIZEOF_VOID_P EQUAL 8)
LIST (APPEND _IPP_POSSIBLE_LIB_SUFFIXES lib/$ENV{IPP_ARCH_PLATFORM})


#----- IPP runtime library
IF("${IPP_FIND_COMPONENTS}" STREQUAL "")
    LIST(APPEND IPP_FIND_COMPONENTS core i s cv)
ENDIF("${IPP_FIND_COMPONENTS}" STREQUAL "")


#----- Component options
set(_IPP_COMPONENT_OPTIONS
    cc
    cce9
    cck0
    ccl9
    ccn8
    ccy8
    ch
    che9
    chk0
    chl9
    chn8
    chy8
    core
    cv
    cve9
    cvk0
    cvl9
    cvn8
    cvy8
    dc
    dce9
    dck0
    dcl9
    dcn8
    dcy8
    i
    ie9
    ik0
    il9
    in8
    iy8
    s
    se9
    sk0
    sl9
    sn8
    sy8
    vm
    vme9
    vmk0
    vml9
    vmn8
    vmy8
)


#----- Find components
FOREACH (_IPP_COMPONENT ${IPP_FIND_COMPONENTS})
    IF(NOT "${_IPP_COMPONENT_OPTIONS}" MATCHES "${_IPP_COMPONENT}")
        MESSAGE(WARNING "${_IPP_COMPONENT} is not listed as a real component")
    ENDIF()

    STRING (TOUPPER ${_IPP_COMPONENT} _IPP_COMPONENT_UPPER)
    SET (_IPP_LIBRARY_BASE IPP_${_IPP_COMPONENT_UPPER}_LIBRARY)

    SET (_IPP_LIBRARY_NAME ipp${_IPP_COMPONENT})

    FIND_LIBRARY (${_IPP_LIBRARY_BASE}
        NAMES ${_IPP_LIBRARY_NAME}
        HINTS ${IPP_ROOT_DIR}
        PATH_SUFFIXES ${_IPP_POSSIBLE_LIB_SUFFIXES}
        DOC "IPP ${_IPP_COMPONENT} library")

    MARK_AS_ADVANCED (${_IPP_LIBRARY_BASE})

    SET (IPP_${_IPP_COMPONENT_UPPER}_FOUND TRUE)

    IF (NOT ${_IPP_LIBRARY_BASE})
        # Component missing: record it for a later report
        LIST (APPEND _IPP_MISSING_COMPONENTS ${_IPP_COMPONENT})
        SET (IPP_${_IPP_COMPONENT_UPPER}_FOUND FALSE)
    ENDIF (NOT ${_IPP_LIBRARY_BASE})

    SET (IPP_${_IPP_COMPONENT}_FOUND ${IPP_${_IPP_COMPONENT_UPPER}_FOUND})

    IF (${_IPP_LIBRARY_BASE})
        # setup the IPP_<COMPONENT>_LIBRARIES variable
        SET (IPP_${_IPP_COMPONENT_UPPER}_LIBRARIES ${${_IPP_LIBRARY_BASE}})
        LIST (APPEND IPP_LIBRARIES ${${_IPP_LIBRARY_BASE}})
    ELSE (${_IPP_LIBRARY_BASE})
        LIST (APPEND _IPP_MISSING_LIBRARIES ${_IPP_LIBRARY_BASE})
    ENDIF (${_IPP_LIBRARY_BASE})
ENDFOREACH (_IPP_COMPONENT ${IPP_FIND_COMPONENTS})


#----- Missing components
IF (DEFINED _IPP_MISSING_COMPONENTS AND _IPP_CHECK_COMPONENTS)
    IF (NOT IPP_FIND_QUIETLY)
        MESSAGE (STATUS "One or more IPP components were not found:")
        # Display missing components indented, each on a separate line
        FOREACH (_IPP_MISSING_COMPONENT ${_IPP_MISSING_COMPONENTS})
            MESSAGE (STATUS "  " ${_IPP_MISSING_COMPONENT})
        ENDFOREACH (_IPP_MISSING_COMPONENT ${_IPP_MISSING_COMPONENTS})
    ENDIF (NOT IPP_FIND_QUIETLY)
ENDIF (DEFINED _IPP_MISSING_COMPONENTS AND _IPP_CHECK_COMPONENTS)


#----- Determine library's version
SET (_IPP_VERSION_HEADER ${IPP_INCLUDE_DIR}/ippversion.h)

IF (EXISTS ${_IPP_VERSION_HEADER})
    FILE (READ ${_IPP_VERSION_HEADER} _IPP_VERSION_CONTENTS)

    STRING (REGEX REPLACE ".*#define IPP_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1"
        IPP_VERSION_MAJOR "${_IPP_VERSION_CONTENTS}")
    STRING (REGEX REPLACE ".*#define IPP_VERSION_MINOR[ \t]+([0-9]+).*" "\\1"
        IPP_VERSION_MINOR "${_IPP_VERSION_CONTENTS}")
    STRING (REGEX REPLACE ".*#define IPP_VERSION_UPDATE[ \t]+([0-9]+).*" "\\1"
        IPP_VERSION_PATCH "${_IPP_VERSION_CONTENTS}")

    SET (IPP_VERSION ${IPP_VERSION_MAJOR}.${IPP_VERSION_MINOR}.${IPP_VERSION_PATCH})
    SET (IPP_VERSION_COMPONENTS 3)
ENDIF (EXISTS ${_IPP_VERSION_HEADER})

#----- Components
IF (NOT _IPP_CHECK_COMPONENTS)
    SET (_IPP_FPHSA_ADDITIONAL_ARGS HANDLE_COMPONENTS)
ENDIF (NOT _IPP_CHECK_COMPONENTS)

IF (CMAKE_VERSION VERSION_GREATER 2.8.2)
    LIST (APPEND _IPP_FPHSA_ADDITIONAL_ARGS VERSION_VAR IPP_VERSION)
ENDIF (CMAKE_VERSION VERSION_GREATER 2.8.2)

MARK_AS_ADVANCED (IPP_ROOT_DIR IPP_INCLUDE_DIR IPP_LIBRARY IPP_BINARY_DIR)

FIND_PACKAGE_HANDLE_STANDARD_ARGS (IPP REQUIRED_VARS IPP_ROOT_DIR
    IPP_INCLUDE_DIR ${_IPP_MISSING_LIBRARIES}
    ${_IPP_FPHSA_ADDITIONAL_ARGS})

