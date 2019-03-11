#
# Module for locating Intel's Math Kernel Library (MKL).
#
# Customizable variables:
#   MKL_ROOT_DIR
#       Specifies MKL's root directory.
#
#   MKL_64BIT_INTEGER
#       Enable MKL with 64-bit integer
#
#   MKL_THREADING
#       MKL threading model (options: sequential, OpenMP, TBB)
#
# Read-only variables:
#   MKL_FOUND
#       Indicates whether the library has been found.
#
#   MKL_INCLUDE_DIRS
#       Specifies MKL's include directory.
#
#   MKL_LIBRARIES
#       Specifies MKL libraries that should be passed to target_link_libararies.
#
#   MKL_<COMPONENT>_LIBRARIES
#       Specifies the libraries of a specific <COMPONENT>.
#
#   MKL_<COMPONENT>_FOUND
#       Indicates whether the specified <COMPONENT> was found.
#
#   MKL_CXX_LINK_FLAGS
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
# FITNESS FOR A PARTMKLLAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

INCLUDE (FindPackageHandleStandardArgs)

IF (CMAKE_VERSION VERSION_GREATER 2.8.7)
  SET (_MKL_CHECK_COMPONENTS FALSE)
ELSE (CMAKE_VERSION VERSION_GREATER 2.8.7)
  SET (_MKL_CHECK_COMPONENTS TRUE)
ENDIF (CMAKE_VERSION VERSION_GREATER 2.8.7)


#----- MKL installation root
FIND_PATH (MKL_ROOT_DIR
  NAMES include/mkl.h
  PATHS ${MKL_ROOT_DIR}
        ENV MKL_ROOT_DIR
        ENV MKLROOT
  DOC "MKL root directory")


#----- MKL include directory
FIND_PATH (MKL_INCLUDE_DIR
  NAMES mkl.h
  HINTS ${MKL_ROOT_DIR}
  PATH_SUFFIXES include
  DOC "MKL include directory")
SET (MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})


#----- Library suffix
IF (CMAKE_SIZEOF_VOID_P EQUAL 8)
  SET (_MKL_POSSIBLE_LIB_SUFFIXES lib/intel64/${_MKL_COMPILER})
  SET (_MKL_POSSIBLE_BIN_SUFFIXES bin/intel64/${_MKL_COMPILER})
ELSE (CMAKE_SIZEOF_VOID_P EQUAL 8)
  SET (_MKL_POSSIBLE_LIB_SUFFIXES lib/ia32/${_MKL_COMPILER})
  SET (_MKL_POSSIBLE_BIN_SUFFIXES bin/ia32/${_MKL_COMPILER})
ENDIF (CMAKE_SIZEOF_VOID_P EQUAL 8)
LIST (APPEND _MKL_POSSIBLE_LIB_SUFFIXES lib/$ENV{MKL_ARCH_PLATFORM})


#----- MKL runtime library
IF("${MKL_FIND_COMPONENTS}" STREQUAL "")
    LIST(APPEND MKL_FIND_COMPONENTS rt)
ENDIF("${MKL_FIND_COMPONENTS}" STREQUAL "")


#----- Component options
set(_MKL_COMPONENT_OPTIONS
    ao_worker
    avx2
    avx512_mic
    avx512
    avx
    blacs_intelmpi_ilp64
    blacs_intelmpi_lp64
    blacs_openmpi_ilp64
    blacs_openmpi_lp64
    blacs_sgimpt_ilp64
    blacs_sgimpt_lp64
    cdft_core
    core
    def
    gf_ilp64
    gf_lp64
    gnu_thread
    intel_ilp64
    intel_lp64
    intel_thread
    mc3
    mc
    rt
    scalapack_ilp64
    scalapack_lp64
    sequential
    mkl_thread
    vml_avx2
    vml_avx512_mic
    vml_avx512
    vml_avx
    vml_cmpt
    vml_def
    vml_mc2
    vml_mc3
    vml_mc
)


#----- Find components
FOREACH (_MKL_COMPONENT ${MKL_FIND_COMPONENTS})
    IF(NOT "${_MKL_COMPONENT_OPTIONS}" MATCHES "${_MKL_COMPONENT}")
        MESSAGE(WARNING "${_MKL_COMPONENT} is not listed as a real component")
    ENDIF()

    STRING (TOUPPER ${_MKL_COMPONENT} _MKL_COMPONENT_UPPER)
    SET (_MKL_LIBRARY_BASE MKL_${_MKL_COMPONENT_UPPER}_LIBRARY)

    SET (_MKL_LIBRARY_NAME mkl_${_MKL_COMPONENT})

    FIND_LIBRARY (${_MKL_LIBRARY_BASE}
        NAMES ${_MKL_LIBRARY_NAME}
        HINTS ${MKL_ROOT_DIR}
        PATH_SUFFIXES ${_MKL_POSSIBLE_LIB_SUFFIXES}
        DOC "MKL ${_MKL_COMPONENT} library")

    MARK_AS_ADVANCED (${_MKL_LIBRARY_BASE})

    SET (MKL_${_MKL_COMPONENT_UPPER}_FOUND TRUE)

    IF (NOT ${_MKL_LIBRARY_BASE})
        # Component missing: record it for a later report
        LIST (APPEND _MKL_MISSING_COMPONENTS ${_MKL_COMPONENT})
        SET (MKL_${_MKL_COMPONENT_UPPER}_FOUND FALSE)
    ENDIF (NOT ${_MKL_LIBRARY_BASE})

    SET (MKL_${_MKL_COMPONENT}_FOUND ${MKL_${_MKL_COMPONENT_UPPER}_FOUND})

    IF (${_MKL_LIBRARY_BASE})
        # setup the MKL_<COMPONENT>_LIBRARIES variable
        SET (MKL_${_MKL_COMPONENT_UPPER}_LIBRARIES ${${_MKL_LIBRARY_BASE}})
        LIST (APPEND MKL_LIBRARIES ${${_MKL_LIBRARY_BASE}})
    ELSE (${_MKL_LIBRARY_BASE})
        LIST (APPEND _MKL_MISSING_LIBRARIES ${_MKL_LIBRARY_BASE})
    ENDIF (${_MKL_LIBRARY_BASE})
ENDFOREACH (_MKL_COMPONENT ${MKL_FIND_COMPONENTS})


#----- Missing components
IF (DEFINED _MKL_MISSING_COMPONENTS AND _MKL_CHECK_COMPONENTS)
    IF (NOT MKL_FIND_QUIETLY)
        MESSAGE (STATUS "One or more MKL components were not found:")
        # Display missing components indented, each on a separate line
        FOREACH (_MKL_MISSING_COMPONENT ${_MKL_MISSING_COMPONENTS})
            MESSAGE (STATUS "  " ${_MKL_MISSING_COMPONENT})
        ENDFOREACH (_MKL_MISSING_COMPONENT ${_MKL_MISSING_COMPONENTS})
    ENDIF (NOT MKL_FIND_QUIETLY)
ENDIF (DEFINED _MKL_MISSING_COMPONENTS AND _MKL_CHECK_COMPONENTS)


#----- Determine library's version
SET (_MKL_VERSION_HEADER ${MKL_INCLUDE_DIR}/mkl_version.h)

IF (EXISTS ${_MKL_VERSION_HEADER})
    FILE (READ ${_MKL_VERSION_HEADER} _MKL_VERSION_CONTENTS)

    STRING (REGEX REPLACE ".*#define __INTEL_MKL__[ \t]+([0-9]+).*" "\\1"
        MKL_VERSION_MAJOR "${_MKL_VERSION_CONTENTS}")
    STRING (REGEX REPLACE ".*#define __INTEL_MKL_MINOR__[ \t]+([0-9]+).*" "\\1"
        MKL_VERSION_MINOR "${_MKL_VERSION_CONTENTS}")
    STRING (REGEX REPLACE ".*#define __INTEL_MKL_UPDATE__[ \t]+([0-9]+).*" "\\1"
        MKL_VERSION_PATCH "${_MKL_VERSION_CONTENTS}")

    SET (MKL_VERSION ${MKL_VERSION_MAJOR}.${MKL_VERSION_MINOR}.${MKL_VERSION_PATCH})
    SET (MKL_VERSION_COMPONENTS 3)
ENDIF (EXISTS ${_MKL_VERSION_HEADER})


#----- Threading
SET (MKL_BINARY_DIR ${MKL_BINARY_DIR} CACHE PATH "MKL binary directory")
IF (NOT MKL_BINARY_DIR)
    SET (_MKL_UPDATE_BINARY_DIR TRUE)
ELSE (NOT MKL_BINARY_DIR)
    SET (_MKL_UPDATE_BINARY_DIR FALSE)
ENDIF (NOT MKL_BINARY_DIR)
SET (_MKL_BINARY_DIR_HINTS ${_MKL_POSSIBLE_BIN_SUFFIXES})
IF (MKL_BINARY_DIR AND _MKL_UPDATE_BINARY_DIR)
    SET (_MKL_BINARY_DIR ${MKL_BINARY_DIR})
    UNSET (MKL_BINARY_DIR CACHE)

    IF (_MKL_BINARY_DIR)
        GET_FILENAME_COMPONENT (MKL_BINARY_DIR ${_MKL_BINARY_DIR} PATH)
    ENDIF (_MKL_BINARY_DIR)
ENDIF (MKL_BINARY_DIR AND _MKL_UPDATE_BINARY_DIR)


#----- Components
IF (NOT _MKL_CHECK_COMPONENTS)
    SET (_MKL_FPHSA_ADDITIONAL_ARGS HANDLE_COMPONENTS)
ENDIF (NOT _MKL_CHECK_COMPONENTS)
IF (CMAKE_VERSION VERSION_GREATER 2.8.2)
    LIST (APPEND _MKL_FPHSA_ADDITIONAL_ARGS VERSION_VAR MKL_VERSION)
ENDIF (CMAKE_VERSION VERSION_GREATER 2.8.2)


#----- Threading
IF(DEFINED MKL_THREADING)
    STRING(TOUPPER "${MKL_THREADING}" _MKL_THREADING)
    IF("${_MKL_THREADING}" STREQUAL "SEQUENTIAL")
        SET(MKL_CXX_LINK_FLAGS "${MKL_CXX_LINK_FLAGS} -mkl=sequential")
    ELSEIF("${_MKL_THREADING}" STREQUAL "OPENMP")
        FIND_PACKAGE(OpenMP REQUIRED QUIET)
        LIST(APPEND MKL_LIBRARIES ${OpenMP_LIBRARIES})
        LIST(APPEND MKL_INCLUDE_DIRS ${OpenMP_INCLUDE_DIRS})
        SET(MKL_CXX_LINK_FLAGS "${MKL_CXX_LINK_FLAGS} --no-as-needed -mkl=parallel ${OpenMP_CXX_FLAGS}")
        IF(NOT "${CMAKE_EXE_LINKER_FLAGS}" STREQUAL "")
            SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        ELSE(NOT "${CMAKE_EXE_LINKER_FLAGS}" STREQUAL "")
            SET(CMAKE_EXE_LINKER_FLAGS "${OpenMP_EXE_LINKER_FLAGS}")
        ENDIF(NOT "${CMAKE_EXE_LINKER_FLAGS}" STREQUAL "")
    ELSEIF("${_MKL_THREADING}" STREQUAL "TBB")
        FIND_PACKAGE(TBB REQUIRED QUIET)
        LIST(APPEND MKL_LIBRARIES ${TBB_LIBRARIES})
        LIST(APPEND MKL_INCLUDE_DIRS ${TBB_INCLUDE_DIRS})
        SET(MKL_CXX_LINK_FLAGS "${MKL_CXX_LINK_FLAGS} -mkl=parallel")
    ELSE("${_MKL_THREADING}" STREQUAL "SEQUENTIAL")
        MESSAGE(WARNING "MKL Threading options: sequential, OpenMP, TBB")
        MESSAGE(FATAL_ERROR "Unknown MKL_THREADING model: ${_MKL_THREADING}")
    ENDIF()
    UNSET(_MKL_THREADING)
ENDIF(DEFINED MKL_THREADING)


#----- 64 bit integer support
if(DEFINED MKL_64BIT_INTEGER)
    LIST(APPEND MKL_DEFINITIONS MKL_ILP64)
endif()


#----- Threading
if(NOT MSVC)
    SET(CMAKE_THREAD_PREFER_PTHREADS ON)
endif()
FIND_PACKAGE(Threads QUIET)
UNSET(CMAKE_THREAD_PREFER_PTHREADS)


#----- libm and libdl libraries
if(NOT MSVC)
    FIND_LIBRARY(MKL_m_LIBRARY
        NAMES m
        DOC "MKL m library")
    FIND_LIBRARY(MKL_dl_LIBRARY
        NAMES dl
        DOC "MKL m library")
    MARK_AS_ADVANCED(MKL_m_LIBRARY MKL_dl_LIBRARY)
    foreach(NAME m dl)
        if(MKL_${NAME}_LIBRARY)
            LIST(APPEND MKL_LIBRARIES ${MKL_${NAME}_LIBRARY})
        endif()
    endforeach()
    if(NOT "${CMAKE_SHARED_LINKER_FLAGS}" STREQUAL "")
        if(NOT APPLE AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined -Wl,--no-as-needed")
        endif()
    else(NOT "${CMAKE_SHARED_LINKER_FLAGS}" STREQUAL "")
        if(NOT APPLE AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            SET(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined -Wl,--no-as-needed")
        endif()
    endif(NOT "${CMAKE_SHARED_LINKER_FLAGS}" STREQUAL "")
endif()

if(Threads_FOUND)
    LIST(APPEND MKL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
endif()

MARK_AS_ADVANCED (MKL_ROOT_DIR MKL_INCLUDE_DIR MKL_LIBRARY MKL_BINARY_DIR)

FIND_PACKAGE_HANDLE_STANDARD_ARGS (MKL REQUIRED_VARS MKL_ROOT_DIR
    MKL_INCLUDE_DIR ${_MKL_MISSING_LIBRARIES}
    ${_MKL_FPHSA_ADDITIONAL_ARGS})

