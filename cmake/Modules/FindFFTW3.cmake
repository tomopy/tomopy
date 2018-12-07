#
# Module for locating Fast Fourier Transform 3 Library (FFTW3).
#
# Customizable variables:
#   FFTW3_ROOT
#       Specifies FFTW3's root directory.
##
# Read-only variables:
#   FFTW3_FOUND
#       Indicates whether the library has been found.
#
#   FFTW3_INCLUDE_DIRS
#       Specifies FFTW3's include directory.
#
#   FFTW3_LIBRARIES
#       Specifies FFTW3 libraries that should be passed to target_link_libararies.
#
#   FFTW3_<COMPONENT>_LIBRARIES
#       Specifies the libraries of a specific <COMPONENT>.
#
#   FFTW3_<COMPONENT>_FOUND
#       Indicates whether the specified <COMPONENT> was found.
#
#   FFTW3_CXX_LINK_FLAGS
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

INCLUDE (FindPackageHandleStandardArgs)

IF (CMAKE_VERSION VERSION_GREATER 2.8.7)
  SET (_FFTW3_CHECK_COMPONENTS FALSE)
ELSE (CMAKE_VERSION VERSION_GREATER 2.8.7)
  SET (_FFTW3_CHECK_COMPONENTS TRUE)
ENDIF (CMAKE_VERSION VERSION_GREATER 2.8.7)


#----- FFTW3 default components
IF(NOT FFTW3_FIND_COMPONENTS)
    set(FFTW3_FIND_COMPONENTS double)
ENDIF(NOT FFTW3_FIND_COMPONENTS)


#----- FFTW3 installation root
FIND_PATH (FFTW3_ROOT_DIR
  NAMES include/fftw3.h
  PATHS ${FFTW3_ROOT_DIR}
        ENV FFTW3_ROOT
        ENV FFTW3_ROOT_DIR
        ENV FFTW3ROOT
  DOC "FFTW3 root directory")


#----- FFTW3 include directory
FIND_PATH (FFTW3_INCLUDE_DIR
  NAMES fftw3.h
  HINTS ${FFTW3_ROOT_DIR}
  PATH_SUFFIXES include
  DOC "FFTW3 include directory")
SET (FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})


#----- Component options
set(_FFTW3_COMPONENT_OPTIONS
    single
    double
    quad
    long
    threads
    omp
    mpi
)
set(_FFTW3_COMPONENT_single_SUFFIX          "f" )
set(_FFTW3_COMPONENT_double_SUFFIX          ""  )
set(_FFTW3_COMPONENT_quad_SUFFIX            "q" )
set(_FFTW3_COMPONENT_long_SUFFIX            "l" )
set(_FFTW3_COMPONENT_threads_SUFFIX         "_threads" )
set(_FFTW3_COMPONENT_omp_SUFFIX             "_omp"  )
set(_FFTW3_COMPONENT_mpi_SUFFIX             "_mpi" )

foreach(_VARIANT mpi omp threads)

    foreach(_PRECISION single double quad long)
        list(APPEND _FFTW3_COMPONENT_OPTIONS ${_VARIANT}_${_PRECISION})
    endforeach()

    set(_FFTW3_COMPONENT_${_VARIANT}_single_SUFFIX   "f_${_VARIANT}" )
    set(_FFTW3_COMPONENT_${_VARIANT}_double_SUFFIX    "${_VARIANT}"  )
    set(_FFTW3_COMPONENT_${_VARIANT}_quad_SUFFIX     "q_${_VARIANT}" )
    set(_FFTW3_COMPONENT_${_VARIANT}_long_SUFFIX     "l_${_VARIANT}" )

endforeach()


#----- Find components
FOREACH (_FFTW3_COMPONENT ${FFTW3_FIND_COMPONENTS})

    IF(NOT "${_FFTW3_COMPONENT_OPTIONS}" MATCHES "${_FFTW3_COMPONENT}")
        MESSAGE(WARNING "${_FFTW3_COMPONENT} is not listed as a real component")
        continue()
    ENDIF()

    STRING (TOUPPER ${_FFTW3_COMPONENT} _FFTW3_COMPONENT_UPPER)

    SET (_FFTW3_LIBRARY_BASE FFTW3_${_FFTW3_COMPONENT_UPPER}_LIBRARY)
    SET (_FFTW3_LIBRARY_NAME fftw3${_FFTW3_COMPONENT_${_FFTW3_COMPONENT}_SUFFIX})

    FIND_LIBRARY (${_FFTW3_LIBRARY_BASE}
        NAMES ${_FFTW3_LIBRARY_NAME}
        HINTS ${FFTW3_ROOT_DIR}
        PATH_SUFFIXES ${_FFTW3_POSSIBLE_LIB_SUFFIXES}
        DOC "FFTW3 ${_FFTW3_COMPONENT} library")

    MARK_AS_ADVANCED (${_FFTW3_LIBRARY_BASE})

    SET (FFTW3_${_FFTW3_COMPONENT_UPPER}_FOUND TRUE)

    IF (NOT ${_FFTW3_LIBRARY_BASE})
        # Component missing: record it for a later report
        LIST (APPEND _FFTW3_MISSING_COMPONENTS ${_FFTW3_COMPONENT})
        SET (FFTW3_${_FFTW3_COMPONENT_UPPER}_FOUND FALSE)
    ENDIF (NOT ${_FFTW3_LIBRARY_BASE})

    SET (FFTW3_${_FFTW3_COMPONENT}_FOUND ${FFTW3_${_FFTW3_COMPONENT_UPPER}_FOUND})

    IF (${_FFTW3_LIBRARY_BASE})
        # setup the FFTW3_<COMPONENT>_LIBRARIES variable
        SET (FFTW3_${_FFTW3_COMPONENT_UPPER}_LIBRARIES ${${_FFTW3_LIBRARY_BASE}})
        LIST (APPEND FFTW3_LIBRARIES ${${_FFTW3_LIBRARY_BASE}})
    ELSE (${_FFTW3_LIBRARY_BASE})
        LIST (APPEND _FFTW3_MISSING_LIBRARIES ${_FFTW3_LIBRARY_BASE})
    ENDIF (${_FFTW3_LIBRARY_BASE})

ENDFOREACH (_FFTW3_COMPONENT ${FFTW3_FIND_COMPONENTS})

#----- Missing components
IF (DEFINED _FFTW3_MISSING_COMPONENTS AND _FFTW3_CHECK_COMPONENTS)
    IF (NOT FFTW3_FIND_QUIETLY)
        MESSAGE (STATUS "One or more FFTW3 components were not found:")
        # Display missing components indented, each on a separate line
        FOREACH (_FFTW3_MISSING_COMPONENT ${_FFTW3_MISSING_COMPONENTS})
            MESSAGE (STATUS "  " ${_FFTW3_MISSING_COMPONENT})
        ENDFOREACH (_FFTW3_MISSING_COMPONENT ${_FFTW3_MISSING_COMPONENTS})
    ENDIF (NOT FFTW3_FIND_QUIETLY)
ENDIF (DEFINED _FFTW3_MISSING_COMPONENTS AND _FFTW3_CHECK_COMPONENTS)


#----- Components
IF (NOT _FFTW3_CHECK_COMPONENTS)
    SET (_FFTW3_FPHSA_ADDITIONAL_ARGS HANDLE_COMPONENTS)
ENDIF (NOT _FFTW3_CHECK_COMPONENTS)

IF (CMAKE_VERSION VERSION_GREATER 2.8.2)
    LIST (APPEND _FFTW3_FPHSA_ADDITIONAL_ARGS VERSION_VAR FFTW3_VERSION)
ENDIF (CMAKE_VERSION VERSION_GREATER 2.8.2)


MARK_AS_ADVANCED (FFTW3_ROOT_DIR FFTW3_INCLUDE_DIR FFTW3_LIBRARY)

FIND_PACKAGE_HANDLE_STANDARD_ARGS (FFTW3 REQUIRED_VARS
    FFTW3_INCLUDE_DIR ${_FFTW3_MISSING_LIBRARIES}
    ${_FFTW3_FPHSA_ADDITIONAL_ARGS})

UNSET(_FFTW3_LIBRARY_BASE)
UNSET(_FFTW3_LIBRARY_NAME)
