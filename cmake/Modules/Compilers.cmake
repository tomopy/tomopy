# ~~~
################################################################################
#
#        Compilers
#
################################################################################
#
#   sets (cached):
#
#       CMAKE_C_COMPILER_IS_<TYPE>
#       CMAKE_CXX_COMPILER_IS_<TYPE>
#
#   where TYPE is:
#       - GNU
#       - CLANG
#       - INTEL
#       - INTEL_ICC
#       - INTEL_ICPC
#       - PGI
#       - XLC
#       - HP_ACC
#       - MIPS
#       - MSVC
# ~~~

# include guard
if(__compilers_is_loaded)
  return()
endif()
set(__compilers_is_loaded ON)

include(CheckLanguage)
include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckCSourceRuns)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)

# ##############################################################################
# macro converting string to list
# ##############################################################################
macro(to_list _VAR _STR)
  string(REPLACE "  " " " ${_VAR} "${_STR}")
  string(REPLACE " " ";" ${_VAR} "${_STR}")
endmacro(
  to_list
  _VAR
  _STR)

# ##############################################################################
# macro converting string to list
# ##############################################################################
macro(to_string _VAR _STR)
  string(REPLACE ";" " " ${_VAR} "${_STR}")
endmacro(
  to_string
  _VAR
  _STR)

# ##############################################################################
# Macro to add to string
# ##############################################################################
macro(add _VAR _FLAG)
  if(NOT "${_FLAG}" STREQUAL "")
    if("${${_VAR}}" STREQUAL "")
      set(${_VAR} "${_FLAG}")
    else()
      set(${_VAR} "${${_VAR}} ${_FLAG}")
    endif()
  endif()
endmacro()

# ##############################################################################
# macro to remove duplicates from string
# ##############################################################################
macro(set_no_duplicates _VAR)
  if(NOT "${ARGN}" STREQUAL "")
    set(${_VAR} "${ARGN}")
  endif()
  # remove the duplicates
  if(NOT "${${_VAR}}" STREQUAL "")
    # create list of flags
    to_list(_VAR_LIST "${${_VAR}}")
    list(REMOVE_DUPLICATES _VAR_LIST)
    to_string(${_VAR} "${_VAR_LIST}")
  endif(NOT "${${_VAR}}" STREQUAL "")
endmacro(set_no_duplicates _VAR)

# ##############################################################################
# check C flag
# ##############################################################################
macro(ADD_C_FLAG_IF_AVAIL FLAG)
  if(NOT "${FLAG}" STREQUAL "")
    string(REGEX REPLACE "^-" "c_" FLAG_NAME "${FLAG}")
    string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
    string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
    string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
    check_c_compiler_flag("${FLAG}" ${FLAG_NAME})
    if(${FLAG_NAME})
      list(APPEND ${PROJECT_NAME}_C_FLAGS "${FLAG}")
    endif()
  endif()
endmacro()

# ##############################################################################
# check CXX flag
# ##############################################################################
macro(ADD_CXX_FLAG_IF_AVAIL FLAG)
  if(NOT "${FLAG}" STREQUAL "")
    string(REGEX REPLACE "^-" "cxx_" FLAG_NAME "${FLAG}")
    string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
    string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
    string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
    check_cxx_compiler_flag("${FLAG}" ${FLAG_NAME})
    if(${FLAG_NAME})
      list(APPEND ${PROJECT_NAME}_CXX_FLAGS "${FLAG}")
    endif()
  endif()
endmacro()

# ##############################################################################
# determine compiler types for each language
# ##############################################################################
foreach(LANG C CXX)

  macro(SET_COMPILER_VAR VAR _BOOL)
    set(CMAKE_${LANG}_COMPILER_IS_${VAR}
        ${_BOOL}
        CACHE BOOL "CMake ${LANG} compiler identification (${VAR})")
    mark_as_advanced(CMAKE_${LANG}_COMPILER_IS_${VAR})
  endmacro()

  if(("${LANG}" STREQUAL "C" AND CMAKE_COMPILER_IS_GNUCC)
     OR ("${LANG}" STREQUAL "CXX" AND CMAKE_COMPILER_IS_GNUCXX))

    # GNU compiler
    set_compiler_var(GNU ON)

  elseif(CMAKE_${LANG}_COMPILER MATCHES "icc.*")

    # Intel icc compiler
    set_compiler_var(INTEL ON)
    set_compiler_var(INTEL_ICC ON)

  elseif(CMAKE_${LANG}_COMPILER MATCHES "icpc.*")

    # Intel icpc compiler
    set_compiler_var(INTEL ON)
    set_compiler_var(INTEL_ICPC ON)

  elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "Clang" OR CMAKE_${LANG}_COMPILER_ID
                                                      MATCHES "AppleClang")

    # Clang/LLVM compiler
    set_compiler_var(CLANG ON)

  elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "PGI")

    # PGI compiler
    set_compiler_var(PGI ON)

  elseif(CMAKE_${LANG}_COMPILER MATCHES "xlC" AND UNIX)

    # IBM xlC compiler
    set_compiler_var(XLC ON)

  elseif(CMAKE_${LANG}_COMPILER MATCHES "aCC" AND UNIX)

    # HP aC++ compiler
    set_compiler_var(HP_ACC ON)

  elseif(
    CMAKE_${LANG}_COMPILER MATCHES "CC"
    AND CMAKE_SYSTEM_NAME MATCHES "IRIX"
    AND UNIX)

    # IRIX MIPSpro CC Compiler
    set_compiler_var(MIPS ON)

  elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "Intel")

    set_compiler_var(INTEL ON)
    set(CTYPE ICC)
    if("${LANG}" STREQUAL "CXX")
      set(CTYPE ICPC)
    endif()
    set_compiler_var(INTEL_${CTYPE} ON)

  elseif(CMAKE_${LANG}_COMPILER MATCHES "MSVC")

    # Windows Visual Studio compiler
    set_compiler_var(MSVC ON)

  endif()

  # set other to no
  foreach(
    TYPE
    GNU
    INTEL
    INTEL_ICC
    INTEL_ICPC
    CLANG
    PGI
    XLC
    HP_ACC
    MIPS
    MSVC)
    if(NOT ${CMAKE_${LANG}_COMPILER_IS_${TYPE}})
      set_compiler_var(${TYPE} OFF)
    endif()
  endforeach()

  if(APPLE)
    set(CMAKE_INCLUDE_SYSTEM_FLAG_${LANG} "-isystem ")
  endif(APPLE)

endforeach()
