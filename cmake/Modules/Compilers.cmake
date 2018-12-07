
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
#

# include guard
if(__compilers_is_loaded)
    return()
endif()
set(__compilers_is_loaded ON)

include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckCSourceRuns)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)

################################################################################
# macro generate a test compile file
################################################################################
macro(generate_test_project)
    if(EXISTS ${CMAKE_SOURCE_DIR}/cmake/Templates/compile-test.cc.in)
        set(HEADER_FILE "stdlib.h")
        configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/compile-test.cc.in
            ${CMAKE_BINARY_DIR}/CMakeFiles/compile-testing/compile-test.cc @ONLY)
    endif()
endmacro()


################################################################################
# macro converting string to list
################################################################################
macro(to_list _VAR _STR)
    STRING(REPLACE "  " " " ${_VAR} "${_STR}")
    STRING(REPLACE " " ";" ${_VAR} "${_STR}")
endmacro(to_list _VAR _STR)


################################################################################
# macro converting string to list
################################################################################
macro(to_string _VAR _STR)
    STRING(REPLACE ";" " " ${_VAR} "${_STR}")
endmacro(to_string _VAR _STR)


################################################################################
#   Macro to add to string
################################################################################
macro(add _VAR _FLAG)
    if(NOT "${_FLAG}" STREQUAL "")
        if("${${_VAR}}" STREQUAL "")
            set(${_VAR} "${_FLAG}")
        else()
            set(${_VAR} "${${_VAR}} ${_FLAG}")
        endif()
    endif()
endmacro()


################################################################################
# macro to remove duplicates from string
################################################################################
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


################################################################################
# function for test compiling with flags
################################################################################
function(test_compile _LANG _VAR _FLAG)
    # recursion requires this
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/cmake/Templates/CMakeLists.txt.in)
        return()
    endif()
    # generate test file
    generate_test_project()
    set(LANG "${_LANG}")
    set(COMPILE_FLAGS "${_FLAG}")
    set(COMPILER "${CMAKE_${LANG}_COMPILER}")
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/CMakeLists.txt.in
        ${CMAKE_BINARY_DIR}/CMakeFiles/compile-testing/CMakeLists.txt @ONLY)
    # try compiling with flag
    try_compile(RET
        ${CMAKE_BINARY_DIR}/CMakeFiles/compile-testing
        ${CMAKE_BINARY_DIR}/CMakeFiles/compile-testing
        CompileTest
        CMAKE_FLAGS
            -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
        OUTPUT_VARIABLE RET_OUT)
    # add flag if successful
    set(${_VAR} ${RET} PARENT_SCOPE)
endfunction(test_compile _LANG _VAR _FLAGS)


################################################################################
# macro for adding C/C++ compiler flags to variable
################################################################################
macro(add_flags _LANG _VAR _FLAGS)
    if("${${_VAR}}" STREQUAL "${_FLAGS}")
        set(_VAR_GOOD)
    else("${${_VAR}}" STREQUAL "${_FLAGS}")
        set(_VAR_GOOD "${${_VAR}}")
    endif("${${_VAR}}" STREQUAL "${_FLAGS}")

    if(WIN32)
        set(WERR_FLAG "/Werror")
    else(WIN32)
        set(WERR_FLAG "-Werror")
    endif(WIN32)
    test_compile(${_LANG} HAS_WERROR "${WERR_FLAG}")
    set(WARNING_AS_ERROR "")
    if(HAS_WERROR)
        set(WARNING_AS_ERROR "${WERR_FLAG}")
    endif(HAS_WERROR)

    # test whole string
    test_compile(${_LANG} COMPILE_SUCCESS "${WARNING_AS_ERROR} ${_FLAGS}")
    if(COMPILE_SUCCESS)
        # add whole string if worked
        add(_VAR_GOOD "${_FLAGS}")
    else(COMPILE_SUCCESS)
        # test individually
        to_list(_LANGFLAGS "${_FLAGS}")
        foreach(_FLAG ${_LANGFLAGS})
            # check individual flag
            test_compile(${_LANG} COMPILE_SUCCESS
                "${WARNING_AS_ERROR} ${_FLAG}")
            if(COMPILE_SUCCESS)
                # add individual flag
                add(_VAR_GOOD "${_FLAG}")
            endif(COMPILE_SUCCESS)
        endforeach(_FLAG ${_LANGFLAGS})
    endif(COMPILE_SUCCESS)

    # set the variable to the working flags
    set(${_VAR} "${_VAR_GOOD}")
    # remove the duplicates
    set_no_duplicates(${_VAR})

endmacro(add_flags _LANG _VAR _FLAGS)


################################################################################
# macro for adding C compiler flags to variable
################################################################################
macro(add_c_flags _VAR _FLAGS)        
    # cache the flags to test
    set(CACHED_C_${_VAR}_TEST_FLAGS "${_FLAGS}" CACHE STRING
        "Possible C flags for ${_VAR}")
    mark_as_advanced(CACHED_C_${_VAR}_TEST_FLAGS)
    # if flags were changed or not previously processed
    if(NOT "${CACHED_C_${_VAR}_TEST_FLAGS}" STREQUAL "${_FLAGS}" OR
            NOT DEFINED CACHED_C_${_VAR}_GOOD_FLAGS)
        # test flags
        add_flags(C "${_VAR}" "${_FLAGS}")
        # cache the valid flags
        set(CACHED_C_${_VAR}_GOOD_FLAGS "${${_VAR}}" CACHE STRING
            "Valid C flags for ${_VAR}" FORCE)
    endif(NOT "${CACHED_C_${_VAR}_TEST_FLAGS}" STREQUAL "${_FLAGS}" OR
        NOT DEFINED CACHED_C_${_VAR}_GOOD_FLAGS)
    # set the ${_VAR} to the valid flags
    set(${_VAR} "${CACHED_C_${_VAR}_GOOD_FLAGS}")
    # cache the flags that were tested
    set(CACHED_C_${_VAR}_TEST_FLAGS "${_FLAGS}" CACHE STRING
        "Possible C flags for ${_VAR}" FORCE)
endmacro(add_c_flags _VAR _FLAGS)


################################################################################
# macro for adding C++ compiler flags to variable
################################################################################
macro(add_cxx_flags _VAR _FLAGS)
    # cache the flags to test
    set(CACHED_CXX_${_VAR}_TEST_FLAGS "${_FLAGS}" CACHE STRING
        "Possible C++ flags for ${_VAR}")
    mark_as_advanced(CACHED_CXX_${_VAR}_TEST_FLAGS)
    # if flags were changed or not previously processed
    if(NOT "${CACHED_CXX_${_VAR}_TEST_FLAGS}" STREQUAL "${_FLAGS}" OR
            NOT DEFINED CACHED_CXX_${_VAR}_GOOD_FLAGS)
        # test flags
        add_flags(CXX "${_VAR}" "${_FLAGS}")
        # cache the valid flags
        set(CACHED_CXX_${_VAR}_GOOD_FLAGS "${${_VAR}}" CACHE STRING
            "Valid C++ flags for ${_VAR}" FORCE)
    endif(NOT "${CACHED_CXX_${_VAR}_TEST_FLAGS}" STREQUAL "${_FLAGS}" OR
        NOT DEFINED CACHED_CXX_${_VAR}_GOOD_FLAGS)
    # set the ${_VAR} to the valid flags
    set(${_VAR} "${CACHED_CXX_${_VAR}_GOOD_FLAGS}")
    # cache the flags that were tested
    set(CACHED_CXX_${_VAR}_TEST_FLAGS "${_FLAGS}" CACHE STRING
        "Possible C++ flags for ${_VAR}" FORCE)
endmacro(add_cxx_flags _VAR _FLAGS)


################################################################################
# determine compiler types for each language
################################################################################
foreach(LANG C CXX)

    macro(SET_COMPILER_VAR VAR _BOOL)
        set(CMAKE_${LANG}_COMPILER_IS_${VAR} ${_BOOL} CACHE BOOL
            "CMake ${LANG} compiler identification (${VAR})")
        mark_as_advanced(CMAKE_${LANG}_COMPILER_IS_${VAR})
    endmacro()

    if(("${LANG}" STREQUAL "C" AND CMAKE_COMPILER_IS_GNUCC)
        OR
       ("${LANG}" STREQUAL "CXX" AND CMAKE_COMPILER_IS_GNUCXX))

        # GNU compiler
        SET_COMPILER_VAR(       GNU                 ON)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "icc.*")

        # Intel icc compiler
        SET_COMPILER_VAR(       INTEL               ON)
        SET_COMPILER_VAR(       INTEL_ICC           ON)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "icpc.*")

        # Intel icpc compiler
        SET_COMPILER_VAR(       INTEL               ON)
        SET_COMPILER_VAR(       INTEL_ICPC          ON)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "Clang" OR
           CMAKE_${LANG}_COMPILER_ID MATCHES "AppleClang")

        # Clang/LLVM compiler
        SET_COMPILER_VAR(       CLANG               ON)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "PGI")

        # PGI compiler
        SET_COMPILER_VAR(       PGI                 ON)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "xlC" AND UNIX)

        # IBM xlC compiler
        SET_COMPILER_VAR(       XLC                 ON)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "aCC" AND UNIX)

        # HP aC++ compiler
        SET_COMPILER_VAR(       HP_ACC              ON)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "CC" AND
           CMAKE_SYSTEM_NAME MATCHES "IRIX" AND UNIX)

        # IRIX MIPSpro CC Compiler
        SET_COMPILER_VAR(       MIPS                ON)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "Intel")

        SET_COMPILER_VAR(       INTEL               ON)

        set(CTYPE ICC)
        if("${LANG}" STREQUAL "CXX")
            set(CTYPE ICPC)
        endif("${LANG}" STREQUAL "CXX")

        SET_COMPILER_VAR(       INTEL_${CTYPE}      ON)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "MSVC")

        # Windows Visual Studio compiler
        SET_COMPILER_VAR(       MSVC                ON)

    endif()

    # set other to no
    foreach(TYPE GNU INTEL INTEL_ICC INTEL_ICPC CLANG PGI XLC HP_ACC MIPS MSVC)
        if(NOT ${CMAKE_${LANG}_COMPILER_IS_${TYPE}})
            SET_COMPILER_VAR(${TYPE} OFF)
        endif()
    endforeach()

    if(APPLE)
        set(CMAKE_INCLUDE_SYSTEM_FLAG_${LANG} "-isystem ")
    endif(APPLE)

endforeach()
