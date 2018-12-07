################################################################################
#
#        Handles the build settings
#
################################################################################

include(GNUInstallDirs)
include(CheckCCompilerFlag)


# ---------------------------------------------------------------------------- #
# check C flag
macro(ADD_C_FLAG_IF_AVAIL FLAG)
    if(NOT "${FLAG}" STREQUAL "")
        string(REGEX REPLACE "^-" "c_" FLAG_NAME "${FLAG}")
        string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
        check_c_compiler_flag("${FLAG}" ${FLAG_NAME})
        if(${FLAG_NAME})
            add(${PROJECT_NAME}_C_FLAGS "${FLAG}")
        endif()
    endif()
endmacro()


# ---------------------------------------------------------------------------- #
#
set(SANITIZE_TYPE leak CACHE STRING "-fsantitize=<TYPE>")
set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_C_STANDARD 11 CACHE STRING "C language standard")


# ---------------------------------------------------------------------------- #
# set the output directory (critical on Windows)
#
foreach(_TYPE ARCHIVE LIBRARY RUNTIME)
    # if ${PROJECT_NAME}_OUTPUT_DIR is not defined, set to CMAKE_BINARY_DIR
    if(NOT DEFINED ${PROJECT_NAME}_OUTPUT_DIR OR "${${PROJECT_NAME}_OUTPUT_DIR}" STREQUAL "")
        set(${PROJECT_NAME}_OUTPUT_DIR ${CMAKE_BINARY_DIR})
    endif(NOT DEFINED ${PROJECT_NAME}_OUTPUT_DIR OR "${${PROJECT_NAME}_OUTPUT_DIR}" STREQUAL "")
    # set the CMAKE_{ARCHIVE,LIBRARY,RUNTIME}_OUTPUT_DIRECTORY variables
    if(WIN32)
        # on Windows, separate types into different directories
        string(TOLOWER "${_TYPE}" _LTYPE)
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${${PROJECT_NAME}_OUTPUT_DIR}/outputs/${_LTYPE})
    else(WIN32)
        # on UNIX, just set to same directory
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${${PROJECT_NAME}_OUTPUT_DIR})
    endif(WIN32)
endforeach(_TYPE ARCHIVE LIBRARY RUNTIME)


# ---------------------------------------------------------------------------- #
#  debug macro
#
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DDEBUG)
else("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DNDEBUG)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")


# ---------------------------------------------------------------------------- #
# used by configure_package_*
set(LIBNAME tomopy)


# ---------------------------------------------------------------------------- #
# set the compiler flags
add_c_flag_if_avail("-W")
add_c_flag_if_avail("-Wall")
add_c_flag_if_avail("-Wextra")
add_c_flag_if_avail("-Wno-unused-parameter")
add_c_flag_if_avail("-Wunused-but-set-parameter")
add_c_flag_if_avail("-Wno-unused-variable")
add_c_flag_if_avail("-fPIC")
add_c_flag_if_avail("-std=c11")
if(NOT c_std_c11)
    add_c_flag_if_avail("-std=c99")
endif()

# SIMD OpenMP
add_c_flag_if_avail("-fopenmp-simd")

if(TOMOPY_USE_ARCH)
    add_c_flag_if_avail("-march")
    add_c_flag_if_avail("-msse2")
    add_c_flag_if_avail("-msse3")
    add_c_flag_if_avail("-msse4")
    add_c_flag_if_avail("-mavx")
    add_c_flag_if_avail("-mavx2")
    add_c_flag_if_avail("-xHOST")

    if(TOMOPY_USE_AVX512)
        add_c_flag_if_avail("-mavx512f")
        add_c_flag_if_avail("-mavx512pf")
        add_c_flag_if_avail("-mavx512er")
        add_c_flag_if_avail("-mavx512cd")
        add_c_flag_if_avail("-axMIC-AVX512")
    endif()

endif()


# ---------------------------------------------------------------------------- #
# user customization
add_c_flag_if_avail("${CFLAGS}")
add_c_flag_if_avail("$ENV{CFLAGS}")
