################################################################################
#
#        Handles the build settings
#
################################################################################

include(GNUInstallDirs)
include(Compilers)

# ---------------------------------------------------------------------------- #
# compiler features
set(CMAKE_CXX_COMPILE_FEATURES
    cxx_std_11 cxx_lambdas cxx_thread_local cxx_constexpr
    cxx_decltype cxx_nullptr cxx_variable_templates
    cxx_deleted_functions cxx_auto_type cxx_alias_templates)

# ---------------------------------------------------------------------------- #
#
set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_C_STANDARD 11 CACHE STRING "C language standard")
set(CMAKE_CXX_STANDARD 11 CACHE STRING "CXX language standard")
set(CMAKE_C_STANDARD_REQUIRED ON CACHE BOOL "Require the C language standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require the CXX language standard")
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING "CUDA language standard")
set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL "Require the CUDA language standard")

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
    list(APPEND ${PROJECT_NAME}_DEFINITIONS DEBUG)
else("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    list(APPEND ${PROJECT_NAME}_DEFINITIONS NDEBUG)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")


# ---------------------------------------------------------------------------- #
# used by configure_package_*
set(LIBNAME tomopy)


# ---------------------------------------------------------------------------- #
# set the compiler flags
add_c_flag_if_avail("-W")
if(NOT WIN32)
    add_c_flag_if_avail("-Wall")
endif()
add_c_flag_if_avail("-Wextra")
add_c_flag_if_avail("-Wno-attributes")
add_c_flag_if_avail("-Wno-unused-variable")
add_c_flag_if_avail("-Wno-unknown-pragmas")
add_c_flag_if_avail("-Wno-unused-parameter")
add_c_flag_if_avail("-Wno-reserved-id-macro")
add_c_flag_if_avail("-Wunused-but-set-parameter")

# SIMD OpenMP
add_c_flag_if_avail("-fopenmp-simd")
add_cxx_flag_if_avail("-fopenmp-simd")

# Intel floating-point model
add_c_flag_if_avail("-fp-model=precise")
add_cxx_flag_if_avail("-fp-model=precise")

add_cxx_flag_if_avail("-W")
if(NOT WIN32)
    add_cxx_flag_if_avail("-Wall")
endif()
add_cxx_flag_if_avail("-Wextra")
add_cxx_flag_if_avail("-Wno-attributes")
add_cxx_flag_if_avail("-Wno-unused-value")
add_cxx_flag_if_avail("-Wno-unused-variable")
add_cxx_flag_if_avail("-Wno-unknown-pragmas")
add_cxx_flag_if_avail("-Wno-unused-parameter")
add_cxx_flag_if_avail("-Wno-reserved-id-macro")
add_cxx_flag_if_avail("-Wno-implicit-fallthrough")
add_cxx_flag_if_avail("-Wunused-but-set-parameter")
add_cxx_flag_if_avail("-faligned-new")

if(TOMOPY_USE_ARCH)
    if(CMAKE_C_COMPILER_IS_INTEL)
        add_c_flag_if_avail("-xHOST")
        if(TOMOPY_USE_AVX512)
            add_c_flag_if_avail("-axMIC-AVX512")
        endif()
    else()
        add_c_flag_if_avail("-march=native")
        add_c_flag_if_avail("-mtune=native")
        add_c_flag_if_avail("-msse2")
        add_c_flag_if_avail("-msse3")
        add_c_flag_if_avail("-mssse3")
        add_c_flag_if_avail("-msse4")
        add_c_flag_if_avail("-msse4.1")
        add_c_flag_if_avail("-msse4.2")
        add_c_flag_if_avail("-mavx")
        add_c_flag_if_avail("-mavx2")
        if(TOMOPY_USE_AVX512)
            add_c_flag_if_avail("-mavx512f")
            add_c_flag_if_avail("-mavx512pf")
            add_c_flag_if_avail("-mavx512er")
            add_c_flag_if_avail("-mavx512cd")
            add_c_flag_if_avail("-mavx512vl")
            add_c_flag_if_avail("-mavx512bw")
            add_c_flag_if_avail("-mavx512dq")
            add_c_flag_if_avail("-mavx512ifma")
            add_c_flag_if_avail("-mavx512vbmi")
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_IS_INTEL)
        add_cxx_flag_if_avail("-xHOST")
        if(TOMOPY_USE_AVX512)
            add_cxx_flag_if_avail("-axMIC-AVX512")
        endif()
    else()
        add_cxx_flag_if_avail("-march=native")
        add_cxx_flag_if_avail("-mtune=native")
        add_cxx_flag_if_avail("-msse2")
        add_cxx_flag_if_avail("-msse3")
        add_cxx_flag_if_avail("-mssse3")
        add_cxx_flag_if_avail("-msse4")
        add_cxx_flag_if_avail("-msse4.1")
        add_cxx_flag_if_avail("-msse4.2")
        add_cxx_flag_if_avail("-mavx")
        add_cxx_flag_if_avail("-mavx2")
        if(TOMOPY_USE_AVX512)
            add_cxx_flag_if_avail("-mavx512f")
            add_cxx_flag_if_avail("-mavx512pf")
            add_cxx_flag_if_avail("-mavx512er")
            add_cxx_flag_if_avail("-mavx512cd")
            add_cxx_flag_if_avail("-mavx512vl")
            add_cxx_flag_if_avail("-mavx512bw")
            add_cxx_flag_if_avail("-mavx512dq")
            add_cxx_flag_if_avail("-mavx512ifma")
            add_cxx_flag_if_avail("-mavx512vbmi")
        endif()
    endif()
endif()


if(TOMOPY_USE_SANITIZER)
    add_c_flag_if_avail("-fsanitize=${SANITIZER_TYPE}")
    add_cxx_flag_if_avail("-fsanitize=${SANITIZER_TYPE}")

    if(c_fsanitize_${SANITIZER_TYPE} AND cxx_fsanitize_${SANITIZER_TYPE})
        if("${SANITIZER_TYPE}" STREQUAL "memory")
            list(APPEND EXTERNAL_LIBRARIES msan)
        elseif("${SANITIZER_TYPE}" STREQUAL "thread")
            list(APPEND EXTERNAL_LIBRARUES tsan)
        elseif("${SANITIZER_TYPE}" STREQUAL "leak")
            list(APPEND EXTERNAL_LIBRARIES lsan)
        endif()
    else()
        unset(SANITIZER_TYPE CACHE)
        set(TOMOPY_USE_SANITIZER OFF)
    endif()
endif()

if(TOMOPY_USE_COVERAGE)
    add_c_flag_if_avail("-ftest-coverage")
    if(c_ftest_coverage)
        list(APPEND ${PROJECT_NAME}_C_FLAGS "-fprofile-arcs")
    endif()
    add_cxx_flag_if_avail("-ftest-coverage")
    if(cxx_ftest_coverage)
        list(APPEND ${PROJECT_NAME}_CXX_FLAGS "-fprofile-arcs")
        add(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs")
        add_feature(CMAKE_EXE_LINKER_FLAGS "Linker flags")
    endif()
endif()

# ---------------------------------------------------------------------------- #
# user customization
to_list(_CFLAGS "${CFLAGS};$ENV{CFLAGS}")
foreach(_FLAG ${_CFLAGS})
    if(TOMOPY_USER_FLAGS)
        list(APPEND ${PROJECT_NAME}_C_FLAGS "${_FLAG}")
    else()
        add_c_flag_if_avail("${_FLAG}")
    endif()
endforeach()

to_list(_CXXFLAGS "${CXXFLAGS};$ENV{CXXFLAGS}")
foreach(_FLAG ${_CXXFLAGS})
    if(TOMOPY_USER_FLAGS)
        list(APPEND ${PROJECT_NAME}_CXX_FLAGS "${_FLAG}")
    else()
        add_cxx_flag_if_avail("${_FLAG}")
    endif()
endforeach()


