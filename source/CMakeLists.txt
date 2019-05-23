################################################################################
#                                                                              #
#                               General                                        #
#                                                                              #
################################################################################
# ensure PIC flag is set
set(PYTHON_SITE_DIR ${CMAKE_INSTALL_PREFIX}/source)
set(BUILD_STATIC_LIBS     ON  CACHE BOOL "Build static libraries")
set(BUILD_SHARED_LIBS     OFF CACHE BOOL "Build shared libraries")
set(PTL_DEVELOPER_INSTALL OFF CACHE BOOL "Install headers, cmake export, and shared libs" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON
    CACHE BOOL "Position independent code" FORCE)
set(CMAKE_INSTALL_LIBDIR ${PYTHON_SITE_DIR}/tomopy/sharedlibs
    CACHE PATH "Installation directory of libraries" FORCE)

#------------------------------------------------------------------------------#
#
#   PTL submodule
#
#------------------------------------------------------------------------------#
checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
    RELATIVE_PATH source/PTL WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

if(TOMOPY_USE_PTL)
    add_subdirectory(PTL)
    if(BUILD_STATIC_LIBS)
        list(APPEND EXTERNAL_LIBRARIES ptl-static)
    else()
        list(APPEND EXTERNAL_LIBRARIES ptl-shared)
    endif()
    list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_PTL)
endif()

#------------------------------------------------------------------------------#
#
#   TomoPy Python module
#
#------------------------------------------------------------------------------#

# Locate sources and headers for this project headers are included for IDEs
file(GLOB tomo_headers
    ${CMAKE_CURRENT_LIST_DIR}/include/*.h
    ${CMAKE_CURRENT_LIST_DIR}/include/*.hh
    ${CMAKE_CURRENT_LIST_DIR}/include/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/*.icc)
file(GLOB tomo_sources
    ${CMAKE_CURRENT_LIST_DIR}/c/*.c
    ${CMAKE_CURRENT_LIST_DIR}/cxx/*.cc)

# core sources
set(LIBRARY_SOURCES ${tomo_headers} ${tomo_sources})

# target properties based on build
if(TOMOPY_USE_CUDA)
    file(GLOB CUDA_LIBRARY_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/gpu/*.cu)

    list(APPEND LIBRARY_SOURCES ${CUDA_LIBRARY_SOURCES})

    list(APPEND TARGET_INCLUDE_DIRECTORIES
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    set(CUDA_PROPERTIES
        CUDA_STANDARD               ${CMAKE_CXX_STANDARD}
        CUDA_STANDARD_REQUIRED      ON
        CUDA_SEPARABLE_COMPILATION  ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# create library
add_library(libtomopy SHARED ${LIBRARY_SOURCES})

# link library
target_link_libraries(libtomopy PUBLIC ${EXTERNAL_LIBRARIES}
    PRIVATE ${EXTERNAL_PRIVATE_LIBRARIES})

# where to place the library
set(OUTPUT_PATH ${CMAKE_BINARY_DIR})

# target properties
set_target_properties(libtomopy
    PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/sharedlibs
    LIBRARY_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/sharedlibs
    ARCHIVE_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/sharedlibs
    INCLUDE_DIRECTORIES         "${TARGET_INCLUDE_DIRECTORIES}"
    C_STANDARD                  ${CMAKE_C_STANDARD}
    C_STANDARD_REQUIRED         ${CMAKE_C_STANDARD_REQUIRED}
    CXX_STANDARD                ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED       ${CMAKE_CXX_STANDARD_REQUIRED}
    ${CUDA_PROPERTIES})

target_compile_definitions(libtomopy PUBLIC
    ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(libtomopy PUBLIC
    $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CUDA_FLAGS}>)

# Install the compiled library
install(TARGETS libtomopy
    DESTINATION ${PYTHON_SITE_DIR}/tomopy/sharedlibs
    COMPONENT python)


#------------------------------------------------------------------------------#
#
#   TomoPy Python file configuration
#
#------------------------------------------------------------------------------#
if(NOT SKBUILD)
    # helper macro
    macro(CONFIG_INSTALL RELATIVE_PATH)
        foreach(_SOURCE_FILE ${ARGN})
            string(REPLACE "${CMAKE_CURRENT_LIST_DIR}" "${PROJECT_BINARY_DIR}" _BINARY_FILE
                "${_SOURCE_FILE}")
            configure_file(${_SOURCE_FILE} ${_BINARY_FILE} COPYONLY)
            install(FILES ${_BINARY_FILE}
                DESTINATION ${PYTHON_SITE_DIR}/tomopy/${RELATIVE_PATH}
                COMPONENT python)
        endforeach()
    endmacro()

    # Copy over pure python module, python testing, and setup files
    file(GLOB _GENERAL "${CMAKE_CURRENT_LIST_DIR}/tomopy/*.py")
    file(GLOB _DATA    "${CMAKE_CURRENT_LIST_DIR}/tomopy/data/*.tif"
                       "${CMAKE_CURRENT_LIST_DIR}/tomopy/data/*.h5")
    file(GLOB _MISC    "${CMAKE_CURRENT_LIST_DIR}/tomopy/misc/*.py")
    file(GLOB _PREP    "${CMAKE_CURRENT_LIST_DIR}/tomopy/prep/*.py")
    file(GLOB _RECON   "${CMAKE_CURRENT_LIST_DIR}/tomopy/recon/*.py")
    file(GLOB _SIM     "${CMAKE_CURRENT_LIST_DIR}/tomopy/sim/*.py")
    file(GLOB _UTIL    "${CMAKE_CURRENT_LIST_DIR}/tomopy/util/*.py")

    config_install("" ${_GENERAL})
    config_install("data"  ${_DATA})
    config_install("misc"  ${_MISC})
    config_install("prep"  ${_PREP})
    config_install("recon" ${_RECON})
    config_install("sim"   ${_SIM})
    config_install("util"  ${_UTIL})

    # copy of test directory to build directory
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/test ${PROJECT_BINARY_DIR}/test
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # copy over setup.* files to build directory
    file(GLOB PYSETUP "${PROJECT_SOURCE_DIR}/setup.*")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PYSETUP} ${PROJECT_BINARY_DIR}/
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endif()

#------------------------------------------------------------------------------#
#
#   Code coverage
#
#------------------------------------------------------------------------------#
if(TOMOPY_USE_COVERAGE)
    foreach(_SRC_FILE ${LIBRARY_SOURCES})
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}" _BIN_FILE "${_SRC_FILE}")
        configure_file(${_SRC_FILE} ${_BIN_FILE} COPYONLY)
    endforeach()
endif()
