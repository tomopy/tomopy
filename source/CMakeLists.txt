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
set(CMAKE_INSTALL_LIBDIR ${PYTHON_SITE_DIR}/tomopy/util/extern
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

# where to place the library
set(OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/libtomo)

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
