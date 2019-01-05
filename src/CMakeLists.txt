################################################################################
#                                                                              #
#                               General                                        #
#                                                                              #
################################################################################
# ensure PIC flag is set
set(PYTHON_SITE_DIR ${CMAKE_INSTALL_PREFIX})
set(PYBIND11_INSTALL    OFF CACHE BOOL "PyBind11 installation"  FORCE)
set(PYBIND11_TEST       OFF CACHE BOOL "PyBind11 testing"       FORCE)
set(BUILD_STATIC_LIBS   ON  CACHE BOOL "Build static libraries" FORCE)
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "Build shared libraries" FORCE)
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
    RELATIVE_PATH src/PTL WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_subdirectory(PTL)
list(APPEND EXTERNAL_LIBRARIES ptl-static)

# Locate sources and headers for this project headers are included for IDEs
file(GLOB tomo_headers
    ${PROJECT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_LIST_DIR}/gpu/*.h
    ${CMAKE_CURRENT_LIST_DIR}/cxx/*.hh
    ${CMAKE_CURRENT_LIST_DIR}/gpu/*.hh)
file(GLOB tomo_sources
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/cxx/*.cc)

# include dirs
set(TARGET_INCLUDE_DIRECTORIES
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/PTL/source
    ${CMAKE_CURRENT_LIST_DIR}/cxx
    ${CMAKE_CURRENT_LIST_DIR}/gpu
    ${TiMemory_INCLUDE_DIRS}
    ${EXTERNAL_INCLUDE_DIRS})


################################################################################
#                                                                              #
#                           TomoPy (Python)                                    #
#                                                                              #
################################################################################

# helper macro
macro(CONFIG_INSTALL RELATIVE_PATH)
    foreach(_SOURCE_FILE ${ARGN})
        string(REPLACE "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}" _BINARY_FILE
            "${_SOURCE_FILE}")
        configure_file(${_SOURCE_FILE} ${_BINARY_FILE} COPYONLY)
        install(FILES ${_BINARY_FILE}
            DESTINATION ${PYTHON_SITE_DIR}/tomopy/${RELATIVE_PATH}
            COMPONENT python)
    endforeach(_SOURCE_FILE ${ARGN})
endmacro()

# Copy over pure python module, python testing, and setup files
file(GLOB _GENERAL "${PROJECT_SOURCE_DIR}/tomopy/*.py")
file(GLOB _DATA    "${PROJECT_SOURCE_DIR}/tomopy/data/*.tif"
                   "${PROJECT_SOURCE_DIR}/tomopy/data/*.h5")
file(GLOB _MISC    "${PROJECT_SOURCE_DIR}/tomopy/misc/*.py")
file(GLOB _PREP    "${PROJECT_SOURCE_DIR}/tomopy/prep/*.py")
file(GLOB _RECON   "${PROJECT_SOURCE_DIR}/tomopy/recon/*.py")
file(GLOB _SIM     "${PROJECT_SOURCE_DIR}/tomopy/sim/*.py")
file(GLOB _UTIL    "${PROJECT_SOURCE_DIR}/tomopy/util/*.py")

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

# core sources
set(LIBRARY_SOURCES ${tomo_headers} ${tomo_sources})

# add the library
if(NOT TOMOPY_USE_CUDA)
    add_library(libtomopy SHARED ${LIBRARY_SOURCES})
else()
    file(GLOB CUDA_LIBRARY_HEADERS
        "${CMAKE_CURRENT_LIST_DIR}/gpu/*.h"
        "${CMAKE_CURRENT_LIST_DIR}/gpu/*.hh"
        "${CMAKE_CURRENT_LIST_DIR}/gpu/*.icc")

    file(GLOB CUDA_CXX_LIBRARY_SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/gpu/*.cc")
    set_source_files_properties(${CUDA_CXX_LIBRARY_SOURCES}
        PROPERTIES
            LANGUAGE CXX
            COMPILE_FLAGS ${${PROJECT_NAME}_CXX_FLAGS})

    file(GLOB CUDA_CU_LIBRARY_SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/gpu/*.cu")
    set_source_files_properties(${CUDA_CU_LIBRARY_SOURCES}
        PROPERTIES
	    LANGUAGE CXX
            COMPILE_FLAGS ${${PROJECT_NAME}_CXX_FLAGS})

    cuda_add_library(libtomopy SHARED
        ${LIBRARY_SOURCES}
        ${CUDA_LIBRARY_HEADERS}
        ${CUDA_CU_LIBRARY_SOURCES}
        ${CUDA_CXX_LIBRARY_SOURCES})
endif()

# link library
target_link_libraries(libtomopy ${EXTERNAL_LIBRARIES} ${EXTERNAL_CUDA_LIBRARIES})

# target properties
set_target_properties(libtomopy
    PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy/sharedlibs
    LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy/sharedlibs
    ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy/sharedlibs
    LANGUAGE                    CXX
    LINKER_LANGUAGE             CXX
    INCLUDE_DIRECTORIES         "${TARGET_INCLUDE_DIRECTORIES}"
)

# Install the compiled library
install(TARGETS libtomopy
    DESTINATION ${PYTHON_SITE_DIR}/tomopy/sharedlibs
    COMPONENT development)

# make sure the generated target gets put into the source tree for testing
if(UNIX)
    add_custom_command(TARGET libtomopy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:libtomopy>
        ${PROJECT_SOURCE_DIR}/tomopy/sharedlibs/$<TARGET_FILE_NAME:libtomopy>)
else()
    add_custom_command(TARGET libtomopy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:libtomopy>
        ${PROJECT_SOURCE_DIR}/tomopy/sharedlibs/)
endif()

#------------------------------------------------------------------------------#
#
#   PyBind11 submodule
#
#------------------------------------------------------------------------------#
if(TOMOPY_USE_PYBIND11)
    # ensure the submodule is checked out
    checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
        RELATIVE_PATH src/pybind11 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    add_subdirectory(pybind11)
    include_directories(${TARGET_INCLUDE_DIRECTORIES})
    pybind11_add_module(tomocxx ${CMAKE_CURRENT_LIST_DIR}/cxx/tomocxx.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cxx/tomocxx.hpp)
    target_link_libraries(tomocxx PUBLIC libtomopy)
    set_target_properties(tomocxx PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy
        LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy
        ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy
    )
    install(TARGETS tomocxx
        DESTINATION ${PYTHON_SITE_DIR}/tomopy
        COMPONENT development)
    if(UNIX)
        add_custom_command(TARGET tomocxx POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:tomocxx>
            ${PROJECT_SOURCE_DIR}/tomopy/$<TARGET_FILE_NAME:tomocxx>)
    else()
        add_custom_command(TARGET tomocxx POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tomocxx>
            ${PROJECT_SOURCE_DIR}/tomopy/)
    endif()
endif()

if(TOMOPY_USE_COVERAGE)
    foreach(_SRC_FILE ${LIBRARY_SOURCES})
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}" _BIN_FILE "${_SRC_FILE}")
        configure_file(${_SRC_FILE} ${_BIN_FILE} COPYONLY)
    endforeach()
endif()
