
include(CheckCXXCompilerFlag)

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL
    "Position independent code" FORCE)

################################################################################
#                                                                              #
#                               General                                        #
#                                                                              #
################################################################################

if(TOMOPY_USE_TIMEMORY)
    add_definitions(-DTOMOPY_USE_TIMEMORY)
endif(TOMOPY_USE_TIMEMORY)

# libraries to install
set(INSTALL_LIBRARIES )

set(PYTHON_SITE_DIR ${CMAKE_INSTALL_PREFIX})
set(CMAKE_INSTALL_LIBDIR ${PYTHON_SITE_DIR}/tomopy/sharedlibs CACHE PATH
    "Installation directory of libraries" FORCE)

#------------------------------------------------------------------------------#
#
#   PyBind11 submodule
#
#------------------------------------------------------------------------------#
if(TOMOPY_USE_PYBIND11)
    # ensure the submodule is checked out
    checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
        RELATIVE_PATH pybind11 WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
    add_subdirectory(pybind11)
endif()

#------------------------------------------------------------------------------#
#
#   PTL submodule
#
#------------------------------------------------------------------------------#
checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
    RELATIVE_PATH PTL WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
add_subdirectory(PTL)

if(BUILD_STATIC_LIBS)
    list(APPEND EXTERNAL_LIBRARIES ptl-static)
else()
    list(APPEND EXTERNAL_LIBRARIES ptl-shared)
endif()

#------------------------------------------------------------------------------#
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
file(GLOB tomo_headers ${PROJECT_SOURCE_DIR}/include/*.h)
file(GLOB tomo_sources ${CMAKE_CURRENT_LIST_DIR}/*.c)
file(GLOB tomo_cxxhdrs ${CMAKE_CURRENT_LIST_DIR}/cxx/*.hh)
file(GLOB tomo_cxxsrcs ${CMAKE_CURRENT_LIST_DIR}/cxx/*.cc)

# set language
set_source_files_properties(${tomo_cxxsrcs} PROPERTIES LANGUAGE CXX
    COMPILE_FLAGS ${${PROJECT_NAME}_CXX_FLAGS})
set_source_files_properties(${tomo_sources} PROPERTIES LANGUAGE C
    COMPILE_FLAGS ${${PROJECT_NAME}_C_FLAGS})


#------------------------------------------------------------------------------#
# tasking library
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/PTL/source
    ${CMAKE_CURRENT_LIST_DIR}/cxx
    ${CMAKE_CURRENT_LIST_DIR}/gpu)
include_directories(${TiMemory_INCLUDE_DIRS})


################################################################################
#                                                                              #
#                           TomoPy (Python)                                    #
#                                                                              #
################################################################################

#------------------------------------------------------------------------------#
#   Copy over pure python module, python testing, and setup files
#
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

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/test ${PROJECT_BINARY_DIR}/test
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
file(GLOB PYSETUP "${PROJECT_SOURCE_DIR}/setup.*")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${PYSETUP} ${PROJECT_BINARY_DIR}/
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

#------------------------------------------------------------------------------#
#
# core sources
set(LIBRARY_SOURCES ${tomo_headers} ${tomo_sources}
                    ${tomo_cxxhdrs} ${tomo_cxxsrcs})

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
    #SUFFIX ${PYTHON_MODULE_EXTENSION}
    RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy/sharedlibs
    LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy/sharedlibs
    ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy/sharedlibs
    LANGUAGE                    CXX
    LINKER_LANGUAGE             CXX
)

#------------------------------------------------------------------------------#
# add the module
#if(TOMOPY_USE_PYBIND11)
#    set(LIBRARY_SOURCES libtomopy.cpp libtomopy.hpp)
#
#    pybind11_add_module(libtomopy ${LIBRARY_SOURCES})
#
#    set_target_properties(libtomopy
#        PROPERTIES
#        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tomopy/sharedlibs
#        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tomopy/sharedlibs
#        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tomopy/sharedlibs
#        LANGUAGE CXX
#        LINKER_LANGUAGE CXX)
#
#    target_link_libraries(libtomopy PUBLIC ${EXTERNAL_LIBRARIES})
#endif()

list(APPEND INSTALL_LIBRARIES libtomopy)

#------------------------------------------------------------------------------#
# Install the targets and export
#
# compiled library
install(TARGETS ${INSTALL_LIBRARIES}
    DESTINATION ${PYTHON_SITE_DIR}/tomopy/sharedlibs
    COMPONENT development)

add_custom_command(TARGET libtomopy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:libtomopy>
        ${PROJECT_SOURCE_DIR}/tomopy/sharedlibs/
)
