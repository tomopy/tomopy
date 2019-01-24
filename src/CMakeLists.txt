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

if(TOMOPY_USE_PTL)
    add_subdirectory(PTL)
    list(APPEND EXTERNAL_LIBRARIES ptl-static)
    list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_PTL)
endif()

#------------------------------------------------------------------------------#
#
#   TomoPy Python module
#
#------------------------------------------------------------------------------#

# Locate sources and headers for this project headers are included for IDEs
file(GLOB tomo_headers
    ${PROJECT_SOURCE_DIR}/include/*.h
    ${PROJECT_SOURCE_DIR}/include/*.hh
    ${PROJECT_SOURCE_DIR}/include/*.hpp
    ${PROJECT_SOURCE_DIR}/include/*.icc)
file(GLOB tomo_sources
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/cxx/*.cc)

# core sources
set(LIBRARY_SOURCES ${tomo_headers} ${tomo_sources})

# add the library
if(TOMOPY_USE_CUDA)
    file(GLOB CUDA_LIBRARY_SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/gpu/*.cc"
        "${CMAKE_CURRENT_LIST_DIR}/gpu/*.cu")

    list(APPEND LIBRARY_SOURCES ${CUDA_LIBRARY_SOURCES})

    list(APPEND TARGET_INCLUDE_DIRECTORIES
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    set(CUDA_PROPERTIES
        CUDA_STANDARD               11
        CUDA_STANDARD_REQUIRED      ON
        #CUDA_PTX_COMPILATION        OFF
        CUDA_SEPARABLE_COMPILATION  ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
endif()

# create library
add_library(libtomopy SHARED ${LIBRARY_SOURCES})

# link library
target_link_libraries(libtomopy ${EXTERNAL_LIBRARIES} ${EXTERNAL_CUDA_LIBRARIES})

# target properties
set_target_properties(libtomopy
    PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy/sharedlibs
    LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy/sharedlibs
    ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy/sharedlibs
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
    COMPONENT development)

# make sure the generated target gets put into the source tree for testing
add_custom_command(TARGET libtomopy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:libtomopy>
    ${PROJECT_SOURCE_DIR}/tomopy/sharedlibs/)


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

    pybind11_add_module(tomocxx ${CMAKE_CURRENT_LIST_DIR}/cxx/tomocxx.cpp
        ${PROJECT_SOURCE_DIR}/include/tomocxx.hpp)

    target_link_libraries(tomocxx PUBLIC libtomopy)

    set_target_properties(tomocxx PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy
        LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy
        ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/tomopy)

    target_compile_definitions(tomocxx PUBLIC
        ${${PROJECT_NAME}_DEFINITIONS})

    get_target_property(_INCLUDE_DIRS tomocxx INCLUDE_DIRECTORIES)
    target_include_directories(tomocxx PRIVATE
        ${_INCLUDE_DIRS} ${TARGET_INCLUDE_DIRECTORIES})

    target_compile_options(tomocxx PUBLIC
        $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CUDA_FLAGS}>)

    install(TARGETS tomocxx
        DESTINATION ${PYTHON_SITE_DIR}/tomopy
        COMPONENT development)

    add_custom_command(TARGET tomocxx POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tomocxx>
        ${PROJECT_SOURCE_DIR}/tomopy/)
endif()


#------------------------------------------------------------------------------#
#
#   TomoPy Python file configuration
#
#------------------------------------------------------------------------------#

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
