################################################################################
#                                                                              #
#                               General                                        #
#                                                                              #
################################################################################
# ensure PIC flag is set
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Position independent code" FORCE)
set(PYTHON_SITE_DIR ${CMAKE_INSTALL_PREFIX})
set(CMAKE_INSTALL_LIBDIR ${PYTHON_SITE_DIR}/tomopy/sharedlibs CACHE PATH
    "Installation directory of libraries" FORCE)

# Locate sources and headers for this project headers are included for IDEs
file(GLOB tomo_headers ${CMAKE_CURRENT_LIST_DIR}/include/*.h)
file(GLOB tomo_sources ${CMAKE_CURRENT_LIST_DIR}/libtomopy/c/*.c)

# set language and project include dirs of source files
set_source_files_properties(${tomo_sources}
    PROPERTIES
        LANGUAGE            C
        COMPILE_FLAGS       ${${PROJECT_NAME}_C_FLAGS}
        INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/include)


################################################################################
#                                                                              #
#                           TomoPy (Python)                                    #
#                                                                              #
################################################################################

if(NOT SKBUILD)
    # helper macro
    macro(CONFIG_INSTALL RELATIVE_PATH)
        foreach(_SOURCE_FILE ${ARGN})
            string(REPLACE "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}" _BINARY_FILE
                "${_SOURCE_FILE}")
            configure_file(${_SOURCE_FILE} ${_BINARY_FILE} COPYONLY)
            install(FILES ${_BINARY_FILE}
                DESTINATION ${PYTHON_SITE_DIR}/src/tomopy/${RELATIVE_PATH}
                COMPONENT python)
        endforeach(_SOURCE_FILE ${ARGN})
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
endif(NOT SKBUILD)

# core sources
set(LIBRARY_SOURCES ${tomo_headers} ${tomo_sources})

# add the library
add_library(libtomopy SHARED ${LIBRARY_SOURCES})

# link library
target_link_libraries(libtomopy ${EXTERNAL_LIBRARIES})

# target properties
set_target_properties(libtomopy
    PROPERTIES
        PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/src/tomopy/sharedlibs
        LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/src/tomopy/sharedlibs
        ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/src/tomopy/sharedlibs
        LANGUAGE                    C
        LINKER_LANGUAGE             C)

# Install the compiled library
install(TARGETS libtomopy
    DESTINATION ${PYTHON_SITE_DIR}/src/tomopy/sharedlibs
    COMPONENT development)
