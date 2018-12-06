#
# Project settings
#

################################################################################

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
    set(CMAKE_BUILD_TYPE Release)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" _CONFIG)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(${PROJECT_NAME}_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE PATH 
    "${PROJECT_NAME} installation prefix")

add_feature(CMAKE_C_FLAGS_${_CONFIG} "C compiler build-specific flags")
add_feature(CMAKE_CXX_FLAGS_${_CONFIG} "C++ compiler build-specific flags")

################################################################################
#
#   Non-python installation directories
#
################################################################################

if(${PROJECT_NAME}_DEVELOPER_INSTALL)

    set(${PROJECT_NAME}_INSTALL_DATAROOTDIR ${CMAKE_INSTALL_DATAROOTDIR})
    if(NOT IS_ABSOLUTE ${${PROJECT_NAME}_INSTALL_DATAROOTDIR})
        set(${PROJECT_NAME}_INSTALL_DATAROOTDIR "${${PROJECT_NAME}_INSTALL_PREFIX}/share"
            CACHE PATH "Installation root directory for data" FORCE)
    endif()

    set(${PROJECT_NAME}_INSTALL_CMAKEDIR ${${PROJECT_NAME}_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
        CACHE PATH "Installation for CMake config" FORCE)
    set(${PROJECT_NAME}_INSTALL_INCLUDEDIR ${${PROJECT_NAME}_INSTALL_PREFIX}/include
        CACHE PATH "Installation for include directories" FORCE)
    set(${PROJECT_NAME}_INSTALL_LIBDIR ${${PROJECT_NAME}_INSTALL_PREFIX}/${LIBDIR_DEFAULT}
        CACHE PATH "Installation for libraries" FORCE)
    set(${PROJECT_NAME}_INSTALL_BINDIR ${${PROJECT_NAME}_INSTALL_PREFIX}/bin
        CACHE PATH "Installation for executables" FORCE)
    set(${PROJECT_NAME}_INSTALL_MANDIR ${${PROJECT_NAME}_INSTALL_DATAROOTDIR}/man
        CACHE PATH "Installation for executables" FORCE)
    set(${PROJECT_NAME}_INSTALL_DOCDIR ${${PROJECT_NAME}_INSTALL_DATAROOTDIR}/doc
        CACHE PATH "Installation for executables" FORCE)

else(${PROJECT_NAME}_DEVELOPER_INSTALL)

    # cmake installation folder
    set(${PROJECT_NAME}_INSTALL_CMAKEDIR  ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
        CACHE PATH "Installation directory for CMake package config files")
    # the rest of the installation folders
    foreach(_TYPE in DATAROOT INCLUDE LIB BIN MAN DOC)
        set(${PROJECT_NAME}_INSTALL_${_TYPE}DIR ${CMAKE_INSTALL_${_TYPE}DIR})
    endforeach(_TYPE in DATAROOT INCLUDE LIB BIN MAN DOC)

endif(${PROJECT_NAME}_DEVELOPER_INSTALL)

# create the full path version and generic path versions
foreach(_TYPE in DATAROOT CMAKE INCLUDE LIB BIN MAN DOC)
    # set the absolute versions
    if(NOT IS_ABSOLUTE "${${PROJECT_NAME}_INSTALL_${_TYPE}DIR}")
        set(${PROJECT_NAME}_INSTALL_FULL_${_TYPE}DIR ${CMAKE_INSTALL_PREFIX}/${${PROJECT_NAME}_INSTALL_${_TYPE}DIR})
    else(NOT IS_ABSOLUTE "${${PROJECT_NAME}_INSTALL_${_TYPE}DIR}")
        set(${PROJECT_NAME}_INSTALL_FULL_${_TYPE}DIR ${${PROJECT_NAME}_INSTALL_${_TYPE}DIR})
    endif(NOT IS_ABSOLUTE "${${PROJECT_NAME}_INSTALL_${_TYPE}DIR}")

    # generic "PROJECT_INSTALL_" variables (used by documentation)"
    set(PROJECT_INSTALL_${_TYPE}DIR ${${PROJECT_NAME}_INSTALL_${TYPE}DIR})
    set(PROJECT_INSTALL_FULL_${_TYPE}DIR ${${PROJECT_NAME}_INSTALL_FULL_${TYPE}DIR})

endforeach(_TYPE in DATAROOT CMAKE INCLUDE LIB BIN MAN DOC)

