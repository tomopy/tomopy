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
set(${PROJECT_NAME}_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE PATH "${PROJECT_NAME} installation prefix")

add_feature(CMAKE_C_FLAGS_${_CONFIG} "C compiler build-specific flags")
