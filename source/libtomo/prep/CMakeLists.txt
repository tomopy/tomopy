set(HEADERS "${tomopy_SOURCE_DIR}/include/libtomo/prep.h"
            "${tomopy_SOURCE_DIR}/include/libtomo/stripe.h")

tomopy_add_library(
  tomo-prep
  SHARED
  prep.c
  stripe.c
  stripes_detect3d.c
  ${HEADERS})

find_package(OpenMP REQUIRED COMPONENTS C)
target_link_libraries(tomo-prep PRIVATE OpenMP::OpenMP_C)
if (WIN32)
target_compile_options(
  tomo-prep PRIVATE $<$<COMPILE_LANGUAGE:C>:/openmp:experimental>)
endif()

target_include_directories(
  tomo-prep
  PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_compile_definitions(tomo-prep PRIVATE ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-prep PRIVATE $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>)

set_target_properties(tomo-prep PROPERTIES
      C_STANDARD 99  # C99 or later requried for boolean type support
      C_STANDARD_REQUIRED ON)

install(TARGETS tomo-prep EXPORT libtomoTargets)

install(
  FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtomo
  COMPONENT Headers)
