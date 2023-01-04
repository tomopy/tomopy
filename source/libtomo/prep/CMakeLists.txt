set(HEADERS "${tomopy_SOURCE_DIR}/include/libtomo/prep.h"
            "${tomopy_SOURCE_DIR}/include/libtomo/stripe.h")

tomopy_add_library(tomo-prep SHARED prep.c stripe.c ${HEADERS})

target_include_directories(
  tomo-prep
  PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_compile_definitions(tomo-prep PRIVATE ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-prep PRIVATE $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>)

install(TARGETS tomo-prep EXPORT libtomoTargets)

install(
  FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtomo
  COMPONENT Headers)
