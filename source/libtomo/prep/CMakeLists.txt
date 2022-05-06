set(HEADERS "${tomopy_SOURCE_DIR}/include/libtomo/prep.h"
            "${tomopy_SOURCE_DIR}/include/libtomo/stripe.h")

add_library(tomo-prep prep.c stripe.c ${HEADERS})

target_include_directories(
  tomo-prep
  PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/include/libtomo>
         $<INSTALL_INTERFACE:include/libtomo>
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_compile_definitions(tomo-prep PUBLIC ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-prep PUBLIC $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>)

install(TARGETS tomo-prep)

install(
  FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtomo
  COMPONENT Headers)
