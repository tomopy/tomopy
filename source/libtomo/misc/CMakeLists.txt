set(HEADERS "${tomopy_SOURCE_DIR}/include/libtomo/morph.h"
            "${tomopy_SOURCE_DIR}/include/libtomo/remove_ring.h")

add_library(tomo-misc SHARED morph.c remove_ring.c ${HEADERS})

target_include_directories(
  tomo-misc
  PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/include/libtomo>
         $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/libtomo>
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# link library to math.h
if(NOT WIN32)
  target_link_libraries(tomo-misc PUBLIC m)
endif()

target_compile_definitions(tomo-misc PRIVATE ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-misc PUBLIC $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>)

install(TARGETS tomo-misc EXPORT libtomoTargets)

install(
  FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtomo
  COMPONENT Headers)

list(APPEND libtomo_TARGETS tomo-misc)
