set(HEADERS "${tomopy_SOURCE_DIR}/include/libtomo/morph.h"
            "${tomopy_SOURCE_DIR}/include/libtomo/remove_ring.h")

add_library(tomo-misc SHARED morph.c remove_ring.c ${HEADERS})

target_include_directories(
  tomo-misc
  PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/include/libtomo>
         $<INSTALL_INTERFACE:include/libtomo>
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# link library to math.h
if(NOT WIN32)
  target_link_libraries(tomo-misc PUBLIC m)
endif()

target_compile_definitions(tomo-misc PUBLIC ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-misc PUBLIC $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>)

install(TARGETS tomo-misc)
install(
  FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtomo
  COMPONENT Headers)
