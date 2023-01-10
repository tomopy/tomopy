set(HEADERS
    "${tomopy_SOURCE_DIR}/include/libtomo/morph.h"
    "${tomopy_SOURCE_DIR}/include/libtomo/remove_ring.h"
    "${tomopy_SOURCE_DIR}/include/libtomo/median_filt3d.h")

tomopy_add_library(
  tomo-misc
  SHARED
  morph.c
  remove_ring.c
  median_filt3d.c
  utils.c
  utils.h
  ${HEADERS})

find_package(OpenMP REQUIRED COMPONENTS C)
target_link_libraries(tomo-misc PRIVATE OpenMP::OpenMP_C)

target_include_directories(
  tomo-misc
  PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# link library to math.h
if(NOT WIN32)
  target_link_libraries(tomo-misc PUBLIC m)
endif()

target_compile_definitions(tomo-misc PRIVATE ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-misc PRIVATE $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>)

install(TARGETS tomo-misc EXPORT libtomoTargets)

install(
  FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtomo
  COMPONENT Headers)
