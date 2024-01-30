set(HEADERS
    "${tomopy_SOURCE_DIR}/include/libtomo/morph.h"
    "${tomopy_SOURCE_DIR}/include/libtomo/remove_ring.h"
    "${tomopy_SOURCE_DIR}/include/libtomo/inpainter.h"
    "${tomopy_SOURCE_DIR}/include/libtomo/median_filt3d.h")

tomopy_add_library(
  tomo-misc
  SHARED
  morph.c
  remove_ring.c
  inpainter.c
  median_filt3d.c
  ${HEADERS})

find_package(OpenMP REQUIRED COMPONENTS C)
target_link_libraries(tomo-misc PRIVATE OpenMP::OpenMP_C)
if (WIN32)
target_compile_options(
  tomo-misc PRIVATE $<$<COMPILE_LANGUAGE:C>:/openmp:experimental>)
endif()

target_include_directories(
  tomo-misc
  PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# link library to math.h
if(NOT WIN32)
  target_link_libraries(tomo-misc PRIVATE m)
endif()

install(TARGETS tomo-misc EXPORT libtomoTargets)

install(
  FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtomo
  COMPONENT Headers)
