set(HEADERS "${tomopy_SOURCE_DIR}/include/libtomo/project.h"
            "${tomopy_SOURCE_DIR}/include/libtomo/recon.h")

add_library(
  tomo-recon SHARED
  art.c
  bart.c
  fbp.c
  grad.c
  osem.c
  ospml_hybrid.c
  ospml_quad.c
  project.c
  tikh.c
  tv.c
  utils.c
  utils.h
  vector.c
  ${HEADERS})

target_include_directories(
  tomo-recon
  PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/include/libtomo>
         $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/libtomo>
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# link libraries to math.h
if(NOT WIN32)
  target_link_libraries(tomo-recon PUBLIC m)
endif()

target_compile_definitions(tomo-recon PRIVATE ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-recon PUBLIC $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>)

install(TARGETS tomo-recon EXPORT libtomoTargets)

install(
  FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtomo
  COMPONENT Headers)
