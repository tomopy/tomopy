add_library(
  tomo-recon
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
  vector.c)

target_include_directories(
  tomo-recon PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/source/include/recon>
                    $<INSTALL_INTERFACE:include>)

# link libraries to math.h
if(NOT WIN32)
  target_link_libraries(tomo-recon PUBLIC m)
endif()

target_compile_definitions(tomo-recon PUBLIC ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-recon PUBLIC $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>)

install(TARGETS tomo-recon)
