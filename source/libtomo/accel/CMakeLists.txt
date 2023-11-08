set(HEADERS "${tomopy_SOURCE_DIR}/include/libtomo/accel.h"
            "${tomopy_SOURCE_DIR}/include/libtomo/macros.h")

tomopy_add_library(
  tomo-accel
  SHARED
  cxx/common.cc
  cxx/mlem.cc
  cxx/sirt.cc
  common.hh
  constants.hh
  cuda.h
  data.hh
  macros.hh
  profiler.hh
  typedefs.hh
  utils.hh
  ${HEADERS})

target_include_directories(
  tomo-accel
  PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

if(TOMOPY_USE_CUDA)

  target_sources(tomo-accel PRIVATE gpu/common.cu gpu/mlem.cu gpu/sirt.cu
                                    gpu/utils.cu)

  target_include_directories(tomo-accel
                             PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  target_link_libraries(tomo-accel PRIVATE tomopy::cuda-npp)

  set(CUDA_PROPERTIES
      CUDA_STANDARD
      ${CMAKE_CXX_STANDARD}
      CUDA_STANDARD_REQUIRED
      ON
      CUDA_SEPARABLE_COMPILATION
      ON
      CUDA_RESOLVE_DEVICE_SYMBOLS
      ON)

  set_target_properties(tomo-accel PROPERTIES ${CUDA_PROPERTIES})

endif(TOMOPY_USE_CUDA)

target_link_libraries(tomo-accel PRIVATE ${TOMOPY_EXTERNAL_LIBRARIES}
                                         ${TOMOPY_EXTERNAL_PRIVATE_LIBRARIES} ptl-static)

target_compile_definitions(tomo-accel PRIVATE ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-accel
  PRIVATE $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
          $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>
          $<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CUDA_FLAGS}>)

install(TARGETS tomo-accel EXPORT libtomoTargets)

install(
  FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtomo
  COMPONENT Headers)
