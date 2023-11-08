set(HEADERS "${tomopy_SOURCE_DIR}/include/libtomo/filters.h"
            "${tomopy_SOURCE_DIR}/include/libtomo/gridrec.h")

if(TOMOPY_CXX_GRIDREC)
  set_source_files_properties("gridrec.c" PROPERTIES LANGUAGE CXX)
endif()

tomopy_add_library(tomo-gridrec SHARED gridrec.c pal.h ${HEADERS})

if(TOMOPY_USE_MKL)
  find_package(MKL REQUIRED)
  target_include_directories(tomo-gridrec PRIVATE ${MKL_INCLUDE_DIRS})
  target_link_options(tomo-gridrec PRIVATE
                      $<$<COMPILE_LANGUAGE:CXX>:${MKL_CXX_LINK_FLAGS}>)
  target_link_libraries(tomo-gridrec PRIVATE ${MKL_LIBRARIES})
  target_compile_definitions(tomo-gridrec PRIVATE TOMOPY_USE_MKL)
else()
  if(TOMOPY_CXX_GRIDREC)
    message(
      WARNING "TOMOPY_CXX_GRIDREC with FFTW3 is not officially supported.")
  endif()
  # FFTW doesn't officially support CMAKE, they prefer PkgConfig.
  find_package(PkgConfig REQUIRED)
  pkg_search_module(FFTW REQUIRED fftw3f IMPORTED_TARGET)
  # Must link with thread aware fftw3f in order to make planner thread safe
  target_link_libraries(tomo-gridrec PRIVATE fftw3f_threads fftw3f)
  target_compile_definitions(tomo-gridrec PRIVATE TOMOPY_USE_FFTW)
endif()

# link library to math.h
if(NOT WIN32)
  target_link_libraries(tomo-gridrec PUBLIC m)
endif()

target_include_directories(
  tomo-gridrec
  PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_compile_definitions(tomo-gridrec PRIVATE ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-gridrec PRIVATE $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
                       $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>)

set_property(TARGET tomo-gridrec PROPERTY CXX_STANDARD 14)

install(TARGETS tomo-gridrec EXPORT libtomoTargets)

install(
  FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtomo
  COMPONENT Headers)
