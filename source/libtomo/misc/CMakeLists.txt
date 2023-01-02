set(HEADERS "${tomopy_SOURCE_DIR}/include/libtomo/morph.h"
            "${tomopy_SOURCE_DIR}/include/libtomo/remove_ring.h"
            "${tomopy_SOURCE_DIR}/include/libtomo/median_filt3d.h")

tomopy_add_library(tomo-misc SHARED morph.c remove_ring.c median_filt3d.c ${HEADERS})

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_STATIC_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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
