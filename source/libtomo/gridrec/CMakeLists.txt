set(HEADERS "${tomopy_SOURCE_DIR}/include/libtomo/filters.h"
            "${tomopy_SOURCE_DIR}/include/libtomo/gridrec.h")

if(TOMOPY_CXX_GRIDREC)
  set_source_files_properties("gridrec.c" PROPERTIES LANGUAGE CXX)
endif()

add_library(tomo-gridrec SHARED gridrec.c pal.h ${HEADERS})

target_link_libraries(tomo-gridrec PRIVATE ${TOMOPY_EXTERNAL_LIBRARIES}
                                           ${TOMOPY_EXTERNAL_PRIVATE_LIBRARIES})

target_include_directories(
  tomo-gridrec
  PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/include/libtomo>
         $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/libtomo>
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_compile_definitions(tomo-gridrec PRIVATE ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-gridrec PUBLIC $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
                      $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>)

install(TARGETS tomo-gridrec EXPORT libtomoTargets)

install(
  FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtomo
  COMPONENT Headers)
