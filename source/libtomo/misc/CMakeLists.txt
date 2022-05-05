add_library(tomo-misc morph.c remove_ring.c)

target_include_directories(
  tomo-misc PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/source/include/misc>
                   $<INSTALL_INTERFACE:include>)

# link library to math.h
if(NOT WIN32)
  target_link_libraries(tomo-misc PUBLIC m)
endif()

target_compile_definitions(tomo-misc PUBLIC ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-misc PUBLIC $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>)

install(TARGETS tomo-misc)
