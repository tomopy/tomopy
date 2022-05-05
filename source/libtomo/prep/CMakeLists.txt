add_library(tomo-prep prep.c stripe.c)

target_include_directories(
  tomo-prep PUBLIC $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/source/include/prep>
                   $<INSTALL_INTERFACE:include>)

target_compile_definitions(tomo-prep PUBLIC ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(
  tomo-prep PUBLIC $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>)

install(TARGETS tomo-prep)
