# Locate sources for this project headers are included for IDEs
file(GLOB accel_tomo_sources
    ${CMAKE_CURRENT_LIST_DIR}/cxx/*.cc)

# core sources
set(ACCEL_LIBRARY_SOURCES ${tomo_headers} ${accel_tomo_sources})

# create library
add_library(libtomopy-accel SHARED ${ACCEL_LIBRARY_SOURCES})

target_link_libraries(libtomopy-accel PUBLIC ${EXTERNAL_LIBRARIES}
    PRIVATE ${EXTERNAL_PRIVATE_LIBRARIES})

# target properties
set_target_properties(libtomopy-accel
    PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/sharedlibs
    LIBRARY_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/sharedlibs
    ARCHIVE_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/sharedlibs
    INCLUDE_DIRECTORIES         "${TARGET_INCLUDE_DIRECTORIES}"
    C_STANDARD                  ${CMAKE_C_STANDARD}
    C_STANDARD_REQUIRED         ${CMAKE_C_STANDARD_REQUIRED}
    CXX_STANDARD                ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED       ${CMAKE_CXX_STANDARD_REQUIRED}
    ${CUDA_PROPERTIES})

target_compile_definitions(libtomopy-accel PUBLIC
    ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(libtomopy-accel PUBLIC
    $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CUDA_FLAGS}>)

install(TARGETS libtomopy-accel
    DESTINATION ${PYTHON_SITE_DIR}/tomopy/sharedlibs
    COMPONENT python)