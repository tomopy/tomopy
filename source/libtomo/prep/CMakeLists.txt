# Locate sources for this project headers are included for IDEs
file(GLOB prep_tomo_sources
    ${CMAKE_CURRENT_LIST_DIR}/*.c)

file(GLOB prep_tomo_headers
    ${CMAKE_CURRENT_LIST_DIR}/../../include/prep/*.h)

# core sources
set(PREP_LIBRARY_SOURCES ${prep_tomo_headers} ${prep_tomo_sources})

# create library
add_library(libtomopy-prep SHARED ${PREP_LIBRARY_SOURCES})

target_link_libraries(libtomopy-prep PUBLIC ${EXTERNAL_LIBRARIES}
    PRIVATE ${EXTERNAL_PRIVATE_LIBRARIES})

list(APPEND TARGET_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../include/prep")

# target properties
set_target_properties(libtomopy-prep
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

target_compile_definitions(libtomopy-prep PUBLIC
    ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(libtomopy-prep PUBLIC
    $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CUDA_FLAGS}>)

install(TARGETS libtomopy-prep
    DESTINATION ${PYTHON_SITE_DIR}/tomopy/sharedlibs COMPONENT python)