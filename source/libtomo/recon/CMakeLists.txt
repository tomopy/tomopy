# Locate sources for this project headers are included for IDEs
file(GLOB recon_tomo_sources
    ${CMAKE_CURRENT_LIST_DIR}/*.c)

# core sources
set(RECON_LIBRARY_SOURCES ${recon_tomo_sources})

add_library(libtomopy-recon SHARED ${RECON_LIBRARY_SOURCES})

list(APPEND TARGET_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../include/recon")

# target properties
set_target_properties(libtomopy-recon
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

target_link_libraries(libtomopy-recon PUBLIC ${EXTERNAL_LIBRARIES}
    PRIVATE ${EXTERNAL_PRIVATE_LIBRARIES})

target_compile_definitions(libtomopy-recon PUBLIC
    ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(libtomopy-recon PUBLIC
    $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CUDA_FLAGS}>)

install(TARGETS libtomopy-recon
    DESTINATION ${PYTHON_SITE_DIR}/tomopy/sharedlibs
    COMPONENT python)