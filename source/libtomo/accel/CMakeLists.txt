# target properties based on build

file(GLOB accel_sources
    ${CMAKE_CURRENT_LIST_DIR}/cxx/*.cc)

set(ACCEL_LIBRARY_SOURCES ${accel_sources})

if(TOMOPY_USE_CUDA)

    file(GLOB CUDA_LIBRARY_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/gpu/*.cu)

    list(APPEND ACCEL_LIBRARY_SOURCES ${CUDA_LIBRARY_SOURCES})
    
    list(APPEND TARGET_INCLUDE_DIRECTORIES
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

endif(TOMOPY_USE_CUDA)

add_library(libtomopy-accel SHARED ${ACCEL_LIBRARY_SOURCES})

target_link_libraries(libtomopy-accel PUBLIC ${EXTERNAL_LIBRARIES}
    PRIVATE ${EXTERNAL_PRIVATE_LIBRARIES})

set(CUDA_PROPERTIES
    CUDA_STANDARD               ${CMAKE_CXX_STANDARD}
    CUDA_STANDARD_REQUIRED      ON
    CUDA_SEPARABLE_COMPILATION  ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

list(APPEND TARGET_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../include/accel")

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



