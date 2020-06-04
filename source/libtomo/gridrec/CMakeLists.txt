# If on Windows, build gridrec.cc
if(WIN32)
# Locate sources for this project headers are included for IDEs
file(GLOB gridrec_tomo_sources
    ${CMAKE_CURRENT_LIST_DIR}/cxx/*.cc)   

else(WIN32)

file(GLOB gridrec_tomo_sources
    ${CMAKE_CURRENT_LIST_DIR}/c/*.c)  

endif(WIN32)

# core sources
set(GRIDREC_SOURCES ${tomo_headers} ${GRIDREC_tomo_sources})

# create library
add_library(libtomopy-gridrec SHARED ${GRIDREC_SOURCES})

# link library
target_link_libraries(libtomopy-gridrec PUBLIC ${EXTERNAL_LIBRARIES}
    PRIVATE ${EXTERNAL_PRIVATE_LIBRARIES})

# where to place the library
set(OUTPUT_PATH ${CMAKE_BINARY_DIR})

# target properties
set_target_properties(libtomopy-gridrec
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

target_compile_definitions(libtomopy-gridrec PUBLIC
    ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(libtomopy-gridrec PUBLIC
    $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CUDA_FLAGS}>)

install(TARGETS libtomopy-gridrec
    DESTINATION ${PYTHON_SITE_DIR}/tomopy/sharedlibs
    COMPONENT python)
