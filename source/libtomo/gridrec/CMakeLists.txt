if(TOMOPY_CXX_GRIDREC)
    SET_SOURCE_FILES_PROPERTIES("gridrec.c" PROPERTIES LANGUAGE CXX)
endif()

file(GLOB gridrec_tomo_sources
    ${CMAKE_CURRENT_LIST_DIR}/*.c)
file(GLOB gridrec_tomo_headers
    ${CMAKE_CURRENT_LIST_DIR}/../../include/gridrec/*.h
    ${CMAKE_CURRENT_LIST_DIR}/*.h)

# core sources
set(GRIDREC_SOURCES ${gridrec_tomo_headers} ${gridrec_tomo_sources})

# create library
add_library(libtomopy-gridrec SHARED ${GRIDREC_SOURCES})

# link library
target_link_libraries(libtomopy-gridrec PUBLIC ${EXTERNAL_LIBRARIES}
    PRIVATE ${EXTERNAL_PRIVATE_LIBRARIES})

# where to place the library
set(OUTPUT_PATH ${CMAKE_BINARY_DIR})

list(APPEND TARGET_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../include/gridrec")
list(APPEND TARGET_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}")

# target properties
set_target_properties(libtomopy-gridrec
    PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/util/extern
    LIBRARY_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/util/extern
    ARCHIVE_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/util/extern
    INCLUDE_DIRECTORIES         "${TARGET_INCLUDE_DIRECTORIES}"
    C_STANDARD                  ${CMAKE_C_STANDARD}
    C_STANDARD_REQUIRED         ${CMAKE_C_STANDARD_REQUIRED}
    CXX_STANDARD                ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED       ${CMAKE_CXX_STANDARD_REQUIRED})

target_compile_definitions(libtomopy-gridrec PUBLIC
    ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(libtomopy-gridrec PUBLIC
    $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>)

install(TARGETS libtomopy-gridrec
    DESTINATION ${PYTHON_SITE_DIR}/tomopy/util/extern
    COMPONENT python)
