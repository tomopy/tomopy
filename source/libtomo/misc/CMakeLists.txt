
# Locate sources for this project headers are included for IDEs
file(GLOB misc_tomo_sources
    ${CMAKE_CURRENT_LIST_DIR}/*.c)

# core sources
set(MISC_LIBRARY_SOURCES ${misc_tomo_sources})

# create library
add_library(libtomopy-misc SHARED ${MISC_LIBRARY_SOURCES})

# link library to math.h
if (NOT WIN32)
    target_link_libraries(libtomopy-misc PUBLIC m)
endif() 

# where to place the library
set(OUTPUT_PATH ${CMAKE_BINARY_DIR})

list(APPEND TARGET_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../include/misc")

# target properties
set_target_properties(libtomopy-misc
    PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/util/extern
    LIBRARY_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/util/extern
    ARCHIVE_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/util/extern
    INCLUDE_DIRECTORIES         "${TARGET_INCLUDE_DIRECTORIES}"
    C_STANDARD                  ${CMAKE_C_STANDARD}
    C_STANDARD_REQUIRED         ${CMAKE_C_STANDARD_REQUIRED})

target_compile_definitions(libtomopy-misc PUBLIC
    ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(libtomopy-misc PUBLIC
    $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>)

install(TARGETS libtomopy-misc
    DESTINATION ${PYTHON_SITE_DIR}/tomopy/util/extern
    COMPONENT python)
