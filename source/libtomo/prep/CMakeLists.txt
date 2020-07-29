# Locate sources for this project headers are included for IDEs
file(GLOB prep_tomo_sources
    ${CMAKE_CURRENT_LIST_DIR}/*.c)

# core sources
set(PREP_LIBRARY_SOURCES ${prep_tomo_sources})

# create library
add_library(libtomopy-prep SHARED ${PREP_LIBRARY_SOURCES})

list(APPEND TARGET_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../include/prep")

# target properties
set_target_properties(libtomopy-prep
    PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/util/extern
    LIBRARY_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/util/extern
    ARCHIVE_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/util/extern
    INCLUDE_DIRECTORIES         "${TARGET_INCLUDE_DIRECTORIES}"
    C_STANDARD                  ${CMAKE_C_STANDARD}
    C_STANDARD_REQUIRED         ${CMAKE_C_STANDARD_REQUIRED})

target_compile_definitions(libtomopy-prep PUBLIC
    ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(libtomopy-prep PUBLIC
    $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>)

install(TARGETS libtomopy-prep
    DESTINATION ${PYTHON_SITE_DIR}/tomopy/util/extern COMPONENT python)
