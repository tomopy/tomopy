find_package(pybind11 REQUIRED)

pybind11_add_module(normalize MODULE normalize.cxx)

set_target_properties(normalize
PROPERTIES
PREFIX ""
RUNTIME_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/util/extern
LIBRARY_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/util/extern
ARCHIVE_OUTPUT_DIRECTORY    ${OUTPUT_PATH}/tomopy/util/extern
INCLUDE_DIRECTORIES         "${CMAKE_CURRENT_LIST_DIR}/../../include/prep"
)

target_compile_options(normalize PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_C_FLAGS}>)

target_link_libraries(normalize PUBLIC libtomopy-prep)

install(TARGETS normalize
    DESTINATION ${PYTHON_SITE_DIR}/tomopy/util/extern
    COMPONENT python)
