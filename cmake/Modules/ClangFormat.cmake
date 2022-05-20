################################################################################
#
#        Creates a 'format' target that runs clang-format
#
################################################################################

find_program(CLANG_FORMATTER
    NAMES
        clang-format-8.0
        clang-format-7.0
        clang-format-6.0
        clang-format)

if(CLANG_FORMATTER)
    file(GLOB headers
        ${PROJECT_SOURCE_DIR}/include/libtomo/*.h
        ${PROJECT_SOURCE_DIR}/include/libtomo/*.h
        ${PROJECT_SOURCE_DIR}/include/libtomo/*.hh
        ${PROJECT_SOURCE_DIR}/include/libtomo/*.hpp)
    file(GLOB sources
        ${PROJECT_SOURCE_DIR}/source/libtomo/*/*.c
        ${PROJECT_SOURCE_DIR}/source/libtomo/*/*.h
        ${PROJECT_SOURCE_DIR}/source/libtomo/*/*.hh
        ${PROJECT_SOURCE_DIR}/source/libtomo/*/*.cc
        ${PROJECT_SOURCE_DIR}/source/libtomo/*/cxx/*.cc
        ${PROJECT_SOURCE_DIR}/source/libtomo/*/gpu/*.cu)

    # avoid conflicting format targets
    set(FORMAT_NAME format)
    if(TARGET format)
        set(FORMAT_NAME format-libtomo)
    endif()

    add_custom_target(${FORMAT_NAME}
        COMMAND ${CLANG_FORMATTER} -i ${headers} ${sources}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Running '${CLANG_FORMATTER}'..."
        SOURCES ${headers} ${sources})
endif()
