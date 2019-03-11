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
    file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.h)
    file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.c)
    add_custom_target(format
        COMMAND ${CLANG_FORMATTER} -i ${headers} ${sources}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Running '${CLANG_FORMATTER}'..."
        SOURCES ${headers} ${sources})
endif()
