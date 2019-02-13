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
        ${PROJECT_SOURCE_DIR}/source/include/*.h
        ${PROJECT_SOURCE_DIR}/source/include/*.h
        ${PROJECT_SOURCE_DIR}/source/include/*.hh
        ${PROJECT_SOURCE_DIR}/source/include/*.hpp)
    file(GLOB sources
        ${PROJECT_SOURCE_DIR}/source/c/*.c
        ${PROJECT_SOURCE_DIR}/source/cxx/*.cc
        ${PROJECT_SOURCE_DIR}/source/cxx/*.cpp
        ${PROJECT_SOURCE_DIR}/source/gpu/*.cc
        ${PROJECT_SOURCE_DIR}/source/gpu/*.cu
        ${PROJECT_SOURCE_DIR}/test/*.cc
        ${PROJECT_SOURCE_DIR}/test/*.cu
        ${PROJECT_SOURCE_DIR}/test/*.cpp)

    # avoid conflicting format targets
    set(FORMAT_NAME format)
    if(TARGET format)
        set(FORMAT_NAME format-tomopy)
    endif()

    add_custom_target(${FORMAT_NAME}
        COMMAND ${CLANG_FORMATTER} -i ${headers} ${sources}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Running '${CLANG_FORMATTER}'..."
        SOURCES ${headers} ${sources})
endif()
