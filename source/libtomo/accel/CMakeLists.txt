add_library(tomo-accel
    cxx/common.cc
    cxx/mlem.cc
    cxx/sirt.cc
    common.hh
    constants.hh
    cuda.h
    data.hh
    macros.h
    macros.hh
    profiler.hh
    typedefs.hh
    utils.hh
    ${tomopy_SOURCE_DIR}/source/include/accel/cxx_extern.h
)

target_include_directories(
    tomo-accel PUBLIC
    $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/source/include/accel>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

if(TRUE)
    # FIXME: Need PTL headers regardless of whether we use PTL
    target_include_directories(
    tomo-accel PRIVATE
        ${tomopy_SOURCE_DIR}/source/PTL/source
    )
endif()

if(TOMOPY_USE_CUDA)

    target_sources(tomo-accel PRIVATE
        gpu/common.cu
        gpu/mlem.cu
        gpu/sirt.cu
        gpu/utils.cu
    )

    target_include_directories(tomo-accel PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    target_link_libraries(tomo-accel
        PUBLIC tomopy::cuda-npp
    )

    set(CUDA_PROPERTIES
        CUDA_STANDARD               ${CMAKE_CXX_STANDARD}
        CUDA_STANDARD_REQUIRED      ON
        CUDA_SEPARABLE_COMPILATION  ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    set_target_properties(tomo-accel
        PROPERTIES
        ${CUDA_PROPERTIES})

endif(TOMOPY_USE_CUDA)

target_link_libraries(tomo-accel
    PUBLIC ${TOMOPY_EXTERNAL_LIBRARIES}
    PRIVATE ${TOMOPY_EXTERNAL_PRIVATE_LIBRARIES})

target_compile_definitions(tomo-accel PUBLIC
    ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(tomo-accel PUBLIC
    $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CUDA_FLAGS}>)

install(TARGETS tomo-accel)



