
add_option(BUILD_TESTS "Build compiled tests" OFF)
if(NOT BUILD_TESTS)
    return()
endif()
  
if(TOMOPY_USE_OPENCV AND OpenCV_FOUND)
    add_executable(test_opencv test_opencv.cc)
    target_link_libraries(test_opencv ${EXTERNAL_LIBRARIES})
    target_link_libraries(test_opencv ${EXTERNAL_LIBRARIES})
    target_compile_definitions(test_opencv PUBLIC
        ${${PROJECT_NAME}_DEFINITIONS})

    target_include_directories(test_opencv PRIVATE
        ${TARGET_INCLUDE_DIRECTORIES})

    target_compile_options(test_opencv PUBLIC
        $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>)
endif()

if(TOMOPY_USE_CUDA)
    add_executable(test_nppi test_nppi.cu)
    target_link_libraries(test_nppi ${EXTERNAL_LIBRARIES} ${EXTERNAL_CUDA_LIBRARIES})
    target_compile_definitions(test_nppi PUBLIC
        ${${PROJECT_NAME}_DEFINITIONS})

    target_include_directories(test_nppi PRIVATE
        ${TARGET_INCLUDE_DIRECTORIES})

    target_compile_options(test_nppi PUBLIC
        $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CUDA_FLAGS}>)
endif()
