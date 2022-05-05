if(TOMOPY_CXX_GRIDREC)
    SET_SOURCE_FILES_PROPERTIES("gridrec.c" PROPERTIES LANGUAGE CXX)
endif()

add_library(tomo-gridrec gridrec.c pal.h)

target_link_libraries(tomo-gridrec PUBLIC ${TOMOPY_EXTERNAL_LIBRARIES}
    PRIVATE ${TOMOPY_EXTERNAL_PRIVATE_LIBRARIES})

target_include_directories(tomo-gridrec PUBLIC
    $<BUILD_INTERFACE:${tomopy_SOURCE_DIR}/source/include/gridrec>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(tomo-gridrec PUBLIC
    ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(tomo-gridrec PUBLIC
    $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>)

install(TARGETS tomo-gridrec)
