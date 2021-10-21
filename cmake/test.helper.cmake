function(add_default_test TARGET_NAME)
    if (ORDERED_DITHERING_TESTS)
        file(GLOB_RECURSE UT_SRC "unittest/*.cc")
        set(UT_NAME "${TARGET_NAME}_unittest")
        add_executable(${UT_NAME} ${UT_SRC})
        target_include_directories(${UT_NAME}
                PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
        target_link_libraries(${UT_NAME}
            PRIVATE ordered_dithering_copts_common ${TARGET_NAME} ordered_dithering_unittest_common)

        configure_lto(${UT_NAME})

        add_test(
            NAME ${UT_NAME}
            COMMAND ${UT_NAME} $ENV{EXTRA_TEST_ARGS})
    endif()
endfunction()
