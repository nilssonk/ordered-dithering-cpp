function(configure_tidy TARGET_NAME)
    if (ORDERED_DITHERING_CLANG_TIDY)
        clang_tidy_check(${TARGET_NAME})
    endif()
endfunction()