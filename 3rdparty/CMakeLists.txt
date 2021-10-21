include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 8.0.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
    sg14
    GIT_REPOSITORY https://github.com/WG21-SG14/SG14.git
    GIT_SHALLOW TRUE
)
if (NOT sg14_POPULATED)
    FetchContent_Populate(sg14)

    add_library(sg14 INTERFACE)
    target_include_directories(sg14 SYSTEM
        INTERFACE $<BUILD_INTERFACE:${sg14_SOURCE_DIR}>)
endif()