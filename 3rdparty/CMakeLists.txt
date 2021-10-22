include(FindPkgConfig)

pkg_check_modules(VIPS vips REQUIRED)
pkg_check_modules(VIPS_CPP vips-cpp REQUIRED)
add_library(vips INTERFACE)
target_include_directories(vips SYSTEM
    INTERFACE ${VIPS_INCLUDE_DIRS} ${VIPS_CPP_INCLUDE_DIRS})
target_compile_options(vips SYSTEM
    INTERFACE ${VIPS_CFLAGS} ${VIPS_CPP_CFLAGS})
target_compile_definitions(vips
    INTERFACE ${VIPS_DEFINITIONS} ${VIPS_CPP_DEFINITIONS})
target_link_libraries(vips
    INTERFACE ${VIPS_CPP_LINK_LIBRARIES} ${VIPS_LINK_LIBRARIES})

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