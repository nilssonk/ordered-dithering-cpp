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