cmake_minimum_required(VERSION 3.14)

project(ordered_dithering VERSION 0.1 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(lto.helper)
include(test.helper)
include(with.helper)
include(tidy.helper)

option(ORDERED_DITHERING_CLANG_TIDY "Build with clang-tidy" ON)
option(ORDERED_DITHERING_TESTS "Build and add tests" ON)
option(ORDERED_DITHERING_COVERAGE "Build with coverage support" OFF)
option(ORDERED_DITHERING_LTO "Build with Link-Time Optimization" OFF)
option(ORDERED_DITHERING_TIMING "Build with processing stage timing enabled" OFF)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
else()
    set(ORDERED_DITHERING_CLANG_TIDY OFF)
    # Disable DeathTests
    set(ENV{EXTRA_TEST_ARGS} --gtest_filter=-*DeathTest*)
endif()

WithMsg(tests ${ORDERED_DITHERING_TESTS})
WithMsg(clang-tidy ${ORDERED_DITHERING_CLANG_TIDY})
WithMsg(coverage ${ORDERED_DITHERING_COVERAGE})
WithMsg(LTO ${ORDERED_DITHERING_LTO})
WithMsg("processing stage timing" ${ORDERED_DITHERING_TIMING})

if (ORDERED_DITHERING_CLANG_TIDY)
    include(clang-tidy)
endif()

set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --child-silent-after-fork=yes --leak-check=full --error-exitcode=5")

if (ORDERED_DITHERING_TESTS)
    include(CTest)
    enable_testing()
endif()

set(ORDERED_DITHERING_EXTRA_CFLAGS -O3 -ffast-math -fvisibility=hidden)

set_property(GLOBAL PROPERTY ORDERED_DITHERING_RELEASE_EXTRA_CFLAGS ${ORDERED_DITHERING_EXTRA_CFLAGS})

add_library(ordered_dithering_copts_common INTERFACE)
target_compile_options(ordered_dithering_copts_common INTERFACE
    -pedantic
    -W
    -Wall
    -Wextra
    -Wcast-align
    -Wcast-qual
    -Wmissing-declarations
    -Wwrite-strings
    -Wundef
    -Wswitch-enum
    -Wshadow
    -Werror
    -Wstrict-aliasing=2
    $<$<C_COMPILER_ID:Clang>:-Wshadow-all>
    $<$<CXX_COMPILER_ID:Clang>:-Wshadow-all>
    $<$<STREQUAL:$<CONFIG>,Debug>:-ggdb>
    $<$<STREQUAL:$<CONFIG>,Release>:${ORDERED_DITHERING_EXTRA_CFLAGS}>
    $<$<BOOL:${ORDERED_DITHERING_COVERAGE}>:-g -fprofile-arcs -ftest-coverage>)
target_link_libraries(ordered_dithering_copts_common
    INTERFACE
    $<$<BOOL:${ORDERED_DITHERING_COVERAGE}>:
        $<IF:$<STREQUAL:${CMAKE_CXX_COMPILER_ID},GNU>,
            gcov, # GCC
            --coverage # Clang
        >
    >)


add_subdirectory(3rdparty)
add_subdirectory(main)
