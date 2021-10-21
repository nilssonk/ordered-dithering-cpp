project(ordered_dithering_main VERSION 0.1 LANGUAGES CXX)

include(FindImageMagick)

find_package(ImageMagick COMPONENTS Magick++ REQUIRED)

file(GLOB_RECURSE INC "include/*.hh")
file(GLOB_RECURSE SRC "src/*.cc")

add_library(magick INTERFACE)
target_include_directories(magick SYSTEM
    INTERFACE ${ImageMagick_INCLUDE_DIRS})
target_compile_definitions(magick
    INTERFACE ${ImageMagick_DEFINITIONS})
target_link_libraries(magick
    INTERFACE ${ImageMagick_LIBRARIES})

add_executable(ordered_dithering_main)
target_sources(ordered_dithering_main PRIVATE ${INC} ${SRC})
target_link_libraries(ordered_dithering_main PRIVATE
    ordered_dithering_copts_common
    ordered_dithering_copts_magick
    magick
    fmt
    sg14)

configure_lto(ordered_dithering_main)
configure_tidy(ordered_dithering_main)

