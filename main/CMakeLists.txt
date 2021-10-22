project(ordered_dithering_main VERSION 0.1 LANGUAGES CXX)

file(GLOB_RECURSE INC "include/*.hh")
file(GLOB_RECURSE SRC "src/*.cc")

add_executable(ordered_dithering_main)
target_sources(ordered_dithering_main PRIVATE ${INC} ${SRC})
target_link_libraries(ordered_dithering_main PRIVATE
    ordered_dithering_copts_common
    vips
    fmt
    sg14)

configure_lto(ordered_dithering_main)
configure_tidy(ordered_dithering_main)

