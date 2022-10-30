include(FetchContent)
FetchContent_Declare(
    asio_fetch
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio
    GIT_TAG asio-1-24-0
    GIT_SHALLOW TRUE
)
FetchContent_GetProperties(asio_fetch)
if(NOT asio_fetch_POPULATED)
    FetchContent_Populate(asio_fetch)

    add_library(asio INTERFACE)
    target_include_directories(asio SYSTEM INTERFACE ${asio_fetch_SOURCE_DIR}/asio/include)
    target_compile_definitions(asio
        INTERFACE
        ASIO_DISABLE_VISIBILITY
        ASIO_NO_DEPRECATED
        ASIO_NO_DYNAMIC_BUFFER_V1
        ASIO_NO_TS_EXECUTORS
        ASIO_NO_TYPEID
        ASIO_STANDALONE)
endif()