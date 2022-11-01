#ifndef MONOTONIC_BUFFER_RESOURCE_HH
#define MONOTONIC_BUFFER_RESOURCE_HH

#include <cstddef>
#include <cstdint>
#include <memory_resource>

class MonotonicBufferResource final : public std::pmr::memory_resource {
    std::byte * data_;
    size_t      size_;

    void * // NOLINTNEXTLINE(modernize-use-trailing-return-type)
    do_allocate(std::size_t bytes, std::size_t alignment) final;
    void // NOLINTNEXTLINE(modernize-use-trailing-return-type)
    do_deallocate(void * p, std::size_t bytes, std::size_t alignment) final;
    bool // NOLINTNEXTLINE(modernize-use-trailing-return-type)
    do_is_equal(const std::pmr::memory_resource & other) const noexcept final;

public:
    MonotonicBufferResource(std::byte * data, size_t size)
        : data_{data}, size_{size}
    {
    }
};

#endif
