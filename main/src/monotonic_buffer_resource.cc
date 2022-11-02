#include "monotonic_buffer_resource.hh"

void * // NOLINTNEXTLINE(modernize-use-trailing-return-type)
MonotonicBufferResource::do_allocate(std::size_t bytes, std::size_t alignment)
{
    auto const
        align_diff = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        (alignment - reinterpret_cast<uintptr_t>(data_) % alignment) %
        alignment;
    auto const alloc_size = bytes + align_diff;
    if (alloc_size > size_) {
        return nullptr;
    }

    auto * const alloc_base = data_ + align_diff;
    data_ += alloc_size;
    size_ -= alloc_size;
    return alloc_base;
}

void // NOLINTNEXTLINE(modernize-use-trailing-return-type)
MonotonicBufferResource::do_deallocate(void *      p,
                                       std::size_t bytes,
                                       std::size_t alignment)
{
    (void)p;
    (void)bytes;
    (void)alignment;
}

bool // NOLINTNEXTLINE(modernize-use-trailing-return-type)
MonotonicBufferResource::do_is_equal(
    const std::pmr::memory_resource & other) const noexcept
{
    (void)other;
    return false;
}