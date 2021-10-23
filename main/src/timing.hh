#ifndef TIMING_HH_INCLUDED_
#define TIMING_HH_INCLUDED_

#ifdef TIMING_ENABLED

#include <chrono>

template<typename F>
[[nodiscard]] inline auto
timed_exec(std::string_view const name, F && f)
{
    using std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using ms = std::chrono::milliseconds;

    auto const start = steady_clock::now();
    if constexpr (std::is_same_v<std::invoke_result_t<F>, void>) {
        f();
        auto const end = steady_clock::now();

        auto const duration = duration_cast<ms>(end - start);
        fmt::print("{} duration: {}ms\n", name, duration.count());
    } else {
        auto const result = f();
        auto const end = steady_clock::now();

        auto const duration = duration_cast<ms>(end - start);
        fmt::print("{} duration: {}ms\n", name, duration.count());

        return result;
    }
}

#else

template<typename F>
[[nodiscard]] inline auto
timed_exec(std::string_view const /*unused*/, F && f)
{
    return f();
}

#endif

#endif
