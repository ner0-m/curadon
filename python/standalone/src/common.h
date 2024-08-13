#pragma once

struct __half;

namespace nanobind {
template <>
struct ndarray_traits<__half> {
    static constexpr bool is_complex = false;
    static constexpr bool is_float = true;
    static constexpr bool is_bool = false;
    static constexpr bool is_int = false;
    static constexpr bool is_signed = true;
};
}; // namespace nanobind
