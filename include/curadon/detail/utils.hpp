#pragma once

#include <type_traits>

namespace curad::utils {
/// Divide x by y and round up.
///
/// This is only really valid for integer types, but I'm to lazy to enforce that right now
template <class T, class U>
std::common_type_t<T, U> round_up_division(T x, U y) {
    // TODO: 1 + ((x - 1) / y); avoid overflow, but is only allowed if x != 0
    return (x + y - 1) / y;
}
} // namespace curad::utils
