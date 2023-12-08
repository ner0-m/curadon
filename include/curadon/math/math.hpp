#pragma once

#include <cmath>
#include <cstdint>

namespace curad::math {
template <class T, class U>
__host__ __device__ auto min(const T &x, const U &y) -> decltype(x < y ? x : y) {
    return x < y ? x : y;
}

template <class T, class U>
__host__ __device__ auto max(const T &x, const U &y) -> decltype(x > y ? x : y) {
    return x > y ? x : y;
}

template <class T>
__host__ __device__ auto floor(const T &x) -> std::int64_t {
    return ::floor(x);
    // int xi = static_cast<std::int64_t>(x);
    // return x < xi ? xi - 1 : xi;
}

template <class T>
__host__ __device__ auto ceil(const T &x) -> std::int64_t {
    return ::ceil(x);
}

template <class T>
__host__ __device__ auto abs(const T &x) -> T {
    return x < T{0} ? -x : x;
}

template <typename T>
__host__ __device__ int sign(T val) {
    return (T{0} < val) - (val < T{0});
}
} // namespace curad::math
