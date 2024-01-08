#pragma once

#include "curadon/detail/vec.hpp"
#include "curadon/utils.hpp"

namespace curad {
template <class T>
class degree;

template <class T>
class radian {
  public:
    __host__ __device__ radian(T radian)
        : radian_(radian) {}

    __host__ __device__ degree<T> to_degree() const noexcept;

    __host__ __device__ const T &value() const noexcept { return radian_; }

    __host__ __device__ T &value() noexcept { return radian_; }

  private:
    T radian_;
};

template <class T>
class degree {
  public:
    __host__ __device__ degree(T degree)
        : degree_(degree) {}

    __host__ __device__ radian<T> to_radian() const noexcept {
        return radian<T>(value() * M_PI / 180.0);
    }

    __host__ __device__ const T &value() const noexcept { return degree_; }

    __host__ __device__ T &value() noexcept { return degree_; }

  private:
    T degree_;
};

template <class T>
__host__ __device__ degree<T> radian<T>::to_degree() const noexcept {
    return value() * 180.0 / M_PI;
}

} // namespace curad

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
