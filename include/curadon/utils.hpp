#pragma once

#include "curadon/math/vector.hpp"
#include "curadon/utils.hpp"

#include "cuda/std/cmath"

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

namespace detail {
template <class T, std::int64_t Dim>
__host__ __device__ auto rotate(T phi, Vec<T, Dim> v) -> Vec<T, Dim> {
    auto s = cuda::std::sin(phi);
    auto c = cuda::std::cos(phi);

    return {v[0] * c - v[1] * s, v[0] * s + v[1] * c};
}

template <class T>
__host__ __device__ auto perpendicular(Vec<T, 2> v) -> Vec<T, 2> {
    return {-v[1], v[0]};
}
} // namespace detail
} // namespace curad
