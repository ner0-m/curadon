#pragma once

#include "curadon/math/vector.hpp"

namespace curad {
template <typename T, std::int64_t Dim>
class ray {
  public:
    __host__ __device__ ray(const Vec<T, Dim> &origin, const Vec<T, Dim> &direction)
        : origin_(origin), direction_(normalized(direction)), inv_direction_(T{1} / direction) {}

    __host__ __device__ const Vec<T, Dim> &origin() const { return origin_; }
    __host__ __device__ Vec<T, Dim> &origin() { return origin_; }

    __host__ __device__ const Vec<T, Dim> &direction() const { return direction_; }
    __host__ __device__ Vec<T, Dim> &direction() { return direction_; }

    __host__ __device__ const Vec<T, Dim> &inv_direction() const { return inv_direction_; }
    __host__ __device__ Vec<T, Dim> &inv_direction() { return inv_direction_; }

    __host__ __device__ Vec<T, Dim> at(T t) const { return origin_ + t * direction_; }

  private:
    Vec<T, Dim> origin_;
    Vec<T, Dim> direction_;
    Vec<T, Dim> inv_direction_;
};
} // namespace curad
