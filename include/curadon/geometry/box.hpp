#pragma once

#include "curadon/math/vector.hpp"
#include <cstdio>

namespace curad {
template <class T, std::int64_t Dim>
struct box {
  public:
    __host__ __device__ box(const Vec<T, Dim> &min, const Vec<T, Dim> &max) : corners{min, max} {}

    __host__ __device__ const Vec<T, Dim> &min() const noexcept { return corners[0]; }

    __host__ __device__ Vec<T, Dim> &min() noexcept { return corners[0]; }

    __host__ __device__ const Vec<T, Dim> &max() const noexcept { return corners[1]; }

    __host__ __device__ Vec<T, Dim> &max() noexcept { return corners[1]; }

    __host__ __device__ const Vec<T, Dim> &operator[](std::size_t i) const noexcept {
        return corners[i];
    }

    __host__ __device__ Vec<T, Dim> &operator[](std::size_t i) noexcept { return corners[i]; }

    __host__ __device__ Vec<T, Dim> center() const noexcept { return (min() + max()) * 0.5; }

    __host__ __device__ Vec<T, Dim> center_voxel() const noexcept { return floor(center()); }

  private:
    Vec<T, Dim> corners[2];
};
} // namespace curad
