#pragma once

#include <limits>

#include "curadon/detail/vec.hpp"
#include "curadon/types.hpp"

#include <tuple>

namespace curad::fp::kernel {
template <i64 Dim>
__host__ __device__ std::tuple<bool, float, float>
intersection(const vec<float, Dim> &boxmin, const vec<float, Dim> &boxmax,
             const vec<float, Dim> &ro, const vec<float, Dim> &rd) {
    float tmin = std::numeric_limits<float>::min();
    float tmax = std::numeric_limits<float>::max();

#pragma unroll
    for (int d = 0; d < Dim; ++d) {
        auto t1 = __fdividef(boxmin[d] - ro[d], rd[d]);
        auto t2 = __fdividef(boxmax[d] - ro[d], rd[d]);

        tmin = fminf(fmaxf(t1, tmin), fmaxf(t2, tmin));
        tmax = fmaxf(fminf(t1, tmax), fminf(t2, tmax));
    }

    return std::make_tuple(tmin <= tmax, tmin, tmax);
}

template <i64 Dim>
__host__ __device__ std::tuple<float, float>
intersection_2d(vec2u aabb_max, const vec<float, Dim> &ro, const vec<float, Dim> &rd) {
    float tmin = std::numeric_limits<float>::min();
    float tmax = std::numeric_limits<float>::max();

#pragma unroll
    for (int d = 0; d < Dim; ++d) {
        auto t1 = __fdividef(-1 - ro[d], rd[d]);
        auto t2 = __fdividef(aabb_max[d] - ro[d], rd[d]);

        tmin = fminf(fmaxf(t1, tmin), fmaxf(t2, tmin));
        tmax = fmaxf(fminf(t1, tmax), fminf(t2, tmax));
    }

    return std::make_tuple(tmin, tmax);
}
} // namespace curad::fp::kernel
