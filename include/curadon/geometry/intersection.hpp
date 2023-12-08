#pragma once

#include "curadon/geometry/box.hpp"
#include "curadon/geometry/ray.hpp"
#include "curadon/math/vector.hpp"
#include <cstdint>

namespace curad {
struct IntersectionResult {
    bool hit;
    float tmin;
    float tmax;

    __host__ __device__ operator bool() const { return hit; }
};

template <class T, std::int64_t Dim>
__host__ __device__ IntersectionResult intersect(const box<T, Dim> &aabb, const ray<T, Dim> &ray) {
    T tmin = -INFINITY;
    T tmax = INFINITY;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        const T t1 = (aabb.min()[i] - ray.origin()[i]) * ray.inv_direction()[i];
        const T t2 = (aabb.max()[i] - ray.origin()[i]) * ray.inv_direction()[i];

        tmin = math::min(math::max(t1, tmin), math::max(t2, tmin));
        tmax = math::max(math::min(t1, tmax), math::min(t2, tmax));
    }

    return {tmin <= tmax, tmin, tmax};
}
} // namespace curad
