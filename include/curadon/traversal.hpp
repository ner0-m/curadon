#pragma once

#include "curadon/geometry/intersection.hpp"
#include "curadon/math/vector.hpp"

#include <cstdint>
#include <iostream>

namespace curad {
template <std::int64_t Dim>
__host__ __device__ auto coord_to_idx(const Vec<std::int64_t, Dim> &coord,
                                      const Vec<std::int64_t, Dim> &strides) -> std::uint64_t {
    auto idx = 0;
#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        idx += coord[i] * strides[i];
    }
    return idx;
}

namespace joseph {

template <class T, std::int64_t Dim>
__host__ __device__ Vec<T, Dim> floor_voxel(const Vec<T, Dim> &x) {
    return ::curad::floor(x - T{0.5});
}

template <class T, std::int64_t Dim>
__host__ __device__ Vec<T, Dim> ceil_voxel(const Vec<T, Dim> &x) {
    return ::curad::ceil(x - T{0.5});
}

template <class T, std::int64_t Dim>
struct weight_result {
    Vec<T, Dim> weight;
    Vec<T, Dim> complement_weight;
};

template <class T, std::int64_t Dim>
__host__ __device__ weight_result<T, Dim> interpol_weights(const Vec<T, Dim> &cur_pos,
                                                           const Vec<T, Dim> &floored_pos,
                                                           const std::int64_t driving_dir) {
    auto complement_weight = (cur_pos - floored_pos) - T{0.5};
    auto weight = T{1} - complement_weight;

    complement_weight[driving_dir] = T{1};
    weight[driving_dir] = T{1};
    return {weight, complement_weight};
}

template <class T, std::int64_t Dim>
__host__ __device__ Vec<T, Dim> clip(const Vec<T, Dim> &x, const box<T, Dim> &aabb) {
    return min(aabb.max(), max(aabb.min(), x));
}

template <std::int64_t Dim>
__host__ __device__ bool all(const Vec<bool, Dim> &x) {
    bool result = true;

#pragma unroll
    for (std::size_t i = 0; i < Dim; ++i) {
        result &= x[i];
    }
    return result;
}

template <class T, std::int64_t Dim>
__host__ __device__ bool is_in_aabb(const Vec<T, Dim> &x, const box<T, Dim> &aabb) {
    return all(x >= aabb.min() && x < aabb.max());
}

template <class T, std::int64_t Dim, class U>
__host__ __device__ box<T, Dim> expand(const box<T, Dim> &aabb, std::size_t driving_dir,
                                       const U &delta) {
    auto b = aabb;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        if (i != driving_dir) {
            b.min()[i] -= delta;
            b.max()[i] += delta;
        }
    }
    return b;
}

template <class T, std::int64_t Dim, class Fn>
__host__ __device__ void compute_ray_coeffs(const box<T, Dim> &aabb, const ray<T, Dim> &r, Fn fn) {
    const auto driving_dir = argmax(abs(r.direction()));

    // for intersection expand in non-driving direction to ensure rays close to aabb still are
    // considered
    auto b = expand(aabb, driving_dir, T{0.5});
    auto hit = intersect(b, r);

    if (hit) {
        // TODO clamp to aabb_min and aabb_max
        const auto entry = r.origin() + hit.tmin * r.direction();
        const auto exit = r.origin() + hit.tmax * r.direction();

        const auto step = r.direction() / math::abs(r.direction()[driving_dir]);

        const auto intersection_length = norm(step);

        Vec<T, Dim> cur_pos = entry;

        const auto dist = cur_pos[driving_dir] - math::floor(cur_pos[driving_dir]);
        cur_pos += step * (0.5f - dist);

        cur_pos[driving_dir] = math::floor(cur_pos[driving_dir]) + 0.5f;

        const auto nsteps = static_cast<std::size_t>(
            math::ceil(math::abs(exit[driving_dir] - cur_pos[driving_dir])));

        for (int i = 0; i < nsteps; ++i) {
            const auto floored_voxel = floor_voxel(cur_pos);
            const auto ceiled_voxel = ceil_voxel(cur_pos);

            auto weights = interpol_weights(cur_pos, floored_voxel, driving_dir);
            // printf("weights: %f %f\n", weights.weight[0], weights.weight[1]);
            // printf("complement_weights: %f %f\n", weights.complement_weight[0],
            //        weights.complement_weight[1]);

            if constexpr (Dim == 2) {
                auto interpol = [&aabb, &fn](auto v1, auto v2, auto w1, auto w2) {
                    auto coord = Vec<T, Dim>{v1, v2};

                    // TODO: check if [v1, v2] is in aabb
                    auto in_aabb = is_in_aabb(coord, aabb);

                    // Ensure that weight is zero, if coord is outside of AABB, and we don't
                    // have branches
                    auto w = w1 * w2;

                    // pass coordinate and weight to outside function
                    fn(in_aabb, coord.template as<std::int64_t>(), w);
                };

                interpol(floored_voxel[0], ceiled_voxel[1], weights.weight[0],
                         weights.complement_weight[1]);
                interpol(ceiled_voxel[0], floored_voxel[1], weights.complement_weight[0],
                         weights.weight[1]);

            } else if constexpr (Dim == 3) {
            }

            // advance iterations
            cur_pos = cur_pos + step;
        }
    }
}

} // namespace joseph
} // namespace curad
