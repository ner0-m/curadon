#pragma once

#include "curadon/detail/vec.hpp"

namespace curad::geometry {
template <class T>
__host__ __device__ vec<T, 3> rotate_yzy(const vec<T, 3> &v, T phi, T theta, T psi) {
    const auto cphi = std::cos(phi);
    const auto sphi = std::sin(phi);
    const auto ctheta = std::cos(theta);
    const auto stheta = std::sin(theta);
    const auto spsi = std::sin(psi);
    const auto cpsi = std::cos(psi);

    vec<T, 3> res = v;

    // clang-format off
    // This is the combined rotation matrix
    // [ cos(phi)*cos(psi)*cos(theta) - sin(phi)*sin(psi), -sin(theta)*cos(phi),  sin(phi)*cos(psi) + sin(psi)*cos(phi)*cos(theta)],
    // [                              sin(theta)*cos(psi),           cos(theta),                               sin(psi)*sin(theta)],
    // [-sin(phi)*cos(psi)*cos(theta) - sin(psi)*cos(phi),  sin(phi)*sin(theta), -sin(phi)*sin(psi)*cos(theta) + cos(phi)*cos(psi)]]
    res[0] = v[0] * (cphi * cpsi * ctheta - sphi * spsi) - v[1] * stheta * cphi + v[2] * (sphi * cpsi + spsi * cphi * ctheta);
    res[1] = v[0] * (stheta * cpsi) + v[1] * ctheta + v[2] * spsi * stheta;
    res[2] = v[0] * (-sphi * cpsi * ctheta - spsi * cphi) + v[1] * sphi * stheta + v[2] * (-sphi * spsi * ctheta + cpsi * cphi);
    // clang-format on

    return res;
}

template <class T>
__host__ __device__ vec<T, 3> rotate_roll_pitch_yaw(const vec<T, 3> &v, T roll, T pitch, T yaw) {
    const auto croll = std::cos(roll);
    const auto sroll = std::sin(roll);
    const auto cpitch = std::cos(pitch);
    const auto spitch = std::sin(pitch);
    const auto syaw = std::sin(yaw);
    const auto cyaw = std::cos(yaw);

    vec<T, 3> res = v;

    res[0] = croll * cpitch * v[0] + (croll * spitch * syaw - sroll * cyaw) * v[1] +
             (croll * spitch * cyaw + sroll * syaw) * v[2];
    res[1] = sroll * cpitch * v[0] + (sroll * spitch * syaw + croll * cyaw) * v[1] +
             (sroll * spitch * cyaw - croll * syaw) * v[2];
    res[2] = -spitch * v[0] + cpitch * syaw * v[1] + cpitch * cyaw * v[2];

    return res;
}

/// Rotate `v` clockwise around `t` by phi radian
__inline__ __host__ __device__ vec2f rotate(const vec2f &v, f32 phi, const vec2f &t) {
#ifdef __CUDA_ARCH__
    const auto cs = __cosf(phi);
    const auto sn = __sinf(phi);
#else
    const auto cs = std::cos(phi);
    const auto sn = std::sin(phi);
#endif

    return vec2f{v.x() * cs + v.y() * sn - t.x(), -v.x() * sn + v.y() * cs - t.y()};
}

__inline__ __host__ __device__ vec2f rotate(const vec2f &v, f32 phi) {
    return rotate(v, phi, vec2f{0.0, 0.0});
}
} // namespace curad::geometry
