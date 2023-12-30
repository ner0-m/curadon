#pragma once

#include "curadon/math/vector.hpp"

namespace curad::geometry {
template <class T>
__host__ __device__ Vec<T, 3> rotate_yzy(const Vec<T, 3> &v, T phi, T theta, T psi) {
    const auto cos_phi = std::cos(phi);
    const auto sin_phi = std::sin(phi);
    const auto ctheta = std::cos(theta);
    const auto stheta = std::sin(theta);
    const auto spsi = std::sin(psi);
    const auto cpsi = std::cos(psi);

    Vec<T, 3> res = v;

    // Rotate around  y
    res[0] = v[0] * cos_phi + v[2] * sin_phi;
    res[2] = -v[0] * sin_phi + v[2] * cos_phi;

    // Rotate around z
    res[0] = res[0] * ctheta - res[1] * stheta;
    res[1] = res[0] * stheta + res[1] * ctheta;

    // Rotate around y again
    res[0] = res[0] * cpsi + res[2] * spsi;
    res[2] = -res[0] * spsi + res[2] * cpsi;

    return res;
}

template <class T>
__host__ __device__ Vec<T, 2> rotate(const Vec<T, 2> &v, T phi) {
    const auto cos_phi = std::cos(phi);
    const auto sin_phi = std::sin(phi);

    return Vec<T, 2>{v[0] * cos_phi - v[1] * sin_phi, v[0] * sin_phi + v[1] * cos_phi};
}
} // namespace curad::geometry
