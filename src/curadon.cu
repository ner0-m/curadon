#include "curadon/cuda/stream.hpp"
#include "curadon/curadon.h"

#include <cstdio>
#include <iostream>
#include <type_traits>
#include <vector>

#include "curadon/cuda/device_span.hpp"
#include "curadon/cuda/device_uvector.hpp"
#include "curadon/cuda/stream_view.hpp"
#include "curadon/geometry/intersection.hpp"
#include "curadon/math/vector.hpp"

namespace curad {


// __global__ void cuda_hello(device_span<float> vol, Vec<std::int64_t, 2> aabb_min,
//                            Vec<std::int64_t, 2> aabb_max, Vec<float, 2> ro, Vec<float, 2> rd) {
//     const auto driving_dir = argmax(rd);
//
//     box<2> aabb{aabb_min, aabb_max};
//     ray<float, 2> r{ro, rd};
//
//     joseph::compute_ray_coeffs(aabb, r, [&](std::uint64_t idx, float intersection_length) {
//         vol[idx] = intersection_length;
//     });
// }

std::vector<float> test_kernel() {
    // using Vec2 = Vec<float, 2>;
    // Vec<std::int64_t, 2> aabb_min{0, 0};
    // Vec<std::int64_t, 2> aabb_max{5, 5};
    //
    // Vec2 ro{2.5, 7.5};
    // Vec2 rd{0 - 1};
    // rd.normalize();
    //
    // curad::device_uvector<float> vol(5 * 5, curad::default_stream);
    //
    // cuda_hello<<<1, 1>>>(vol, aabb_min, aabb_max, ro, rd);
    // cudaDeviceSynchronize();

    // std::vector<float> result(vol.size());
    // cudaMemcpy(result.data(), vol.data(), vol.size() * sizeof(float), cudaMemcpyDefault);
    std::vector<float> result;
    return result;
}

// void forward(view volume, view sinogram) { cuda_hello<<<1, 1>>>(); }
//
// void backward(view sinogram, view volume) {}
} // namespace curad
