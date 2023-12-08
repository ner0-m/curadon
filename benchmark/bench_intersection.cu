#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <cstdio>
#include <iostream>
#include <type_traits>
#include <vector>

#include "curadon/cuda/device_span.hpp"
#include "curadon/cuda/device_uvector.hpp"
#include "curadon/cuda/stream.hpp"
#include "curadon/cuda/stream_view.hpp"
#include "curadon/curadon.h"
#include "curadon/math/math.hpp"
#include "curadon/math/vector.hpp"
#include "curadon/geometry/intersection.hpp"

namespace curad {

__global__ void cuda_hello2(Vec<std::int64_t, 2> aabb_min, Vec<std::int64_t, 2> aabb_max,
                            device_span<float> tmins) {
    const auto size = tmins.size();

    using Vec2f = Vec<float, 2>;
    Vec2f ro{-2.5, 2.5};
    Vec2f rd{1, 0};
    rd.normalize();

    const auto i = threadIdx.x + blockIdx.x * blockDim.x;
    float tmin = 100000000;

    if (i < size) {
        const auto driving_dir = argmax(rd);

        auto hit = intersect2<float>(aabb_min, aabb_max, ro, rd);

        if (hit && hit.tmin < tmin) {
            tmin = hit.tmin;
        }
    }
    tmins[i] = tmin;
}

__global__ void cuda_hello(Vec<std::int64_t, 2> aabb_min, Vec<std::int64_t, 2> aabb_max,
                           device_span<float> tmins) {
    const auto size = tmins.size();

    using Vec2f = Vec<float, 2>;
    Vec2f ro{-2.5, 2.5};
    Vec2f rd{1, 0};
    rd.normalize();

    const auto i = threadIdx.x + blockIdx.x * blockDim.x;
    float tmin = 100000000;

    if (i < size) {
        const auto driving_dir = argmax(rd);

        auto hit = intersect<float>(aabb_min, aabb_max, ro, rd);

        if (hit && hit.tmin < tmin) {
            tmin = hit.tmin;
        }
    }
    tmins[i] = tmin;
}
} // namespace curad

// TEST_CASE("Benchmark intersection GPU") {
//     using Vec2f = curad::Vec<float, 2>;
//     using Vec2i = curad::Vec<std::int64_t, 2>;
//     Vec2i aabb_min{0, 0};
//     Vec2i aabb_max{5, 5};
//
//     constexpr std::size_t size = 1 << 12;
//     curad::device_uvector<Vec2f> ros(size, curad::default_stream);
//     curad::device_uvector<Vec2f> rds(size, curad::default_stream);
//
//     for (int i = 0; i < size; ++i) {
//         ros.set_element(i, {-2.5, 2.5}, curad::default_stream);
//
//         Vec2f tmp = {1, 0};
//         tmp.normalize();
//         rds.set_element(i, tmp, curad::default_stream);
//     }
//
//     ankerl::nanobench::Bench b;
//     b.title("Random Number Generators")
//         .unit("rays")
//         .batch(size)
//         .relative(true)
//         .performanceCounters(true);
//
//     curad::device_uvector<float> tmins(1, curad::default_stream);
//
//     const auto block = 128;
//     const auto grid = (size + block - 1) / block;
//
//     b.run("Classical", [&] {
//         curad::cuda_hello<<<grid, block>>>(aabb_min, aabb_max, ros, rds, tmins);
//         cudaDeviceSynchronize();
//         ankerl::nanobench::doNotOptimizeAway(tmins);
//     });
//
//     tmins = curad::device_uvector<float>(1, curad::default_stream);
//
//     b.run("New", [&] {
//         curad::cuda_hello2<<<grid, block>>>(aabb_min, aabb_max, ros, rds, tmins);
//         cudaDeviceSynchronize();
//         ankerl::nanobench::doNotOptimizeAway(tmins);
//     });
// }

// TEST_CASE("Benchmark intersection CPU") {
//     using Vec2f = curad::Vec<float, 2>;
//     using Vec2i = curad::Vec<std::int64_t, 2>;
//     Vec2i aabb_min{0, 0};
//     Vec2i aabb_max{5, 5};
//
//     Vec2f ro{-2.5, 2.5};
//     Vec2f rd{1, 0};
//     rd.normalize();
//
//     ankerl::nanobench::Bench b;
//     b.title("Random Number Generators").unit("rays").relative(true).performanceCounters(true);
//
//     b.run("Classic CPU", [&] {
//         auto hit = intersect(aabb_min, aabb_max, ro, rd);
//         ankerl::nanobench::doNotOptimizeAway(hit);
//     });
//
//     b.run("New CPU", [&] {
//         auto hit = intersect2(aabb_min, aabb_max, ro, rd);
//         ankerl::nanobench::doNotOptimizeAway(hit);
//     });
// }

TEST_CASE("Single GPU") {
    using Vec2f = curad::Vec<float, 2>;
    using Vec2i = curad::Vec<std::int64_t, 2>;
    Vec2i aabb_min{0, 0};
    Vec2i aabb_max{5, 5};

    constexpr std::size_t size = 1 << 16;
    // curad::device_uvector<Vec2f> ros(size, curad::default_stream);
    // curad::device_uvector<Vec2f> rds(size, curad::default_stream);
    //
    // for (int i = 0; i < size; ++i) {
    //     ros.set_element(i, {-2.5, 2.5}, curad::default_stream);
    //
    //     Vec2f tmp = {1, 0};
    //     tmp.normalize();
    //     rds.set_element(i, tmp, curad::default_stream);
    // }

    const auto block = 256;
    const auto grid = (size + block - 1) / block;

    curad::device_uvector<float> tmins(1, curad::default_stream);
    curad::cuda_hello<<<grid, block>>>(aabb_min, aabb_max, tmins);
    cudaDeviceSynchronize();

    // tmins = curad::device_uvector<float>(size, curad::default_stream);
    curad::cuda_hello2<<<grid, block>>>(aabb_min, aabb_max, tmins);
    cudaDeviceSynchronize();

    ankerl::nanobench::doNotOptimizeAway(tmins);
}
