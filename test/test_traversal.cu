#include "curadon/geometry/box.hpp"
#include "curadon/geometry/ray.hpp"
#include "curadon/traversal.hpp"
#include "doctest/doctest.h"

#include <iomanip>

TEST_CASE("Joseph: basic traversal") {
    curad::box<float, 2> aabb = {{0, 0}, {10, 10}};
    curad::ray<float, 2> r = {{0.2, -1}, {0, 1}};

    auto visited_counter = 0;
    auto in_aabb_counter = 0;
    curad::joseph::compute_ray_coeffs(aabb, r, [&] __host__(auto in_aabb, auto coord, auto coeff) {
        ++visited_counter;
        if (in_aabb) {
            ++in_aabb_counter;
        }
    });

    CHECK_EQ(visited_counter, 20);
    CHECK_EQ(in_aabb_counter, 10);
}

// TEST_CASE("Joseph: basic traversal") {
//     curad::box<2> aabb = {{0, 0}, {10, 10}};
//
//     for (int i = 0; i < 10; ++i) {
//         curad::ray<float, 2> r = {{-1, i + .5f}, {1, 0}};
//
//         auto counter = 0;
//         curad::joseph::compute_ray_coeffs(aabb, r,
//                                           [&] __host__(auto idx, auto coeff) { ++counter; });
//
//         CHECK_EQ(counter, 10);
//     }
//
//     for (int i = 0; i < 10; ++i) {
//         curad::ray<float, 2> r = {{11, i + .5f}, {-1, 0}};
//
//         auto counter = 0;
//         curad::joseph::compute_ray_coeffs(aabb, r,
//                                           [&] __host__(auto idx, auto coeff) { ++counter; });
//
//         CHECK_EQ(counter, 10);
//     }
//
//     for (int i = 0; i < 10; ++i) {
//         curad::ray<float, 2> r = {{i + .5f, -1}, {0, 1}};
//
//         auto counter = 0;
//         curad::joseph::compute_ray_coeffs(aabb, r,
//                                           [&] __host__(auto idx, auto coeff) { ++counter; });
//
//         CHECK_EQ(counter, 10);
//     }
//
//     for (int i = 0; i < 10; ++i) {
//         curad::ray<float, 2> r = {{i + .5f, 11}, {0, -1}};
//
//         auto counter = 0;
//         curad::joseph::compute_ray_coeffs(aabb, r,
//                                           [&] __host__(auto idx, auto coeff) { ++counter; });
//
//         CHECK_EQ(counter, 10);
//     }
// }
