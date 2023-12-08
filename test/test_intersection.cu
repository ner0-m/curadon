#include "curadon/geometry/intersection.hpp"
#include "curadon/geometry/ray.hpp"
#include "doctest/doctest.h"
#include <cstdint>

TEST_CASE("Check rays through corners are considered as intersecting") {
    using Vec2i = curad::Vec<std::int64_t, 2>;
    using Vec2f = curad::Vec<float, 2>;
    using ray2f = curad::ray<float, 2>;

    const curad::box<float, 2> aabb{{0, 0}, {10, 10}};

    const ray2f r{Vec2f{5, 5}, Vec2f{1, -1}};
    auto hit = curad::intersect(aabb, r);

    CHECK_UNARY(hit);
    CHECK_EQ(hit.tmin, doctest::Approx(1));
    CHECK_EQ(hit.tmax, doctest::Approx(1));
}
TEST_CASE("Check rays through corners are considered as intersecting") {
    using Vec2i = curad::Vec<std::int64_t, 2>;
    using Vec2f = curad::Vec<float, 2>;
    using ray2f = curad::ray<float, 2>;

    const curad::box<float, 2> aabb{{0, 0}, {10, 10}};

    SUBCASE("Ray through lower left corner 1") {
        const ray2f r{Vec2f{-1, 1}, Vec2f{1, -1}};
        auto hit = curad::intersect(aabb, r);

        CHECK_UNARY(hit);
        CHECK_EQ(hit.tmin, doctest::Approx(1));
        CHECK_EQ(hit.tmax, doctest::Approx(1));
    }

    SUBCASE("Ray through lower left corner 2") {
        const ray2f r{Vec2f{1, -1}, Vec2f{-1, 1}};
        auto hit = curad::intersect(aabb, r);

        CHECK_UNARY(hit);
        CHECK_EQ(hit.tmin, doctest::Approx(1));
        CHECK_EQ(hit.tmax, doctest::Approx(1));
    }

    SUBCASE("Ray through lower right corner 1") {
        const ray2f r{Vec2f{11, 1}, Vec2f{-1, -1}};
        auto hit = curad::intersect(aabb, r);

        CHECK_UNARY(hit);
        CHECK_EQ(hit.tmin, doctest::Approx(1));
        CHECK_EQ(hit.tmax, doctest::Approx(1));
    }
    SUBCASE("Ray through lower right corner 1") {
        const ray2f r{Vec2f{9, -1}, Vec2f{1, 1}};
        auto hit = curad::intersect(aabb, r);

        CHECK_UNARY(hit);
        CHECK_EQ(hit.tmin, doctest::Approx(1));
        CHECK_EQ(hit.tmax, doctest::Approx(1));
    }

    SUBCASE("Ray through upper left corner 1") {
        const ray2f r{Vec2f{-1, 9}, Vec2f{1, 1}};
        auto hit = curad::intersect(aabb, r);

        CHECK_UNARY(hit);
        CHECK_EQ(hit.tmin, doctest::Approx(1));
        CHECK_EQ(hit.tmax, doctest::Approx(1));
    }

    SUBCASE("Ray through upper left corner 2") {
        const ray2f r{Vec2f{1, 11}, Vec2f{-1, -1}};
        auto hit = curad::intersect(aabb, r);

        CHECK_UNARY(hit);
        CHECK_EQ(hit.tmin, doctest::Approx(1));
        CHECK_EQ(hit.tmax, doctest::Approx(1));
    }

    SUBCASE("Ray through upper right corner 1") {
        const ray2f r{Vec2f{9, 11}, Vec2f{1, -1}};
        auto hit = curad::intersect(aabb, r);

        CHECK_UNARY(hit);
        CHECK_EQ(hit.tmin, doctest::Approx(1));
        CHECK_EQ(hit.tmax, doctest::Approx(1));
    }

    SUBCASE("Ray through upper right corner 2") {
        const ray2f r{Vec2f{11, 9}, Vec2f{-1, 1}};
        auto hit = curad::intersect(aabb, r);

        CHECK_UNARY(hit);
        CHECK_EQ(hit.tmin, doctest::Approx(1));
        CHECK_EQ(hit.tmax, doctest::Approx(1));
    }
}

// TEST_CASE("Check curad::rays through edges are considered as intersecting") {
//     using Vec2i = curad::Vec<std::int64_t, 2>;
//     using Vec2f = curad::Vec<float, 2>;
//
//     const Vec2i aabb_min{0, 0};
//     const Vec2i aabb_max{10, 10};
//
//     SUBCASE("Ray lower edge 1") {
//         const Vec2f ray_origin{-1, 0};
//         Vec2f ray_dir{1, 0};
//         ray_dir.normalize();
//
//         auto hit = curad::intersect(aabb_min, aabb_max, ray_origin, ray_dir);
//
//         CHECK_UNARY(hit);
//         CHECK_EQ(hit.tmin, doctest::Approx(1));
//         CHECK_EQ(hit.tmax, doctest::Approx(11));
//     }
//
//     SUBCASE("Ray lower edge 2") {
//         const Vec2f ray_origin{11, 0};
//         Vec2f ray_dir{-1, 0};
//         ray_dir.normalize();
//
//         auto hit = curad::intersect(aabb_min, aabb_max, ray_origin, ray_dir);
//
//         CHECK_UNARY(hit);
//         CHECK_EQ(hit.tmin, doctest::Approx(1));
//         CHECK_EQ(hit.tmax, doctest::Approx(11));
//     }
//
//     SUBCASE("Ray upper edge 1") {
//         const Vec2f ray_origin{-1, 10};
//         Vec2f ray_dir{1, 0};
//         ray_dir.normalize();
//
//         auto hit = curad::intersect(aabb_min, aabb_max, ray_origin, ray_dir);
//
//         CHECK_UNARY(hit);
//         CHECK_EQ(hit.tmin, doctest::Approx(1));
//         CHECK_EQ(hit.tmax, doctest::Approx(11));
//     }
//
//     SUBCASE("Ray upper edge 1") {
//         const Vec2f ray_origin{11, 10};
//         Vec2f ray_dir{-1, 0};
//         ray_dir.normalize();
//
//         auto hit = curad::intersect(aabb_min, aabb_max, ray_origin, ray_dir);
//
//         CHECK_UNARY(hit);
//         CHECK_EQ(hit.tmin, doctest::Approx(1));
//         CHECK_EQ(hit.tmax, doctest::Approx(11));
//     }
// }
