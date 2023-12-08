#include "doctest/doctest.h"

#include <cmath>
#include <iomanip>
#include <type_traits>

#include "curadon/fan.hpp"

TEST_CASE("fan_beam") {
    curad::equally_spaced_angles<float> space(0, 2 * M_PI, 4);

    // auto source_center = 1.f;
    auto center_det = 3.f;
    auto geometry = curad::fan_beam_geometry<float, curad::equally_spaced_angles<float>>(
        space, 1.f, center_det, 3);

    auto ref_point = geometry.det_ref_point(0);
    CHECK_EQ(ref_point[0], doctest::Approx(center_det));
    CHECK_EQ(ref_point[1], doctest::Approx(0));

    ref_point = geometry.det_ref_point(1);
    CHECK_EQ(ref_point[0], doctest::Approx(0));
    CHECK_EQ(ref_point[1], doctest::Approx(center_det));

    ref_point = geometry.det_ref_point(2);
    CHECK_EQ(ref_point[0], doctest::Approx(-center_det));
    CHECK_EQ(ref_point[1], doctest::Approx(0));

    ref_point = geometry.det_ref_point(3);
    CHECK_EQ(ref_point[0], doctest::Approx(0));
    CHECK_EQ(ref_point[1], doctest::Approx(-center_det));

    auto pose = 1;
    auto r = geometry(pose, 0);
    printf("r.dir = (%f, %f)\n", r.direction()[0], r.direction()[1]);
    // CHECK_EQ(r.origin()[0], -1);
    // CHECK_EQ(r.origin()[1], 0);

    r = geometry(pose, 1);
    printf("r.dir = (%f, %f)\n", r.direction()[0], r.direction()[1]);
    // CHECK_EQ(r.origin()[0], -1);
    // CHECK_EQ(r.origin()[1], 0);

    r = geometry(pose, 2);
    printf("r.dir = (%f, %f)\n", r.direction()[0], r.direction()[1]);
    CHECK_EQ(r.origin()[0], -1);
    CHECK_EQ(r.origin()[1], 0);

    // CHECK_EQ(r.direction()[0], 1);
    // CHECK_EQ(r.direction()[1], 0);
    //
    // r = geometry(1, 0);
    // CHECK_EQ(r.origin()[0], 0);
    // CHECK_EQ(r.origin()[1], -1);
    //
    // CHECK_EQ(r.direction()[0], 0);
    // CHECK_EQ(r.direction()[1], 1);

    // r = geometry(2, 0);
    // CHECK_EQ(r.origin()[0], 1);
    // CHECK_EQ(r.origin()[1], 0);
    //
    // CHECK_EQ(r.direction()[0], -1);
    // CHECK_EQ(r.direction()[1], 0);

    // r = geometry(3, 0);
    // CHECK_EQ(r.origin()[0], -1);
    // CHECK_EQ(r.origin()[1], 0);
    //
    // CHECK_EQ(r.direction()[0], 1);
    // CHECK_EQ(r.direction()[1], 0);
}
