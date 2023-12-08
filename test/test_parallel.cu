#include "curadon/cuda/device_span.hpp"
#include "doctest/doctest.h"

#include "curadon/parallel.hpp"

TEST_CASE("Test rotate") {
    auto v = curad::Vec<float, 2>{1, 0};

    auto rot1 = curad::detail::rotate(0.f, v);
    CHECK_EQ(rot1[0], doctest::Approx(1));
    CHECK_EQ(rot1[1], doctest::Approx(0));

    auto rot2 = curad::detail::rotate(curad::degree{90.f}.to_radian().value(), v);
    CHECK_EQ(rot2[0], doctest::Approx(0));
    CHECK_EQ(rot2[1], doctest::Approx(1));

    auto rot3 = curad::detail::rotate(curad::degree{180.f}.to_radian().value(), v);
    CHECK_EQ(rot3[0], doctest::Approx(-1));
    CHECK_EQ(rot3[1], doctest::Approx(0));

    auto rot4 = curad::detail::rotate(curad::degree{270.f}.to_radian().value(), v);
    CHECK_EQ(rot4[0], doctest::Approx(0));
    CHECK_EQ(rot4[1], doctest::Approx(-1));

    auto rot5 = curad::detail::rotate(curad::degree{360.f}.to_radian().value(), v);
    CHECK_EQ(rot5[0], doctest::Approx(1));
    CHECK_EQ(rot5[1], doctest::Approx(0));
}

TEST_CASE("Test flat_panel_det_1D with 3 pixels") {
    auto detector = curad::flat_panel_det_1D(curad::Vec<float, 2>{0, 1}, 3);

    CHECK_EQ(detector.axis().x(), 0);
    CHECK_EQ(detector.axis().y(), 1);

    CHECK_EQ(detector.surface_normal(0).x(), -1);
    CHECK_EQ(detector.surface_normal(0).y(), 0);
    CHECK_EQ(detector.surface_normal(1).x(), -1);
    CHECK_EQ(detector.surface_normal(1).y(), 0);
    CHECK_EQ(detector.surface_normal(-1).x(), -1);
    CHECK_EQ(detector.surface_normal(-1).y(), 0);

    auto p1 = detector.surface(0);
    auto p2 = detector.surface(1);
    auto p3 = detector.surface(2);
    CAPTURE(p1.x());
    CAPTURE(p1.y());

    CAPTURE(p2.x());
    CAPTURE(p2.y());

    CAPTURE(p3.x());
    CAPTURE(p3.y());

    CHECK_EQ(p1.x(), doctest::Approx(0));
    CHECK_EQ(p1.y(), doctest::Approx(-1));

    CHECK_EQ(p2.x(), doctest::Approx(0));
    CHECK_EQ(p2.y(), doctest::Approx(0));

    CHECK_EQ(p3.x(), doctest::Approx(0));
    CHECK_EQ(p3.y(), doctest::Approx(1));
}

TEST_CASE("Test det_to_source 0 degree") {
    auto pose = curad::parallel_pose_2d(curad::degree(0.f).to_radian(), 3);

    auto ref_point1 = pose.det_ref_point();
    CAPTURE(ref_point1.x());
    CAPTURE(ref_point1.y());
    CHECK_EQ(ref_point1.x(), doctest::Approx(1));
    CHECK_EQ(ref_point1.y(), doctest::Approx(0));

    auto dir1 = pose.det_to_source(-1);
    CAPTURE(dir1.y());
    CAPTURE(dir1.x());

    CHECK_EQ(dir1.x(), doctest::Approx(-1));
    CHECK_EQ(dir1.y(), doctest::Approx(0));
}

TEST_CASE("Test det_to_source 90 degree") {
    auto pose = curad::parallel_pose_2d(curad::degree(90.f).to_radian(), 3);

    auto ref_point1 = pose.det_ref_point();
    CAPTURE(ref_point1.x());
    CAPTURE(ref_point1.y());
    CHECK_EQ(ref_point1.x(), doctest::Approx(0));
    CHECK_EQ(ref_point1.y(), doctest::Approx(1));

    auto dir1 = pose.det_to_source(-1);
    CAPTURE(dir1.y());
    CAPTURE(dir1.x());

    CHECK_EQ(dir1.x(), doctest::Approx(0));
    CHECK_EQ(dir1.y(), doctest::Approx(-1));
}

TEST_CASE("Test det_to_source 180 degree") {
    auto pose = curad::parallel_pose_2d(curad::degree(180.f).to_radian(), 3);

    auto ref_point1 = pose.det_ref_point();
    CAPTURE(ref_point1.x());
    CAPTURE(ref_point1.y());
    CHECK_EQ(ref_point1.x(), doctest::Approx(-1));
    CHECK_EQ(ref_point1.y(), doctest::Approx(0));

    auto dir1 = pose.det_to_source(-1);
    CAPTURE(dir1.y());
    CAPTURE(dir1.x());

    CHECK_EQ(dir1.x(), doctest::Approx(1));
    CHECK_EQ(dir1.y(), doctest::Approx(0));
}

TEST_CASE("Test det_to_source 270 degree") {
    auto pose = curad::parallel_pose_2d(curad::degree(270.f).to_radian(), 3);

    auto ref_point1 = pose.det_ref_point();
    CAPTURE(ref_point1.x());
    CAPTURE(ref_point1.y());
    CHECK_EQ(ref_point1.x(), doctest::Approx(0));
    CHECK_EQ(ref_point1.y(), doctest::Approx(-1));

    auto dir1 = pose.det_to_source(-1);
    CAPTURE(dir1.y());
    CAPTURE(dir1.x());

    CHECK_EQ(dir1.x(), doctest::Approx(0));
    CHECK_EQ(dir1.y(), doctest::Approx(1));
}

TEST_CASE("test parallel_2d_geometry") {
    curad::equally_spaced_angles<float> space(0, M_PI, 4);
    auto geometry = curad::parallel_2d_geometry<float, curad::equally_spaced_angles<float>>(space, 3);

    auto ray = geometry(0, 1);
    CHECK_EQ(ray.direction().x(), doctest::Approx(1));
    CHECK_EQ(ray.direction().y(), doctest::Approx(0));

    ray = geometry(1, 1);
    CHECK_EQ(ray.direction().x(), doctest::Approx(0));
    CHECK_EQ(ray.direction().y(), doctest::Approx(1));

    ray = geometry(2, 1);
    CHECK_EQ(ray.direction().x(), doctest::Approx(-1));
    CHECK_EQ(ray.direction().y(), doctest::Approx(0));

    ray = geometry(3, 1);
    CHECK_EQ(ray.direction().x(), doctest::Approx(0));
    CHECK_EQ(ray.direction().y(), doctest::Approx(-1));
}
