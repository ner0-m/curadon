#include "curadon/bmp.hpp"
#include "curadon/fan.hpp"
#include "curadon/forward.hpp"
#include "curadon/geometry/ray.hpp"
#include "curadon/parallel.hpp"

#include "doctest/doctest.h"

#include <cmath>
#include <iomanip>
#include <type_traits>

namespace doctest {
template <typename T>
struct StringMaker<::curad::projection_view<T, 1>> {
    static String convert(const ::curad::projection_view<T, 1> &in) {
        std::ostringstream oss;
        oss.precision(3);

        oss << "proj [angles <" << in.nangles() << ">, det size <" << in.shape()[0] << ">]\n";
        for (int i = 0; i < in.nangles(); ++i) {
            oss << "[";
            for (int j = 0; j < in.shape()[0] - 1; ++j) {
                oss << std::setw(5) << in(i, j) << " ";
            }
            oss << std::setw(5) << in(i, in.shape()[0] - 1) << "]\n";
        }

        return oss.str().c_str();
    }
};
} // namespace doctest

TEST_CASE("Forward: single projection") {
    // curad::box<float, 2> aabb = {{-6, -6}, {6, 6}};
    curad::box<float, 2> aabb = {{-8, -8}, {8, 8}};

    std::vector<float> voldata(
        {{0.0, 0.0,       0.0,       0.0,       0.0, 0.0,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.0,       0.0, 0.0,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.0,       0.0, 0.0,       0.0,       0.2,
          0.2, 0.2,       0.0,       0.0,       0.0, 0.0,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.0,       0.0, 0.2,       0.2,       0.2,
          0.2, 0.2,       0.2,       0.2,       0.0, 0.0,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.0,       0.0, 0.2,       0.2,       0.3,
          0.3, 0.3,       0.2,       0.2,       0.0, 0.0,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.0,       0.2, -1.49e-08, 0.3,       0.3,
          0.3, 0.3,       0.3,       0.2,       0.2, 0.0,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.0,       0.2, -1.49e-08, 0.1,       0.3,
          0.3, 0.3,       0.3,       0.2,       0.2, 0.0,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.2,       0.2, -1.49e-08, 0.1,       0.3,
          0.3, 0.3,       0.1,       0.2,       0.2, 0.2,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.2,       0.2, -1.49e-08, -1.49e-08, 0.3,
          0.3, 0.1,       -1.49e-08, -1.49e-08, 0.2, 0.2,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.2,       0.2, -1.49e-08, -1.49e-08, -1.49e-08,
          0.2, -1.49e-08, -1.49e-08, -1.49e-08, 0.2, 0.2,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.2,       0.2, 0.2,       -1.49e-08, -1.49e-08,
          0.2, -1.49e-08, -1.49e-08, -1.49e-08, 0.2, 0.2,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.2,       0.2, 0.2,       -1.49e-08, -1.49e-08,
          0.2, 0.2,       -1.49e-08, 0.2,       0.2, 0.2,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.0,       0.2, 0.2,       -1.49e-08, -1.49e-08,
          0.2, 0.2,       0.2,       0.2,       0.2, 0.0,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.0,       0.2, 0.2,       0.2,       -1.49e-08,
          0.2, 0.2,       0.2,       0.2,       0.2, 0.0,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.0,       0.0, 0.2,       0.2,       0.2,
          0.2, 0.2,       0.2,       0.2,       0.0, 0.0,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.0,       0.0, 0.2,       0.2,       0.2,
          0.2, 0.2,       0.2,       0.2,       0.0, 0.0,       0.0,       0.0,
          0.0, 0.0,       0.0,       0.0,       0.0, 0.0,       0.0,       0.2,
          0.2, 0.2,       0.0,       0.0,       0.0, 0.0,       0.0,       0.0}});

    // std::vector<float> voldata({{
    //     0.000, 0.000, 0.000, 0.000, 0.000,  0.000,  0.000,  0.000,  0.000,  0.000, 0.000, 0.000,
    //     0.000, 0.000, 0.000, 0.000, 0.000,  0.200,  0.200,  0.200,  0.000,  0.000, 0.000, 0.000,
    //     0.000, 0.000, 0.000, 0.000, 0.200,  0.200,  0.200,  0.200,  0.200,  0.000, 0.000, 0.000,
    //     0.000, 0.000, 0.000, 0.200, 0.200,  0.200,  0.300,  0.200,  0.200,  0.200, 0.000, 0.000,
    //     0.000, 0.000, 0.200, 0.200, -0.000, 0.100,  0.300,  0.100,  0.200,  0.200, 0.200, 0.000,
    //     0.000, 0.000, 0.200, 0.200, -0.000, -0.000, 0.100,  -0.000, -0.000, 0.200, 0.200, 0.000,
    //     0.000, 0.000, 0.200, 0.200, -0.000, -0.000, -0.200, -0.000, -0.000, 0.200, 0.200, 0.000,
    //     0.000, 0.000, 0.200, 0.200, 0.200,  -0.000, -0.200, -0.000, -0.000, 0.200, 0.200, 0.000,
    //     0.000, 0.000, 0.200, 0.200, 0.200,  -0.000, -0.000, -0.000, 0.200,  0.200, 0.200, 0.000,
    //     0.000, 0.000, 0.000, 0.200, 0.200,  0.200,  0.200,  0.200,  0.200,  0.200, 0.000, 0.000,
    //     0.000, 0.000, 0.000, 0.000, 0.200,  0.200,  0.200,  0.200,  0.200,  0.000, 0.000, 0.000,
    //     0.000, 0.000, 0.000, 0.000, 0.000,  0.200,  0.200,  0.200,  0.000,  0.000, 0.000, 0.000,
    // }});

    curad::device_span<float> vol_span(voldata.data(), voldata.size());
    // curad::volume<float, 2> vol{vol_span, {12, 12}};
    curad::volume<float, 2> vol{vol_span, {16, 16}};

    const auto num_angles = 100;
    const auto det_size = 20;
    std::vector<float> projdata(num_angles * det_size, 0);

    curad::device_span<float> proj_span(projdata.data(), projdata.size());
    curad::projection_view<float, 1> proj(proj_span, {det_size, num_angles});

    curad::equally_spaced_angles<float> space(0, M_PI, num_angles);
    // auto geometry =
    //     curad::parallel_2d_geometry<float, curad::equally_spaced_angles<float>>(space, det_size);
    auto geometry = curad::fan_beam_geometry<float, curad::equally_spaced_angles<float>>(
        space, 1000,50, det_size);

    curad::forward(aabb, vol, proj, geometry,
                   [&] __host__(auto angle_idx, auto range_idx, bool in_aabb,
                                curad::Vec<std::int64_t, 2> volcoord, float coeff) {
                       // proj(angle_idx, range_idx) = 1;
                       proj(angle_idx, range_idx) +=
                           static_cast<int>(in_aabb) * vol(volcoord) * coeff;
                   });

    // CAPTURE(proj);
    curad::write("test.pgm", projdata.data(), projdata.size(), det_size, num_angles);
    // CHECK(false);
}

TEST_CASE("read_data") {
    auto [data, width, height] = curad::read("/home/david/src/work/curadon/data/phantom_0256.pgm");

    std::vector<float> voldata(data.size());
    for (int i = 0; i < voldata.size(); i++) {
        voldata[i] = data[i];
    }

    curad::device_span<float> vol_span(voldata.data(), voldata.size());
    curad::volume<float, 2> vol{vol_span, {width, height}};
    auto aabb = vol.aabb();

    const auto num_angles = 360;
    const auto det_size = static_cast<std::int64_t>(std::sqrt(2) * std::max(width, height));
    std::vector<float> projdata(num_angles * det_size, 0);

    curad::device_span<float> proj_span(projdata.data(), projdata.size());
    curad::projection_view<float, 1> proj(proj_span, {det_size, num_angles});

    curad::equally_spaced_angles<float> space(0, M_PI, num_angles);
    // auto geometry =
    //     curad::parallel_2d_geometry<float, curad::equally_spaced_angles<float>>(space, det_size);
    auto geometry = curad::fan_beam_geometry<float, curad::equally_spaced_angles<float>>(
        space, 10 * width, 2 * width, det_size);

    curad::forward(aabb, vol, proj, geometry,
                   [&] __host__(auto angle_idx, auto range_idx, bool in_aabb,
                                curad::Vec<std::int64_t, 2> volcoord, float coeff) {
                       proj(angle_idx, range_idx) +=
                           static_cast<int>(in_aabb) * vol(volcoord) * coeff;
                   });

    curad::write("test.pgm", projdata.data(), projdata.size(), det_size, num_angles);
}
