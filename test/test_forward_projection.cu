#include "curadon/bmp.hpp"
#include "curadon/forward.hpp"

#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "show.hpp"

TEST_CASE("forward_projection_3d") {
    auto [data, width, height, depth, ignore1, ignore2, ignore3] = curad::easy::read("phantom.txt");

    thrust::host_vector<float> host_volume(width * height * depth, 0);
    std::copy(data.begin(), data.end(), host_volume.begin());

    thrust::device_vector<float> volume = host_volume;
    auto volume_ptr = thrust::raw_pointer_cast(volume.data());

    // compute angles
    const std::size_t nangles = 300;
    std::vector<float> angles(nangles);
    const float step = 2.f * M_PI / nangles;
    thrust::sequence(angles.begin(), angles.end(), 0.f, step);

    const std::size_t det_width = static_cast<std::size_t>(std::ceil(width * std::sqrt(2)));
    const std::size_t det_height = static_cast<std::size_t>(std::ceil(height * std::sqrt(2)));
    thrust::device_vector<float> sinogram(det_width * det_height * nangles, 0);
    auto sinogram_ptr = thrust::raw_pointer_cast(sinogram.data());

    const auto det_shape = curad::Vec<std::uint64_t, 2>{det_width, det_height};

    const auto vol_shape = curad::Vec<std::uint64_t, 3>{width, height, depth};

    const auto vol_spacing = curad::Vec<float, 3>{1, 1, 1};
    const auto vol_size = vol_shape * vol_spacing;

    const auto vol_offset = curad::Vec<float, 3>{0, 0, 0};

    const float DSD = width * 10;
    const float DSO = DSD * 0.7;

    curad::device_volume<float> vol_span(volume_ptr, vol_shape, vol_spacing, vol_offset);
    curad::device_measurement<float> sino_span(sinogram_ptr, {det_width, det_height, nangles});
    sino_span.set_distance_source_to_detector(DSD).set_distance_source_to_object(DSO).set_angles(
        angles);

    curad::fp::forward_3d(vol_span, sino_span);

    thrust::host_vector<float> host_sino = sinogram;

    const auto max = *std::max_element(host_sino.begin(), host_sino.end());
    std::transform(host_sino.begin(), host_sino.end(), host_sino.begin(),
                   [&](auto x) { return x / max; });
    // draw(data.data(), 0, width, height, depth);
    draw(host_sino.data(), 0, det_width, det_height, nangles);
}

TEST_CASE("forward_projection_2d") {
    auto [data, width, height, depth, ignore1, ignore2, ignore3] =
        curad::easy::read("phantom_2d.txt");

    // const auto width = 64;
    // const auto height = 64;
    // const auto depth = 1;
    thrust::host_vector<float> host_volume(width * height * depth, 0);

    // const auto box_width = 50;
    // const auto x_offset = (width - box_width) / 2;
    // const auto y_offset = (height - box_width) / 2;
    // for (int i = x_offset; i < x_offset + box_width; ++i) {
    //     for (int j = y_offset; j < y_offset + box_width; ++j) {
    //         host_volume[i * width + j] = 1;
    //     }
    // }
    std::copy(data.begin(), data.end(), host_volume.begin());

    thrust::device_vector<float> volume = host_volume;
    auto volume_ptr = thrust::raw_pointer_cast(volume.data());

    // compute angles
    const std::size_t nangles = 8;
    const float step = 22.5;
    std::vector<float> angles(nangles);
    thrust::sequence(angles.begin(), angles.end(), 0.f, step);
    thrust::transform(angles.begin(), angles.end(), angles.begin(),
                      [](auto x) { return x * M_PI / 180; });

    // const std::size_t det_width = static_cast<std::size_t>(std::ceil(width * std::sqrt(2)));
    const std::size_t det_width = width;
    thrust::device_vector<float> sinogram(det_width * nangles, 0);
    auto sinogram_ptr = thrust::raw_pointer_cast(sinogram.data());

    const auto det_shape = det_width;
    const float det_spacing = 1;

    const auto vol_shape = curad::Vec<std::uint64_t, 2>{width, height};

    const auto vol_spacing = curad::Vec<float, 2>{1, 1};
    const auto vol_size = vol_shape * vol_spacing;

    const auto vol_offset = curad::Vec<float, 2>{0, 0};

    const float DSD = width * 100;
    const float DSO = DSD * 0.95;

    curad::fp::forward_2d(volume_ptr, vol_shape, vol_size, vol_spacing, vol_offset, sinogram_ptr,
                          det_shape, det_spacing, angles, DSD, DSO);

    thrust::host_vector<float> host_sino = sinogram;

    const auto max = *std::max_element(host_sino.begin(), host_sino.end());
    std::transform(host_sino.begin(), host_sino.end(), host_sino.begin(),
                   [&](auto x) { return x / max; });
    // draw(data.data(), 0, width, height, depth);
    draw(host_sino.data(), 0, det_width, nangles, 1);
}
