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

#include "read.hpp"
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

    const auto det_shape = curad::vec<std::uint64_t, 2>{det_width, det_height};

    const auto vol_shape = curad::vec<std::uint64_t, 3>{width, height, depth};

    const auto vol_spacing = curad::vec<float, 3>{1, 1, 1};
    const auto vol_size = vol_shape * vol_spacing;

    const auto vol_offset = curad::vec<float, 3>{0, 0, 0};

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
    auto [data, width, height, ign1, ign2, ign3, ign4] = curad::easy::read("phantom_2d.txt");

    thrust::host_vector<float> host_volume(width * height, 0);

    std::copy(data.begin(), data.end(), host_volume.begin());

    thrust::device_vector<float> volume = host_volume;
    auto volume_ptr = thrust::raw_pointer_cast(volume.data());

    // compute angles
    const std::size_t nangles = 360;
    const float step = 2.f * M_PI / nangles;
    std::vector<float> angles(nangles);
    thrust::sequence(angles.begin(), angles.end(), 0.f, step);

    // const std::size_t det_width = static_cast<std::size_t>(std::ceil(width * std::sqrt(2)));
    const std::size_t det_width = static_cast<int>(width * std::sqrt(2));
    thrust::device_vector<float> sinogram(det_width * nangles, 0);
    auto sinogram_ptr = thrust::raw_pointer_cast(sinogram.data());

    const auto det_shape = det_width;
    const float det_spacing = 1;

    const auto vol_shape = curad::vec<std::uint64_t, 2>{width, height};

    const auto vol_spacing = curad::vec<float, 2>{1, 1};
    const auto vol_size = vol_shape * vol_spacing;

    const auto vol_offset = curad::vec<float, 2>{0, 0};

    const float DSD = 1280;
    const float DSO = 1408;

    std::cout << "vol_shape: " << vol_shape[0] << ", " << vol_shape[1] << "\n";
    std::cout << "vol_extent: " << vol_size[0] << ", " << vol_size[1] << "\n";
    std::cout << "vol_spacing: " << vol_spacing[0] << ", " << vol_spacing[1] << "\n";
    std::cout << "vol_offset: " << vol_offset[0] << ", " << vol_offset[1] << "\n";
    std::cout << "sino_shape: " << angles.size() << "\n";
    std::cout << "det_shape: " << det_shape << "\n";
    std::cout << "det_spacing: " << det_spacing << "\n";
    std::cout << "DSO: " << DSO << "\n";
    std::cout << "DSD: " << DSD << "\n";

    curad::image_2d vol_span(volume_ptr, vol_shape, vol_spacing, vol_offset);
    curad::measurement_2d sino_span(sinogram_ptr, det_shape, angles.size(), det_spacing);
    sino_span.set_angles(angles);
    sino_span.set_distance_source_to_object(DSO);
    sino_span.set_distance_source_to_detector(DSD);

    curad::fp::forward_2d(vol_span, sino_span);

    thrust::host_vector<float> host_sino = sinogram;

    const auto max = *std::max_element(host_sino.begin(), host_sino.end());
    std::transform(host_sino.begin(), host_sino.end(), host_sino.begin(),
                   [&](auto x) { return x / max; });
    // draw(data.data(), 0, width, height, depth);
    draw(host_sino.data(), 0, det_width, nangles, 1);
}
