#include "doctest/doctest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "read.hpp"
#include "show.hpp"

#include "curadon/backward.hpp"
#include "curadon/measurement.hpp"
#include "curadon/types.hpp"

TEST_CASE("backward_3d") {
    const auto volsize = 64;
    auto [data, width, height, nangles, angles, DSO, DSD] = curad::easy::read("sino_3d.txt");

    std::transform(angles.begin(), angles.end(), angles.begin(),
                   [](float x) { return x * M_PI / 180.; });

    thrust::host_vector<float> host_sino(width * height * nangles, 0);
    std::copy(data.begin(), data.end(), host_sino.begin());

    thrust::device_vector<float> sino = host_sino;
    auto sino_ptr = thrust::raw_pointer_cast(sino.data());

    thrust::device_vector<float> volume(volsize * volsize * volsize, 0);
    auto volume_ptr = thrust::raw_pointer_cast(volume.data());
    gpuErrchk(cudaDeviceSynchronize());

    auto det_shape = curad::vec2u{width, height};
    auto vol_shape = curad::vec3u{volsize, volsize, volsize};

    // auto vol_spacing = curad::Vec<float, 3>{1, 1, 1};
    auto vol_spacing = curad::vec3f{3, 3, 3};
    auto vol_offset = curad::vec3f{0, 0, 0};

    curad::device_volume<float> vol_span(volume_ptr, vol_shape, vol_spacing, vol_offset);
    curad::device_measurement<float> sino_span(sino_ptr, {width, height}, DSD, DSO, angles);

    curad::bp::backproject_3d(vol_span, sino_span);

    thrust::host_vector<float> vol_host = volume;

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto max = *std::max_element(vol_host.begin(), vol_host.end());
    std::transform(vol_host.begin(), vol_host.end(), vol_host.begin(),
                   [&](auto x) { return x / max; });

    draw(thrust::raw_pointer_cast(vol_host.data()), 0, volsize, volsize, volsize);
}

TEST_CASE("backward_2d") {
    const auto volsize = 512;
    auto [data, width, _, nangles, angles, DSO, DSD] = curad::easy::read("sino_2d.txt");

    std::transform(angles.begin(), angles.end(), angles.begin(),
                   [](float x) { return x * M_PI / 180.; });

    // Copy to thrust device vector
    thrust::host_vector<float> host_sino(width * nangles, 0);
    std::copy(data.begin(), data.end(), host_sino.begin());

    // Copy to device
    thrust::device_vector<float> sino = host_sino;
    auto sino_ptr = thrust::raw_pointer_cast(sino.data());

    // Allocate device memory for volume
    thrust::device_vector<float> volume(volsize * volsize, 0);
    auto volume_ptr = thrust::raw_pointer_cast(volume.data());
    gpuErrchk(cudaDeviceSynchronize());

    auto det_shape = width;
    auto vol_shape = curad::vec2u{volsize, volsize};

    auto vol_spacing = curad::vec2f{1, 1};
    auto vol_offset = curad::vec2f{0, 0};

    const auto source = curad::vec2f{0, -DSO};

    std::cout << "DSO: " << DSO << std::endl;
    std::cout << "width: " << width << std::endl;
    std::cout << "nangles: " << nangles << std::endl;

    curad::image_2d vol_span(volume_ptr, vol_shape, vol_spacing, vol_offset);
    curad::measurement_2d sino_span(sino_ptr, det_shape, angles.size());
    sino_span.set_angles(angles);
    sino_span.set_distance_source_to_object(DSO);
    sino_span.set_distance_source_to_detector(DSD);

    curad::bp::backproject_2d(vol_span, sino_span);

    // Copy result back to host
    thrust::host_vector<float> vol_host = volume;

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Just make sure values are between 0 and 1
    auto max = *std::max_element(vol_host.begin(), vol_host.end());
    std::transform(vol_host.begin(), vol_host.end(), vol_host.begin(),
                   [&](auto x) { return x / max; });

    draw(thrust::raw_pointer_cast(vol_host.data()), 0, volsize, volsize, 1);
}
