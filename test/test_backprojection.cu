#include "doctest/doctest.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "show.hpp"

#include "curadon/backward.hpp"

TEST_CASE("test_backprojection") {

    const auto volsize = 64;
    auto [data, width, height, nangles, angles, DSO, DSD] = curad::easy::read("demofile2.txt");

    thrust::host_vector<float> host_sino(width * height * nangles, 0);
    std::copy(data.begin(), data.end(), host_sino.begin());

    thrust::device_vector<float> sino = host_sino;
    auto sino_ptr = thrust::raw_pointer_cast(sino.data());

    thrust::device_vector<float> volume(volsize * volsize * volsize, 0);
    auto volume_ptr = thrust::raw_pointer_cast(volume.data());
    gpuErrchk(cudaDeviceSynchronize());

    auto det_shape = curad::Vec<std::uint64_t, 2>{width, height};
    auto vol_shape = curad::Vec<std::uint64_t, 3>{volsize, volsize, volsize};

    // auto vol_spacing = curad::Vec<float, 3>{1, 1, 1};
    auto vol_spacing = curad::Vec<float, 3>{3, 3, 3};
    auto vol_offset = curad::Vec<float, 3>{0, 0, 0};

    curad::device_volume<float> vol_span(volume_ptr, vol_shape, vol_spacing, vol_offset);
    curad::device_measurement<float> sino_span(sino_ptr, {width, height}, DSD, DSO, angles);

    curad::backproject_3d(vol_span, sino_span);

    thrust::host_vector<float> vol_host = volume;

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto max = *std::max_element(vol_host.begin(), vol_host.end());
    std::transform(vol_host.begin(), vol_host.end(), vol_host.begin(),
                   [&](auto x) { return x / max; });

    draw(thrust::raw_pointer_cast(vol_host.data()), 0, volsize, volsize, volsize);
}
