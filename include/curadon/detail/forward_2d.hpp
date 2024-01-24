#pragma once

#include "curadon/detail/device_span.hpp"
#include "curadon/detail/error.h"
#include "curadon/detail/image_2d.hpp"
#include "curadon/detail/measurement_2d.hpp"
#include "curadon/detail/rotation.hpp"
#include "curadon/detail/texture_cache.hpp"
#include "curadon/detail/utils.hpp"
#include "curadon/detail/vec.hpp"

#include "curadon/detail/intersection.h"

#include <cmath>
#include <cstdio>
#include <type_traits>

#include <cuda_runtime_api.h>

// TODO: remove
#include <numeric>

namespace curad::fp {
namespace kernel {

static constexpr u64 pixels_u_per_block_2d = 16;
static constexpr u64 num_projections_per_kernel_2d = 128;

__constant__ vec2f dev_u_origins_2d[num_projections_per_kernel_2d];
__constant__ vec2f dev_delta_us_2d[num_projections_per_kernel_2d];
__constant__ vec2f dev_sources_2d[num_projections_per_kernel_2d];

template <class T>
__global__ void kernel_forward_2d(device_span_2d<T> sinogram, vec<u64, 2> vol_shape,
                                  vec2f vol_spacing, vec2f vol_offset, cudaTextureObject_t tex,
                                  u64 det_shape, i64 cur_proj, i64 num_projections) {
    // Calculate texture coordinates
    const auto idx_u = blockIdx.x * blockDim.x + threadIdx.x;

    const auto proj_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto global_proj_idx = proj_idx + cur_proj;

    if (global_proj_idx >= num_projections && idx_u >= det_shape) {
        return;
    }

    const auto width = vol_shape[0];
    const auto height = vol_shape[1];

    const auto source = dev_sources_2d[proj_idx];
    const auto uv_origin = dev_u_origins_2d[proj_idx];
    const auto delta_u = dev_delta_us_2d[proj_idx];
    const auto det_point = uv_origin + idx_u * delta_u;

    // Intersect with volume
    const auto [tmin, tmax] = intersection_2d(vol_shape, source, det_point);

    // if ray volume intersection is empty exit
    if (tmin > tmax - 1e-6) {
        sinogram(idx_u, proj_idx) = 0.0f;
        return;
    }

    auto ro = source + tmin * det_point;
    auto rd = det_point * (tmax - tmin);

    const int n_steps = __float2int_rn(fmax(fabs(rd.x()), fabs(rd.y())));

    auto v = rd / fmax(fabs(rd.x()), fabs(rd.y()));

    f32 accumulator = 0.f;
    for (int j = 0; j < n_steps; j++) {
        accumulator += tex2D<f32>(tex, ro.x(), ro.y());
        ro += v;
    }

    const float n = hypot(v.x() * vol_spacing[0], v.y() * vol_spacing[1]);
    sinogram(idx_u, proj_idx) = accumulator * n;
}

void setup_constants(span<f32> angles, vec2f init_source, u64 det_shape, f32 det_spacing, f32 DOD,
                     vec2f vol_extent, vec2f vol_spacing, vec2f vol_offset) {
    // TODO: can we do this smartly?
    thrust::device_vector<f32> d_angles(angles.data(), angles.data() + angles.size());
    thrust::host_vector<f32> h_angles = d_angles;

    std::vector<vec2f> u_origins(angles.size());
    std::vector<vec2f> delta_us(angles.size());
    std::vector<vec2f> sources(angles.size());

    auto det_extent = det_shape * det_spacing;

    // Assume the initial detector point, at
    vec2f init_det_origin{-det_extent / 2.f + det_spacing * .5f, -DOD};

    // The next detector pixel
    vec2f init_delta_u = init_det_origin + vec2f{det_spacing, 0.f};

    for (int i = 0; i < angles.size(); ++i) {
        auto angle = h_angles[i];

        // Apply geometry transformation, such that volume origin coincidence with world origin,
        // the volume voxels are unit size, for all projections, the image stays the same

        // 1) apply roll, pitch, yaw of detector
        // TODO

        // 3) Rotate according to current position clockwise
        auto source = ::curad::geometry::rotate(init_source, angle);
        auto det_origin = ::curad::geometry::rotate(init_det_origin, angle, source);
        auto delta_u = ::curad::geometry::rotate(init_delta_u, angle, source);

        // 4) move everything such that volume origin coincides with world origin
        source = source - vol_offset + 0.5f * vol_extent;

        // 5) scale such that volume voxels are unit size
        det_origin = det_origin / vol_spacing;
        delta_u = delta_u / vol_spacing;
        source = source / vol_spacing;

        // 6) Apply center of rotation correction
        // TODO

        // 7) store in host vector
        u_origins[i] = det_origin;
        delta_us[i] = delta_u - det_origin;
        sources[i] = source;
    }

    gpuErrchk(cudaMemcpyToSymbol(dev_u_origins_2d, u_origins.data(),
                                 sizeof(curad::vec2f) * kernel::num_projections_per_kernel_2d, 0,
                                 cudaMemcpyDefault));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpyToSymbol(dev_delta_us_2d, delta_us.data(),
                                 sizeof(curad::vec2f) * kernel::num_projections_per_kernel_2d, 0,
                                 cudaMemcpyDefault));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpyToSymbol(dev_sources_2d, sources.data(),
                                 sizeof(curad::vec2f) * kernel::num_projections_per_kernel_2d, 0,
                                 cudaMemcpyDefault));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
} // namespace kernel

template <class T, class U>
void forward_2d(image_2d<T> volume, measurement_2d<U> sinogram) {
    auto det_shape = sinogram.detector_shape();
    auto det_spacing = sinogram.spacing();
    auto angles = sinogram.angles();
    auto nangles = sinogram.nangles();
    auto DSD = sinogram.distance_source_to_detector();
    auto DSO = sinogram.distance_source_to_object();

    auto vol_shape = volume.shape();
    auto vol_extent = volume.extent();
    auto vol_spacing = volume.spacing();
    auto vol_offset = volume.offset();

    texture_config tex_config{vol_shape[0], vol_shape[1], 0, false};
    texture_cache_key key = {static_cast<void *>(volume.device_data()), tex_config};

    auto [_, inserted] = get_texture_cache().try_emplace(key, tex_config);
    auto &tex = get_texture_cache().at(key);

    if (inserted) {
        tex.write(volume.device_data());
    }

    const int num_kernel_calls =
        utils::round_up_division(nangles, kernel::num_projections_per_kernel_2d);

    for (int i = 0; i < num_kernel_calls; ++i) {
        const auto proj_idx = i * kernel::num_projections_per_kernel_2d;
        const auto num_projections_left = nangles - proj_idx;
        const auto num_projections =
            std::min<int>(kernel::num_projections_per_kernel_2d, num_projections_left);

        const u64 div_u = kernel::pixels_u_per_block_2d;
        dim3 grid(utils::round_up_division(det_shape, div_u),
                  utils::round_up_division(num_projections, div_u));
        dim3 block(div_u, div_u);

        auto source = sinogram.source();
        kernel::setup_constants(angles.subspan(proj_idx, num_projections), source, det_shape,
                                det_spacing, sinogram.distance_object_to_detector(), vol_extent,
                                vol_spacing, vol_offset);

        auto sino_slice = sinogram.slice(proj_idx, num_projections);
        kernel::kernel_forward_2d<<<grid, block>>>(sino_slice, vol_shape, vol_spacing, vol_offset,
                                                   tex.tex(), det_shape, proj_idx, nangles);
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

} // namespace curad::fp
