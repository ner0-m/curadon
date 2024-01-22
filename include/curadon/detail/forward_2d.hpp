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

static constexpr u64 pixels_u_per_block_2d = 32;
static constexpr u64 num_projections_per_kernel_2d = 256;

__constant__ vec2f dev_u_origins_2d[num_projections_per_kernel_2d];
__constant__ vec2f dev_delta_us_2d[num_projections_per_kernel_2d];
__constant__ vec2f dev_sources_2d[num_projections_per_kernel_2d];

template <class T>
__global__ void kernel_forward_2d(device_span_2d<T> sinogram, vec<u64, 2> vol_shape, f32 DSD,
                                  f32 DSO, i64 total_projections, cudaTextureObject_t tex,
                                  f32 accuracy) {
    const auto idx_u = threadIdx.x + blockIdx.x * blockDim.x;

    // TODO: make this less strict, enable this kernel to be called multiple times
    const auto proj_number = threadIdx.y + blockIdx.y * blockDim.y;

    const auto det_shape = sinogram.shape()[0];

    if (idx_u >= det_shape || proj_number >= total_projections) {
        return;
    }

    const auto uv_origin = dev_u_origins_2d[proj_number];
    const auto delta_u = dev_delta_us_2d[proj_number];
    const auto source = dev_sources_2d[proj_number];

    vec2f det_point = uv_origin + idx_u * delta_u;
    auto dir = det_point - source;

    // Compute intersection with AABB, by construction of our method, the volume origin is at the
    // origin, we increase the aabb just a touch due to the interpolation method
    auto [tmin, tmax] = intersection_2d(vol_shape, source, dir);

    // we don't hit the aabb, so just write zeros and leave
    if (tmin > tmax - 1e-6) {
        sinogram(idx_u, proj_number) = 0;
        return;
    }

    auto rs = source;
    rs += tmin * dir;
    // TODO: make this an operator on vector
    dir.x() *= (tmax - tmin);
    dir.y() *= (tmax - tmin);

    const int nsteps = __float2int_rn(fmaxf(fabs(dir.x()), fabs(dir.y())));

    vec2f step_length;
    step_length.x() = __fdividef(dir.x(), fmaxf(fabs(dir.x()), fabs(dir.y())));
    step_length.y() = __fdividef(dir.y(), fmaxf(fabs(dir.x()), fabs(dir.y())));

    f32 accumulator = 0;
    for (int j = 0; j < nsteps; j++) {
        accumulator += tex2D<f32>(tex, rs.x() + 0.5, rs.y() + 0.5);
        rs += step_length;
    }

    // TODO: multiply vx and vy by vol_spacing
    const float intersection_length = hypot(step_length.x(), step_length.y());
    sinogram(idx_u, proj_number) = accumulator * intersection_length;
}

void setup_constants(span<f32> angles, vec2f init_source, u64 det_shape, f32 det_spacing, f32 DOD,
                     vec2f vol_extent, vec2f vol_spacing, vec2f vol_offset) {
    std::vector<vec2f> u_origins(angles.size());
    std::vector<vec2f> delta_us(angles.size());
    std::vector<vec2f> sources(angles.size());

    for (int i = 0; i < angles.size(); ++i) {
        auto angle = angles[i];

        // Assume detector origin is at the bottom left corner, i.e. detector point (0, 0)
        vec2f init_det_origin{-det_spacing * (det_shape / 2.f) + det_spacing * 0.5f, 0.f};

        // detector point (1)
        vec2f init_delta_u = init_det_origin + vec2f{det_spacing, 0.f};

        // Apply geometry transformation, such that volume origin coincidence with world origin,
        // the volume voxels are unit size, for all projections, the image stays the same

        // 1) apply roll, pitch, yaw of detector
        // TODO

        // 2) translate to real detector position
        init_det_origin[1] += DOD;
        init_delta_u[1] += DOD;

        // 3) Rotate according to current position clockwise
        auto det_origin = ::curad::geometry::rotate(init_det_origin, -angle);
        auto delta_u = ::curad::geometry::rotate(init_delta_u, -angle);
        auto source = ::curad::geometry::rotate(init_source, -angle);

        // 4) move everything such that volume origin coincides with world origin
        const auto translation = vol_extent / 2.f - vol_spacing / 2;
        det_origin = det_origin - vol_offset + translation;
        delta_u = delta_u - vol_offset + translation;
        source = source - vol_offset + translation;

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

    // TODO: make this configurable
    const f32 accuracy = 1.f;

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

        auto source = sinogram.source();
        kernel::setup_constants(angles.subspan(proj_idx, num_projections), source, det_shape,
                                det_spacing, sinogram.distance_object_to_detector(), vol_extent,
                                vol_spacing, vol_offset);

        const u64 div_u = kernel::pixels_u_per_block_2d;
        dim3 grid(utils::round_up_division(det_shape, div_u),
                  utils::round_up_division(num_projections, div_u));
        dim3 block(div_u, div_u);

        // Move sinogram ptr ahead
        auto sino_slice = sinogram.slice(proj_idx, num_projections);
        kernel::kernel_forward_2d<<<grid, block>>>(sino_slice, vol_shape, DSD, DSO, nangles,
                                                   tex.tex(), accuracy);
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

} // namespace curad::fp
