#pragma once

#include "curadon/detail/device_span.hpp"
#include "curadon/detail/error.h"
#include "curadon/detail/image_2d.hpp"
#include "curadon/detail/measurement_2d.hpp"
#include "curadon/detail/rotation.hpp"
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

static constexpr u64 pixels_u_per_block_2d = 8;
static constexpr u64 num_projections_per_block_2d = 8;
static constexpr u64 num_projections_per_kernel_2d = 360;

__constant__ vec2f dev_u_origins_2d[num_projections_per_kernel_2d];
__constant__ vec2f dev_delta_us_2d[num_projections_per_kernel_2d];
__constant__ vec2f dev_sources_2d[num_projections_per_kernel_2d];

/// tex is the volume
template <class T>
__global__ void kernel_forward_2d(device_span_2d<T> sinogram, vec<u64, 2> vol_shape, f32 DSD,
                                  f32 DSO, i64 total_projections, cudaTextureObject_t tex,
                                  f32 accuracy) {
    const auto idx_u = threadIdx.x + blockIdx.x * blockDim.x;

    // TODO: make this less strict, enable this kernel to be called multiple times
    const auto proj_number = threadIdx.y + blockIdx.y * blockDim.y;

    const auto det_shape = sinogram.shape()[0];

    if (idx_u >= det_shape || proj_number >= total_projections)
        return;

    const auto uv_origin = dev_u_origins_2d[proj_number];
    const auto delta_u = dev_delta_us_2d[proj_number];
    const auto source = dev_sources_2d[proj_number];

    // The detector point this thread is working on
    // const auto det_point = uv_origin + idx_u * delta_u + idx_v * delta_v;
    vec2f det_point;
    det_point.x() = uv_origin.x() + idx_u * delta_u.x();
    det_point.y() = uv_origin.y() + idx_u * delta_u.y();

    // direction from source to detector point
    auto dir = det_point - source;

    // how many steps to take along dir should we walk?
    auto dir_len = norm(dir);
    const auto nsamples = static_cast<i64>(::ceil(__fdividef(dir_len, accuracy)));

    dir /= dir_len;

    const vec2f boxmin{-1, -1};
    const vec2f boxmax{vol_shape[0] + 1, vol_shape[1] + 1};
    auto [hit, tmin, tmax] = intersection(boxmin, boxmax, source, dir);

    if (!hit) {
        sinogram(idx_u, proj_number) = 0.f;
        return;
    }

    // tmin and tmax are both within [0, 1], hence, we compute how many of the nsamples are within
    // this region and only walk that many samples
    const auto nsteps = static_cast<int>(ceilf((tmax - tmin) / dir_len * nsamples));
    const auto step_length = (tmax - tmin) / (f32)nsteps;

    vec2f p;
    f32 accumulator = 0;

    for (f32 t = tmin; t <= tmax; t += step_length) {
        p = dir * t + source;
        accumulator += tex2D<f32>(tex, p.x(), p.y());
    }

    // TODO: accumulator * dir_len * vol_spacing
    sinogram(idx_u, proj_number) = accumulator;
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

    // TODO: bind volume to texture
    cudaTextureObject_t tex;

    // allocate cuarray with size of volume
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<f32>();
    cudaArray_t array;
    gpuErrchk(cudaMallocArray(&array, &channelDesc, vol_shape[0], vol_shape[1]));
    gpuErrchk(cudaPeekAtLastError());

    const auto size = vol_shape[0] * sizeof(f32);
    gpuErrchk(cudaMemcpy2DToArray(array, 0, 0, volume.device_data(), size, size, vol_shape[1],
                                  cudaMemcpyDefault));
    gpuErrchk(cudaStreamSynchronize(0));
    gpuErrchk(cudaPeekAtLastError());

    // bind texture
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = array;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;

    gpuErrchk(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
    gpuErrchk(cudaPeekAtLastError());

    cudaResourceDesc desc;
    cudaGetTextureObjectResourceDesc(&desc, tex);

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
                  utils::round_up_division(num_projections, kernel::num_projections_per_block_2d));
        dim3 block(div_u, kernel::num_projections_per_block_2d);

        // Move sinogram ptr ahead
        auto sino_slice = sinogram.slice(proj_idx, num_projections);
        kernel::kernel_forward_2d<<<grid, block>>>(sino_slice, vol_shape, DSD, DSO, nangles, tex,
                                                   accuracy);
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaFreeArray(array);
    cudaDestroyTextureObject(tex);
}

} // namespace curad::fp
