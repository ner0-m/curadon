#pragma once

#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <texture_indirect_functions.h>
#include <vector>

#include <thrust/device_vector.h>

#include "curadon/detail/error.h"
#include "curadon/detail/image_2d.hpp"
#include "curadon/detail/measurement_2d.hpp"
#include "curadon/detail/rotation.hpp"
#include "curadon/detail/texture_cache.hpp"
#include "curadon/detail/utils.hpp"
#include "curadon/detail/vec.hpp"

namespace curad::bp {
namespace kernel {
static constexpr i64 num_projections_per_kernel_2d = 512;

template <typename T>
__global__ void backward_2d(T *__restrict__ volume, vec2u vol_shape, vec2f vol_spacing,
                            vec2f vol_offset, cudaTextureObject_t texture, u64 det_shape,
                            f32 det_spacing, f32 DSD, f32 DSO, f32 *__restrict__ angles,
                            u64 cur_projection, u64 nangles) {
    // Calculate image coordinates
    const u64 x = blockIdx.x * blockDim.x + threadIdx.x;

    const u64 y = blockIdx.y * blockDim.y + threadIdx.y;

    const u64 tid = threadIdx.y * blockDim.x + threadIdx.x;

    const f32 vol_extend_x = vol_shape[0] / 2.0f;
    const f32 vol_extend_y = vol_shape[1] / 2.0f;
    const f32 det_extend_u = det_shape / 2.0f;

    const f32 inv_u_spacing = __fdividef(1.0f, det_spacing);

    if (x >= vol_shape[0] || y >= vol_shape[1]) {
        return;
    }

    // keep sin and cos packed together to save one memory load in the main loop
    __shared__ float2 sincos[num_projections_per_kernel_2d];

    for (int i = tid; i < nangles; i += blockDim.x * blockDim.y) {
        if (i < num_projections_per_kernel_2d) {
            float2 tmp;
            tmp.x = -__sinf(angles[cur_projection + i]);
            tmp.y = __cosf(angles[cur_projection + i]);
            sincos[i] = tmp;
        }
    }
    __syncthreads();

    // Compute x and y coordinates in world space (without rotation)
    const f32 real_x = (f32(x) - vol_extend_x) * vol_spacing[0] + vol_offset[0] + 0.5f;
    const f32 real_y = (f32(y) - vol_extend_y) * vol_spacing[1] + vol_offset[1] + 0.5f;

    const f32 scaled_real_x = real_x * inv_u_spacing;
    const f32 scaled_real_y = real_y * inv_u_spacing;

    f32 projf = 0.5f; // projected index in float (shifted by 0.5)
    f32 accum = 0.0f;

#pragma unroll(16)
    for (int proj_idx = 0; proj_idx < num_projections_per_kernel_2d; proj_idx++) {
        if (proj_idx + cur_projection > nangles) {
            break;
        }

        const auto sin = sincos[proj_idx].x;
        const auto cos = sincos[proj_idx].y;

        // Similar to equation (2) from "CUDA and OpenCL Implementations of 3D CT Reconstruction for
        // Biomedical Imaging", but adapted to produce equal output to TorchRadon and ASTRA
        const f32 denominator = fmaf(cos, -real_y, sin * real_x + DSO);
        const f32 dist_denom = __fdividef(DSD, denominator);

        const f32 u = fmaf(cos * scaled_real_x + sin * scaled_real_y, dist_denom, det_extend_u);

        accum += tex1DLayered<float>(texture, u, projf) * dist_denom;
        projf += 1.0f;
    }

    volume[x + vol_shape[0] * y] += accum * inv_u_spacing;
}
} // namespace kernel

template <class T, class U>
void backproject_2d(image_2d<T> volume, measurement_2d<U> sino) {
    auto vol_shape = volume.shape();
    auto vol_extent = volume.extent();
    auto vol_spacing = volume.spacing();
    auto vol_offset = volume.offset();

    auto sino_ptr = sino.device_data();
    auto det_shape = sino.detector_shape();
    auto det_spacing = sino.spacing();
    auto DSD = sino.distance_source_to_detector();
    auto DSO = sino.distance_source_to_object();
    auto source = sino.source();
    auto angles = sino.angles();
    auto nangles = sino.nangles();

    texture_config tex_config{det_shape, 0, kernel::num_projections_per_kernel_2d, true};

    const int num_kernel_calls =
        utils::round_up_division(nangles, kernel::num_projections_per_kernel_2d);
    for (int i = 0; i < num_kernel_calls; ++i) {
        const auto proj_idx = i * kernel::num_projections_per_kernel_2d;
        const auto num_projections_left = nangles - proj_idx;
        const auto num_projections =
            std::min<int>(kernel::num_projections_per_kernel_2d, num_projections_left);

        // Copy projection data necessary for the next kernel to cuda array
        const auto offset = proj_idx * det_shape;
        auto cur_proj_ptr = sino_ptr + offset;

        texture_cache_key key = {static_cast<void *>(cur_proj_ptr), tex_config};
        auto [_, inserted] = get_texture_cache().try_emplace(key, tex_config);
        auto &tex = get_texture_cache().at(key);

        if (inserted) {
            tex.write(cur_proj_ptr, num_projections);
        }

        int divx = 16;
        int divy = 16;
        dim3 threads_per_block(divx, divy);

        int block_x = utils::round_up_division(vol_shape[0], divx);
        int block_y = utils::round_up_division(vol_shape[1], divy);

        dim3 num_blocks(block_x, block_y);
        kernel::backward_2d<<<num_blocks, threads_per_block>>>(
            volume.device_data(), vol_shape, vol_spacing, vol_offset, tex.tex(), det_shape,
            det_spacing, DSD, DSO, angles.data(), proj_idx, nangles);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
}
} // namespace curad::bp
