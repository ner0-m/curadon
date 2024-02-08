#pragma once

#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <texture_indirect_functions.h>
#include <vector>

#include <thrust/device_vector.h>

#include "curadon/detail/device_span.hpp"
#include "curadon/detail/error.h"
#include "curadon/detail/plan/plan_2d.hpp"
#include "curadon/detail/utils.hpp"
#include "curadon/detail/vec.hpp"
#include "curadon/pool.hpp"

namespace curad::bp {
namespace kernel {

template <typename T>
__global__ void backward_2d(T *__restrict__ volume, vec2u vol_shape, vec2f vol_spacing,
                            vec2f vol_offset, cudaTextureObject_t texture, u64 det_shape,
                            f32 det_spacing, f32 DSD, f32 DSO, f32 *__restrict__ angles,
                            u64 cur_projection, u64 nangles, u64 num_projections) {
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
    extern __shared__ float2 sincos[];

    for (int i = tid; i < nangles; i += blockDim.x * blockDim.y) {
        if (i < num_projections) {
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

    f32 accum = 0.0f;

#pragma unroll(16)
    for (int proj_idx = 0; proj_idx < num_projections; proj_idx++) {
        if (proj_idx + cur_projection >= nangles) {
            break;
        }

        const auto sin = sincos[proj_idx].x;
        const auto cos = sincos[proj_idx].y;

        // Similar to equation (2) from "CUDA and OpenCL Implementations of 3D CT Reconstruction for
        // Biomedical Imaging", but adapted to produce equal output to TorchRadon and ASTRA
        const f32 denominator = fmaf(cos, -real_y, sin * real_x + DSO);
        const f32 dist_denom = __fdividef(DSD, denominator);

        const f32 u = fmaf(cos * scaled_real_x + sin * scaled_real_y, dist_denom, det_extend_u);
        accum += tex1DLayered<float>(texture, u, proj_idx) * dist_denom;
    }

    volume[x + vol_shape[0] * y] += accum * inv_u_spacing;
}
} // namespace kernel

template <class T, class U>
void backproject_2d_async(device_span_2d<T> volume, device_span_2d<U> sino, forward_plan_2d &plan,
                          cuda::stream_view stream) {
    auto vol_shape = plan.vol_shape();

    auto sino_ptr = sino.device_data();
    auto det_count = plan.det_count();
    auto nangles = plan.nangles();

    auto event = cuda::get_next_event();

    const int num_kernel_calls =
        utils::round_up_division(nangles, plan.num_projections_per_kernel());

    auto &tex = plan.backward_tex();

    for (int i = 0; i < num_kernel_calls; ++i) {
        const auto proj_idx = i * plan.num_projections_per_kernel();
        const auto num_projections_left = nangles - proj_idx;
        const auto num_projections =
            std::min<int>(plan.num_projections_per_kernel(), num_projections_left);

        // Copy projection data necessary for the next kernel to cuda array
        const auto offset = proj_idx * det_count;
        auto cur_proj_ptr = sino_ptr + offset;

        auto loop_stream = cuda::get_next_stream();

        tex.write_1dlayered(cur_proj_ptr, det_count, num_projections, loop_stream);

        int divx = 16;
        int divy = 16;
        dim3 threads_per_block(divx, divy);

        int block_x = utils::round_up_division(vol_shape[0], divx);
        int block_y = utils::round_up_division(vol_shape[1], divy);

        dim3 num_blocks(block_x, block_y);
        kernel::backward_2d<<<num_blocks, threads_per_block,
                              sizeof(f32) * plan.num_projections_per_kernel(), loop_stream>>>(
            volume.device_data(), plan.vol_shape(), plan.vol_spacing(), plan.vol_offset(),
            tex.tex(), det_count, plan.det_spacing(), plan.DSD(), plan.DSO(), plan.angles().data(),
            proj_idx, nangles, plan.num_projections_per_kernel());

        gpuErrchk(cudaGetLastError());

        event.record(loop_stream);
        stream.wait_for_event(event);
    }
}

template <class T, class U>
void backproject_2d_sync(device_span_2d<T> volume, device_span_2d<U> sinogram,
                         forward_plan_2d &plan, cuda::stream_view stream) {
    backproject_2d_async(volume, sinogram, plan, stream);
    stream.synchronize();
}

template <class T, class U>
void backproject_2d(device_span_2d<T> volume, device_span_2d<U> sinogram, forward_plan_2d &plan) {
    auto stream = cuda::get_next_stream();
    backproject_2d_sync(volume, sinogram, plan, stream);
}
} // namespace curad::bp
