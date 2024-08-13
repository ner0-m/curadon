#pragma once

#include "curadon/detail/device_span.hpp"
#include "curadon/detail/error.h"
#include "curadon/detail/plan/plan_2d.hpp"
#include "curadon/detail/utils.hpp"
#include "curadon/detail/vec.hpp"
#include "curadon/pool.hpp"

#include "curadon/detail/intersection.h"

#include <cmath>
#include <cstdio>
#include <type_traits>

#include <cuda_runtime_api.h>

namespace curad::fp {
namespace kernel {

template <class T>
__global__ void kernel_forward_2d(device_span_2d<T> sinogram, vec<u64, 2> vol_shape,
                                  vec2f vol_spacing, vec2f vol_offset, cudaTextureObject_t tex,
                                  u64 det_shape, i64 cur_proj, i64 num_projections,
                                  span<vec2f> u_origins, span<vec2f> delta_us,
                                  span<vec2f> sources) {
    // Calculate texture coordinates
    const auto idx_u = blockIdx.x * blockDim.x + threadIdx.x;

    const auto proj_idx = threadIdx.y + blockIdx.y * blockDim.y;

    if (proj_idx >= num_projections || idx_u >= det_shape) {
        return;
    }

    const auto width = vol_shape[0];
    const auto height = vol_shape[1];

    const auto glob_proj_idx = cur_proj + proj_idx;

    const auto source = sources[glob_proj_idx];
    const auto uv_origin = u_origins[glob_proj_idx];
    const auto delta_u = delta_us[glob_proj_idx];

    // TODO: to support curved detector, change this to equal-angled instead
    // of equal spaced. i.e. given a field of view (e.g. 30), and n detector pixels (e.g. n = 10),
    // then each detector pixel is alpha (e.g. 30 / 10 = 3) degrees further away.
    // => calculate detector point with this => some trigonometric ratio
    // maybe: tan (alpha) = opposite / adjacent => tan(alpha) * adjacent = opposite
    // (where adjacent is the source to detector distance, and opposite the distance
    // from detector center to detector pixel)
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

    const f32 n = hypot(v.x() * vol_spacing[0], v.y() * vol_spacing[1]);
    sinogram(idx_u, proj_idx) = accumulator * n;
}
} // namespace kernel

template <class T, class U>
void forward_2d_async(device_span_2d<T> volume, device_span_2d<U> sinogram, plan_2d &plan,
                      cuda::stream_view stream) {
    auto det_shape = plan.det_count();
    auto nangles = plan.nangles();

    auto &tex = plan.forward_tex();
    tex.write_2d(volume.device_data(), tex.config().width, tex.config().height, stream);

    // Create event to observe texture write
    auto tex_event = cuda::get_next_event();
    tex_event.record(stream);

    const int num_kernel_calls =
        utils::round_up_division(nangles, plan.num_projections_per_kernel());

    auto event = cuda::get_next_event();

    for (int i = 0; i < num_kernel_calls; ++i) {
        const auto proj_idx = i * plan.num_projections_per_kernel();
        const auto num_projections_left = nangles - proj_idx;
        const auto num_projections =
            std::min<int>(plan.num_projections_per_kernel(), num_projections_left);

        dim3 block(plan.forward_block_x, plan.forward_block_y);
        dim3 grid(utils::round_up_division(det_shape, block.x),
                  utils::round_up_division(num_projections, block.y));

        auto loop_stream = cuda::get_next_stream();

        // Ensure that texture copy already happend
        loop_stream.wait_for_event(tex_event);

        auto sino_slice = sinogram.slice(proj_idx, num_projections);
        kernel::kernel_forward_2d<<<grid, block, 0, loop_stream>>>(
            sino_slice, plan.vol_shape(), plan.vol_spacing(), plan.vol_offset(), tex.tex(),
            det_shape, proj_idx, num_projections, plan.u_origins(), plan.delta_us(),
            plan.sources());

        gpuErrchk(cudaGetLastError());

        event.record(loop_stream);
        stream.wait_for_event(event);
    }
}

template <class T, class U>
void forward_2d_sync(device_span_2d<T> volume, device_span_2d<U> sinogram, plan_2d &plan,
                     cuda::stream_view stream) {
    forward_2d_async(volume, sinogram, plan, stream);
    stream.synchronize();
}

template <class T, class U>
void forward_2d(device_span_2d<T> volume, device_span_2d<U> sinogram, plan_2d &plan) {
    auto stream = cuda::get_next_stream();
    forward_2d_sync(volume, sinogram, plan, stream);
}
} // namespace curad::fp
