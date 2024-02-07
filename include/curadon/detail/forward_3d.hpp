#pragma once

#include "curadon/detail/error.h"
#include "curadon/detail/image_3d.hpp"
#include "curadon/detail/intersection.h"
#include "curadon/detail/measurement_3d.hpp"
#include "curadon/detail/rotation.hpp"
#include "curadon/detail/texture_utils.hpp"
#include "curadon/detail/utils.hpp"
#include "curadon/detail/vec.hpp"

#include <cmath>
#include <cstdio>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda_runtime_api.h>

namespace curad::fp {
namespace kernel {

static constexpr u64 pixels_u_per_block_3d = 8;
static constexpr u64 pixels_v_per_block_3d = 8;
static constexpr u64 projections_per_block_3d = 8;

static constexpr i64 num_projects_per_kernel_3d = 128;

__constant__ vec3f dev_u_origins_3d[num_projects_per_kernel_3d];
__constant__ vec3f dev_delta_us_3d[num_projects_per_kernel_3d];
__constant__ vec3f dev_delta_vs_3d[num_projects_per_kernel_3d];
__constant__ vec3f dev_sources_3d[num_projects_per_kernel_3d];

/// tex is the volume
template <class T>
__global__ void kernel_forward_3d(device_span_3d<T> sinogram, vec3u vol_shape, f32 DSD, f32 DSO,
                                  cudaTextureObject_t tex, f32 accuracy, u64 start_proj) {
    const auto idx_u = threadIdx.x + blockIdx.x * blockDim.x;
    const auto idx_v = threadIdx.y + blockIdx.y * blockDim.y;

    const auto local_proj_idx = threadIdx.z + blockIdx.z * blockDim.z;
    const auto proj_idx = start_proj + local_proj_idx;

    if (idx_u >= sinogram.shape()[0] || idx_v >= sinogram.shape()[1] ||
        proj_idx >= sinogram.shape()[2]) {
        return;
    }

    const auto uv_origin = dev_u_origins_3d[local_proj_idx];
    const auto delta_u = dev_delta_us_3d[local_proj_idx];
    const auto delta_v = dev_delta_vs_3d[local_proj_idx];
    const auto source = dev_sources_3d[local_proj_idx];

    // The detector point this thread is working on
    // const auto det_point = uv_origin + idx_u * delta_u + idx_v * delta_v;
    vec3f det_point;
    det_point.x() = uv_origin.x() + idx_u * delta_u.x() + idx_v * delta_v.x();
    det_point.y() = uv_origin.y() + idx_u * delta_u.y() + idx_v * delta_v.y();
    det_point.z() = uv_origin.z() + idx_u * delta_u.z() + idx_v * delta_v.z();

    // direction from source to detector point
    auto dir = det_point - source;

    // how many steps to take along dir should we walk
    // TODO: This walks from the source to the  detector all the way, this is hyper inefficient,
    // clean this up (i.e. interect with AABB of volume)
    const auto nsamples = static_cast<i64>(::ceil(__fdividef(norm(dir), accuracy)));

    const vec3f boxmin{-1, -1, -1};
    const vec3f boxmax{vol_shape[0] + 1, vol_shape[1] + 1, vol_shape[2] + 1};
    auto [hit, tmin, tmax] = intersection(boxmin, boxmax, source, dir);

    if (!hit) {
        return;
    }

    // tmin and tmax are both within [0, 1], hence, we compute how many of the nsamples are within
    // this region and only walk that many samples
    const auto nsteps = static_cast<int>(ceilf((tmax - tmin) * nsamples));
    const auto step_length = (tmax - tmin) / nsteps;

    vec3f t;
    f32 accumulator = 0;

    // TODO: i should start at a min t value, which is certainly not 0!
    for (f32 i = tmin; i <= tmax; i += step_length) {
        t = dir * i + source;
        accumulator += tex3D<f32>(tex, t.x() + 0.5f, t.y() + 0.5f, t.z() + 0.5f);
    }

    f32 delta_length = norm(dir /* * vol_spacing */);
    sinogram(idx_u, idx_v, proj_idx) = accumulator;
}
} // namespace kernel

namespace detail {
template <class T, class U>
void setup_constants(device_volume<T> vol, device_measurement<U> sino, u64 proj_idx, u64 num_proj) {
    const auto vol_shape = vol.shape();
    const auto vol_size = vol.extent();
    const auto vol_spacing = vol.spacing();
    const auto vol_offset = vol.offset();

    const auto det_shape = sino.shape();
    const auto det_spacing = sino.spacing();
    const auto DSD = sino.distance_source_to_detector();
    const auto DSO = sino.distance_source_to_object();

    const auto nangles = sino.nangles();

    std::vector<vec3f> host_uv_origins;
    std::vector<vec3f> host_deltas_us;
    std::vector<vec3f> host_delta_vs;
    std::vector<vec3f> host_sources;

    // distance object to detector
    const auto DOD = DSD - DSO;
    for (int i = proj_idx; i < proj_idx + num_proj; ++i) {
        vec3f init_source({0, 0, -DSO});

        // Assume detector origin is at the bottom left corner, i.e. detector point (0, 0)
        vec3f init_det_origin{-det_spacing[0] * (det_shape[0] / 2.f) + det_spacing[0] * 0.5f, // u
                              -det_spacing[1] * (det_shape[1] / 2.f) + det_spacing[1] * 0.5f, // v
                              0.f};

        // detector point (1,0)
        vec3f init_delta_u = init_det_origin + vec3f{det_spacing[0], 0.f, 0.f};
        // detector point (0, 1)
        vec3f init_delta_v = init_det_origin + vec3f{0.f, det_spacing[1], 0.f};

        // Apply geometry transformation, such that volume origin coincidence with world origin,
        // the volume voxels are unit size, for all projections, the image stays the same

        // 1) apply roll, pitch, yaw of detector
        init_det_origin = ::curad::geometry::rotate_roll_pitch_yaw(init_det_origin, sino.roll(),
                                                                   sino.pitch(), sino.yaw());
        init_delta_u = ::curad::geometry::rotate_roll_pitch_yaw(init_delta_u, sino.roll(),
                                                                sino.pitch(), sino.yaw());
        init_delta_v = ::curad::geometry::rotate_roll_pitch_yaw(init_delta_v, sino.roll(),
                                                                sino.pitch(), sino.yaw());

        // 2) translate to real detector position
        init_det_origin[2] += DOD;
        init_delta_u[2] += DOD;
        init_delta_v[2] += DOD;

        // 3) Rotate according to current position
        auto det_origin =
            ::curad::geometry::rotate_yzy(init_det_origin, sino.phi(i), sino.theta(i), sino.psi(i));
        auto delta_u =
            ::curad::geometry::rotate_yzy(init_delta_u, sino.phi(i), sino.theta(i), sino.psi(i));
        auto delta_v =
            ::curad::geometry::rotate_yzy(init_delta_v, sino.phi(i), sino.theta(i), sino.psi(i));
        auto source =
            ::curad::geometry::rotate_yzy(init_source, sino.phi(i), sino.theta(i), sino.psi(i));

        // 4) move everything such that volume origin coincides with world origin
        const auto translation = vol_size / 2.f - vol_spacing / 2;
        det_origin = det_origin - vol_offset + translation;
        delta_u = delta_u - vol_offset + translation;
        delta_v = delta_v - vol_offset + translation;
        source = source - vol_offset + translation;

        // 5) scale such that volume voxels are unit size
        det_origin = det_origin / vol_spacing;
        delta_u = delta_u / vol_spacing;
        delta_v = delta_v / vol_spacing;
        source = source / vol_spacing;

        // 6) Apply center of rotation correction, This doesn't work for non-standard trajectories,
        // TODO: check it?
        auto COR = sino.center_of_rotation_correction();
        auto cor_x = COR * std::sin(sino.phi(i)) / vol_spacing[0];
        auto cor_z = COR * std::cos(sino.phi(i)) / vol_spacing[2];

        det_origin[0] += cor_x;
        det_origin[2] += cor_z;
        delta_u[0] += cor_x;
        delta_u[2] += cor_z;
        delta_v[0] += cor_x;
        delta_v[2] += cor_z;
        source[0] += cor_x;
        source[2] += cor_z;

        // 7) store in host vector
        host_uv_origins.push_back(det_origin);
        host_deltas_us.push_back(delta_u - det_origin);
        host_delta_vs.push_back(delta_v - det_origin);
        host_sources.push_back(source);
    }

    // upload uv_origin, delta_u, delta_v, sources
    gpuErrchk(cudaMemcpyToSymbol(kernel::dev_u_origins_3d, host_uv_origins.data(),
                                 sizeof(curad::vec3f) * host_uv_origins.size(), 0,
                                 cudaMemcpyDefault));

    gpuErrchk(cudaMemcpyToSymbol(kernel::dev_delta_us_3d, host_deltas_us.data(),
                                 sizeof(curad::vec3f) * host_deltas_us.size(), 0,
                                 cudaMemcpyDefault));

    gpuErrchk(cudaMemcpyToSymbol(kernel::dev_delta_vs_3d, host_delta_vs.data(),
                                 sizeof(curad::vec3f) * host_delta_vs.size(), 0,
                                 cudaMemcpyDefault));

    gpuErrchk(cudaMemcpyToSymbol(kernel::dev_sources_3d, host_sources.data(),
                                 sizeof(curad::vec3f) * host_sources.size(), 0, cudaMemcpyDefault));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
} // namespace detail

template <class T, class U>
void forward_3d(device_volume<T> vol, device_measurement<U> sino) {
    auto volume = vol.device_data();
    const auto vol_shape = vol.shape();
    const auto vol_extent = vol.extent();
    const auto vol_spacing = vol.spacing();
    const auto vol_offset = vol.offset();

    auto sinogram = sino.device_data();
    const auto angles = sino.angles();
    const auto det_shape = sino.shape();
    const auto det_spacing = sino.spacing();
    const auto DSD = sino.distance_source_to_detector();
    const auto DSO = sino.distance_source_to_object();

    const auto nangles = sino.nangles();

    // TODO: make this configurable
    const f32 accuracy = 1.f;

    cudaArray_t array = ::curad::detail::allocate_cuarray(vol_shape[0], vol_shape[1], vol_shape[2]);

    cudaTextureObject_t tex;
    ::curad::detail::bind_texture_to_array(&tex, array);

    // Copy to cuda array
    ::curad::detail::copy_projections_to_array(vol.kernel_span(), array);

    auto num_kernel_calls = utils::round_up_division(nangles, kernel::num_projects_per_kernel_3d);
    for (int i = 0; i < num_kernel_calls; ++i) {
        auto proj_idx = i * kernel::num_projects_per_kernel_3d;

        auto projections_left = nangles - (i * kernel::num_projects_per_kernel_3d);
        const auto num_projections =
            std::min<int>(kernel::num_projects_per_kernel_3d, projections_left);

        // upload uv_origin, delta_u, delta_v, sources for the given projections
        detail::setup_constants(vol, sino, proj_idx, num_projections);

        const u64 div_u = kernel::pixels_u_per_block_3d;
        const u64 div_v = kernel::pixels_v_per_block_3d;

        dim3 grid(utils::round_up_division(det_shape[0], div_u),
                  utils::round_up_division(det_shape[1], div_v),
                  utils::round_up_division(num_projections, kernel::projections_per_block_3d));
        dim3 block(div_u, div_v, kernel::projections_per_block_3d);

        auto sino_span = sino.kernel_span();

        kernel::kernel_forward_3d<<<grid, block>>>(sino_span, vol_shape, DSD, DSO, tex, accuracy,
                                                   proj_idx);

        // TODO: These should be moved to the outside of the loop, but currently, I'm not 100% sure
        // it's safe
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    cudaDestroyTextureObject(tex);
    cudaFreeArray(array);
}

} // namespace curad::fp
