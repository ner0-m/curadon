#pragma once

#include "curadon/detail/error.h"
#include "curadon/detail/texture.hpp"
#include "curadon/device_span.hpp"
#include "curadon/math/vector.hpp"
#include "curadon/rotation.h"
#include "curadon/utils.hpp"

#include <cstdint>
#include <cuda_runtime_api.h>

namespace curad {

static constexpr i64 num_projects_per_kernel = 32;

static constexpr i64 num_voxels_per_thread = 8;

__constant__ vec3f dev_vol_origin[num_projects_per_kernel];
__constant__ vec3f dev_delta_x[num_projects_per_kernel];
__constant__ vec3f dev_delta_y[num_projects_per_kernel];
__constant__ vec3f dev_delta_z[num_projects_per_kernel];

// TODO: have constant memory array for sources!

template <class T>
__global__ void kernel_backprojection_3d(device_span_3d<T> volume, vec3f source, f32 DSD, f32 DSO,
                                         vec2u det_shape, i64 cur_projection, i64 total_projections,
                                         cudaTextureObject_t tex) {
    auto idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    auto idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    auto start_idx_z = blockIdx.z * num_voxels_per_thread + threadIdx.z;

    if (idx_x >= volume.shape()[0] || idx_y >= volume.shape()[1] ||
        start_idx_z >= volume.shape()[2])
        return;

    // Each thread has an array of volume voxels, that we first read, then
    // work on, and then write back once
    f32 local_volume[num_voxels_per_thread];

    // First step, load all values into the local_volume array
#pragma unroll
    for (int z = 0; z < num_voxels_per_thread; ++z) {
        const auto idx_z = start_idx_z + z;
        if (idx_z >= volume.shape()[2]) {
            break;
        }

        local_volume[z] = volume(idx_x, idx_y, idx_z);
    }

    // Second step, for all projections this kernel performs, do the backprojection
    // for all voxels this thread is associated with
    for (int proj = 0; proj < num_projects_per_kernel; ++proj) {
        auto idx_proj = cur_projection * num_projects_per_kernel + proj;

        if (idx_proj >= total_projections) {
            break;
        }

        auto vol_origin = dev_vol_origin[proj];
        auto delta_x = dev_delta_x[proj];
        auto delta_y = dev_delta_y[proj];
        auto delta_z = dev_delta_z[proj];

#pragma unroll
        for (int z = 0; z < num_voxels_per_thread; ++z) {
            const auto idx_z = start_idx_z + z;

            if (idx_z >= volume.shape()[2]) {
                break;
            }

            // Compute world coordinates of the working voxel
            // move idx_x delta_x from origin, same for y and z
            auto P = vol_origin + idx_x * delta_x + idx_y * delta_y + idx_z * delta_z;

            // Compute line from source to P
            auto dir = P - source;

            // Compute intersection of detector with dir
            auto t = __fdividef(DSO - DSD - source[2], dir[2]);

            // Coordinates are from [-det_shape / 2, det_shape / 2], hence shift it to be
            // strictly positive
            auto u = (dir[0] * t + source[0]) + det_shape[0] / 2;
            auto v = (dir[1] * t + source[1]) + det_shape[1] / 2;

            auto sample = tex3D<f32>(tex, u, v, proj + 0.5f);

            local_volume[z] += sample;
        }
    }

    // Last step, write local volume back to global one
#pragma unroll
    for (int z = 0; z < num_voxels_per_thread; ++z) {
        const auto idx_z = start_idx_z + z;
        if (idx_z >= volume.shape()[2]) {
            break;
        }

        volume(idx_x, idx_y, idx_z) = local_volume[z];
    }
}

namespace detail {
template <class T, class U>
void setup_constants(device_volume<T> vol, device_measurement<U> sino, u64 start_proj,
                     u64 num_projections) {
    const auto vol_extent = vol.extent();
    const auto vol_spacing = vol.spacing();
    const auto vol_offset = vol.offset();

    std::vector<curad::vec3f> vol_origins;
    std::vector<curad::vec3f> delta_xs;
    std::vector<curad::vec3f> delta_ys;
    std::vector<curad::vec3f> delta_zs;

    for (int i = start_proj; i < start_proj + num_projections; ++i) {
        // TODO: check what am I still missing to make this feature complete? Check with tigre

        curad::vec3f init_vol_origin = -vol_extent / 2.f + vol_spacing / 2.f + vol_offset;
        auto vol_origin =
            curad::geometry::rotate_yzy(init_vol_origin, sino.phi(i), sino.theta(i), sino.psi(i));
        vol_origins.push_back(vol_origin);

        curad::vec3f init_delta;
        init_delta = init_vol_origin;
        init_delta[0] += vol_spacing[0];
        init_delta =
            curad::geometry::rotate_yzy(init_delta, sino.phi(i), sino.theta(i), sino.psi(i));
        delta_xs.push_back(init_delta - vol_origin);

        init_delta = init_vol_origin;
        init_delta[1] += vol_spacing[1];
        init_delta =
            curad::geometry::rotate_yzy(init_delta, sino.phi(i), sino.theta(i), sino.psi(i));
        delta_ys.push_back(init_delta - vol_origin);

        init_delta = init_vol_origin;
        init_delta[2] += vol_spacing[2];
        init_delta =
            curad::geometry::rotate_yzy(init_delta, sino.phi(i), sino.theta(i), sino.psi(i));
        delta_zs.push_back(init_delta - vol_origin);
    }

    cudaMemcpyToSymbol(curad::dev_vol_origin, vol_origins.data(),
                       sizeof(curad::vec3f) * curad::num_projects_per_kernel, 0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(curad::dev_delta_x, delta_xs.data(),
                       sizeof(curad::vec3f) * curad::num_projects_per_kernel, 0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(curad::dev_delta_y, delta_ys.data(),
                       sizeof(curad::vec3f) * curad::num_projects_per_kernel, 0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(curad::dev_delta_z, delta_zs.data(),
                       sizeof(curad::vec3f) * curad::num_projects_per_kernel, 0, cudaMemcpyDefault);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
} // namespace detail

template <class T, class U>
void backproject_3d(device_volume<T> volume, device_measurement<U> sinogram) {
    const auto nangles = sinogram.nangles();
    const auto sino_width = sinogram.shape()[0];
    const auto sino_height = sinogram.shape()[1];

    // allocate cuda array for sinogram
    auto array_cu = detail::allocate_cuarray(sino_width, sino_height, num_projects_per_kernel);

    cudaTextureObject_t tex;

    // Bind cuda array containing the sinogram to the texture
    detail::bind_texture_to_array(&tex, array_cu);

    auto num_kernel_calls =
        (nangles + curad::num_projects_per_kernel - 1) / curad::num_projects_per_kernel;
    for (int i = 0; i < num_kernel_calls; ++i) {
        auto proj_idx = i * curad::num_projects_per_kernel;

        auto projections_left = nangles - (i * curad::num_projects_per_kernel);

        // On how many projections do we work on this call? either num_projects_per_kernel, or
        // what ever is left
        const auto num_projections =
            std::min<int>(curad::num_projects_per_kernel, projections_left);

        // Copy projection data necessary for the next kernel to cuda array
        const auto sub_sino = sinogram.slice(proj_idx, num_projections);
        detail::copy_projections_to_array(sub_sino, array_cu);

        // kernel uses variables stored in __constant__ memory, e.g. volume origin, volume
        // deltas Compute them here and upload them
        detail::setup_constants(volume, sinogram, proj_idx, num_projections);

        auto vol_span = volume.kernel_span();

        const auto source = sinogram.source();
        const auto DSD = sinogram.distance_source_to_detector();
        const auto DSO = sinogram.distance_source_to_object();
        const auto det_shape = sinogram.detector_shape();

        int divx = 16;
        int divy = 32;
        int divz = curad::num_voxels_per_thread;

        dim3 threads_per_block(divx, divy, 1);

        const auto vol_shape = volume.shape();
        int block_x = (vol_shape[0] + divx - 1) / divx;
        int block_y = (vol_shape[1] + divy - 1) / divy;
        int block_z = (vol_shape[2] + divz - 1) / divz;
        dim3 num_blocks(block_x, block_y, block_z);
        kernel_backprojection_3d<<<num_blocks, threads_per_block>>>(vol_span, source, DSD, DSO,
                                                                    det_shape, i, nangles, tex);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    cudaDestroyTextureObject(tex);
}

} // namespace curad
