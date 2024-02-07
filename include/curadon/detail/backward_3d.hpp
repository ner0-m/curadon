#pragma once

#include "curadon/detail/error.h"
#include "curadon/detail/image_3d.hpp"
#include "curadon/detail/measurement_3d.hpp"
#include "curadon/detail/rotation.hpp"
#include "curadon/detail/texture_utils.hpp"
#include "curadon/detail/utils.hpp"
#include "curadon/detail/vec.hpp"

#include <cstdint>
#include <cuda_runtime_api.h>

namespace curad::bp {

namespace kernel {
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

    auto fn = [&](auto p, auto i) {
        // 1. Rotate point according to current projection geometry
        p = curad::geometry::rotate_yzy(p, sino.phi(i), sino.theta(i), sino.psi(i));

        // 2. Apply detector offset
        p[0] += sino.offset()[0];
        p[1] += sino.offset()[1];

        // 3. Apply detector rotation
        // 3.1 Move point to origin (only z is necessary)
        p[2] -= sino.distance_object_to_detector();
        // 3.2 Apply roll, pitch, yaw
        p = curad::geometry::rotate_roll_pitch_yaw(p, sino.roll(), sino.pitch(), sino.yaw());
        // 3.3 Move point back
        p[2] += sino.distance_object_to_detector();

        return p;
    };

    for (int i = start_proj; i < start_proj + num_projections; ++i) {
        // TODO: check what am I still missing to make this feature complete? Check with tigre

        // 1. Create initial point (in this case image(0, 0, 0))
        curad::vec3f init_vol_origin = -vol_extent / 2.f + vol_spacing / 2.f + vol_offset;

        auto vol_origin = fn(init_vol_origin, i);

        // Store it
        vol_origins.push_back(vol_origin);

        curad::vec3f init_delta = init_vol_origin;
        init_delta[0] += vol_spacing[0];
        auto delta_x = fn(init_delta, i);
        delta_xs.push_back(delta_x - vol_origin);

        init_delta = init_vol_origin;
        init_delta[1] += vol_spacing[1];
        auto delta_y = fn(init_delta, i);
        delta_ys.push_back(delta_y - vol_origin);

        init_delta = init_vol_origin;
        init_delta[2] += vol_spacing[2];
        auto delta_z = fn(init_delta, i);
        delta_zs.push_back(delta_z - vol_origin);
    }

    cudaMemcpyToSymbol(dev_vol_origin, vol_origins.data(),
                       sizeof(curad::vec3f) * num_projects_per_kernel, 0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(dev_delta_x, delta_xs.data(), sizeof(curad::vec3f) * num_projects_per_kernel,
                       0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(dev_delta_y, delta_ys.data(), sizeof(curad::vec3f) * num_projects_per_kernel,
                       0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(dev_delta_z, delta_zs.data(), sizeof(curad::vec3f) * num_projects_per_kernel,
                       0, cudaMemcpyDefault);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
} // namespace kernel

template <class T, class U>
void backproject_3d(device_volume<T> volume, device_measurement<U> sinogram) {
    const auto nangles = sinogram.nangles();
    const auto sino_width = sinogram.shape()[0];
    const auto sino_height = sinogram.shape()[1];

    // allocate cuda array for sinogram
    auto array_cu =
        detail::allocate_cuarray(sino_width, sino_height, kernel::num_projects_per_kernel);

    cudaTextureObject_t tex;

    // Bind cuda array containing the sinogram to the texture
    detail::bind_texture_to_array(&tex, array_cu);

    auto num_kernel_calls =
        (nangles + kernel::num_projects_per_kernel - 1) / kernel::num_projects_per_kernel;
    for (int i = 0; i < num_kernel_calls; ++i) {
        auto proj_idx = i * kernel::num_projects_per_kernel;

        auto projections_left = nangles - (i * kernel::num_projects_per_kernel);

        // On how many projections do we work on this call? either num_projects_per_kernel, or
        // what ever is left
        const auto num_projections =
            std::min<int>(kernel::num_projects_per_kernel, projections_left);

        // Copy projection data necessary for the next kernel to cuda array
        const auto sub_sino = sinogram.slice(proj_idx, num_projections);
        detail::copy_projections_to_array(sub_sino, array_cu);

        // kernel uses variables stored in __constant__ memory, e.g. volume origin, volume
        // deltas Compute them here and upload them
        kernel::setup_constants(volume, sinogram, proj_idx, num_projections);

        auto vol_span = volume.kernel_span();

        auto source = sinogram.source();

        // Apply roll, pitch, yaw of detector to source
        // 1. Move back, such that detector is at origin
        source[2] -= sinogram.distance_object_to_detector();
        // 2. Apply roll, pitch, yaw
        source = geometry::rotate_roll_pitch_yaw(source, sinogram.roll(), sinogram.pitch(),
                                                 sinogram.yaw());
        // 3. Move source to original position
        source[2] += sinogram.distance_object_to_detector();

        const auto DSD = sinogram.distance_source_to_detector();
        const auto DSO = sinogram.distance_source_to_object();
        const auto det_shape = sinogram.detector_shape();

        int divx = 16;
        int divy = 32;
        int divz = kernel::num_voxels_per_thread;

        dim3 threads_per_block(divx, divy, 1);

        const auto vol_shape = volume.shape();
        int block_x = (vol_shape[0] + divx - 1) / divx;
        int block_y = (vol_shape[1] + divy - 1) / divy;
        int block_z = (vol_shape[2] + divz - 1) / divz;
        dim3 num_blocks(block_x, block_y, block_z);
        kernel::kernel_backprojection_3d<<<num_blocks, threads_per_block>>>(
            vol_span, source, DSD, DSO, det_shape, i, nangles, tex);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    cudaDestroyTextureObject(tex);
    cudaFreeArray(array_cu);
}

} // namespace curad::bp
