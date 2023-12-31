#pragma once

#include "curadon/cuda/error.h"
#include "curadon/device_span.hpp"
#include "curadon/math/vector.hpp"
#include "curadon/rotation.h"
#include "curadon/utils.hpp"

#include <cstdint>
#include <cuda_runtime_api.h>

namespace curad {

static constexpr std::int64_t num_projects_per_kernel = 32;

static constexpr std::int64_t num_voxels_per_thread = 8;

__constant__ Vec<float, 3> dev_vol_origin[num_projects_per_kernel];
__constant__ Vec<float, 3> dev_delta_x[num_projects_per_kernel];
__constant__ Vec<float, 3> dev_delta_y[num_projects_per_kernel];
__constant__ Vec<float, 3> dev_delta_z[num_projects_per_kernel];

// TODO: have constant memory array for sources!

template <class T>
__global__ void kernel_backprojection(device_span_3d<T> volume, Vec<float, 3> source, float DSD,
                                      float DSO, Vec<std::uint64_t, 2> det_shape,
                                      std::int64_t cur_projection, std::int64_t total_projections,
                                      cudaTextureObject_t tex) {
    auto idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    auto idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    auto start_idx_z = blockIdx.z * num_voxels_per_thread + threadIdx.z;

    if (idx_x >= volume.shape()[0] || idx_y >= volume.shape()[1] ||
        start_idx_z >= volume.shape()[2])
        return;

    // Each thread has an array of volume voxels, that we first read, then
    // work on, and then write back once
    float local_volume[num_voxels_per_thread];

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

            // Coordinates are from [-det_shape / 2, det_shape / 2], hence shift it to be strictly
            // positive
            auto u = (dir[0] * t + source[0]) + det_shape[0] / 2;
            auto v = (dir[1] * t + source[1]) + det_shape[1] / 2;

            auto sample = tex3D<float>(tex, u, v, proj + 0.5f);

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
cudaArray_t allocate_cuarray(std::size_t width, std::size_t height, std::size_t depth) {
    const cudaExtent extent_alloc = make_cudaExtent(width, height, depth);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    cudaMalloc3DArray(&array, &channelDesc, extent_alloc);
    gpuErrchk(cudaPeekAtLastError());

    return array;
}

template <class T>
void copy_projections_to_array(device_span_3d<T> sino, cudaArray_t array_cu) {
    // Copy to cuda array
    cudaMemcpy3DParms copyParams = {0};

    auto ptr = sino.device_data();
    const auto width = sino.shape()[0];
    const auto height = sino.shape()[1];
    const auto width_bytes = sizeof(T) * width;
    const auto num_projections = sino.shape()[2];

    copyParams.srcPtr = make_cudaPitchedPtr((void *)ptr, width_bytes, width, height);

    const cudaExtent extent = make_cudaExtent(width, height, num_projections);
    gpuErrchk(cudaPeekAtLastError());
    copyParams.dstArray = array_cu;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDefault;
    cudaMemcpy3DAsync(&copyParams, 0); // TODO: use stream pool
    cudaStreamSynchronize(0);
    gpuErrchk(cudaPeekAtLastError());
}

void bind_texture_to_array(cudaTextureObject_t *tex, cudaArray_t array_cu) {
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = array_cu;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.addressMode[2] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(tex, &texRes, &texDescr, NULL);
    gpuErrchk(cudaPeekAtLastError());
}

void setup_constants(std::size_t start_proj, std::size_t num_projections, Vec<float, 3> vol_size,
                     Vec<float, 3> vol_spacing, Vec<float, 3> vol_offset, span<float> angles) {
    std::vector<curad::Vec<float, 3>> vol_origins;
    std::vector<curad::Vec<float, 3>> delta_xs;
    std::vector<curad::Vec<float, 3>> delta_ys;
    std::vector<curad::Vec<float, 3>> delta_zs;

    for (int j = start_proj; j < start_proj + num_projections; ++j) {
        float angle = angles[j] * M_PI / 180.f;

        curad::Vec<float, 3> init_vol_origin = -vol_size / 2.f + vol_spacing / 2.f + vol_offset;
        auto vol_origin = curad::geometry::rotate_yzy(init_vol_origin, angle, 0.f, 0.f);
        vol_origins.push_back(vol_origin);

        curad::Vec<float, 3> init_delta;
        init_delta = init_vol_origin;
        init_delta[0] += vol_spacing[0];
        init_delta = curad::geometry::rotate_yzy(init_delta, angle, 0.f, 0.f);
        delta_xs.push_back(init_delta - vol_origin);

        init_delta = init_vol_origin;
        init_delta[1] += vol_spacing[1];
        init_delta = curad::geometry::rotate_yzy(init_delta, angle, 0.f, 0.f);
        delta_ys.push_back(init_delta - vol_origin);

        init_delta = init_vol_origin;
        init_delta[2] += vol_spacing[2];
        init_delta = curad::geometry::rotate_yzy(init_delta, angle, 0.f, 0.f);
        delta_zs.push_back(init_delta - vol_origin);
    }

    cudaMemcpyToSymbol(curad::dev_vol_origin, vol_origins.data(),
                       sizeof(curad::Vec<float, 3>) * curad::num_projects_per_kernel, 0,
                       cudaMemcpyDefault);
    cudaMemcpyToSymbol(curad::dev_delta_x, delta_xs.data(),
                       sizeof(curad::Vec<float, 3>) * curad::num_projects_per_kernel, 0,
                       cudaMemcpyDefault);
    cudaMemcpyToSymbol(curad::dev_delta_y, delta_ys.data(),
                       sizeof(curad::Vec<float, 3>) * curad::num_projects_per_kernel, 0,
                       cudaMemcpyDefault);
    cudaMemcpyToSymbol(curad::dev_delta_z, delta_zs.data(),
                       sizeof(curad::Vec<float, 3>) * curad::num_projects_per_kernel, 0,
                       cudaMemcpyDefault);
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
        detail::setup_constants(proj_idx, num_projections, volume.extent(), volume.spacing(),
                                volume.offset(), sinogram.phi());

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
        kernel_backprojection<<<num_blocks, threads_per_block>>>(vol_span, source, DSD, DSO,
                                                                 det_shape, i, nangles, tex);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    cudaDestroyTextureObject(tex);
}

} // namespace curad
