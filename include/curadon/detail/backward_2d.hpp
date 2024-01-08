#pragma once

#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

#include "curadon/detail/error.h"
#include "curadon/detail/rotation.hpp"
#include "curadon/detail/utils.hpp"
#include "curadon/detail/vec.hpp"

namespace curad::bp {
namespace kernel {
static constexpr i64 num_projections_per_kernel_2d = 32;

static constexpr i64 num_voxels_per_thread_2d = 8;

__constant__ vec2f dev_vol_origin_2d[num_projections_per_kernel_2d];
__constant__ vec2f dev_delta_x_2d[num_projections_per_kernel_2d];
__constant__ vec2f dev_delta_y_2d[num_projections_per_kernel_2d];

template <class T>
__global__ void backward_2d(T *volume, i64 vol_stride, vec2u vol_shape, cudaTextureObject_t tex,
                            u64 det_shape, i64 cur_projection, i64 nprojections, vec2f source,
                            f32 DSO, f32 DSD) {
    const auto idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto start_y = blockIdx.y * num_voxels_per_thread_2d + threadIdx.y;

    if (idx_x >= vol_shape[0] || start_y >= vol_shape[1]) {
        return;
    }

    f32 local_volume[num_voxels_per_thread_2d];

    // load volume into local_volume
#pragma unroll
    for (i64 y = 0; y < num_voxels_per_thread_2d; y++) {
        const auto idx_y = start_y + y;

        if (idx_y >= vol_shape[1]) {
            break;
        }
        local_volume[y] = volume[idx_x + idx_y * vol_stride];
    }

    // Do the thing
    for (int proj = 0; proj < num_projections_per_kernel_2d; ++proj) {
        auto idx_proj = cur_projection + proj;

        if (idx_proj >= nprojections) {
            break;
        }

        const auto vol_origin = dev_vol_origin_2d[proj];
        const auto delta_x = dev_delta_x_2d[proj];
        const auto delta_y = dev_delta_y_2d[proj];

#pragma unroll
        for (i64 y = 0; y < num_voxels_per_thread_2d; y++) {
            const auto idx_y = start_y + y;

            if (idx_y >= vol_shape[1]) {
                break;
            }

            auto P = vol_origin + idx_x * delta_x + idx_y * delta_y;

            // Compute line from source to P
            auto dir = P - source;

            // Compute intersection of detector with dir
            auto t = __fdividef(DSO - DSD - source[1], dir[1]);

            // Coordinates are from [-det_shape / 2, det_shape / 2], hence shift it to be
            // strictly positive
            auto u = (dir[0] * t + source[0]) + det_shape / 2;

            auto sample = tex1DLayered<f32>(tex, u, proj);

            local_volume[y] += sample;
        }
    }

    // Write back local_volume to volume
#pragma unroll
    for (i64 y = 0; y < num_voxels_per_thread_2d; y++) {
        const auto idx_y = start_y + y;

        if (idx_y >= vol_shape[1]) {
            break;
        }

        volume[idx_x + idx_y * vol_stride] = local_volume[y];
    }
}

void setup_constants(vec2f vol_extent, vec2f vol_spacing, vec2f vol_offset,
                     std::vector<float> angles, u64 start_proj, u64 num_projections) {
    std::vector<curad::vec2f> vol_origins;
    std::vector<curad::vec2f> delta_xs;
    std::vector<curad::vec2f> delta_ys;

    for (int i = start_proj; i < start_proj + num_projections; ++i) {
        // TODO: check what am I still missing to make this feature complete? Check with tigre

        auto init_vol_origin = -vol_extent / 2.f + vol_spacing / 2.f + vol_offset;
        auto vol_origin = curad::geometry::rotate(init_vol_origin, angles[i]);
        vol_origins.push_back(vol_origin);

        auto init_delta = init_vol_origin;
        init_delta[0] += vol_spacing[0];
        init_delta = curad::geometry::rotate(init_delta, angles[i]);
        delta_xs.push_back(init_delta - vol_origin);

        init_delta = init_vol_origin;
        init_delta[1] += vol_spacing[1];
        init_delta = curad::geometry::rotate(init_delta, angles[i]);
        delta_ys.push_back(init_delta - vol_origin);
    }

    cudaMemcpyToSymbol(dev_vol_origin_2d, vol_origins.data(),
                       sizeof(curad::vec2f) * kernel::num_projections_per_kernel_2d, 0,
                       cudaMemcpyDefault);
    cudaMemcpyToSymbol(dev_delta_x_2d, delta_xs.data(),
                       sizeof(curad::vec2f) * kernel::num_projections_per_kernel_2d, 0,
                       cudaMemcpyDefault);
    cudaMemcpyToSymbol(dev_delta_y_2d, delta_ys.data(),
                       sizeof(curad::vec2f) * kernel::num_projections_per_kernel_2d, 0,
                       cudaMemcpyDefault);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
} // namespace kernel

class Texture {
  public:
    Texture(cudaResourceDesc texRes, cudaTextureDesc texDescr) {
        gpuErrchk(cudaCreateTextureObject(&handle_, &texRes, &texDescr, NULL));
    }

    cudaTextureObject_t handle() const { return handle_; }

    ~Texture() { gpuErrchk(cudaDestroyTextureObject(handle_)); }

  private:
    cudaTextureObject_t handle_;
};

template <class T, class U>
void backproject_2d(T *volume_ptr, vec2u vol_shape, vec2f vol_spacing, vec2f vol_offset,
                    vec2f vol_extent, U *sino_ptr, u64 det_shape, f32 DSD, f32 DSO, vec2f source,
                    std::vector<f32> angles) {
    const auto nangles = angles.size();

    // allocate cuarray with size of volume
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<f32>();
    cudaArray_t array;

    // IMPORTANT: height 0 required for 1Dlayered
    auto extentDesc = make_cudaExtent(det_shape, 0, kernel::num_projections_per_kernel_2d);
    gpuErrchk(cudaMalloc3DArray(&array, &channelDesc, extentDesc, cudaArrayLayered));
    gpuErrchk(cudaPeekAtLastError());

    // Create texture
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

    Texture tex(texRes, texDescr);

    const int num_kernel_calls =
        utils::round_up_division(nangles, kernel::num_projections_per_kernel_2d);
    for (int i = 0; i < num_kernel_calls; ++i) {
        const auto proj_idx = i * kernel::num_projections_per_kernel_2d;
        // std::cout << "kernel call: " << proj_idx << " / " << nangles << "\n";
        const auto num_projections_left = nangles - proj_idx;
        const auto num_projections =
            std::min<int>(kernel::num_projections_per_kernel_2d, num_projections_left);

        // Copy projection data necessary for the next kernel to cuda array
        const auto offset = proj_idx * det_shape;
        auto cur_proj_ptr = sino_ptr + offset;
        const auto size = det_shape * sizeof(U);
        cudaMemcpy3DParms mParams = {0};
        mParams.srcPtr = make_cudaPitchedPtr(cur_proj_ptr, det_shape * sizeof(U), det_shape, 1);
        mParams.kind = cudaMemcpyHostToDevice;

        // Important! non-zero height required for memcpy to do anything
        mParams.extent = make_cudaExtent(det_shape, 1, kernel::num_projections_per_kernel_2d);
        mParams.dstArray = array;
        gpuErrchk(cudaMemcpy3D(&mParams));
        gpuErrchk(cudaPeekAtLastError());

        // kernel uses variables stored in __constant__ memory, e.g. volume origin, volume
        // deltas Compute them here and upload them
        kernel::setup_constants(vol_extent, vol_spacing, vol_offset, angles, proj_idx,
                                num_projections);

        int divx = 16;
        int divy = kernel::num_voxels_per_thread_2d;

        dim3 threads_per_block(divx, 1);

        int block_x = utils::round_up_division(vol_shape[0], divx);
        int block_y = utils::round_up_division(vol_shape[1], divy);
        dim3 num_blocks(block_x, block_y);
        kernel::backward_2d<<<num_blocks, threads_per_block>>>(volume_ptr, vol_shape[0], vol_shape,
                                                               tex.handle(), det_shape, proj_idx,
                                                               nangles, source, DSD, DSO);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    cudaFreeArray(array);
}
} // namespace curad::bp
