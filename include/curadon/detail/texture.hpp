#pragma once

#include "curadon/detail/device_span.hpp"
#include "curadon/detail/error.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstring>
#include <cuda_runtime.h>

namespace curad {
struct texture_config {
    texture_config() = default;
    texture_config(const texture_config &other) = default;
    texture_config(texture_config &&other) = default;

    usize device_id;

    u64 width;
    u64 height;
    u64 depth;

    bool is_layered;

    // TODO: Need members for device, precision and channels

    bool operator==(const texture_config &other) const {
        return device_id == other.device_id && depth == other.depth && height == other.height &&
               width == other.width && is_layered == other.is_layered;
    }
};

struct texture {
  public:
    texture() = default;

    explicit texture(const texture_config &config)
        : config_(config) {
        cudaChannelFormatDesc channelDesc;
        channelDesc = cudaCreateChannelDesc<f32>();

        auto allocation_type = config_.is_layered ? cudaArrayLayered : cudaArrayDefault;

        const cudaExtent extent = make_cudaExtent(config_.width, config_.height, config_.depth);
        gpuErrchk(cudaMalloc3DArray(&storage_, &channelDesc, extent, allocation_type));

        // Create resource descriptor
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = storage_;

        // Specify texture object parameters
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));

        texDesc.addressMode[0] = cudaAddressModeBorder;
        texDesc.addressMode[1] = cudaAddressModeBorder;
        texDesc.addressMode[2] = cudaAddressModeBorder;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        gpuErrchk(cudaCreateTextureObject(&texture_, &resDesc, &texDesc, NULL));

        gpuErrchk(cudaCreateSurfaceObject(&surface_, &resDesc));
    }

    // texture(const texture &) = delete;
    // texture &operator=(const texture &) = delete;

    // TODO: improve this, this should take an nd_span kind of thing
    void write(f32 *data) { this->write(data, config_.depth); }

    void write(f32 *data, u64 depth) {
        // if using a single channel use cudaMemcpy to copy data into array
        cudaMemcpy3DParms cpy_params = {0};

        cpy_params.srcPos = make_cudaPos(0, 0, 0);
        cpy_params.dstPos = make_cudaPos(0, 0, 0);

        const auto width = config_.width;
        const auto size = width * sizeof(float);

        cpy_params.srcPtr =
            make_cudaPitchedPtr(data, size, width, std::max<u64>(config_.height, 1));
        cpy_params.dstArray = this->storage_;

        cpy_params.extent =
            make_cudaExtent(width, std::max<u64>(config_.height, 1), std::max<u64>(depth, 1));

        // cpy_params.kind = cudaMemcpyDefault;
        cpy_params.kind = cudaMemcpyDeviceToDevice;
        gpuErrchk(cudaMemcpy3D(&cpy_params));
    }

    ~texture() {
        if (texture_ != 0) {
            gpuErrchk(cudaDestroyTextureObject(texture_));
        }
        if (storage_ != nullptr) {
            gpuErrchk(cudaFreeArray(storage_));
        }
    }

    cudaTextureObject_t tex() const { return texture_; }

    texture_config config() const { return config_; }

  private:
    cudaArray_t storage_ = nullptr;
    cudaTextureObject_t texture_ = 0;
    texture_config config_;

    // TODO: torch-radon uses cudaSurfaceObject_t to write e.g. half-precision types
    cudaSurfaceObject_t surface_ = 0;
};
} // namespace curad
