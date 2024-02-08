#pragma once

#include "curadon/detail/device_span.hpp"
#include "curadon/detail/error.h"
#include "curadon/detail/utils.hpp"
#include "curadon/pool.hpp"
#include "curadon/types.hpp"
#include "curadon/types_half.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstring>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace curad {
enum class precision {
    HALF = 16,
    SINGLE = 32,
};

struct texture_config {
    texture_config() = default;
    texture_config(const texture_config &other) = default;
    texture_config(texture_config &&other) = default;

    usize device_id;

    u64 width;
    u64 height;
    u64 depth;

    bool is_layered;

    precision precision_;

    bool operator==(const texture_config &other) const {
        return device_id == other.device_id && depth == other.depth && height == other.height &&
               width == other.width && is_layered == other.is_layered &&
               precision_ == other.precision_;
    }
};

namespace detail {
__inline__ cudaChannelFormatDesc get_channel_desc(precision p) {
    if (p == precision::SINGLE) {
        return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    } else if (p == precision::HALF) {
        return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
    }
    // TODO error
    throw "Unknown precision";
}
} // namespace detail

struct texture {
  public:
    texture() = default;

    explicit texture(const texture_config &config)
        : config_(config) {

        auto channelDesc = detail::get_channel_desc(config_.precision_);

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

        // Create surface object associated with the same cudaArray
        gpuErrchk(cudaCreateSurfaceObject(&surface_, &resDesc));
    }

    texture(const texture &) = delete;
    texture &operator=(const texture &) = delete;

    template <class T>
    void write_1dlayered(T *data, u64 width, u64 nlayers, cuda::stream_view stream) {
        write(data, width, 1, nlayers, stream);
    }

    template <class T>
    void write(T *data, u64 width, u64 height, u64 depth, cuda::stream_view stream) {
        if (config_.precision_ != precision::SINGLE) {
            // Something wrong, what do I do?
        }

        // if using a single channel use cudaMemcpy to copy data into array
        cudaMemcpy3DParms cpy_params = {0};

        cpy_params.srcPos = make_cudaPos(0, 0, 0);
        cpy_params.dstPos = make_cudaPos(0, 0, 0);

        const auto size = width * sizeof(T);

        cpy_params.srcPtr = make_cudaPitchedPtr(data, size, width, std::max<u64>(height, 1));
        cpy_params.dstArray = this->storage_;

        cpy_params.extent =
            make_cudaExtent(width, std::max<u64>(height, 1), std::max<u64>(depth, 1));

        // cpy_params.kind = cudaMemcpyDefault;
        cpy_params.kind = cudaMemcpyDeviceToDevice;
        gpuErrchk(cudaMemcpy3DAsync(&cpy_params, stream));
    }

    template <class T>
    void write_2d(T *data, u64 width, u64 height, cuda::stream_view stream) {
        auto pitch = width * sizeof(T);
        auto width_bytes = width * sizeof(T);

        // TODO: this isn't working for all cases yet, but for
        gpuErrchk(cudaMemcpy2DToArrayAsync(storage_, 0, 0, data, pitch, width_bytes, height,
                                           cudaMemcpyDefault, stream));
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
