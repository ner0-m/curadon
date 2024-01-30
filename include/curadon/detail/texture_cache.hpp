#pragma once

#include <cstddef>
#include <cstring>
#include <functional>
#include <unordered_map>

#include <cuda.h>
#include <cuda_runtime.h>

#include "curadon/detail/error.h"
#include "curadon/types.hpp"

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
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<f32>();
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
        gpuErrchk(cudaPeekAtLastError());
    }

    // texture(const texture &) = delete;
    // texture &operator=(const texture &) = delete;

    // TODO: improve this, this should take an nd_span kind of thing
    void write(float *data) { this->write(data, config_.depth); }

    void write(float *data, u64 depth) {
        // if using a single channel use cudaMemcpy to copy data into array
        cudaMemcpy3DParms cpy_params = {0};

        cpy_params.srcPos = make_cudaPos(0, 0, 0);
        cpy_params.dstPos = make_cudaPos(0, 0, 0);

        cpy_params.srcPtr = make_cudaPitchedPtr(data, config_.width * sizeof(float), config_.width,
                                                std::max<u64>(config_.height, 1));
        cpy_params.dstArray = this->storage_;

        cpy_params.extent = make_cudaExtent(config_.width, std::max<u64>(config_.height, 1),
                                            std::max<u64>(depth, 1));

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
    // cudaSurfaceObject_t surface_ = 0;
};

using texture_cache_key = std::tuple<void *, texture_config>;
} // namespace curad

namespace curad::detail {
template <class T>
inline void hash_combine(std::size_t &seed, const T &v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
} // namespace curad::detail

namespace std {
template <>
struct std::hash<curad::texture_config> {
    std::size_t operator()(const curad::texture_config &k) const {
        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:
        auto seed = std::hash<::curad::usize>{}(k.device_id);
        ::curad::detail::hash_combine(seed, k.width);
        ::curad::detail::hash_combine(seed, k.height);
        ::curad::detail::hash_combine(seed, k.depth);
        ::curad::detail::hash_combine(seed, k.is_layered);

        return seed;
    }
};

template <>
struct std::hash<curad::texture_cache_key> {
    std::size_t operator()(const curad::texture_cache_key &k) const {
        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:
        auto seed = std::hash<void *>{}(std::get<0>(k));
        ::curad::detail::hash_combine(seed, std::get<1>(k));

        return seed;
    }
};
} // namespace std

namespace curad {
class texture_cache {
  public:
    texture_cache(u64 cache_size)
        : cache_(cache_size) {}

    bool try_emplace(texture_cache_key key, texture_config config) {
        auto [_, inserted] = cache_.try_emplace(std::move(key), std::move(config));
        return inserted;
    }

    texture &at(const texture_cache_key &key) { return cache_.at(key); }

    const texture &at(const texture_cache_key &key) const { return cache_.at(key); }

  private:
    std::unordered_map<texture_cache_key, texture> cache_;
};

inline texture_cache &get_texture_cache() {
    static texture_cache cache(10);
    return cache;
}
} // namespace curad
