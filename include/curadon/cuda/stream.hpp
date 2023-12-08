#pragma once

#include <cuda_runtime_api.h>

#include "curadon/cuda/defines.h"
#include "curadon/cuda/stream_view.hpp"

namespace curad {
class stream {
  public:
    stream() : stream_() { CURADON_CUDA_TRY(cudaStreamCreate(&stream_)); }

    stream(const stream &other) = delete;
    stream &operator=(const stream &other) = delete;

    stream(stream &&other) noexcept = default;
    stream &operator=(stream &&other) noexcept = default;

    ~stream() { CURADON_ASSERT_CUDA_SUCCESS(cudaStreamDestroy(value())); }

    [[nodiscard]]
    bool is_valid() const noexcept {
        return value() != nullptr;
    }

    [[nodiscard]]
    stream_view view() const noexcept {
        return stream_view(value());
    }

    operator stream_view() const noexcept { return view(); }

    [[nodiscard]]
    cudaStream_t value() const noexcept {
        return stream_;
    }

    void synchronize() const { CURADON_CUDA_TRY(cudaStreamSynchronize(value())); }

  private:
    cudaStream_t stream_;
};
} // namespace curad
