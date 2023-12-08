#pragma once

#include "curadon/cuda/defines.h"
#include <cuda_runtime_api.h>

#include <iostream>

namespace curad {

class stream_view {
  public:
    constexpr stream_view() = default;
    constexpr stream_view(stream_view const &) = default;
    constexpr stream_view(stream_view &&) = default;
    constexpr stream_view &operator=(stream_view const &) = default;
    constexpr stream_view &operator=(stream_view &&) = default;
    ~stream_view() = default;

    constexpr stream_view(int) = delete;
    constexpr stream_view(std::nullptr_t) = delete;

    constexpr stream_view(cudaStream_t stream) noexcept : stream_{stream} {}

    [[nodiscard]]
    constexpr cudaStream_t value() const noexcept {
        return stream_;
    }

    constexpr operator cudaStream_t() const noexcept { return value(); }

    [[nodiscard]]
    inline bool is_default() const noexcept;

    void synchronize() const { CURADON_CUDA_TRY(cudaStreamSynchronize(stream_)); }

    void synchronize_no_throw() const noexcept {
        CURADON_ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream_));
    }

  private:
    cudaStream_t stream_{};
};

inline bool operator==(stream_view lhs, stream_view rhs) { return lhs.value() == rhs.value(); }

inline bool operator!=(stream_view lhs, stream_view rhs) { return not(lhs == rhs); }

inline std::ostream &operator<<(std::ostream &os, stream_view stream) {
    os << stream.value();
    return os;
}

static constexpr stream_view default_stream{};

static const stream_view legacy_stream{cudaStreamLegacy};

static const stream_view per_thread_stream{cudaStreamPerThread};

[[nodiscard]]
inline bool stream_view::is_default() const noexcept {
    return value() == legacy_stream.value() || value() == nullptr;
}

} // namespace curad
