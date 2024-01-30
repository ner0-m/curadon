#pragma once

#include <atomic>
#include <ostream>
#include <vector>

#include <cuda_runtime_api.h>

#include "curadon/detail/error.h"
#include "curadon/types.hpp"

namespace curad::cuda {

class stream_view;

class event {
  public:
    // Unique ownership => delete copy
    event(event const &) = delete;
    event(event &&) = default;
    event &operator=(event const &) = delete;
    event &operator=(event &&) = default;

    event() { gpuErrchk(cudaEventCreate(&event_)); }
    ~event() { gpuErrchk(cudaEventDestroy(event_)); }

    [[nodiscard]]
    cudaEvent_t value() const noexcept {
        return event_;
    }

    operator cudaEvent_t() const noexcept { return value(); }

    void record(stream_view stream) const noexcept;

  private:
    cudaEvent_t event_{};
};

class event_view {
  public:
    event_view() = default;
    event_view(event_view const &) = default;
    event_view(event_view &&) = default;
    event_view &operator=(event_view const &) = default;
    event_view &operator=(event_view &&) = default;
    ~event_view() = default;

    event_view(cudaEvent_t event) noexcept
        : event_{event} {}

    [[nodiscard]]
    cudaEvent_t value() const noexcept {
        return event_;
    }

    operator cudaEvent_t() const noexcept { return value(); }

    void record(stream_view stream) const noexcept;

  private:
    cudaEvent_t event_{};
};

class stream {
  public:
    // Unique ownership => delete copy
    stream(stream const &) = delete;
    stream(stream &&) = default;
    stream &operator=(stream const &) = delete;
    stream &operator=(stream &&) = default;

    stream() noexcept { gpuErrchk(cudaStreamCreate(&stream_)); }
    ~stream() { gpuErrchk(cudaStreamDestroy(stream_)); }

    [[nodiscard]]
    cudaStream_t value() const noexcept {
        return stream_;
    }

    operator cudaStream_t() const noexcept { return value(); }

    void synchronize() const { gpuErrchk(cudaStreamSynchronize(stream_)); }

    void wait_for_event(event_view const &event) const {
        gpuErrchk(cudaStreamWaitEvent(stream_, event.value()));
    }

  private:
    cudaStream_t stream_{};
};

class stream_view {
  public:
    stream_view() = default;
    stream_view(stream_view const &) = default;
    stream_view(stream_view &&) = default;
    stream_view &operator=(stream_view const &) = default;
    stream_view &operator=(stream_view &&) = default;
    ~stream_view() = default;

    stream_view(cudaStream_t stream) noexcept
        : stream_{stream} {}

    [[nodiscard]]
    cudaStream_t value() const noexcept {
        return stream_;
    }

    operator cudaStream_t() const noexcept { return value(); }

    void synchronize() const { gpuErrchk(cudaStreamSynchronize(stream_)); }

    void wait_for_event(event_view const &event) const {
        gpuErrchk(cudaStreamWaitEvent(stream_, event.value()));
    }

  private:
    cudaStream_t stream_{};
};

inline void event::record(stream_view stream) const noexcept {
    gpuErrchk(cudaEventRecord(event_, stream.value()));
}

inline void event_view::record(stream_view stream) const noexcept {
    gpuErrchk(cudaEventRecord(event_, stream.value()));
}

static constexpr stream_view cuda_stream_default{};

static const stream_view cuda_stream_legacy{cudaStreamLegacy};

static const stream_view cuda_stream_per_thread{cudaStreamPerThread};

class stream_pool {
  public:
    static constexpr std::size_t default_size{16};

    // This should only be used statically!!
    stream_pool(stream_pool &&) = delete;
    stream_pool(stream_pool const &) = delete;
    stream_pool &operator=(stream_pool &&) = delete;
    stream_pool &operator=(stream_pool const &) = delete;

    explicit stream_pool(usize pool_size = default_size)
        : events_(pool_size) {}

    ~stream_pool() = default;

    [[nodiscard]]
    stream_view get_next_stream() const noexcept {
        return stream_view{events_[(index_++) % events_.size()]};
    }

    [[nodiscard]]
    stream_view get_stream(std::size_t idx) const {
        return stream_view{events_[idx % events_.size()]};
    }

  private:
    std::vector<stream> events_;
    mutable std::atomic_size_t index_{0};
};

class event_pool {
  public:
    static constexpr std::size_t default_size{16};

    event_pool(event_pool &&) = delete;
    event_pool(event_pool const &) = delete;
    event_pool &operator=(event_pool &&) = delete;
    event_pool &operator=(event_pool const &) = delete;

    explicit event_pool(std::size_t pool_size = default_size)
        : events_(pool_size) {}

    ~event_pool() = default;

    [[nodiscard]]
    event_view get_next_event() const noexcept {
        return event_view{events_[(index_++) % events_.size()]};
    }

    [[nodiscard]]
    event_view get_event(std::size_t idx) const {
        return event_view{events_[idx % events_.size()]};
    }

  private:
    std::vector<event> events_;
    mutable std::atomic_size_t index_{0};
};

namespace detail {
inline stream_pool &instance() {
    static stream_pool pool{};
    return pool;
}
} // namespace detail

[[nodiscard]]
inline stream_view get_next_stream() {
    return detail::instance().get_next_stream();
}

[[nodiscard]]
inline stream_view get_stream(std::size_t idx) {
    return detail::instance().get_stream(idx);
}

namespace detail {
inline event_pool &event_pool_instance() {
    static event_pool pool{};
    return pool;
}
} // namespace detail

[[nodiscard]]
inline event_view get_next_event() {
    return detail::event_pool_instance().get_next_event();
}

[[nodiscard]]
inline event_view get_event(std::size_t idx) {
    return detail::event_pool_instance().get_event(idx);
}

} // namespace curad::cuda
