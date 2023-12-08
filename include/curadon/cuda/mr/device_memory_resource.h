#pragma once

#include <cassert>

#include "curadon/cuda/defines.h"
#include "curadon/cuda/device.hpp"
#include "curadon/cuda/stream_view.hpp"

#include <map>
#include <mutex>

namespace curad {
namespace detail {
constexpr bool is_pow2(std::size_t value) { return (0 == (value & (value - 1))); }

constexpr bool is_supported_alignment(std::size_t alignment) { return is_pow2(alignment); }

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    assert(is_supported_alignment(alignment));
    return (value + (alignment - 1)) & ~(alignment - 1);
}
} // namespace detail

namespace mr {
class device_memory_resource {
  public:
    device_memory_resource() = default;
    device_memory_resource(const device_memory_resource &) = default;
    device_memory_resource &operator=(const device_memory_resource &) = default;
    device_memory_resource(device_memory_resource &&) = default;
    device_memory_resource &operator=(device_memory_resource &&) = default;
    ~device_memory_resource() = default;

    void *allocate(std::size_t bytes, stream_view stream = stream_view{}) {
        return do_allocate(detail::align_up(bytes, allocation_size_alignment), stream);
    }

    void deallocate(void *ptr, std::size_t bytes, stream_view stream = stream_view{}) {
        do_deallocate(ptr, detail::align_up(bytes, allocation_size_alignment), stream);
    }

  private:
    static constexpr auto allocation_size_alignment = std::size_t{8};

    virtual void *do_allocate(std::size_t bytes, stream_view stream) = 0;

    virtual void do_deallocate(void *ptr, std::size_t bytes, stream_view stream) = 0;

    [[nodiscard]]
    virtual bool do_is_equal(device_memory_resource const &other) const noexcept {
        return this == &other;
    }

    [[nodiscard]]
    virtual std::pair<std::size_t, std::size_t> do_get_mem_info(stream_view stream) const = 0;
};

class stream_ordered_memory_resource final : public device_memory_resource {
  public:
  private:
    void *do_allocate(std::size_t bytes, stream_view stream) override {
        void *ptr{nullptr};
        CURADON_CUDA_TRY_ALLOC(cudaMallocAsync(&ptr, bytes, stream.value()));
        return ptr;
    }

    void do_deallocate(void *ptr, std::size_t, stream_view stream) override {
        CURADON_ASSERT_CUDA_SUCCESS(cudaFreeAsync(ptr, stream.value()));
    }

    [[nodiscard]]
    std::pair<std::size_t, std::size_t> do_get_mem_info(stream_view stream) const override {
        return std::make_pair(0, 0);
    }
};

namespace detail {

inline device_memory_resource *initial_resource() {
    static stream_ordered_memory_resource mr{};
    return &mr;
}

inline std::mutex &map_lock() {
    static std::mutex map_lock;
    return map_lock;
}

// Must have default visibility, see: https://github.com/rapidsai/rmm/issues/826
__attribute__((visibility("default"))) inline auto &get_map() {
    static std::map<cuda_device_id::value_type, device_memory_resource *> device_id_to_resource;
    return device_id_to_resource;
}

} // namespace detail

inline device_memory_resource *get_per_device_resource(cuda_device_id device_id) {
    std::lock_guard<std::mutex> lock{detail::map_lock()};
    auto &map = detail::get_map();
    // If a resource was never set for `id`, set to the initial resource
    auto const found = map.find(device_id.value());
    return (found == map.end()) ? (map[device_id.value()] = detail::initial_resource())
                                : found->second;
}

inline device_memory_resource *get_current_device_resource() {
    return get_per_device_resource(curad::current_device());
}

} // namespace mr
} // namespace curad
