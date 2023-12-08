#pragma once

#include "curadon/cuda/defines.h"
#include "curadon/cuda/device_buffer.hpp"
#include "curadon/cuda/mr/device_memory_resource.h"
#include "curadon/cuda/stream_view.hpp"

#include <type_traits>

namespace curad {

template <class T>
class device_uvector {
    static_assert(std::is_trivially_copyable<T>::value,
                  "device_uvector only supports types that are trivially copyable.");

  public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = value_type &;
    using const_reference = value_type const &;
    using pointer = value_type *;
    using const_pointer = value_type const *;
    using iterator = pointer;
    using const_iterator = const_pointer;

    device_uvector() = default;
    ~device_uvector() = default;

    device_uvector(device_uvector const &) = delete;
    device_uvector &operator=(device_uvector const &) = delete;

    device_uvector(device_uvector &&) noexcept = default;
    device_uvector &operator=(device_uvector &&) noexcept = default;

    device_uvector(size_type size, stream_view stream,
                   mr::device_memory_resource *mr = mr::get_current_device_resource())
        : storage_(elements_to_bytes(size), stream, mr) {}

    device_uvector(const device_uvector &other, stream_view stream,
                   mr::device_memory_resource *mr = mr::get_current_device_resource())
        : storage_(other.storage_, stream, mr) {}

    [[nodiscard]]
    size_type size() const noexcept {
        return storage_.size() / sizeof(T);
    }

    [[nodiscard]]
    bool is_empty() const noexcept {
        return storage_.is_empty();
    }

    [[nodiscard]]
    mr::device_memory_resource *memory_resource() const noexcept {
        return storage_.memory_resource();
    }

    [[nodiscard]]
    stream_view stream() const noexcept {
        return storage_.stream();
    }

    void set_stream(stream_view stream) noexcept { storage_.set_stream(stream); }

    [[nodiscard]]
    pointer data() noexcept {
        return static_cast<pointer>(storage_.data());
    }

    [[nodiscard]]
    const_pointer data() const noexcept {
        return static_cast<const_pointer>(storage_.data());
    }

    [[nodiscard]]
    pointer element_ptr(std::size_t idx) noexcept {
        assert(idx < size());
        return data() + idx;
    }

    [[nodiscard]]
    const_pointer element_ptr(std::size_t idx) const noexcept {
        assert(idx < size());
        return data() + idx;
    }

    void set_element_async(std::size_t idx, const value_type &value, stream_view stream) {
        CURADON_EXPECTS(idx < size(), curad::out_of_range,
                        "Attempt to access out of bounds element.");

        CURADON_CUDA_TRY(cudaMemcpyAsync(element_ptr(idx), &value, sizeof(value), cudaMemcpyDefault,
                                         stream.value()));
    }

    // Delete the r-value reference overload to prevent asynchronously copying from a literal or
    // implicit temporary value after it is deleted or goes out of scope.
    void set_element_async(std::size_t, value_type const &&, stream_view) = delete;

    void set_element_to_zero_async(std::size_t idx, stream_view stream) {
        CURADON_EXPECTS(idx < size(), curad::out_of_range,
                        "Attempt to access out of bounds element.");
        CURADON_CUDA_TRY(cudaMemsetAsync(element_ptr(idx), 0, sizeof(value_type), stream.value()));
    }

    void set_element(std::size_t idx, const value_type &value, stream_view stream) {
        set_element_async(idx, value, stream);
        stream.synchronize_no_throw();
    }

    [[nodiscard]]
    value_type element(std::size_t idx, stream_view stream) const {
        CURADON_EXPECTS(idx < size(), curad::out_of_range,
                        "Attempt to access out of bounds element.");

        value_type value;
        CURADON_CUDA_TRY(cudaMemcpyAsync(&value, element_ptr(idx), sizeof(value), cudaMemcpyDefault,
                                         stream.value()));
        stream.synchronize();
        return value;
    }

    [[nodiscard]]
    value_type front_element(stream_view stream) const {
        return element(0, stream);
    }

    [[nodiscard]]
    value_type back_element(stream_view stream) const {
        return element(size() - 1, stream);
    }

    void reserve(size_type new_capacity, stream_view stream) {
        storage_.reserve(elements_to_bytes(new_capacity), stream);
    }

    void resize(size_type new_size, stream_view stream) {
        storage_.resize(elements_to_bytes(new_size), stream);
    }

    [[nodiscard]]
    iterator begin() noexcept {
        return data();
    }

    [[nodiscard]]
    const_iterator cbegin() const noexcept {
        return data();
    }

    [[nodiscard]]
    const_iterator begin() const noexcept {
        return cbegin();
    }

    [[nodiscard]]
    iterator end() noexcept {
        return data() + size();
    }

    [[nodiscard]]
    const_iterator cend() const noexcept {
        return data() + size();
    }

    [[nodiscard]]
    const_iterator end() const noexcept {
        return cend();
    }

  private:
    device_buffer storage_;

    size_type elements_to_bytes(size_type nelements) const {
        return nelements * sizeof(value_type);
    }

    size_type bytes_to_elements(size_type bytes) const { return bytes / sizeof(value_type); }
};
} // namespace curad
