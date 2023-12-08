#pragma once

#include "curadon/cuda/mr/device_memory_resource.h"
#include "curadon/cuda/stream_view.hpp"

namespace curad {
class device_buffer {
  public:
    device_buffer() = default;

    // default copies are deleted, as they have no info about a stream
    device_buffer(device_buffer const &) = delete;
    device_buffer &operator=(device_buffer const &) = delete;

    device_buffer(std::size_t size, stream_view stream,
                  mr::device_memory_resource *mr = mr::get_current_device_resource())
        : stream_(stream), mr_(mr) {
        allocate_async(size);
    }

    device_buffer(void const *data, std::size_t size, stream_view stream,
                  mr::device_memory_resource *mr = mr::get_current_device_resource())
        : stream_(stream), mr_(mr) {
        allocate_async(size);
        copy_async(data, size);
    }

    device_buffer(const device_buffer &other, stream_view stream,
                  mr::device_memory_resource *mr = mr::get_current_device_resource())
        : device_buffer(other.data(), other.size(), stream, mr) {}

    device_buffer(device_buffer &&other) noexcept
        : data_(other.data()), size_(other.size()), capacity_(other.capacity()),
          stream_(other.stream()), mr_(other.memory_resource()) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
        other.set_stream(stream_view());
        other.mr_ = nullptr;
    }

    device_buffer &operator=(device_buffer &&other) noexcept {
        if (this != &other) {
            deallocate_async();

            data_ = other.data();
            size_ = other.size();
            capacity_ = other.capacity();
            set_stream(other.stream());
            mr_ = other.memory_resource();

            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
            other.set_stream(stream_view());
            other.mr_ = nullptr;
        }
        return *this;
    }

    ~device_buffer() {
        deallocate_async();
        mr_ = nullptr;
        stream_ = stream_view();
    }

    [[nodiscard]]
    std::size_t size() const noexcept {
        return size_;
    }

    [[nodiscard]]
    std::size_t capacity() const noexcept {
        return capacity_;
    }

    [[nodiscard]]
    void *data() noexcept {
        return data_;
    }

    [[nodiscard]]
    void const *data() const noexcept {
        return data_;
    }

    [[nodiscard]]
    stream_view stream() const noexcept {
        return stream_;
    }

    void set_stream(stream_view stream) noexcept { stream_ = stream; }

    [[nodiscard]]
    mr::device_memory_resource *memory_resource() const noexcept {
        return mr_;
    }

    [[nodiscard]]
    bool is_empty() const noexcept {
        return 0 == size();
    }

    void reserve(std::size_t newcapacity, stream_view stream) {
        set_stream(stream);
        if (newcapacity > capacity()) {
            auto newbuffer = device_buffer(newcapacity, stream, memory_resource());
            const auto oldsize = size();
            CURADON_CUDA_TRY(cudaMemcpyAsync(newbuffer.data(), data(), size(), cudaMemcpyDefault,
                                             stream.value()));
            *this = std::move(newbuffer);
            size_ = oldsize;
        }
    }

    void resize(std::size_t newsize, stream_view stream) {
        set_stream(stream);
        if (newsize <= capacity()) {
            size_ = newsize;
        } else {
            auto newbuffer = device_buffer(newsize, stream, memory_resource());
            CURADON_CUDA_TRY(cudaMemcpyAsync(newbuffer.data(), data(), size(), cudaMemcpyDefault,
                                             stream.value()));
            *this = std::move(newbuffer);
        }
    }

  private:
    void *data_{nullptr};
    std::size_t size_{};
    std::size_t capacity_{};
    stream_view stream_{};
    mr::device_memory_resource *mr_{mr::get_current_device_resource()};

    void allocate_async(std::size_t nbytes) {
        size_ = nbytes;
        capacity_ = nbytes;
        data_ = (nbytes > 0) ? memory_resource()->allocate(nbytes, stream_) : nullptr;
    }

    void deallocate_async() {
        if (capacity_ > 0) {
            memory_resource()->deallocate(data(), capacity(), stream());
        }
        size_ = 0;
        capacity_ = 0;
        data_ = nullptr;
    }

    void copy_async(void const *src, std::size_t bytes) {
        if (bytes > 0) {
            CURADON_EXPECTS(nullptr != src, "Invalid copy from nullptr.");

            CURADON_CUDA_TRY(
                cudaMemcpyAsync(data(), src, bytes, cudaMemcpyDefault, stream().value()));
        }
    }
};
} // namespace curad
