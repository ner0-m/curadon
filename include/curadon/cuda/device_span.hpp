#pragma once

#include "curadon/cuda/device_uvector.hpp"
#include <cstddef>

namespace curad {
template <class T>
class device_span {
  public:
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;

    __host__ __device__ device_span(T *ptr, std::size_t size) : ptr_{ptr}, size_{size} {}

    __host__ device_span(curad::device_uvector<T> &x) : ptr_{x.data()}, size_{x.size()} {}

    __host__ __device__ std::size_t size() const { return size_; }

    __host__ __device__ pointer device_data() { return ptr_; }

    __host__ __device__ const_pointer device_data() const { return ptr_; }

    __host__ __device__ reference operator[](std::size_t i) { return ptr_[i]; }

    __host__ __device__ const_reference operator[](std::size_t i) const { return ptr_[i]; }

    __host__ __device__ device_span<value_type> subspan(std::size_t offset,
                                                        std::size_t length) const {
        return device_span(ptr_ + offset, length);
    }

  private:
    T *ptr_;
    std::size_t size_;
};

} // namespace curad
