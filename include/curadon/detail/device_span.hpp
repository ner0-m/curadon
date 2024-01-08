#pragma once

#include "curadon/types.hpp"
#include "curadon/detail/vec.hpp"

namespace curad {

/// Simple std::span like abstraction, but way less features and kernel ready
template <class T>
class device_span {
  public:
    using value_type = T;
    using size_type = u64;
    using difference_type = i64;
    using pointer = T *;
    using const_pointer = T const *;
    using reference = T &;
    using const_reference = T const &;

    __host__ __device__ device_span(T *data, size_type size)
        : data_(data)
        , size_(size) {}

    __host__ __device__ size_type size() const { return size_; }

    __host__ __device__ u64 nbytes() const { return sizeof(T) * size_; }

    __host__ __device__ pointer data() { return data_; }

    __host__ __device__ const_pointer data() const { return data_; }

    __device__ reference operator[](size_type i) { return data_[i]; }

    __device__ const_reference operator[](size_type i) const { return data_[i]; }

    __host__ __device__ device_span subspan(size_type offset, size_type size) {
        return device_span(data_ + offset, size);
    }

  private:
    pointer data_;
    size_type size_;
};

/// Non-owning span over 3D data stored in device memory, mostly useful for
/// handing something to the kernel
template <class T>
class device_span_3d {
  public:
    using value_type = T;
    using size_type = u64;
    using strides_type = i64;
    using difference_type = i64;
    using pointer = T *;
    using const_pointer = T const *;
    using reference = T &;
    using const_reference = T const &;

    static constexpr int Dim = 3;

    device_span_3d(pointer data, vec<size_type, Dim> shape)
        : data_(data, shape.hprod())
        , shape_(shape) {

        // By default assume row-major
        strides_type running_size = 1;
        for (u64 i = 0; i < Dim; ++i) {
            strides_[i] = running_size;
            running_size = strides_[i] * static_cast<strides_type>(shape_[i]);
        }
    }

    __host__ __device__ size_type ndim() const { return Dim; }

    __host__ __device__ size_type size() const { return data_.size(); }

    __host__ __device__ size_type nbytes() const { return data_.nbytes(); }

    __host__ __device__ pointer device_data() { return data_.data(); }

    __host__ __device__ const_pointer device_data() const { return data_.data(); }

    __host__ __device__ vec<size_type, Dim> shape() const { return shape_; }

    __host__ __device__ vec<strides_type, Dim> strides() const { return strides_; }

    __device__ reference operator()(size_type x, size_type y, size_type z) {
        return data_[x * strides_[0] + y * strides_[1] + z * strides_[2]];
    }

    __device__ const_reference operator()(size_type x, size_type y, size_type z) const {
        return data_[x * strides_[0] + y * strides_[1] + z * strides_[2]];
    }

  private:
    device_span<value_type> data_;

    vec<size_type, Dim> shape_;

    vec<strides_type, Dim> strides_;
};
} // namespace curad
