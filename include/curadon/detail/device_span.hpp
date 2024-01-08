#pragma once

#include "curadon/detail/vec.hpp"
#include "curadon/types.hpp"

namespace curad {
namespace detail {
template <i64 Idx, i64 Dim, class Arg>
__host__ __device__ u64 to_index_recurse(vec<i64, Dim> strides, Arg arg) {
    return strides[Idx] * arg;
}
template <i64 Idx, i64 Dim, class Arg, class... Args>
__host__ __device__ u64 to_index_recurse(vec<i64, Dim> strides, Arg arg, Args... args) {
    return strides[Idx] * arg + to_index_recurse<Idx + 1, Dim>(strides, args...);
}
template <i64 Dim, class... Args>
__host__ __device__ u64 to_index(vec<i64, Dim> strides, Args... args) {
    static_assert(Dim == sizeof...(args));
    return to_index_recurse<0, Dim>(strides, args...);
}
} // namespace detail

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
template <class T, i64 Dim>
class device_span_nd {
  public:
    using value_type = T;
    using size_type = u64;
    using strides_type = i64;
    using difference_type = i64;
    using pointer = T *;
    using const_pointer = T const *;
    using reference = T &;
    using const_reference = T const &;

    device_span_nd(pointer data, vec<size_type, Dim> shape)
        : data_(data, shape.hprod())
        , shape_(shape) {
        // By default assume row-major
        strides_type running_size = 1;
        for (u64 i = 0; i < Dim; ++i) {
            strides_[i] = running_size;
            running_size = strides_[i] * static_cast<strides_type>(shape_[i]);
        }
    }

    device_span_nd(pointer data, vec<size_type, Dim> shape, vec<strides_type, Dim> strides)
        : data_(data, shape.hprod())
        , shape_(shape)
        , strides_(strides) {}

    __host__ __device__ size_type ndim() const { return Dim; }

    __host__ __device__ size_type size() const { return data_.size(); }

    __host__ __device__ size_type nbytes() const { return data_.nbytes(); }

    __host__ __device__ pointer device_data() { return data_.data(); }

    __host__ __device__ const_pointer device_data() const { return data_.data(); }

    __host__ __device__ vec<size_type, Dim> shape() const { return shape_; }

    __host__ __device__ vec<strides_type, Dim> strides() const { return strides_; }

    template <i64 D = Dim, typename std::enable_if_t<D == 3, int> = 0>
    __device__ reference operator()(size_type x, size_type y, size_type z) {
        return data_[detail::to_index(strides(), x, y, z)];
    }

    template <i64 D = Dim, typename std::enable_if_t<D == 3, int> = 0>
    __device__ const_reference operator()(size_type x, size_type y, size_type z) const {
        return data_[detail::to_index(strides(), x, y, z)];
    }

    template <i64 D = Dim, typename std::enable_if_t<D == 2, int> = 0>
    __device__ reference operator()(size_type x, size_type y) {
        return data_[detail::to_index(strides(), x, y)];
    }

    template <i64 D = Dim, typename std::enable_if_t<D == 2, int> = 0>
    __device__ const_reference operator()(size_type x, size_type y) const {
        return data_[detail::to_index(strides(), x, y)];
    }

  private:
    device_span<value_type> data_;

    vec<size_type, Dim> shape_;

    vec<strides_type, Dim> strides_;
};

template <class T>
class device_span_3d : device_span_nd<T, 3> {
    using B = device_span_nd<T, 3>;

  public:
    using value_type = typename B::value_type;
    using size_type = typename B::size_type;
    using strides_type = typename B::strides_type;
    using difference_type = typename B::difference_type;
    using pointer = typename B::pointer;
    using const_pointer = typename B::const_pointer;
    using reference = typename B::reference;
    using const_reference = typename B::const_reference;

    device_span_3d(pointer data, vec<size_type, 3> shape)
        : B(data, shape) {}

    device_span_3d(pointer data, vec<size_type, 3> shape, vec<strides_type, 3> strides)
        : B(data, shape, strides) {}

    using B::device_data;
    using B::nbytes;
    using B::ndim;
    using B::shape;
    using B::size;
    using B::strides;
    using B::operator();
};

template <class T>
class device_span_2d : device_span_nd<T, 2> {
    using B = device_span_nd<T, 2>;

  public:
    using value_type = typename B::value_type;
    using size_type = typename B::size_type;
    using strides_type = typename B::strides_type;
    using difference_type = typename B::difference_type;
    using pointer = typename B::pointer;
    using const_pointer = typename B::const_pointer;
    using reference = typename B::reference;
    using const_reference = typename B::const_reference;

    device_span_2d(pointer data, vec<size_type, 2> shape)
        : B(data, shape) {}

    device_span_2d(pointer data, vec<size_type, 2> shape, vec<strides_type, 2> strides)
        : B(data, shape, strides) {}

    using B::device_data;
    using B::nbytes;
    using B::ndim;
    using B::shape;
    using B::size;
    using B::strides;
    using B::operator();
};

} // namespace curad
