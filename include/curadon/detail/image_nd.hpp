#pragma once

#include "curadon/detail/device_span.hpp"
#include "curadon/detail/vec.hpp"
#include "curadon/types.hpp"

namespace curad::detail {
template <class T, i64 Dim>
class device_volume_nd {
  public:
    device_volume_nd(usize device, T *data, vec<u64, Dim> shape)
        : device_volume_nd<T, Dim>(device, data, shape, vec<f32, Dim>::ones()) {}

    device_volume_nd(usize device, T *data, vec<u64, Dim> shape, vec<f32, Dim> spacing)
        : device_volume_nd<T, Dim>(device, data, shape, vec<f32, Dim>::ones(),
                                   vec<f32, Dim>::zeros()) {}

    device_volume_nd(usize device, T *data, vec<u64, Dim> shape, vec<f32, Dim> spacing,
                     vec<f32, Dim> offset)
        : data_(device, data, shape)
        , spacing_(spacing)
        , extent_(shape * spacing_)
        , offset_(offset) {}

    u64 ndim() const { return data_.ndim(); }

    u64 size() const { return data_.size(); }

    u64 nbytes() const { return data_.nbytes(); }

    T *device_data() { return data_.device_data(); }

    T const *device_data() const { return data_.device_data; }

    usize device_id() const { return data_.device_id(); }

    vec<u64, Dim> shape() const { return data_.shape(); }

    vec<f32, Dim> spacing() const { return spacing_; }

    vec<f32, Dim> extent() const { return extent_; }

    vec<f32, Dim> offset() const { return offset_; }

    auto strides() const { return data_.strides(); }

    template <i64 D = Dim, typename std::enable_if_t<D == 3, int> = 0>
    device_span_3d<T> kernel_span() {
        return device_span_3d<T>(device_id(), data_.device_data(), data_.shape(), data_.strides());
    }

    template <i64 D = Dim, typename std::enable_if_t<D == 2, int> = 0>
    device_span_2d<T> kernel_span() {
        return device_span_2d<T>(device_id(), data_.device_data(), data_.shape(), data_.strides());
    }

  private:
    device_span_nd<T, Dim> data_;

    vec<f32, Dim> spacing_;

    vec<f32, Dim> extent_;

    vec<f32, Dim> offset_;
};
} // namespace curad::detail
