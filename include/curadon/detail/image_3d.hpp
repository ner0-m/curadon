#pragma once

#include "curadon/detail/device_span.hpp"
#include "curadon/math/vector.hpp"
#include "curadon/types.hpp"

namespace curad {
/// Non-onwing span over 3D volume stored in device memory
template <class T>
class device_volume {
  public:
    static constexpr int Dim = 3;

    device_volume(T *data, vec<u64, Dim> shape)
        : device_volume<T>(data, shape, vec<f32, Dim>::ones()) {}

    device_volume(T *data, vec<u64, Dim> shape, vec<f32, Dim> spacing)
        : device_volume<T>(data, shape, vec<f32, Dim>::ones(), vec<f32, Dim>::zeros()) {}

    device_volume(T *data, vec<u64, Dim> shape, vec<f32, Dim> spacing, vec<f32, Dim> offset)
        : data_(data, shape)
        , spacing_(spacing)
        , extent_(shape * spacing_)
        , offset_(offset) {}

    u64 ndim() const { return data_.ndim(); }

    u64 size() const { return data_.size(); }

    u64 nbytes() const { return data_.nbytes(); }

    T *device_data() { return data_.device_data(); }

    T const *device_data() const { return data_.device_data; }

    vec<u64, Dim> shape() const { return data_.shape(); }

    vec<f32, Dim> spacing() const { return spacing_; }

    vec<f32, Dim> extent() const { return extent_; }

    vec<f32, Dim> offset() const { return offset_; }

    device_span_3d<T> kernel_span() { return data_; }

  private:
    device_span_3d<T> data_;

    vec<f32, Dim> spacing_;

    vec<f32, Dim> extent_;

    vec<f32, Dim> offset_;
};
} // namespace curad
