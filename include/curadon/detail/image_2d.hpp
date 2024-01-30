#pragma once

#include "curadon/detail/image_nd.hpp"
#include "curadon/detail/vec.hpp"

namespace curad {
template <class T>
class image_2d : private detail::device_volume_nd<T, 2> {
    using B = detail::device_volume_nd<T, 2>;

  public:
    static constexpr int Dim = 2;

    image_2d(usize device, T *data, vec<u64, Dim> shape)
        : B(device, data, shape, vec<f32, Dim>::ones()) {}

    image_2d(usize device, T *data, vec<u64, Dim> shape, vec<f32, Dim> spacing)
        : B(device, data, shape, vec<f32, Dim>::ones(), vec<f32, Dim>::zeros()) {}

    image_2d(usize device, T *data, vec<u64, Dim> shape, vec<f32, Dim> spacing,
             vec<f32, Dim> offset)
        : B(device, data, shape, spacing, offset) {}

    using B::device_data;
    using B::device_id;
    using B::extent;
    using B::kernel_span;
    using B::nbytes;
    using B::ndim;
    using B::offset;
    using B::shape;
    using B::size;
    using B::spacing;
};
} // namespace curad
