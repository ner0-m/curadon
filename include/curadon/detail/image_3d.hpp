#pragma once

#include "curadon/detail/image_nd.hpp"
#include "curadon/detail/vec.hpp"
#include "curadon/types.hpp"

namespace curad {
/// Non-onwing span over 3D volume stored in device memory
template <class T>
class device_volume : private detail::device_volume_nd<T, 3> {
    using B = detail::device_volume_nd<T, 3>;

  public:
    static constexpr int Dim = 3;

    device_volume(usize device, T *data, vec<u64, Dim> shape)
        : B(device, data, shape, vec<f32, Dim>::ones()) {}

    device_volume(usize device, T *data, vec<u64, Dim> shape, vec<f32, Dim> spacing)
        : B(device, data, shape, vec<f32, Dim>::ones(), vec<f32, Dim>::zeros()) {}

    device_volume(usize device, T *data, vec<u64, Dim> shape, vec<f32, Dim> spacing,
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
    using B::strides;
};

} // namespace curad
