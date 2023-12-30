#pragma once

#include "curadon/math/vector.hpp"
#include <thrust/device_vector.h>
namespace curad {

/// Simple std::span like abstraction, but way less features and kernel ready
template <class T>
class device_span {
  public:
    using value_type = T;
    using size_type = std::uint64_t;
    using difference_type = std::int64_t;
    using pointer = T *;
    using const_pointer = T const *;
    using reference = T &;
    using const_reference = T const &;

    __host__ __device__ device_span(T *data, size_type size)
        : data_(data)
        , size_(size) {}

    __host__ __device__ size_type size() const { return size_; }

    __host__ __device__ std::uint64_t nbytes() const { return sizeof(T) * size_; }

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
    using size_type = std::uint64_t;
    using strides_type = std::int64_t;
    using difference_type = std::int64_t;
    using pointer = T *;
    using const_pointer = T const *;
    using reference = T &;
    using const_reference = T const &;

    static constexpr int Dim = 3;

    device_span_3d(pointer data, Vec<size_type, Dim> shape)
        : data_(data, shape.hprod())
        , shape_(shape) {

        // By default assume row-major
        strides_type running_size = 1;
        for (std::uint64_t i = 0; i < Dim; ++i) {
            strides_[i] = running_size;
            running_size = strides_[i] * static_cast<strides_type>(shape_[i]);
        }
    }

    __host__ __device__ size_type ndim() const { return Dim; }

    __host__ __device__ size_type size() const { return data_.size(); }

    __host__ __device__ size_type nbytes() const { return data_.nbytes(); }

    __host__ __device__ pointer device_data() { return data_.data(); }

    __host__ __device__ const_pointer device_data() const { return data_.data(); }

    __host__ __device__ Vec<size_type, Dim> shape() const { return shape_; }

    __host__ __device__ Vec<strides_type, Dim> strides() const { return strides_; }

    __device__ reference operator()(size_type x, size_type y, size_type z) {
        return data_[x * strides_[0] + y * strides_[1] + z * strides_[2]];
    }

    __device__ const_reference operator()(size_type x, size_type y, size_type z) const {
        return data_[x * strides_[0] + y * strides_[1] + z * strides_[2]];
    }

  private:
    device_span<value_type> data_;

    Vec<size_type, Dim> shape_;

    Vec<strides_type, Dim> strides_;
};

/// Non-onwing span over 3D volume stored in device memory
template <class T>
class device_volume {
  public:
    static constexpr int Dim = 3;

    device_volume(T *data, Vec<std::uint64_t, Dim> shape)
        : device_volume<T>(data, shape, Vec<float, Dim>::ones()) {}

    device_volume(T *data, Vec<std::uint64_t, Dim> shape, Vec<float, Dim> spacing)
        : device_volume<T>(data, shape, Vec<float, Dim>::ones(), Vec<float, Dim>::zeros()) {}

    device_volume(T *data, Vec<std::uint64_t, Dim> shape, Vec<float, Dim> spacing,
                  Vec<float, Dim> offset)
        : data_(data, shape)
        , spacing_(spacing)
        , extent_(shape * spacing_)
        , offset_(offset) {}

    std::uint64_t ndim() const { return data_.ndim(); }

    std::uint64_t size() const { return data_.size(); }

    std::uint64_t nbytes() const { return data_.nbytes(); }

    T *device_data() { return data_.device_data(); }

    T const *device_data() const { return data_.device_data; }

    Vec<std::uint64_t, Dim> shape() const { return data_.shape(); }

    Vec<float, Dim> spacing() const { return spacing_; }

    Vec<float, Dim> extent() const { return extent_; }

    Vec<float, Dim> offset() const { return offset_; }

    device_span_3d<T> kernel_span() { return data_; }

  private:
    device_span_3d<T> data_;

    Vec<float, Dim> spacing_;

    Vec<float, Dim> extent_;

    Vec<float, Dim> offset_;
};

/// Non-onwing span over stack of 2D x-ray projections stored in device memory
/// TODO: This assumes a single detector and source for the completely measurement, relax it at some
/// point
/// TODO: as long as we store thrust::device_vectors here, we can not use this
/// in kernels, which I might want to do? I'm not sure yet
// template <class T>
// class device_measurement {
//   public:
//     static constexpr int Dim = 3;
//     static constexpr int DetectorDim = 2;
//
//     device_measurement(T *data, Vec<std::uint64_t, DetectorDim> det_shape, float DSD, float DSO,
//                        thrust::device_vector<float> phi)
//         : data_(data, {det_shape.x(), det_shape.y(), phi.size()})
//         , det_shape_(det_shape)
//         , det_spacing_(Vec<float, DetectorDim>::ones())
//         , det_offset_(Vec<float, DetectorDim>::zeros())
//         , nangles_(psi.size())
//         , DSD(DSD)
//         , DSO(DSO)
//         , COR(0)
//         , phi_(phi)
//         , theta_()
//         , psi_()
//         , pitch_()
//         , roll_()
//         , yaw_() {}
//
//     __host__ __device__ std::uint64_t projection_dim() const { return DetectorDim; }
//
//     // TODO: Maybe rename, to indicate this is for the complete data
//     __host__ __device__ std::uint64_t size() const { return data_.size(); }
//
//     __host__ __device__ Vec<std::uint64_t, Dim> shape() const { return data_.shape(); }
//
//     __host__ __device__ std::uint64_t nbytes() const { return data_.nbytes(); }
//
//     __host__ __device__ std::uint64_t proj_size() const { return det_shape_.hprod(); }
//
//     __host__ __device__ std::uint64_t nangles() const { return nangles_; }
//
//     __host__ __device__ Vec<float, Dim> det_shape() const { return det_shape_; }
//
//     __host__ __device__ Vec<float, Dim> det_spacing() const { return spacing_; }
//
//     __host__ __device__ Vec<float, Dim> det_extend() const { return extend_; }
//
//     __host__ __device__ float distance_source_detector() const { return DSD; }
//
//     __host__ __device__ float distance_source_object() const { return DSO; }
//
//     __host__ __device__ float distance_source_detecetor() const { return DSD - DSO; }
//
//     // TODO: Maybe remove this, but conviniente if simple circular trajectory
//     __host__ thrust::device_vector<float> angles() const { return phi_; }
//
//     __host__ thrust::device_vector<float> phi() const { return phi_; }
//
//     __host__ thrust::device_vector<float> theta() const { return theta_; }
//
//     __host__ thrust::device_vector<float> psi() const { return psi_; }
//
//     __host__ __device__ T *device_data() { return data_; }
//
//     __host__ __device__ T const *device_data() const { return data_; }
//
//     __host__ __device T *device_proj_data(std::uint64_t proj) {
//         return data_.data() + proj_size() * proj;
//     }
//
//     __host__ __device T const *device_proj_data(std::uint64_t proj) const {
//         return data_.data() + proj_size() * proj;
//     }
//
//     __device__ T &operator()(std::uint64_t u, std::uint64_t v, std::uint64_t proj) {
//         return data_(u, v, proj);
//     }
//
//     __device__ const T &operator()(std::uint64_t u, std::uint64_t v, std::uint64_t proj) const {
//         return data_(u, v, proj);
//     }
//
//   private:
//     device_span_3d<T> data_;
//
//     Vec<std::uint64_t, DetectorDim> det_shape_;
//
//     Vec<float, DetectorDim> det_spacing_;
//
//     Vec<float, DetectorDim> det_extent_;
//
//     Vec<float, Dim> det_offset_;
//
//     // Geometry information about measurements
//     std::uint64_t nangles_;
//
//     /// Distance source to detector
//     float DSD;
//
//     /// Distance source to object / center of rotation
//     float DSO;
//
//     /// Center of rotation correction (TODO: actually use it)
//     float COR = 0;
//
//     // euler angles in radian, TODO: change to device_uvector
//     thrust::device_vector<float> phi_;
//     thrust::device_vector<float> theta_; // TODO: expose
//     thrust::device_vector<float> psi_;   // TODO: expose
//
//     // Detector rotation in radians, TODO: maybe make this a vector? TODO: expose
//     float pitch_;
//     float roll_;
//     float yaw_;
// };
} // namespace curad
