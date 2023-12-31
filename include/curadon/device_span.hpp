#pragma once

#include "curadon/math/vector.hpp"
#include "curadon/span.hpp"

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

template <class T>
class device_measurement {
  public:
    static constexpr int Dim = 3;
    static constexpr int DetectorDim = 2;

    device_measurement(T *data, Vec<std::uint64_t, DetectorDim> shape, float DSD, float DSO,
                       std::vector<float> phi)
        : device_measurement(data, shape, Vec<float, DetectorDim>::ones(), DSD, DSO, phi) {}

    device_measurement(T *data, Vec<std::uint64_t, DetectorDim> shape,
                       Vec<float, DetectorDim> spacing, float DSD, float DSO,
                       std::vector<float> phi)
        : device_measurement(data, shape, Vec<float, DetectorDim>::ones(),
                             Vec<float, DetectorDim>::zeros(), DSD, DSO, phi) {}

    device_measurement(T *data, Vec<std::uint64_t, DetectorDim> shape,
                       Vec<float, DetectorDim> spacing, Vec<float, DetectorDim> offset, float DSD,
                       float DSO, std::vector<float> phi)
        : data_(data, {shape.x(), shape.y(), phi.size()})
        , spacing_(spacing)
        , offset_(offset)
        , extent_(shape * spacing_)
        , nangles_(phi.size())
        , DSD(DSD)
        , DSO(DSO)
        , COR(0)
        , phi_(phi)
        , theta_()
        , psi_()
        , pitch_(0)
        , roll_(0)
        , yaw_(0) {}

    std::uint64_t size() const { return data_.size(); }

    std::uint64_t nbytes() const { return data_.nbytes(); }

    T *device_data() { return data_.device_data(); }

    T const *device_data() const { return data_.device_data(); }

    Vec<std::uint64_t, Dim> shape() const { return data_.shape(); }

    Vec<std::uint64_t, DetectorDim> detector_shape() const {
        return {data_.shape()[0], data_.shape()[1]};
    }

    Vec<float, DetectorDim> spacing() const { return spacing_; }

    Vec<float, DetectorDim> extent() const { return extent_; }

    Vec<float, DetectorDim> offset() const { return offset_; }

    device_span_3d<T> kernel_span() { return data_; }

    std::uint64_t nangles() const { return nangles_; }

    float distance_source_to_detector() const { return DSD; }

    float distance_source_to_object() const { return DSO; }

    float distance_object_to_detector() const { return DSD - DSO; }

    span<float> angles() { return span<float>(phi_.data(), phi_.size()); }

    span<float> phi() { return span<float>(phi_.data(), phi_.size()); }

    span<float> theta() { return span<float>(theta_.data(), theta_.size()); }

    span<float> psi() { return span<float>(psi_.data(), psi_.size()); }

    float pitch() const { return pitch_; }

    float roll() const { return roll_; }

    float yaw() const { return yaw_; }

    Vec<float, Dim> source() const { return {0, 0, -distance_source_to_object()}; }

    device_span_3d<T> slice(std::uint64_t offset, std::uint64_t count = 1) {
        Vec<std::uint64_t, Dim> new_shape{shape()[0], shape()[1], count};
        auto ptr = device_data() + offset * data_.strides()[2];
        return device_span_3d<T>(ptr, new_shape);
    }

  private:
    device_span_3d<T> data_;

    Vec<float, DetectorDim> spacing_;

    Vec<float, DetectorDim> extent_;

    Vec<float, DetectorDim> offset_;

    // Geometry information about measurements
    std::uint64_t nangles_;

    /// Distance source to detector
    float DSD;

    /// Distance source to object / center of rotation
    float DSO;

    // euler angles in radian, TODO: change to device_uvector? But we only need it in the
    // pre-computing phase, so let's think about this a little more
    std::vector<float> phi_;
    std::vector<float> theta_; // TODO: enable in constructor
    std::vector<float> psi_;   // TODO: enable in constructor

    // Detector rotation in radians, TODO: maybe make this a vector?
    float pitch_;
    float roll_;
    float yaw_;

    /// Center of rotation correction (TODO: actually use it)
    float COR = 0;
};

} // namespace curad
