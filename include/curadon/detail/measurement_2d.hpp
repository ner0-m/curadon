#pragma once

#include <vector>

#include "curadon/detail/device_span.hpp"
#include "curadon/detail/span.hpp"
#include "curadon/detail/vec.hpp"
#include "curadon/types.hpp"

namespace curad {
template <class T>
class measurement_2d {
  public:
    static constexpr int Dim = 2;

    measurement_2d(T *data, u64 det_shape, u64 nangles)
        : measurement_2d(data, det_shape, nangles, 1.f) {}

    measurement_2d(T *data, u64 det_shape, u64 nangles, f32 spacing)
        : measurement_2d(data, det_shape, nangles, spacing, 0.f) {}

    measurement_2d(T *data, u64 det_shape, u64 nangles, f32 spacing, f32 offset)
        : data_(data, vec2u{det_shape, nangles})
        , spacing_(spacing)
        , offset_(offset)
        , extent_(detector_shape() * this->spacing())
        , nangles_(nangles)
        , DSD(0)
        , DSO(0)
        , COR(0)
        , angles_(this->nangles(), 0)
        , pitch_(0) {}

    u64 size() const { return data_.size(); }

    u64 nbytes() const { return data_.nbytes(); }

    T *device_data() { return data_.device_data(); }

    T const *device_data() const { return data_.device_data(); }

    vec<u64, Dim> shape() const { return data_.shape(); }

    u64 detector_shape() const { return data_.shape()[0]; }

    f32 spacing() const { return spacing_; }

    f32 extent() const { return extent_; }

    f32 offset() const { return offset_; }

    device_span_2d<T> kernel_span() { return data_; }

    u64 nangles() const { return nangles_; }

    f32 distance_source_to_detector() const { return DSD; }

    f32 distance_source_to_object() const { return DSO; }

    f32 distance_object_to_detector() const { return DSD - DSO; }

    f32 center_of_rotation_correction() const { return COR; }

    span<f32> angles() { return span<f32>(angles_.data(), angles_.size()); }

    f32 pitch() const { return pitch_; }

    vec<f32, Dim> source() const { return {-offset(), -distance_source_to_object()}; }

    device_span_2d<T> slice(u64 offset, u64 count = 1) {
        vec<u64, Dim> new_shape{detector_shape(), count};
        auto ptr = device_data() + offset * data_.strides()[1];
        return device_span_2d<T>(ptr, new_shape);
    }

    // Builder pattern to set many variables:
    measurement_2d<T> &set_spacing(f32 new_spacing) {
        spacing_ = new_spacing;
        extent_ = detector_shape() * new_spacing;
        return *this;
    }

    measurement_2d<T> &set_offset(f32 new_offset) {
        offset_ = new_offset;
        return *this;
    }

    measurement_2d<T> &set_distance_source_to_detector(f32 new_DSD) {
        DSD = new_DSD;
        return *this;
    }

    measurement_2d<T> &set_distance_source_to_object(f32 new_DSO) {
        DSO = new_DSO;
        return *this;
    }

    measurement_2d<T> &set_center_of_rotation_correction(f32 new_COR) {
        COR = new_COR;
        return *this;
    }

    measurement_2d<T> &set_angles(std::vector<f32> new_angles) {
        nangles_ = new_angles.size();
        angles_ = std::move(new_angles);
        return *this;
    }

    measurement_2d<T> &set_pitch(f32 new_pitch) {
        pitch_ = new_pitch;
        return *this;
    }

  private:
    device_span_2d<T> data_;

    f32 spacing_;

    f32 extent_;

    f32 offset_;

    // Geometry information about measurements
    u64 nangles_;

    /// Distance source to detector
    f32 DSD;

    /// Distance source to object / center of rotation
    f32 DSO;

    // euler angles in radian, TODO: change to device_uvector? But we only need it in the
    // pre-computing phase, so let's think about this a little more
    std::vector<f32> angles_;

    // Detector rotation in radians, TODO: maybe make this a vector?
    f32 pitch_;

    /// Center of rotation correction (TODO: actually use it)
    f32 COR = 0;
};
} // namespace curad
