#pragma once

#include <vector>

#include "curadon/detail/device_span.hpp"
#include "curadon/detail/span.hpp"
#include "curadon/detail/vec.hpp"
#include "curadon/types.hpp"

namespace curad {
template <class T>
class device_measurement {
  public:
    static constexpr int Dim = 3;
    static constexpr int DetectorDim = 2;

    device_measurement(T *data, vec<u64, Dim> shape)
        : device_measurement(data, shape, vec<f32, DetectorDim>::ones()) {}

    device_measurement(T *data, vec<u64, Dim> shape, vec<f32, DetectorDim> spacing)
        : device_measurement(data, shape, vec<f32, DetectorDim>::ones(),
                             vec<f32, DetectorDim>::zeros()) {}

    device_measurement(T *data, vec<u64, Dim> shape, vec<f32, DetectorDim> spacing,
                       vec<f32, DetectorDim> offset)
        : data_(data, shape)
        , spacing_(spacing)
        , offset_(offset)
        , extent_(detector_shape() * this->spacing())
        , nangles_(shape[0])
        , DSD(0)
        , DSO(0)
        , COR(0)
        , phi_(nangles(), 0)
        , theta_(nangles(), 0)
        , psi_(nangles(), 0)
        , pitch_(0)
        , roll_(0)
        , yaw_(0) {}

    device_measurement(T *data, vec<u64, DetectorDim> shape, f32 DSD, f32 DSO, std::vector<f32> phi)
        : device_measurement(data, shape, vec<f32, DetectorDim>::ones(), DSD, DSO, phi) {}

    device_measurement(T *data, vec<u64, DetectorDim> shape, vec<f32, DetectorDim> spacing, f32 DSD,
                       f32 DSO, std::vector<f32> phi)
        : device_measurement(data, shape, vec<f32, DetectorDim>::ones(),
                             vec<f32, DetectorDim>::zeros(), DSD, DSO, phi) {}

    device_measurement(T *data, vec<u64, DetectorDim> shape, vec<f32, DetectorDim> spacing,
                       vec<f32, DetectorDim> offset, f32 DSD, f32 DSO, std::vector<f32> phi)
        : data_(data, {shape.x(), shape.y(), phi.size()})
        , spacing_(spacing)
        , offset_(offset)
        , extent_(shape * spacing_)
        , nangles_(phi.size())
        , DSD(DSD)
        , DSO(DSO)
        , COR(0)
        , phi_(phi)
        , theta_(nangles(), 0)
        , psi_(nangles(), 0)
        , pitch_(0)
        , roll_(0)
        , yaw_(0) {}

    u64 size() const { return data_.size(); }

    u64 nbytes() const { return data_.nbytes(); }

    T *device_data() { return data_.device_data(); }

    T const *device_data() const { return data_.device_data(); }

    vec<u64, Dim> shape() const { return data_.shape(); }

    vec<u64, DetectorDim> detector_shape() const { return {data_.shape()[0], data_.shape()[1]}; }

    vec<f32, DetectorDim> spacing() const { return spacing_; }

    vec<f32, DetectorDim> extent() const { return extent_; }

    vec<f32, DetectorDim> offset() const { return offset_; }

    device_span_3d<T> kernel_span() { return data_; }

    u64 nangles() const { return nangles_; }

    f32 distance_source_to_detector() const { return DSD; }

    f32 distance_source_to_object() const { return DSO; }

    f32 distance_object_to_detector() const { return DSD - DSO; }

    f32 center_of_rotation_correction() const { return COR; }

    span<f32> angles() { return span<f32>(phi_.data(), phi_.size()); }

    span<f32> phi() { return span<f32>(phi_.data(), phi_.size()); }

    f32 phi(u64 idx) { return phi_[idx]; }

    span<f32> theta() { return span<f32>(theta_.data(), theta_.size()); }

    f32 theta(u64 idx) { return theta_[idx]; }

    span<f32> psi() { return span<f32>(psi_.data(), psi_.size()); }

    f32 psi(u64 idx) { return psi_[idx]; }

    f32 pitch() const { return pitch_; }

    f32 roll() const { return roll_; }

    f32 yaw() const { return yaw_; }

    vec<f32, Dim> source() const { return {0, 0, -distance_source_to_object()}; }

    device_span_3d<T> slice(u64 offset, u64 count = 1) {
        vec<u64, Dim> new_shape{shape()[0], shape()[1], count};
        auto ptr = device_data() + offset * data_.strides()[2];
        return device_span_3d<T>(ptr, new_shape);
    }

    // Builder pattern to set many variables:
    device_measurement<T> &set_spacing(vec<f32, DetectorDim> new_spacing) {
        spacing_ = new_spacing;
        extent_ = detector_shape() * new_spacing;
        return *this;
    }

    device_measurement<T> &set_offset(vec<f32, DetectorDim> new_offset) {
        offset_ = new_offset;
        return *this;
    }

    device_measurement<T> &set_distance_source_to_detector(f32 new_DSD) {
        DSD = new_DSD;
        return *this;
    }

    device_measurement<T> &set_distance_source_to_object(f32 new_DSO) {
        DSO = new_DSO;
        return *this;
    }

    device_measurement<T> &set_center_of_rotation_correction(f32 new_COR) {
        COR = new_COR;
        return *this;
    }

    device_measurement<T> &set_angles(std::vector<f32> new_angles) {
        std::vector<f32> zeros(new_angles.size(), 0);
        return set_angles(new_angles, zeros, zeros);
    }

    device_measurement<T> &set_angles(std::vector<f32> new_phi, std::vector<f32> new_theta,
                                      std::vector<f32> new_psi) {
        // TODO: check that all have the same size!
        phi_ = new_phi;
        theta_ = new_theta;
        psi_ = new_psi;
        nangles_ = new_phi.size();
        return *this;
    }

    device_measurement<T> &set_pitch(f32 new_pitch) {
        pitch_ = new_pitch;
        return *this;
    }

    device_measurement<T> &set_roll(f32 new_roll) {
        roll_ = new_roll;
        return *this;
    }

    device_measurement<T> &set_yaw(f32 new_yaw) {
        yaw_ = new_yaw;
        return *this;
    }

  private:
    device_span_3d<T> data_;

    vec<f32, DetectorDim> spacing_;

    vec<f32, DetectorDim> extent_;

    vec<f32, DetectorDim> offset_;

    // Geometry information about measurements
    u64 nangles_;

    /// Distance source to detector
    f32 DSD;

    /// Distance source to object / center of rotation
    f32 DSO;

    // euler angles in radian, TODO: change to device_uvector? But we only need it in the
    // pre-computing phase, so let's think about this a little more
    std::vector<f32> phi_;
    std::vector<f32> theta_;
    std::vector<f32> psi_;

    // Detector rotation in radians, TODO: maybe make this a vector?
    f32 pitch_;
    f32 roll_;
    f32 yaw_;

    /// Center of rotation correction (TODO: actually use it)
    f32 COR = 0;
};
} // namespace curad
