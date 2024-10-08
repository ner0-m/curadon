#pragma once

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>

#include "curadon/detail/rotation.hpp"
#include "curadon/detail/span.hpp"
#include "curadon/detail/texture.hpp"
#include "curadon/detail/vec.hpp"
#include "curadon/types.hpp"

namespace curad {
namespace detail {
struct precompute_source_rotate_fn {
    template <class Tuple>
    // __device__ __host__ auto operator()(vec2f &pos, const f32 &angle) const {
    __device__ __host__ auto operator()(Tuple t) const {
        auto &pos = thrust::get<0>(t);
        const auto &angle = thrust::get<1>(t);

        pos = ::curad::geometry::rotate(pos, angle);
    }
};

struct precompute_source_transform_fn {
    vec2f vol_spacing;
    vec2f vol_extent;
    vec2f vol_offset;

    __device__ __host__ auto operator()(const vec2f &pos) const {
        return (pos - vol_offset + 0.5f * vol_extent) / vol_spacing;
    }
};

struct precompute_det_origin_fn {
    vec2f vol_spacing;

    template <class Tuple>
    __device__ __host__ auto operator()(Tuple t) const {
        auto &pos = thrust::get<0>(t);
        const auto &angle = thrust::get<1>(t);
        const auto &source = thrust::get<2>(t);
        pos = ::curad::geometry::rotate(pos, angle, source) / vol_spacing;
    }
};

struct init_det_origin_fn {
    f32 det_extent;
    f32 det_spacing;

    template <class Tuple>
    __device__ __host__ auto operator()(Tuple t) const -> vec2f {
        auto DSO = thrust::get<0>(t);
        auto DSD = thrust::get<1>(t);
        auto DOD = DSD - DSO;
        return {-det_extent / 2.f + det_spacing * .5f, -DOD};
    }
};

struct init_delta_u_fn {
    f32 det_extent;
    f32 det_spacing;

    template <class Tuple>
    __device__ __host__ auto operator()(Tuple t) const -> vec2f {
        auto DSO = thrust::get<0>(t);
        auto DSD = thrust::get<1>(t);
        auto DOD = DSD - DSO;
        return {-det_extent / 2.f + det_spacing * 1.5f, -DOD};
    }
};

struct init_source_fn {
    f32 det_offset;

    __device__ __host__ auto operator()(f32 DSO) const -> vec2f { return {det_offset, DSO}; }
};
} // namespace detail

class plan_2d {
  public:
    plan_2d() = default;

    // don't allow copy, TODO: maybe make this easier, but this should be expensive and hence
    // avoided
    plan_2d(const plan_2d &) = delete;
    plan_2d &operator=(const plan_2d &) = delete;

    plan_2d(plan_2d &&) = default;
    plan_2d &operator=(plan_2d &&) = default;

    plan_2d(usize device, precision vol_prec, vec2u vol_shape, vec2f vol_spacing, vec2f vol_offset,
            precision det_prec, u64 det_count, f32 det_spacing, f32 det_offset,
            thrust::device_vector<f32> DSO, thrust::device_vector<f32> DSD,
            thrust::device_vector<f32> angles)
        : plan_2d(device, vol_prec, vol_shape, vol_spacing, vol_offset, det_prec, det_count,
                  det_spacing, det_offset, std::move(DSO), std::move(DSD), std::move(angles), 0,
                  0) {}

    plan_2d(usize device, precision vol_prec, vec2u vol_shape, vec2f vol_spacing, vec2f vol_offset,
            precision det_prec, u64 det_count, f32 spacing, f32 offset,
            thrust::device_vector<f32> DSO, thrust::device_vector<f32> DSD,
            thrust::device_vector<f32> angles, f32 pitch, f32 COR)
        : device_(device)
        , vol_prec_(vol_prec)
        , det_prec_(det_prec)
        , vol_shape_(vol_shape)
        , vol_spacing_(vol_spacing)
        , vol_extent_(vol_shape_ * vol_spacing_)
        , vol_offset_(vol_offset)
        , det_count_(det_count)
        , det_spacing_(spacing)
        , det_extent_(det_count * spacing)
        , det_offset_(offset)
        , DSD_(std::move(DSD))
        , DSO_(std::move(DSO))
        , angles_(std::move(angles))
        , pitch_(pitch)
        , COR_(COR)
        , forward_tex_({device, vol_shape_[0], vol_shape_[1], 0, false, vol_prec})
        , backward_tex_({device, det_count, 0, num_projections_per_kernel(), true, det_prec}) {
        precompute();
    }

    texture &forward_tex() { return forward_tex_; }

    texture &backward_tex() { return backward_tex_; }

    u64 num_projections_per_kernel() const { return 1024; }

    usize device_id() const { return device_; }

    u64 nangles() const { return angles_.size(); }

    u32 vol_precision() const { return static_cast<u32>(vol_prec_); }

    vec2u vol_shape() const { return vol_shape_; }

    vec2f vol_spacing() const { return vol_spacing_; }

    vec2f vol_extent() const { return vol_extent_; }

    vec2f vol_offset() const { return vol_offset_; }

    u32 det_precision() const { return static_cast<u32>(det_prec_); }

    u64 det_count() const { return det_count_; }

    f32 det_spacing() const { return det_spacing_; }

    f32 det_extent() const { return det_extent_; }

    f32 det_offset() const { return det_offset_; }

    span<f32> DSO() { return {thrust::raw_pointer_cast(DSO_.data()), DSO_.size()}; }

    span<f32> distance_source_to_object() { return DSO(); }

    span<f32> DSD() { return {thrust::raw_pointer_cast(DSD_.data()), DSD_.size()}; }

    span<f32> distance_source_to_detector() { return DSD(); }

    f32 pitch() const { return pitch_; }

    f32 COR() const { return COR_; }

    f32 center_of_rotation_correction() const { return COR(); }

    span<f32> angles() { return {thrust::raw_pointer_cast(angles_.data()), angles_.size()}; }

    span<vec2f> sources() { return {thrust::raw_pointer_cast(sources_.data()), sources_.size()}; }

    span<vec2f> u_origins() {
        return {thrust::raw_pointer_cast(u_origins_.data()), u_origins_.size()};
    }

    span<vec2f> delta_us() {
        return {thrust::raw_pointer_cast(delta_us_.data()), delta_us_.size()};
    }

    u64 forward_block_x = 16;

    u64 forward_block_y = 16;

  private:
    void precompute() {
        u_origins_.resize(angles_.size());
        delta_us_.resize(angles_.size());
        sources_.resize(angles_.size());

        {
            auto first = thrust::make_zip_iterator(thrust::make_tuple(DSO_.begin(), DSD_.begin()));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(DSO_.end(), DSD_.end()));

            thrust::transform(first, last, u_origins_.begin(),
                              detail::init_det_origin_fn{det_extent(), det_spacing()});
            thrust::transform(first, last, delta_us_.begin(),
                              detail::init_delta_u_fn{det_extent(), det_spacing()});
            thrust::transform(DSO_.begin(), DSO_.end(), sources_.begin(),
                              detail::init_source_fn{det_offset()});
        }

        // Rotate source
        {
            auto first =
                thrust::make_zip_iterator(thrust::make_tuple(sources_.begin(), angles_.begin()));
            auto last =
                thrust::make_zip_iterator(thrust::make_tuple(sources_.end(), angles_.end()));

            thrust::for_each(first, last, detail::precompute_source_rotate_fn{});
        }

        // rotate det points (they need the source as rotation point)
        {
            auto first = thrust::make_zip_iterator(
                thrust::make_tuple(u_origins_.begin(), angles_.begin(), sources_.begin()));
            auto last = thrust::make_zip_iterator(
                thrust::make_tuple(u_origins_.end(), angles_.end(), sources_.begin()));

            thrust::for_each(first, last, detail::precompute_det_origin_fn{vol_spacing_});
        }

        {
            auto first = thrust::make_zip_iterator(
                thrust::make_tuple(delta_us_.begin(), angles_.begin(), sources_.begin()));
            auto last = thrust::make_zip_iterator(
                thrust::make_tuple(delta_us_.end(), angles_.end(), sources_.begin()));

            thrust::for_each(first, last, detail::precompute_det_origin_fn{vol_spacing_});
            thrust::transform(delta_us_.begin(), delta_us_.end(), u_origins_.begin(),
                              delta_us_.begin(), thrust::minus<vec2f>{});
        }

        // Do the rest of the transformations of the source
        thrust::transform(
            sources_.begin(), sources_.end(), sources_.begin(),
            detail::precompute_source_transform_fn{vol_spacing_, vol_extent_, vol_offset_});
    }

    usize device_;

    precision vol_prec_;

    precision det_prec_;

    vec2u vol_shape_;

    vec2f vol_spacing_;

    vec2f vol_extent_;

    vec2f vol_offset_;

    /// Number of detector pixels
    u64 det_count_;

    /// Size of each detector pixel
    f32 det_spacing_;

    /// Total size of the detector
    f32 det_extent_;

    /// Offset of the detector
    f32 det_offset_;

    /// Distance source to detector
    thrust::device_vector<f32> DSD_;

    /// Distance source to object / center of rotation
    // f32 DSO_;
    thrust::device_vector<f32> DSO_;

    // euler angles in radian,
    // TODO: change to device_uvector? to avoid call to thrust::uninitialized_fill
    thrust::device_vector<f32> angles_;

    // Detector rotation in radians
    f32 pitch_ = 0;

    /// Center of rotation correction
    f32 COR_ = 0;

    texture forward_tex_;

    texture backward_tex_;

    /// Precomputed origins of each pose
    thrust::device_vector<vec2f> u_origins_ = {};

    /// Precomputed delta for each pose for kernel (how to reach the next detector
    /// cell)
    thrust::device_vector<vec2f> delta_us_ = {};

    /// Precomputed source position for each pose for kernel
    thrust::device_vector<vec2f> sources_ = {};
};
} // namespace curad
