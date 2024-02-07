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
    // __device__ __host__ auto operator()(vec2f &pos, const f32 &angle) const {
    __device__ __host__ auto operator()(Tuple t) const {
        auto &pos = thrust::get<0>(t);
        const auto &angle = thrust::get<1>(t);
        const auto &source = thrust::get<2>(t);
        pos = ::curad::geometry::rotate(pos, angle, source) / vol_spacing;
    }
};
} // namespace detail

class forward_plan_2d {
  public:
    forward_plan_2d() = default;

    // don't allow copy, TODO: maybe make this easier, but this should be expensive and hence
    // avoided
    forward_plan_2d(const forward_plan_2d &) = delete;
    forward_plan_2d &operator=(const forward_plan_2d &) = delete;

    forward_plan_2d(forward_plan_2d &&) = default;
    forward_plan_2d &operator=(forward_plan_2d &&) = default;

    forward_plan_2d(usize device, vec2u vol_shape, vec2f vol_spacing, vec2f vol_offset,
                    u64 det_count, f32 det_spacing, f32 det_offset, f32 DSO, f32 DSD,
                    thrust::device_vector<f32> angles)
        : forward_plan_2d(device, vol_shape, vol_spacing, vol_offset, det_count, det_spacing,
                          det_offset, DSO, DSD, std::move(angles), 0, 0) {}

    forward_plan_2d(usize device, vec2u vol_shape, vec2f vol_spacing, vec2f vol_offset,
                    u64 det_count, f32 spacing, f32 offset, f32 DSO, f32 DSD,
                    thrust::device_vector<f32> angles, f32 pitch, f32 COR)
        : device_(device)
        , vol_shape_(vol_shape)
        , vol_spacing_(vol_spacing)
        , vol_extent_(vol_shape_ * vol_spacing_)
        , vol_offset_(vol_offset)
        , det_count_(det_count)
        , det_spacing_(spacing)
        , det_extent_(det_count * spacing)
        , det_offset_(offset)
        , DSD_(DSD)
        , DSO_(DSO)
        , angles_(std::move(angles))
        , pitch_(pitch)
        , COR_(COR)
        , forward_tex_({device, vol_shape_[0], vol_shape_[1], 0, false})
        , backward_tex_({device, det_count, 0, num_projections_per_kernel(), true}) {
        precompute();
    }

    texture &forward_tex() { return forward_tex_; }

    texture &backward_tex() { return backward_tex_; }

    u64 num_projections_per_kernel() const { return 1024; }

    usize device_id() const { return device_; }

    u64 nangles() const { return angles_.size(); }

    vec2u vol_shape() const { return vol_shape_; }

    vec2f vol_spacing() const { return vol_spacing_; }

    vec2f vol_extent() const { return vol_extent_; }

    vec2f vol_offset() const { return vol_offset_; }

    u64 det_count() const { return det_count_; }

    f32 det_spacing() const { return det_spacing_; }

    f32 det_extent() const { return det_extent_; }

    f32 det_offset() const { return det_offset_; }

    f32 DSO() const { return DSO_; }

    f32 distance_source_to_object() const { return DSO_; }

    f32 DSD() const { return DSD_; }

    f32 distance_source_to_detector() const { return DSD_; }

    f32 DOD() const { return DSD_ - DSO_; }

    f32 distance_object_to_detector() const { return DSD_ - DSO_; }

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
        vec2f init_det_origin{-det_extent_ / 2.f + det_spacing_ * .5f, -DOD()};
        vec2f init_detla_u{-det_extent_ / 2.f + det_spacing_ * 1.5f, -DOD()};
        const vec2f init_source = {det_offset_, DSO_};

        u_origins_.resize(angles_.size());
        delta_us_.resize(angles_.size());
        sources_.resize(angles_.size());

        thrust::fill(u_origins_.begin(), u_origins_.end(), init_det_origin);
        thrust::fill(delta_us_.begin(), delta_us_.end(), init_detla_u);
        thrust::fill(sources_.begin(), sources_.end(), init_source);

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
    f32 DSD_;

    /// Distance source to object / center of rotation
    f32 DSO_;

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
