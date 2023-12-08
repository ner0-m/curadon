#pragma once

#include "curadon/geometry/ray.hpp"
#include "curadon/math/vector.hpp"
#include "curadon/utils.hpp"

#include "cuda/std/cmath"

namespace curad {
template <class T>
struct equally_spaced_angles {
    __host__ __device__ equally_spaced_angles(T start, T end, std::size_t n)
        : start_(start)
        , step_((end - start) / static_cast<T>(n))
        , n_(n) {}

    __host__ __device__ auto operator()(std::size_t i) const noexcept -> T {
        return start_ + step_ * i;
    }

    auto size() const noexcept -> std::size_t { return n_; }

  private:
    T start_;
    T step_;
    std::size_t n_;
};

// template <class T>
// struct vector_based_angles {
//     __host__ __device__ vector_based_angles(::curad::device_span<T> span)
//         : span_(span) {}
//
//     __device__ auto operator()(std::size_t i) const noexcept -> T { return span_[i]; }
//
//     auto size() const noexcept -> std::size_t { return span_.size(); }
//
//   private:
//     curad::device_span<T> span_;
// };

template <class T>
class flat_panel_det_1D {
    static constexpr std::int64_t Dim = 2;

  public:
    __host__ __device__ flat_panel_det_1D(const Vec<T, Dim> &axis, std::size_t width)
        : axis_(axis)
        , normal_(detail::perpendicular(axis))
        , width_(width)
        , spacing_(1.0) {}

    __host__ __device__ flat_panel_det_1D(const Vec<T, Dim> &axis, std::size_t width, T spacing)
        : axis_(axis)
        , normal_(detail::perpendicular(axis))
        , width_(width)
        , spacing_(spacing) {}

    __host__ __device__ auto width() const noexcept -> std::size_t { return width_; }

    __host__ __device__ auto spacing() const noexcept -> T { return spacing_; }

    __host__ __device__ auto axis() const noexcept -> Vec<T, Dim> { return axis_; }

    __host__ __device__ auto surface_normal(std::int64_t) const noexcept -> Vec<T, Dim> {
        return normal_;
    }

    // Return shift to center of ith pixel, assuming 0 to be the first pixel and the reference point
    // being in the center
    __host__ __device__ auto surface(std::int64_t u) const noexcept -> Vec<T, Dim> {
        auto shift = u - (width() / T{2}) + T{0.5};
        return shift * (axis() * spacing());
    }

    __host__ __device__ auto surface_deriv(std::int64_t u) const noexcept -> Vec<T, Dim> {
        return axis_;
    }

  private:
    Vec<T, Dim> axis_;

    Vec<T, Dim> normal_;

    std::size_t width_;

    T spacing_;
};

/// Assume center point to be at (1, 0), and axis at (0, 1)
template <class T>
class parallel_pose_2d {
    static constexpr std::int64_t Dim = 2;

  public:
    __host__ __device__ parallel_pose_2d() = default;

    __host__ __device__ parallel_pose_2d(radian<T> phi, std::size_t det_width)
        : phi_(phi)
        , reference_point_(detail::rotate(phi.value(), Vec<T, Dim>{1, 0}))
        , det_(detail::rotate(phi.value(), Vec<T, Dim>{0, 1}), det_width) {}

    __host__ __device__ parallel_pose_2d(radian<T> phi, std::size_t det_width, T spacing)
        : phi_(phi)
        , reference_point_(detail::rotate(phi.value(), Vec<T, Dim>{1, 0}))
        , det_(detail::rotate(phi.value(), Vec<T, Dim>{0, 1}), det_width, spacing) {}

    __host__ __device__ auto operator()(std::int64_t u) const noexcept -> ray<T, Dim> {
        auto ro = det_point(u);
        auto rd = -det_to_source(u);

        return {ro, rd};
    }

    __host__ __device__ auto det_point(std::int64_t u) const noexcept -> Vec<T, Dim> {
        return det_ref_point() + det_.surface(u);
    }

    __host__ __device__ auto det_to_source(std::int64_t u) const noexcept -> Vec<T, Dim> {
        return det_.surface_normal(u);
    }

    __host__ __device__ auto det_ref_point() const noexcept -> Vec<T, Dim> {
        return reference_point_;
    }

    __host__ __device__ auto phi() const noexcept -> radian<T> { return phi_; }

  private:
    radian<T> phi_{0};

    /// Reference point on the detector, usually the center
    Vec<T, Dim> reference_point_{{1, 0}};

    flat_panel_det_1D<T> det_{{0, 1}, 1};
};

template <class T, class Fn>
class parallel_2d_geometry {
    static constexpr std::int64_t Dim = 2;

  public:
    __host__ __device__ parallel_2d_geometry(Fn fn, std::size_t det_width)
        : fn_(fn)
        , reference_point_init_({1, 0})
        , det_({0, 1}, det_width) {}

    __host__ __device__ auto operator()(std::int64_t idx, std::int64_t u) const noexcept
        -> ray<T, Dim> {
        auto ro = detail::rotate(angle(idx).value(), det_point(idx, u));
        auto rd = detail::rotate(angle(idx).value(), -det_to_source(idx, u));

        return {ro, rd};
    }

    __host__ __device__ auto det_point(std::size_t idx, std::int64_t u) const noexcept
        -> Vec<T, Dim> {
        return detail::rotate(angle(idx).value(), det_ref_point(u) + det_.surface(u));
    }

    __host__ __device__ auto det_to_source(std::size_t idx, std::int64_t u) const noexcept
        -> Vec<T, Dim> {
        return detail::rotate(angle(idx).value(), det_.surface_normal(u));
    }

    __host__ __device__ auto det_ref_point(std::size_t idx) const noexcept -> Vec<T, Dim> {
        return detail::rotate(angle(idx).value(), reference_point_init_);
    }

    __host__ __device__ auto angle(std::size_t idx) const noexcept -> radian<T> { return fn_(idx); }

  private:
    Fn fn_;

    /// Reference point on the detector, usually the center
    Vec<T, Dim> reference_point_init_{{1, 0}};

    flat_panel_det_1D<T> det_{{0, 1}, 1};
};
} // namespace curad
