#pragma once

#include "curadon/parallel.hpp"
#include "curadon/utils.hpp"

namespace curad {
template <class T, std::int64_t Dim>
class geometry {
  public:
  private:
    /// Distance source to detector
    T DSD;

    /// Distance source to origin
    T DSO;

    /// Shape of detector
    Vec<std::uint64_t, Dim> det_shape_;

    /// Spacing of detector

    /// Size in world of detector (shape * spacing)

    /// Offset of detector origin / offset correction

    /// Rotation of detector (roll, pitch, yaw)

    /// Center of rotation correction

    /// Shape of volume

    /// Spacing of volume

    /// Size in world of volume (shape * spacing)

    /// Offset of volume origin
};

template <class T, class Fn>
class fan_beam_geometry {
    static constexpr std::int64_t Dim = 2;

  public:
    __host__ __device__ fan_beam_geometry(Fn fn, T dist_source_center, T dist_center_det,
                                          std::size_t det_width)
        : fn_(fn)
        , source_({-dist_source_center, 0})
        , reference_point_init_({dist_center_det, 0})
        , det_({0, 1}, det_width) {}

    __host__ __device__ auto operator()(std::int64_t idx, std::int64_t u) const noexcept
        -> ray<T, Dim> {
        auto ro = source(idx);
        auto ref = det_point(idx, u);
        auto rd = ref - ro;
        rd = normalized(rd);
        return {ro, rd};
    }

    __host__ __device__ auto source(std::size_t idx) const noexcept -> Vec<T, Dim> {
        return detail::rotate(angle(idx).value(), source_);
    }

    __host__ __device__ auto det_point(std::size_t idx, std::int64_t u) const noexcept
        -> Vec<T, Dim> {
        auto tmp = det_.surface(u);
        return detail::rotate(angle(idx).value(), reference_point_init_ + det_.surface(u));
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

    Vec<T, Dim> source_{{-1, 0}};

    /// Reference point on the detector, usually the center
    Vec<T, Dim> reference_point_init_{{1, 0}};

    flat_panel_det_1D<T> det_{{0, 1}, 1};
};
} // namespace curad
