#pragma once

#include "curadon/cuda/device_buffer.hpp"
#include "curadon/cuda/device_span.hpp"
#include "curadon/cuda/device_uvector.hpp"
#include "curadon/cuda/stream_view.hpp"
#include "curadon/geometry/box.hpp"
#include "curadon/parallel.hpp"
#include "curadon/traversal.hpp"

namespace curad {
template <class T>
T deg2rad(const T &deg) {
    return deg * static_cast<T>(M_PI) / static_cast<T>(180);
}
template <class T>
T rad2deg(const T &rad) {
    return rad * static_cast<T>(180) / static_cast<T>(M_PI);
}

template <std::int64_t Dim>
__host__ __device__ auto compute_c_strides(const Vec<std::size_t, Dim> &shape)
    -> Vec<std::int64_t, Dim> {
    Vec<std::int64_t, Dim> strides;
    strides[0] = 1;
    for (int i = 1; i < Dim; ++i) {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    return strides;
}

template <class T, std::int64_t Dim>
class single_projection_view {
  public:
    __host__ __device__ single_projection_view(device_span<T> data, Vec<std::size_t, Dim> shape,
                                               Vec<std::int64_t, Dim> strides)
        : data_(data)
        , shape_(shape)
        , strides_(strides) {}

    __host__ __device__ std::size_t ndim() const { return shape_.size(); }

    __host__ __device__ auto shape() const noexcept { return shape_; }

    __host__ __device__ auto shape() noexcept { return shape_; }

    __host__ __device__ auto strides() const noexcept { return strides_; }

    __host__ __device__ auto strides() noexcept { return strides_; }

    template <std::int64_t Tmp = Dim>
    __host__ __device__ auto operator()(std::int64_t u) const noexcept
        -> std::enable_if_t<Tmp == 1, T> {
        return data_[u * strides_[0]];
    }

    template <std::int64_t Tmp = Dim>
    __host__ __device__ auto operator()(std::int64_t u) noexcept
        -> std::enable_if_t<Tmp == 1, T &> {
        return data_[u * strides_[0]];
    }

    template <std::int64_t Tmp = Dim>
    __host__ __device__ auto operator()(std::int64_t u, std::int64_t v) const noexcept
        -> std::enable_if_t<Tmp == 2, const T &> {
        return data_[u * strides_[0] + v * strides_[1]];
    }

    template <std::int64_t Tmp = Dim>
    __host__ __device__ auto operator()(std::int64_t u, std::int64_t v) noexcept
        -> std::enable_if_t<Tmp == 2, const T &> {
        return data_[u * strides_[0] + v * strides_[1]];
    }

  private:
    device_span<T> data_;

    Vec<std::size_t, Dim> shape_;

    Vec<std::int64_t, Dim> strides_;
};

template <class T, std::int64_t Dim>
class projection_view {
    constexpr static std::int64_t RangeDim = Dim;
    constexpr static std::int64_t AccessDim = Dim + 1;

  public:
    __host__ __device__ projection_view(device_span<T> data, Vec<std::size_t, AccessDim> shape)
        : data_(data)
        , shape_(shape)
        , strides_(compute_c_strides(shape)) {}

    __host__ __device__ std::size_t ndim() const { return shape_.size(); }

    __host__ __device__ std::size_t nangles() const { return shape_[AccessDim - 1]; }

    /*
     * 1D accesses
     */

    template <std::int64_t Tmp = RangeDim>
    __host__ __device__ auto operator()(std::int64_t angle, std::int64_t u) const noexcept
        -> std::enable_if_t<Tmp == 1, const T &> {
        return data_[offset(angle, u)];
    }

    template <std::int64_t Tmp = RangeDim>
    __host__ __device__ auto operator()(std::int64_t angle, std::int64_t u) noexcept
        -> std::enable_if_t<Tmp == 1, T &> {
        return data_[offset(angle, u)];
    }

    /*
     * 2D accesses
     */
    template <std::int64_t Tmp = RangeDim>
    __host__ __device__ auto operator()(std::int64_t angle, std::int64_t u,
                                        std::int64_t v) const noexcept
        -> std::enable_if_t<Tmp == 2, const T &> {
        return data_[offset(angle, u, v)];
    }

    template <std::int64_t Tmp = RangeDim>
    __host__ __device__ auto operator()(std::int64_t angle, std::int64_t u, std::int64_t v) noexcept
        -> std::enable_if_t<Tmp == 2, T &> {
        return data_[offset(angle, u, v)];
    }

    template <std::int64_t Tmp = RangeDim>
    __host__ __device__ auto offset(std::int64_t angle, std::int64_t u) const noexcept
        -> std::enable_if_t<Tmp == 1, std::int64_t> {
        return u * strides_[0] + angle * strides_[1];
    }

    template <std::int64_t Tmp = RangeDim>
    __host__ __device__ auto offset(std::int64_t angle, std::int64_t u,
                                    std::int64_t v) const noexcept
        -> std::enable_if_t<Tmp == 2, std::int64_t> {
        return u * strides_[0] + v * strides_[1] + angle * strides_[2];
    }

    __host__ __device__ auto shape() const noexcept { return shape_; }

    __host__ __device__ auto shape() noexcept { return shape_; }

    __host__ __device__ auto strides() const noexcept { return strides_; }

    __host__ __device__ auto strides() noexcept { return strides_; }

    __host__ __device__ auto data() const noexcept { return data_; }

    __host__ __device__ auto slice(std::int64_t idx) const noexcept
        -> single_projection_view<T, RangeDim> {
        if constexpr (RangeDim == 1) {
            auto span = device_span<T>(&data()[offset(idx, 0)], shape()[0]);
            return single_projection_view<T, RangeDim>{span, shape().template head<Dim>(),
                                                       strides().template head<Dim>()};
        } else {
            auto span = device_span<T>(&data()[offset(idx, 0, 0)], shape()[0] * shape()[1]);
            return single_projection_view<T, RangeDim>{span, shape().template head<Dim>(),
                                                       strides().template head<Dim>()};
        }
    }

  private:
    device_span<T> data_;

    Vec<std::size_t, AccessDim> shape_;

    Vec<std::int64_t, AccessDim> strides_;
};

template <class T, std::int64_t Dim>
class volume {
  public:
    __host__ __device__ volume(device_span<T> data, Vec<std::size_t, Dim> shape)
        : data_(data)
        , shape_(shape)
        , strides_(compute_c_strides(shape))
        , translation_(Vec<T, Dim>::zero())
        , aabb_({-(shape_.template as<T>() / 2) + translation_,
                 (shape_.template as<T>() / 2) + translation_}) {}

    __host__ __device__ std::size_t ndim() const { return shape_.size(); }

    __host__ __device__ const T &operator()(const Vec<std::int64_t, Dim> &coord) const {
        auto idx = coord_to_idx(coord, strides()) + coord_shift();
        return data_[idx];
    }

    __host__ __device__ auto shape() const noexcept { return shape_; }

    __host__ __device__ auto shape() noexcept { return shape_; }

    __host__ __device__ auto strides() const noexcept { return strides_; }

    __host__ __device__ auto strides() noexcept { return strides_; }

    __host__ __device__ auto aabb() const noexcept -> box<T, Dim> { return aabb_; }

  private:
    __host__ __device__ auto coord_shift() const noexcept -> std::size_t {
        auto translation = (::curad::abs(aabb().min())).template as<std::int64_t>();
        return coord_to_idx(translation, strides());
    }

    device_span<T> data_;

    Vec<std::size_t, Dim> shape_;

    Vec<std::int64_t, Dim> strides_;

    Vec<T, Dim> translation_;

    box<T, Dim> aabb_;
};

template <class T, std::int64_t DomainDim, std::int64_t RangeDim, class Geometry,
          class ForwardFunctor>
__host__ __device__ void forward_projection(std::size_t angle_idx, const box<T, DomainDim> &aabb,
                                            volume<T, DomainDim> vol,
                                            single_projection_view<T, RangeDim> projection,
                                            Geometry geom, ForwardFunctor fn) {
    static_assert(RangeDim == 1 || RangeDim == 2, "Only 1D and 2D projections are supported");

    if constexpr (RangeDim == 1) {
        const auto npixels = projection.shape()[0];
        for (std::size_t i = 0; i < npixels; ++i) {
            const auto ray = geom(angle_idx, i);

            joseph::compute_ray_coeffs(
                aabb, ray,
                [&](bool in_aabb, const curad::Vec<std::int64_t, DomainDim> &volcoord, T coeff) {
                    fn(angle_idx, i, in_aabb, volcoord, coeff);
                });
        }
    } else if constexpr (RangeDim == 2) {
        const auto nx = projection.shape()[0];
        const auto ny = projection.shape()[1];

        std::size_t pixel_idx = 0;
        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {
                const auto ray = geom(pixel_idx, i, j);

                joseph::compute_ray_coeffs(
                    aabb, ray,
                    [&](bool in_aabb, const curad::Vec<std::int64_t, DomainDim> &volcoord,
                        T coeff) { fn(angle_idx, pixel_idx, in_aabb, volcoord, coeff); });
                ++pixel_idx;
            }
        }
    }
}

template <class T, std::int64_t DomainDim, std::int64_t RangeDim, class Geometry,
          class ForwardFunctor>
__host__ __device__ void forward(const box<T, DomainDim> &aabb, volume<T, DomainDim> vol,
                                 projection_view<T, RangeDim> projection, Geometry geom,
                                 ForwardFunctor fn) {
    static_assert(RangeDim == 1 || RangeDim == 2, "Only 1D and 2D projections are supported");

    for (std::size_t i = 0; i < projection.nangles(); ++i) {
        forward_projection(i, aabb, vol, projection.slice(i), geom, fn);
    }
}
} // namespace curad
