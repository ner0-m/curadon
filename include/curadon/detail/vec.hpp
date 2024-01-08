#pragma once

#include <cstdint>

#include "curadon/detail/math.hpp"
#include "curadon/types.hpp"

namespace curad {

template <class T, i64 Dim>
class vec {
  public:
    __host__ __device__ vec() = default;

    __host__ __device__ vec(const T &t)
        : data_{t} {}

    __host__ __device__ vec(const T &x, const T &y)
        : data_{x, y} {}

    __host__ __device__ vec(const T &x, const T &y, const T &z)
        : data_{x, y, z} {}

    __host__ __device__ std::size_t size() const { return Dim; }

    __host__ __device__ T *data() { return data_; }

    __host__ __device__ const T *data() const { return data_; }

    __host__ __device__ T &operator[](i64 i) { return data_[i]; }

    __host__ __device__ const T &operator[](i64 i) const { return data_[i]; }

    __host__ __device__ T &x() { return data_[0]; }

    __host__ __device__ const T &x() const { return data_[0]; }

    __host__ __device__ T &y() { return data_[1]; }

    __host__ __device__ const T &y() const { return data_[1]; }

    __host__ __device__ T &z() { return data_[2]; }

    __host__ __device__ const T &z() const { return data_[2]; }

    __host__ __device__ vec &normalize();

    __host__ __device__ T hprod() const {
        T res = 1;
#pragma unroll
        for (int i = 0; i < Dim; ++i) {
            res += (*this)[i];
        }
        return res;
    }

    __host__ __device__ static vec<T, Dim> zeros() noexcept {
        vec<T, Dim> res;
#pragma unroll
        for (int i = 0; i < Dim; ++i) {
            res[i] = T(0);
        }
        return res;
    }

    __host__ __device__ static vec<T, Dim> ones() noexcept {
        vec<T, Dim> res;
#pragma unroll
        for (int i = 0; i < Dim; ++i) {
            res[i] = T(1);
        }
        return res;
    }

    template <i64 NewDim>
    __host__ __device__ auto head() const noexcept -> vec<T, NewDim> {
        vec<T, NewDim> result;

#pragma unroll
        for (int i = 0; i < NewDim; ++i) {
            result[i] = (*this)[i];
        }
        return result;
    }

    __host__ __device__ auto operator+=(const vec<T, Dim> &other) -> vec & {
#pragma unroll
        for (int i = 0; i < Dim; ++i) {
            (*this)[i] += other[i];
        }
        return *this;
    }

    __host__ __device__ auto operator/=(const T &scalar) -> vec & {
#pragma unroll
        for (int i = 0; i < Dim; ++i) {
            (*this)[i] /= scalar;
        }
        return *this;
    }

    template <class U>
    __host__ __device__ vec<U, Dim> as() const {
        vec<U, Dim> other;

#pragma unroll
        for (int i = 0; i < Dim; ++i) {
            other[i] = static_cast<U>((*this)[i]);
        }

        return other;
    }

  private:
    T data_[Dim];
};

template <class T, i64 Dim>
__host__ __device__ auto operator+(const vec<T, Dim> &x) -> vec<T, Dim> {
    vec<T, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = +x[i];
    }

    return result;
}

template <class T, i64 Dim>
__host__ __device__ auto operator-(const vec<T, Dim> &x) -> vec<T, Dim> {
    vec<T, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = -x[i];
    }

    return result;
}

/* ==================================== */
/*           Addition                   */
/* ==================================== */
template <class T, class U, i64 Dim>
__host__ __device__ auto operator+(const vec<T, Dim> &lhs, const vec<U, Dim> &rhs)
    -> vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    vec<V, Dim> result = lhs;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] += rhs[i];
    }
    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator+(const vec<T, Dim> &lhs, const U &scalar)
    -> vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    vec<V, Dim> result = lhs;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = result[i] + scalar;
    }
    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator+(const U &scalar, const vec<T, Dim> &rhs)
    -> vec<std::common_type_t<T, U>, Dim> {
    return rhs + scalar;
}

/* ==================================== */
/*           Subtraction                */
/* ==================================== */
template <class T, class U, i64 Dim>
__host__ __device__ auto operator-(const vec<T, Dim> &lhs, const vec<U, Dim> &rhs)
    -> vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] - rhs[i];
    }

    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator-(const vec<T, Dim> &lhs, const U &scalar)
    -> vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] - scalar;
    }
    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator-(const T &scalar, const vec<U, Dim> &rhs)
    -> vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = scalar - rhs[i];
    }
    return result;
}

/* ==================================== */
/*           Multiplication             */
/* ==================================== */
template <class T, class U, i64 Dim>
__host__ __device__ auto operator*(const vec<T, Dim> &lhs, const vec<U, Dim> &rhs)
    -> vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator*(const vec<T, Dim> &lhs, const U &scalar)
    -> vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] * scalar;
    }
    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator*(const U &scalar, const vec<T, Dim> &rhs)
    -> vec<std::common_type_t<T, U>, Dim> {
    return rhs * scalar;
}

/* ==================================== */
/*           Division                   */
/* ==================================== */
template <class T, class U, i64 Dim>
__host__ __device__ auto operator/(const vec<T, Dim> &lhs, const vec<U, Dim> &rhs)
    -> vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    vec<V, Dim> result;
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] / rhs[i];
    }
    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator/(const vec<T, Dim> &lhs, const U &scalar)
    -> vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    vec<V, Dim> result = lhs;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = result[i] / scalar;
    }
    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator/(const T &scalar, const vec<U, Dim> &rhs)
    -> vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = scalar / rhs[i];
    }
    return result;
}

/* ==================================== */
/*           Comparison                 */
/* ==================================== */
template <class T, class U, i64 Dim>
__host__ __device__ auto operator==(const vec<T, Dim> &lhs, const vec<U, Dim> &rhs)
    -> vec<bool, Dim> {
    vec<bool, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] == rhs[i];
    }
    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator!=(const vec<T, Dim> &lhs, const vec<U, Dim> &rhs)
    -> vec<bool, Dim> {
    return !(lhs == rhs);
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator<(const vec<T, Dim> &lhs, const vec<U, Dim> &rhs)
    -> vec<bool, Dim> {
    vec<bool, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] < rhs[i];
    }
    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator<=(const vec<T, Dim> &lhs, const vec<U, Dim> &rhs)
    -> vec<bool, Dim> {
    vec<bool, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] <= rhs[i];
    }
    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator>(const vec<T, Dim> &lhs, const vec<U, Dim> &rhs)
    -> vec<bool, Dim> {
    vec<bool, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] > rhs[i];
    }
    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto operator>=(const vec<T, Dim> &lhs, const vec<U, Dim> &rhs)
    -> vec<bool, Dim> {
    vec<bool, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] >= rhs[i];
    }
    return result;
}

/* ==================================== */
/*           Logic                      */
/* ==================================== */
template <class T, class U, i64 Dim>
__host__ __device__ auto operator&&(const vec<T, Dim> &lhs, const vec<U, Dim> &rhs)
    -> vec<bool, Dim> {
    vec<bool, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] && rhs[i];
    }
    return result;
}

/* ==================================== */
/*           Other                      */
/* ==================================== */
template <class T, i64 Dim>
__host__ __device__ auto argmax(const vec<T, Dim> &v) -> std::size_t {
    // TODO: use thrust for this
    auto current_max = v[0];
    auto maxpos = 0;

#pragma unroll
    for (int i = 1; i < Dim; ++i) {
        if (v[i] > current_max) {
            current_max = v[i];
            maxpos = i;
        }
    }
    return maxpos;
}

template <class T, i64 Dim>
__host__ __device__ auto norm(const vec<T, Dim> &v) -> T {
    // TODO: use thrust for this
    T result = 0;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result += v[i] * v[i];
    }

    return sqrt(result);
}

template <class T, i64 Dim>
__host__ __device__ auto normalized(const vec<T, Dim> &v) -> vec<T, Dim> {
    auto copy = v;
    copy.normalize();
    return copy;
}

template <class T, i64 Dim>
__host__ __device__ auto floor(const vec<T, Dim> &v) -> vec<T, Dim> {
    // TODO: use thrust for this
    auto copy = v;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        copy[i] = math::floor(v[i]);
    }
    return copy;
}

template <class T, i64 Dim>
__host__ __device__ auto floori(const vec<T, Dim> &v) -> vec<i64, Dim> {
    // TODO: use thrust for this
    vec<i64, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = static_cast<i64>(math::floor(v[i]));
    }
    return result;
}

template <class T, i64 Dim>
__host__ __device__ auto ceil(const vec<T, Dim> &v) -> vec<T, Dim> {
    // TODO: use thrust for this
    vec<T, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = math::ceil(v[i]);
    }
    return result;
}

template <class T, i64 Dim>
__host__ __device__ auto ceili(const vec<T, Dim> &v) -> vec<i64, Dim> {
    // TODO: use thrust for this
    vec<i64, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = static_cast<i64>(math::ceil(v[i]));
    }
    return result;
}

template <class T, i64 Dim>
__host__ __device__ auto abs(const vec<T, Dim> &v) -> vec<T, Dim> {
    // TODO: use thrust for this
    auto copy = v;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        copy[i] = math::abs(v[i]);
    }
    return copy;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto min(const vec<T, Dim> &x, const vec<U, Dim> &y)
    -> vec<std::common_type_t<T, U>, Dim> {
    // TODO: use thrust for this
    using V = std::common_type_t<T, U>;
    vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = math::min(x[i], y[i]);
    }
    return result;
}

template <class T, class U, i64 Dim>
__host__ __device__ auto max(const vec<T, Dim> &x, const vec<U, Dim> &y)
    -> vec<std::common_type_t<T, U>, Dim> {
    // TODO: use thrust for this
    using V = std::common_type_t<T, U>;
    vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = math::max(x[i], y[i]);
    }
    return result;
}

template <class T, i64 Dim>
__host__ __device__ auto sign(const vec<T, Dim> &v) -> vec<T, Dim> {
    // TODO: use thrust for this
    auto copy = v;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        copy[i] = math::sign(v[i]);
    }
    return copy;
}

/// @brief normalize the current vector
template <class T, i64 Dim>
__host__ __device__ vec<T, Dim> &vec<T, Dim>::normalize() {
    auto norm = ::curad::norm(*this);
    *this /= norm;
    return *this;
}

template <class T>
using vec2 = vec<T, 2>;
template <class T>
using vec3 = vec<T, 3>;

using vec2f = vec<f32, 2>;
using vec3f = vec<f32, 3>;

using vec2i = vec<i64, 2>;
using vec3i = vec<i64, 3>;

using vec2u = vec<u64, 2>;
using vec3u = vec<u64, 3>;
} // namespace curad
