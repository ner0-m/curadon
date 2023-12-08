#pragma once

#include "curadon/math/math.hpp"
#include <cstdint>

namespace curad {

template <class T, std::int64_t Dim>
class Vec;

template <class T, std::int64_t Dim>
class Vec {
  public:
    __host__ __device__ Vec() = default;

    __host__ __device__ Vec(const T &t)
        : data_{t} {}

    __host__ __device__ Vec(const T &x, const T &y)
        : data_{x, y} {}

    __host__ __device__ Vec(const T &x, const T &y, const T &z)
        : data_{x, y, z} {}

    __host__ __device__ std::size_t size() const { return Dim; }

    __host__ __device__ T *data() { return data_; }

    __host__ __device__ const T *data() const { return data_; }

    __host__ __device__ T &operator[](std::int64_t i) { return data_[i]; }

    __host__ __device__ const T &operator[](std::int64_t i) const { return data_[i]; }

    __host__ __device__ T &x() { return data_[0]; }

    __host__ __device__ const T &x() const { return data_[0]; }

    __host__ __device__ T &y() { return data_[1]; }

    __host__ __device__ const T &y() const { return data_[1]; }

    __host__ __device__ T &z() { return data_[2]; }

    __host__ __device__ const T &z() const { return data_[2]; }

    __host__ __device__ Vec &normalize();

    __host__ __device__ static Vec<T, Dim> zero() noexcept {
        Vec<T, Dim> res;
#pragma unroll
        for (int i = 0; i < Dim; ++i) {
            res[i] = T(0);
        }
        return res;
    }

    template <std::int64_t NewDim>
    __host__ __device__ auto head() const noexcept -> Vec<T, NewDim> {
        Vec<T, NewDim> result;

#pragma unroll
        for (int i = 0; i < NewDim; ++i) {
            result[i] = (*this)[i];
        }
        return result;
    }

    __host__ __device__ auto operator+=(const Vec<T, Dim> &other) -> Vec & {
#pragma unroll
        for (int i = 0; i < Dim; ++i) {
            (*this)[i] += other[i];
        }
        return *this;
    }

    __host__ __device__ auto operator/=(const T &scalar) -> Vec & {
#pragma unroll
        for (int i = 0; i < Dim; ++i) {
            (*this)[i] /= scalar;
        }
        return *this;
    }

    template <class U>
    __host__ __device__ Vec<U, Dim> as() const {
        Vec<U, Dim> other;

#pragma unroll
        for (int i = 0; i < Dim; ++i) {
            other[i] = static_cast<U>((*this)[i]);
        }

        return other;
    }

  private:
    T data_[Dim];
};

template <class T, std::int64_t Dim>
__host__ __device__ auto operator+(const Vec<T, Dim> &x) -> Vec<T, Dim> {
    Vec<T, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = +x[i];
    }

    return result;
}

template <class T, std::int64_t Dim>
__host__ __device__ auto operator-(const Vec<T, Dim> &x) -> Vec<T, Dim> {
    Vec<T, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = -x[i];
    }

    return result;
}

/* ==================================== */
/*           Addition                   */
/* ==================================== */
template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator+(const Vec<T, Dim> &lhs, const Vec<U, Dim> &rhs)
    -> Vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    Vec<V, Dim> result = lhs;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] += rhs[i];
    }
    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator+(const Vec<T, Dim> &lhs, const U &scalar)
    -> Vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    Vec<V, Dim> result = lhs;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = result[i] + scalar;
    }
    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator+(const U &scalar, const Vec<T, Dim> &rhs)
    -> Vec<std::common_type_t<T, U>, Dim> {
    return rhs + scalar;
}

/* ==================================== */
/*           Subtraction                */
/* ==================================== */
template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator-(const Vec<T, Dim> &lhs, const Vec<U, Dim> &rhs)
    -> Vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    Vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] - rhs[i];
    }

    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator-(const Vec<T, Dim> &lhs, const U &scalar)
    -> Vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    Vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] - scalar;
    }
    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator-(const T &scalar, const Vec<U, Dim> &rhs)
    -> Vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    Vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = scalar - rhs[i];
    }
    return result;
}

/* ==================================== */
/*           Multiplication             */
/* ==================================== */
template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator*(const Vec<T, Dim> &lhs, const Vec<U, Dim> &rhs)
    -> Vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    Vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator*(const Vec<T, Dim> &lhs, const U &scalar)
    -> Vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    Vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] * scalar;
    }
    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator*(const U &scalar, const Vec<T, Dim> &rhs)
    -> Vec<std::common_type_t<T, U>, Dim> {
    return rhs * scalar;
}

/* ==================================== */
/*           Division                   */
/* ==================================== */
template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator/(const Vec<T, Dim> &lhs, const Vec<U, Dim> &rhs)
    -> Vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    Vec<V, Dim> result;
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] / rhs[i];
    }
    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator/(const Vec<T, Dim> &lhs, const U &scalar)
    -> Vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    Vec<V, Dim> result = lhs;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = result[i] / scalar;
    }
    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator/(const T &scalar, const Vec<U, Dim> &rhs)
    -> Vec<std::common_type_t<T, U>, Dim> {
    using V = std::common_type_t<T, U>;
    Vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = scalar / rhs[i];
    }
    return result;
}

/* ==================================== */
/*           Comparison                 */
/* ==================================== */
template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator==(const Vec<T, Dim> &lhs, const Vec<U, Dim> &rhs)
    -> Vec<bool, Dim> {
    Vec<bool, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] == rhs[i];
    }
    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator!=(const Vec<T, Dim> &lhs, const Vec<U, Dim> &rhs)
    -> Vec<bool, Dim> {
    return !(lhs == rhs);
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator<(const Vec<T, Dim> &lhs, const Vec<U, Dim> &rhs)
    -> Vec<bool, Dim> {
    Vec<bool, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] < rhs[i];
    }
    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator<=(const Vec<T, Dim> &lhs, const Vec<U, Dim> &rhs)
    -> Vec<bool, Dim> {
    Vec<bool, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] <= rhs[i];
    }
    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator>(const Vec<T, Dim> &lhs, const Vec<U, Dim> &rhs)
    -> Vec<bool, Dim> {
    Vec<bool, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] > rhs[i];
    }
    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator>=(const Vec<T, Dim> &lhs, const Vec<U, Dim> &rhs)
    -> Vec<bool, Dim> {
    Vec<bool, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] >= rhs[i];
    }
    return result;
}

/* ==================================== */
/*           Logic                      */
/* ==================================== */
template <class T, class U, std::int64_t Dim>
__host__ __device__ auto operator&&(const Vec<T, Dim> &lhs, const Vec<U, Dim> &rhs)
    -> Vec<bool, Dim> {
    Vec<bool, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = lhs[i] && rhs[i];
    }
    return result;
}

/* ==================================== */
/*           Other                      */
/* ==================================== */
template <class T, std::int64_t Dim>
__host__ __device__ auto argmax(const Vec<T, Dim> &v) -> std::size_t {
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

template <class T, std::int64_t Dim>
__host__ __device__ auto norm(const Vec<T, Dim> &v) -> T {
    // TODO: use thrust for this
    T result = 0;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result += v[i] * v[i];
    }

    return sqrt(result);
}

template <class T, std::int64_t Dim>
__host__ __device__ auto normalized(const Vec<T, Dim> &v) -> Vec<T, Dim> {
    auto copy = v;
    copy.normalize();
    return copy;
}

template <class T, std::int64_t Dim>
__host__ __device__ auto floor(const Vec<T, Dim> &v) -> Vec<T, Dim> {
    // TODO: use thrust for this
    auto copy = v;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        copy[i] = math::floor(v[i]);
    }
    return copy;
}

template <class T, std::int64_t Dim>
__host__ __device__ auto floori(const Vec<T, Dim> &v) -> Vec<std::int64_t, Dim> {
    // TODO: use thrust for this
    Vec<std::int64_t, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = static_cast<std::int64_t>(math::floor(v[i]));
    }
    return result;
}

template <class T, std::int64_t Dim>
__host__ __device__ auto ceil(const Vec<T, Dim> &v) -> Vec<T, Dim> {
    // TODO: use thrust for this
    Vec<T, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = math::ceil(v[i]);
    }
    return result;
}

template <class T, std::int64_t Dim>
__host__ __device__ auto ceili(const Vec<T, Dim> &v) -> Vec<std::int64_t, Dim> {
    // TODO: use thrust for this
    Vec<std::int64_t, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = static_cast<std::int64_t>(math::ceil(v[i]));
    }
    return result;
}

template <class T, std::int64_t Dim>
__host__ __device__ auto abs(const Vec<T, Dim> &v) -> Vec<T, Dim> {
    // TODO: use thrust for this
    auto copy = v;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        copy[i] = math::abs(v[i]);
    }
    return copy;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto min(const Vec<T, Dim> &x, const Vec<U, Dim> &y)
    -> Vec<std::common_type_t<T, U>, Dim> {
    // TODO: use thrust for this
    using V = std::common_type_t<T, U>;
    Vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = math::min(x[i], y[i]);
    }
    return result;
}

template <class T, class U, std::int64_t Dim>
__host__ __device__ auto max(const Vec<T, Dim> &x, const Vec<U, Dim> &y)
    -> Vec<std::common_type_t<T, U>, Dim> {
    // TODO: use thrust for this
    using V = std::common_type_t<T, U>;
    Vec<V, Dim> result;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        result[i] = math::max(x[i], y[i]);
    }
    return result;
}

template <class T, std::int64_t Dim>
__host__ __device__ auto sign(const Vec<T, Dim> &v) -> Vec<T, Dim> {
    // TODO: use thrust for this
    auto copy = v;

#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        copy[i] = math::sign(v[i]);
    }
    return copy;
}

/// @brief normalize the current vector
template <class T, std::int64_t Dim>
__host__ __device__ Vec<T, Dim> &Vec<T, Dim>::normalize() {
    auto norm = ::curad::norm(*this);
    *this /= norm;
    return *this;
}
} // namespace curad
