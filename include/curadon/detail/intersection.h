#pragma once

namespace curad::fp::kernel {
template <std::int64_t Dim>
__host__ __device__ std::tuple<bool, float, float>
intersection(const Vec<float, Dim> &boxmin, const Vec<float, Dim> &boxmax,
             const Vec<float, Dim> &ro, const Vec<float, Dim> &rd) {
    float tmin = std::numeric_limits<float>::min();
    float tmax = std::numeric_limits<float>::max();

    for (int d = 0; d < Dim; ++d) {
        auto t1 = __fdividef(boxmin[d] - ro[d], rd[d]);
        auto t2 = __fdividef(boxmax[d] - ro[d], rd[d]);

        // tmin = fmaxf(fminf(t1, t2),);
        // tmax = fminf(fmaxf(t1, t2));
        tmin = fminf(fmaxf(t1, tmin), fmaxf(t2, tmin));
        tmax = fmaxf(fminf(t1, tmax), fminf(t2, tmax));

        // if (threadIdx.x + blockIdx.x * blockDim.x == 45 &&
        //     threadIdx.z + blockIdx.z * blockDim.z == 7) {
        //     printf("tmin: %f, tmax: %f\n", tmin, tmax);
        // }
    }

    return {tmin <= tmax, tmin, tmax};
}
} // namespace curad::fp::kernel
