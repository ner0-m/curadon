#pragma once

#include "curadon/detail/error.h"
#include "curadon/math/vector.hpp"
#include "curadon/rotation.h"
#include "curadon/utils.hpp"

#include "curadon/detail/intersection.h"

#include <cmath>
#include <cstdio>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda_runtime_api.h>

namespace curad::fp {
namespace kernel {

static constexpr std::uint64_t pixels_u_per_block_2d = 8;
static constexpr std::uint64_t num_projections_per_block_2d = 1;
static constexpr std::uint64_t num_projections_per_kernel_2d = 1;

/// tex is the volume
template <class T>
__global__ void
kernel_forward_2d(T *sinogram, vec<std::uint64_t, 2> vol_shape, float DSD, float DSO,
                  std::uint64_t det_shape, std::int64_t total_projections, cudaTextureObject_t tex,
                  float accuracy,
                  // These should be moved to constant memory
                  vec<float, 2> *uv_origins, vec<float, 2> *delta_us, vec<float, 2> *sources) {
    const auto idx_u = threadIdx.x + blockIdx.x * blockDim.x;

    // TODO: make this less strict, enable this kernel to be called multiple times
    const auto proj_number = threadIdx.y + blockIdx.y * blockDim.y;

    // if (idx_u == 0) {
    //     printf("idx_u: %d, proj_number: %d\n", idx_u, proj_number);
    // }

    if (idx_u >= det_shape || proj_number >= total_projections)
        return;

    const auto print_proj = 0;
    const bool do_print = idx_u == 0 || idx_u == det_shape - 1;

    const std::int64_t stride_u = 1;
    const std::int64_t stride_proj = det_shape;
    const std::int64_t idx = idx_u * stride_u + proj_number * stride_proj;

    const auto uv_origin = uv_origins[proj_number];
    const auto delta_u = delta_us[proj_number];
    const auto source = sources[proj_number];

    // The detector point this thread is working on
    // const auto det_point = uv_origin + idx_u * delta_u + idx_v * delta_v;
    vec<float, 2> det_point;
    det_point.x() = uv_origin.x() + idx_u * delta_u.x();
    det_point.y() = uv_origin.y() + idx_u * delta_u.y();

    // direction from source to detector point
    auto dir = det_point - source;


    // how many steps to take along dir should we walk
    // TODO: This walks from the source to the  detector all the way, this is hyper inefficient,
    // clean this up (i.e. interect with AABB of volume)
    auto dir_len = norm(dir);
    const auto nsamples = static_cast<std::int64_t>(::ceil(__fdividef(dir_len, accuracy)));

    dir /= dir_len;

    if (do_print) {
        // printf("dir: %f %f\n", dir.x(), dir.y());
        // printf("source: %f %f\n", source.x(), source.y());
        // printf("det_point: %f %f\n", det_point.x(), det_point.y());
        printf("proj_number: %d, idx_u: %d, dir: %f %f, source: %f %f, det_point: %f %f\n",
               proj_number, idx_u, dir.x(), dir.y(), source.x(), source.y(), det_point.x(),
               det_point.y());
    }

    const vec<float, 2> boxmin{-1, -1};
    const vec<float, 2> boxmax{vol_shape[0] + 1, vol_shape[1] + 1};
    auto [hit, tmin, tmax] = intersection(boxmin, boxmax, source, dir);

    if (!hit) {
        return;
    }

    // tmin and tmax are both within [0, 1], hence, we compute how many of the nsamples are within
    // this region and only walk that many samples
    const auto nsteps = static_cast<int>(ceilf((tmax - tmin) / dir_len * nsamples));
    const auto step_length = (tmax - tmin) / (float)nsteps;

    if (do_print) {
        printf("proj_number: %d, idx_u: %d, nsteps: %d, tmin: %f, tmax: %f, step_length: %f, "
               "dir_len %f\n",
               proj_number, idx_u, nsteps, tmin, tmax, step_length, dir_len);
    }

    vec<float, 2> p;
    float accumulator = 0;

    auto step = 0;
    for (float t = tmin; t <= tmax; t += step_length) {
        p = dir * t + source;
        const auto partial = tex2D<float>(tex, p.x(), p.y());
        accumulator += partial;
        // if (partial > 0.000 && do_print) {
        //     printf("step: %d, proj_number: %d, idx_u: %d, t: %f, partial: %f, p: %f %f\n", step, proj_number, idx_u,
        //            t, partial, p.x(), p.y());
        // }
        // ++step;
        // __syncthreads();
    }

    if (do_print) {
        printf("proj_number: %d, idx_u: %d, accumulator: %f, delta_length: %f\n", proj_number,
               idx_u, accumulator, dir_len);
    }

    // TODO: dir_len * vol_spacing
    sinogram[idx] = accumulator * dir_len;
}
} // namespace kernel

template <class T, class U>
void forward_2d(T *volume, vec<std::uint64_t, 2> vol_shape, vec<float, 2> vol_size,
                vec<float, 2> vol_spacing, vec<float, 2> vol_offset, U *sinogram,
                std::uint64_t det_shape, float det_spacing, std::vector<float> angles, float DSD,
                float DSO) {

    std::size_t nangles = angles.size();

    // TODO: make this configurable
    const float accuracy = 1.f;

    // TODO: bind volume to texture
    cudaTextureObject_t tex;

    // allocate cuarray with size of volume
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    gpuErrchk(cudaMallocArray(&array, &channelDesc, vol_shape[0], vol_shape[1]));
    gpuErrchk(cudaPeekAtLastError());

    const auto size = vol_shape[0] * vol_shape[1] * sizeof(float);
    gpuErrchk(cudaMemcpyToArray(array, 0, 0, volume, size, cudaMemcpyDefault));
    gpuErrchk(cudaStreamSynchronize(0));
    gpuErrchk(cudaPeekAtLastError());

    // bind texture
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = array;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;

    gpuErrchk(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
    gpuErrchk(cudaPeekAtLastError());

    cudaResourceDesc desc;
    cudaGetTextureObjectResourceDesc(&desc, tex);

    const int num_kernel_calls =
        utils::round_up_division(nangles, kernel::num_projections_per_kernel_2d);
    for (int i = 0; i < num_kernel_calls; ++i) {
        const auto proj_idx = i * kernel::num_projections_per_kernel_2d;
        const auto num_projections_left = nangles - proj_idx;
        const auto num_projections =
            std::min<int>(kernel::num_projections_per_kernel_2d, num_projections_left);

        // TODO: compute uv origins, delta_u, delta_v, sources
        thrust::host_vector<vec<float, 2>> host_uv_origins(num_projections);
        thrust::host_vector<vec<float, 2>> host_deltas_us(num_projections);
        thrust::host_vector<vec<float, 2>> host_sources(num_projections);

        // distance object to detector
        const auto DOD = DSD - DSO;
        for (int i = 0; i < num_projections; ++i) {
            auto angle = angles[proj_idx + i];
            vec<float, 2> init_source({0, -DSO});

            // Assume detector origin is at the bottom left corner, i.e. detector point (0, 0)
            vec<float, 2> init_det_origin{-det_spacing * (det_shape / 2.f) + det_spacing * 0.5f,
                                          0.f};

            // detector point (1)
            vec<float, 2> init_delta_u = init_det_origin + vec<float, 2>{det_spacing, 0.f};

            // Apply geometry transformation, such that volume origin coincidence with world origin,
            // the volume voxels are unit size, for all projections, the image stays the same

            // 1) apply roll, pitch, yaw of detector
            // TODO

            // 2) translate to real detector position
            init_det_origin[1] += DOD;
            init_delta_u[1] += DOD;

            // 3) Rotate according to current position
            auto det_origin = ::curad::geometry::rotate(init_det_origin, angle);
            auto delta_u = ::curad::geometry::rotate(init_delta_u, angle);
            auto source = ::curad::geometry::rotate(init_source, angle);

            // 4) move everything such that volume origin coincides with world origin
            const auto translation = vol_size / 2.f - vol_spacing / 2;
            det_origin = det_origin - vol_offset + translation;
            delta_u = delta_u - vol_offset + translation;
            source = source - vol_offset + translation;

            // 5) scale such that volume voxels are unit size
            det_origin = det_origin / vol_spacing;
            delta_u = delta_u / vol_spacing;
            source = source / vol_spacing;

            // 6) Apply center of rotation correction
            // TODO

            // 7) store in host vector
            host_uv_origins[i] = det_origin;
            host_deltas_us[i] = delta_u - det_origin;
            host_sources[i] = source;
        }

        // upload uv_origin, delta_u, delta_v, sources
        thrust::device_vector<vec<float, 2>> dev_uv_origins = host_uv_origins;
        thrust::device_vector<vec<float, 2>> dev_deltas_us = host_deltas_us;
        thrust::device_vector<vec<float, 2>> dev_sources = host_sources;

        auto uv_origins = thrust::raw_pointer_cast(dev_uv_origins.data());
        auto deltas_us = thrust::raw_pointer_cast(dev_deltas_us.data());
        auto sources = thrust::raw_pointer_cast(dev_sources.data());

        const std::uint64_t div_u = kernel::pixels_u_per_block_2d;

        dim3 grid(utils::round_up_division(det_shape, div_u),
                  utils::round_up_division(num_projections, kernel::num_projections_per_block_2d));
        dim3 block(div_u, kernel::num_projections_per_block_2d);

        // Move sinogram ptr ahead
        auto sinogram_ptr = sinogram + proj_idx * det_shape;
        std::cout << "Calling kernel for projection " << proj_idx << " / " << nangles
                  << " (start angle: " << angles[proj_idx] * 180. / M_PI << ")\n";
        kernel::kernel_forward_2d<<<grid, block>>>(sinogram_ptr, vol_shape, DSD, DSO, det_shape,
                                                   nangles, tex, accuracy, uv_origins, deltas_us,
                                                   sources);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "\n";
    }
}

} // namespace curad::fp
