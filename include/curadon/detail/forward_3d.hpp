#pragma once

#include "curadon/cuda/error.h"
#include "curadon/device_span.hpp"
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

static constexpr std::uint64_t pixels_u_per_block_3d = 8;
static constexpr std::uint64_t pixels_v_per_block_3d = 8;
static constexpr std::uint64_t projections_per_block_3d = 8;

/// tex is the volume
template <class T>
__global__ void kernel_forward(device_span_3d<T> sinogram, Vec<std::uint64_t, 3> vol_shape,
                               float DSD, float DSO, cudaTextureObject_t tex, float accuracy,
                               // These should be moved to constant memory
                               Vec<float, 3> *uv_origins, Vec<float, 3> *delta_us,
                               Vec<float, 3> *delta_vs, Vec<float, 3> *sources) {
    const auto idx_u = threadIdx.x + blockIdx.x * blockDim.x;
    const auto idx_v = threadIdx.y + blockIdx.y * blockDim.y;

    const auto proj_number = threadIdx.z + blockIdx.z * blockDim.z;

    if (idx_u >= sinogram.shape()[0] || idx_v >= sinogram.shape()[1] ||
        proj_number >= sinogram.shape()[2]) {
        return;
    }

    const auto uv_origin = uv_origins[proj_number];
    const auto delta_u = delta_us[proj_number];
    const auto delta_v = delta_vs[proj_number];
    const auto source = sources[proj_number];

    // The detector point this thread is working on
    // const auto det_point = uv_origin + idx_u * delta_u + idx_v * delta_v;
    Vec<float, 3> det_point;
    det_point.x() = uv_origin.x() + idx_u * delta_u.x() + idx_v * delta_v.x();
    det_point.y() = uv_origin.y() + idx_u * delta_u.y() + idx_v * delta_v.y();
    det_point.z() = uv_origin.z() + idx_u * delta_u.z() + idx_v * delta_v.z();

    // direction from source to detector point
    auto dir = det_point - source;

    // how many steps to take along dir should we walk
    // TODO: This walks from the source to the  detector all the way, this is hyper inefficient,
    // clean this up (i.e. interect with AABB of volume)
    const auto nsamples = static_cast<std::int64_t>(::ceil(__fdividef(norm(dir), accuracy)));

    const Vec<float, 3> boxmin{-1, -1, -1};
    const Vec<float, 3> boxmax{vol_shape[0] + 1, vol_shape[1] + 1, vol_shape[2] + 1};
    auto [hit, tmin, tmax] = intersection(boxmin, boxmax, source, dir);

    if (!hit) {
        return;
    }

    // tmin and tmax are both within [0, 1], hence, we compute how many of the nsamples are within
    // this region and only walk that many samples
    const auto nsteps = static_cast<int>(ceilf((tmax - tmin) * nsamples));
    const auto step_length = (tmax - tmin) / nsteps;

    Vec<float, 3> t;
    float accumulator = 0;

    // TODO: i should start at a min t value, which is certainly not 0!
    for (float i = tmin; i <= tmax; i += step_length) {
        t = dir * i + source;
        accumulator += tex3D<float>(tex, t.x() + 0.5f, t.y() + 0.5f, t.z() + 0.5f);
    }

    float delta_length = norm(dir /* * vol_spacing */);
    sinogram(idx_u, idx_v, proj_number) = accumulator;
}
} // namespace kernel

template <class T, class U>
void forward_3d(device_volume<T> vol, device_measurement<U> sino) {
    auto volume = vol.device_data();
    const auto vol_shape = vol.shape();
    const auto vol_size = vol.extent();
    const auto vol_spacing = vol.spacing();
    const auto vol_offset = vol.offset();

    auto sinogram = sino.device_data();
    const auto angles = sino.angles();
    const auto det_shape = sino.shape();
    const auto det_spacing = sino.spacing();
    const auto DSD = sino.distance_source_to_detector();
    const auto DSO = sino.distance_source_to_object();

    const auto nangles = sino.nangles();

    // TODO: make this configurable
    const float accuracy = 1.f;

    // TODO: bind volume to texture
    cudaTextureObject_t tex;

    // allocate cuarray with size of volume
    const cudaExtent extent_alloc = make_cudaExtent(vol_shape[0], vol_shape[1], vol_shape[2]);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    gpuErrchk(cudaMalloc3DArray(&array, &channelDesc, extent_alloc));
    gpuErrchk(cudaPeekAtLastError());

    // Copy to cuda array
    cudaMemcpy3DParms copyParams = {0};

    copyParams.srcPtr =
        make_cudaPitchedPtr((void *)volume, vol_shape[0] * sizeof(T), vol_shape[0], vol_shape[1]);

    copyParams.dstArray = array;
    copyParams.extent = extent_alloc;
    copyParams.kind = cudaMemcpyDefault;
    gpuErrchk(cudaMemcpy3DAsync(&copyParams, 0)); // TODO: use stream pool
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
    texDescr.addressMode[2] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;

    gpuErrchk(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
    gpuErrchk(cudaPeekAtLastError());

    cudaResourceDesc desc;
    cudaGetTextureObjectResourceDesc(&desc, tex);

    // TODO: compute uv origins, delta_u, delta_v, sources
    thrust::host_vector<Vec<float, 3>> host_uv_origins(nangles);
    thrust::host_vector<Vec<float, 3>> host_deltas_us(nangles);
    thrust::host_vector<Vec<float, 3>> host_delta_vs(nangles);
    thrust::host_vector<Vec<float, 3>> host_sources(nangles);

    // distance object to detector
    const auto DOD = DSD - DSO;
    for (int i = 0; i < nangles; ++i) {
        Vec<float, 3> init_source({0, 0, -DSO});

        // Assume detector origin is at the bottom left corner, i.e. detector point (0, 0)
        Vec<float, 3> init_det_origin{
            -det_spacing[0] * (det_shape[0] / 2.f) + det_spacing[0] * 0.5f, // u
            -det_spacing[1] * (det_shape[1] / 2.f) + det_spacing[1] * 0.5f, // v
            0.f};

        // detector point (1,0)
        Vec<float, 3> init_delta_u = init_det_origin + Vec<float, 3>{det_spacing[0], 0.f, 0.f};
        // detector point (0, 1)
        Vec<float, 3> init_delta_v = init_det_origin + Vec<float, 3>{0.f, det_spacing[1], 0.f};

        // Apply geometry transformation, such that volume origin coincidence with world origin,
        // the volume voxels are unit size, for all projections, the image stays the same

        // 1) apply roll, pitch, yaw of detector
        // TODO

        // 2) translate to real detector position
        init_det_origin[2] += DOD;
        init_delta_u[2] += DOD;
        init_delta_v[2] += DOD;

        // 3) Rotate according to current position
        auto det_origin = ::curad::geometry::rotate_yzy(init_det_origin, angles[i], 0.f, 0.f);
        auto delta_u = ::curad::geometry::rotate_yzy(init_delta_u, angles[i], 0.f, 0.f);
        auto delta_v = ::curad::geometry::rotate_yzy(init_delta_v, angles[i], 0.f, 0.f);
        auto source = ::curad::geometry::rotate_yzy(init_source, angles[i], 0.f, 0.f);

        // 4) move everything such that volume origin coincides with world origin
        const auto translation = vol_size / 2.f - vol_spacing / 2;
        det_origin = det_origin - vol_offset + translation;
        delta_u = delta_u - vol_offset + translation;
        delta_v = delta_v - vol_offset + translation;
        source = source - vol_offset + translation;

        // 5) scale such that volume voxels are unit size
        det_origin = det_origin / vol_spacing;
        delta_u = delta_u / vol_spacing;
        delta_v = delta_v / vol_spacing;
        source = source / vol_spacing;

        // 6) Apply center of rotation correction
        // TODO

        // 7) store in host vector
        host_uv_origins[i] = det_origin;
        host_deltas_us[i] = delta_u - det_origin;
        host_delta_vs[i] = delta_v - det_origin;
        host_sources[i] = source;
    }

    // upload uv_origin, delta_u, delta_v, sources
    thrust::device_vector<Vec<float, 3>> dev_uv_origins = host_uv_origins;
    thrust::device_vector<Vec<float, 3>> dev_deltas_us = host_deltas_us;
    thrust::device_vector<Vec<float, 3>> dev_delta_vs = host_delta_vs;
    thrust::device_vector<Vec<float, 3>> dev_sources = host_sources;

    auto uv_origins = thrust::raw_pointer_cast(dev_uv_origins.data());
    auto deltas_us = thrust::raw_pointer_cast(dev_deltas_us.data());
    auto delta_vs = thrust::raw_pointer_cast(dev_delta_vs.data());
    auto sources = thrust::raw_pointer_cast(dev_sources.data());

    const std::uint64_t div_u = kernel::pixels_u_per_block_3d;
    const std::uint64_t div_v = kernel::pixels_v_per_block_3d;

    dim3 grid(utils::round_up_division(det_shape[0], div_u),
              utils::round_up_division(det_shape[1], div_v),
              utils::round_up_division(nangles, kernel::projections_per_block_3d));
    dim3 block(div_u, div_v, kernel::projections_per_block_3d);

    auto sino_span = sino.kernel_span();

    kernel::kernel_forward<<<grid, block>>>(sino_span, vol_shape, DSD, DSO, tex, accuracy,
                                            uv_origins, deltas_us, delta_vs, sources);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

} // namespace curad::fp
