#pragma once

#include "curadon/math/vector.hpp"
#include "curadon/utils.hpp"
#include <cstdint>
#include <cuda_runtime_api.h>

#define gpuErrchk(answer)                                                                          \
    { gpuAssert((answer), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: '%s: %s' (%d) in %s:%d\n", cudaGetErrorName(code),
                cudaGetErrorString(code), code, file, line);
        if (abort) {
            exit(code);
        }
    }
}

namespace curad {
namespace detail {
/// Basically computes the multiplication of R_y(psi) * R_z(theta) * R_x(phi) * v
template <class T>
__host__ __device__ Vec<T, 3> rotate_yzy(Vec<T, 3> v, T phi, T theta, T psi) {
    const auto cphi = std::cos(phi);
    const auto sphi = std::sin(phi);
    const auto ctheta = std::cos(theta);
    const auto stheta = std::sin(theta);
    const auto spsi = std::sin(psi);
    const auto cpsi = std::cos(psi);

    // Rotate around  y
    v[0] = v[0] * cphi - v[2] * sphi;
    v[2] = v[0] * sphi + v[2] * cphi;

    // Rotate around z
    v[0] = v[0] * ctheta - v[1] * stheta;
    v[1] = -v[0] * stheta + v[1] * ctheta;

    // Rotate around y again
    v[0] = v[0] * cpsi - v[2] * spsi;
    v[2] = v[0] * spsi + v[2] * cpsi;
    return v;
}
} // namespace detail

/// Geometry for a single pose, currently still assumes flat panel detector
///
/// The geometry class assumes a left-handed coordinate system. The volume
/// is centered at the origin, the detector is in the positive z direction,
/// and the source in the negative z direction. Rotations are applied according to YZY.
///
/// The origin of the detector is at the center of the detector, u goes in the
/// same direction as the x-axis, v goes in the same direction as the y-axis.
class geometry {
    // Don't really need the option to choose it, so just hardcode it here.
    using T = float;
    static constexpr std::int64_t Dim = 3;
    static constexpr std::int64_t RangeDim = Dim - 1;

  public:
    __host__ __device__ std::uint64_t volume_shape(std::size_t i) const { return vol_shape_[i]; }

    __host__ __device__ std::uint64_t det_shape(std::size_t i) const { return det_shape_[i]; }

    __host__ __device__ T distance_source_to_origin() const { return DSO; }

    __host__ __device__ T distance_source_to_detector() const { return DSD; }

    __host__ __device__ T distance_origin_to_detector() const { return DSD - DSO; }

    __host__ __device__ T rotation_correction() const { return COR_; }

    __host__ __device__ T phi() const { return phi_; }

    __host__ __device__ T psi() const { return psi_; }

    __host__ __device__ T theta() const { return theta_; }

    __host__ __device__ Vec<T, Dim> source() const {
        // TODO: apply roll, pitch, yaw of detector
        Vec<T, Dim> init_source({0, 0, -distance_source_to_origin()});
        return init_source;
    }

    // Return world coordinates of the center of the volume origin (0, 0, 0)
    __host__ __device__ Vec<T, Dim> vol_origin() const {
        // TODO: apply roll, pitch, yaw of detector
        Vec<T, Dim> init_vol_origin = -vol_size_ / 2.f + vol_spacing_ / 2.f + vol_offset_;
        return detail::rotate_yzy(init_vol_origin, phi_, theta_, psi_);
    }

    __host__ __device__ Vec<T, Dim> vol_offset() const { return vol_offset_; }

    /// Return difference between world position of volume (0, 0, 0) and (1, 0, 0)
    __host__ __device__ Vec<T, Dim> vol_origin_delta_x() const { return compute_delta(0); }

    /// Return difference between world position of volume (0, 0, 0) and (0, 1, 0)
    __host__ __device__ Vec<T, Dim> vol_origin_delta_y() const { return compute_delta(1); }

    /// Return difference between world position of volume (0, 0, 0) and (0, 0, 1)
    __host__ __device__ Vec<T, Dim> vol_origin_delta_z() const { return compute_delta(2); }

    __host__ __device__ Vec<T, Dim> det_origin() const {
        // TODO: apply roll, pitch, yaw of detector
        Vec<T, Dim> init_det_origin(
            {det_offset_[0], det_offset_[1], -distance_origin_to_detector()});
        return detail::rotate_yzy(init_det_origin, phi_, theta_, psi_);
    }

    // private:
    __host__ __device__ Vec<T, Dim> compute_delta(std::size_t i) const {
        // TODO: apply roll, pitch, yaw of detector
        Vec<T, Dim> init_delta = -vol_size_ / 2.f + vol_spacing_ / 2.f + vol_offset_;
        init_delta[i] += vol_spacing_[i];
        init_delta = detail::rotate_yzy(init_delta, phi_, theta_, psi_);

        return init_delta - vol_origin();
    }

    /// Distance source to detector
    T DSD;

    /// Distance source to origin
    T DSO;

    /// Shape of detector (u, v)
    Vec<std::uint64_t, RangeDim> det_shape_;

    /// Spacing of detector (spacing of u, then v)
    Vec<T, RangeDim> det_spacing_;

    /// Size in world of detector (shape * spacing)
    Vec<T, RangeDim> det_size_;

    /// Offset of detector origin / offset correction (u, v)
    Vec<T, RangeDim> det_offset_;

    /// Rotation of detector (roll, pitch, yaw)
    Vec<T, Dim> det_rotation_;

    /// Center of rotation correction
    T COR_;

    /// Shape of volume
    Vec<std::uint64_t, Dim> vol_shape_;

    /// Spacing of volume
    Vec<T, Dim> vol_spacing_;

    /// Size in world of volume (shape * spacing)
    Vec<T, Dim> vol_size_;

    /// Offset of volume origin
    Vec<T, Dim> vol_offset_;

    /// First rotation around y-axis
    T phi_;

    /// Rotation around z-axis
    T theta_;

    /// Rotation around y-axis
    T psi_;
};

static constexpr std::int64_t num_projects_per_kernel = 32;

static constexpr std::int64_t num_voxels_per_thread = 8;

__constant__ Vec<float, 3> dev_vol_origin[num_projects_per_kernel];
__constant__ Vec<float, 3> dev_delta_x[num_projects_per_kernel];
__constant__ Vec<float, 3> dev_delta_y[num_projects_per_kernel];
__constant__ Vec<float, 3> dev_delta_z[num_projects_per_kernel];

template <class T>
__global__ void kernel_backprojection(T *volume, std::int64_t stride_x, std::int64_t stride_y,
                                      std::int64_t stride_z, Vec<float, 3> source,
                                      Vec<std::uint64_t, 3> vol_shape, float DSD, float DSO,
                                      Vec<std::uint64_t, 2> det_shape, std::int64_t cur_projection,
                                      std::int64_t total_projections, cudaTextureObject_t tex) {
    auto idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    auto idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    auto start_idx_z = blockIdx.z * num_voxels_per_thread + threadIdx.z;

    if (idx_x >= vol_shape[0] || idx_y >= vol_shape[1] || start_idx_z >= vol_shape[2])
        return;

    // Each thread has an array of volume voxels, that we first read, then
    // work on, and then write back once
    float local_volume[num_voxels_per_thread];

    // First step, load all values into the local_volume array
#pragma unroll
    for (int z = 0; z < num_voxels_per_thread; ++z) {
        const auto idx_z = start_idx_z + z;
        if (idx_z >= vol_shape[2]) {
            break;
        }

        const auto vol_idx = idx_x * stride_x + idx_y * stride_y + idx_z * stride_z;

        local_volume[z] = volume[vol_idx];
    }

    // Second step, for all projections this kernel performs, do the backprojection
    // for all voxels this thread is associated with
    for (int i = 0; i < num_projects_per_kernel; ++i) {
        auto idx_proj = cur_projection * num_projects_per_kernel + i;

        if (idx_proj >= total_projections) {
            break;
        }

        auto vol_origin = dev_vol_origin[i];
        auto delta_x = dev_delta_x[i];
        auto delta_y = dev_delta_y[i];
        auto delta_z = dev_delta_z[i];

#pragma unroll
        for (int z = 0; z < num_voxels_per_thread; ++z) {
            const auto idx_z = start_idx_z + z;

            if (idx_z >= vol_shape[2]) {
                break;
            }

            // Compute world coordinates of the working voxel
            // move idx_x delta_x from origin, same for y and z
            auto P = vol_origin + idx_x * delta_x + idx_y * delta_y + idx_z * delta_z;

            // Compute line from source to P
            auto dir = P - source;

            // Compute intersection of detector with dir
            auto t = __fdividef(DSO - DSD - source[2], dir[2]);

            // Coordinates are from [-det_shape / 2, det_shape / 2], hence shift it to be strictly
            // positive
            auto u = (dir[0] * t + source[0]) + det_shape[0] / 2;
            auto v = (dir[1] * t + source[1]) + det_shape[1] / 2;

            auto sample = tex3D<float>(tex, u, v, i + 0.5f);

            local_volume[z] += sample;
        }
    }

    // Last step, write local volume back to global one
#pragma unroll
    for (int z = 0; z < num_voxels_per_thread; ++z) {
        const auto idx_z = start_idx_z + z;
        if (idx_z >= vol_shape[2]) {
            break;
        }

        const auto vol_idx = idx_x * stride_x + idx_y * stride_y + idx_z * stride_z;

        volume[vol_idx] = local_volume[z];
    }
}

namespace detail {
cudaArray_t allocate_cuarray(std::size_t width, std::size_t height, std::size_t depth) {
    const cudaExtent extent_alloc = make_cudaExtent(width, height, depth);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    cudaMalloc3DArray(&array, &channelDesc, extent_alloc);
    gpuErrchk(cudaPeekAtLastError());

    return array;
}

template <class T>
void copy_projections_to_array(T *sinogram, std::size_t offset, cudaArray_t array_cu,
                               std::size_t width, std::size_t height, std::size_t num_projections) {
    // Copy to cuda array
    cudaMemcpy3DParms copyParams = {0};

    auto ptr = sinogram + offset;
    copyParams.srcPtr = make_cudaPitchedPtr((void *)ptr, width * sizeof(float), width, height);

    const cudaExtent extent = make_cudaExtent(width, height, num_projections);
    gpuErrchk(cudaPeekAtLastError());
    copyParams.dstArray = array_cu;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDefault;
    cudaMemcpy3DAsync(&copyParams, 0); // TODO: use stream pool
    cudaStreamSynchronize(0);
    gpuErrchk(cudaPeekAtLastError());
}

void copy_to_texture(cudaTextureObject_t *tex, cudaArray_t array_cu) {
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = array_cu;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.addressMode[2] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(tex, &texRes, &texDescr, NULL);
    gpuErrchk(cudaPeekAtLastError());
}

void setup_constants(std::size_t start_proj, std::size_t num_projections, Vec<float, 3> vol_size,
                     Vec<float, 3> vol_spacing, Vec<float, 3> vol_offset,
                     const std::vector<float> &angles) {
    std::vector<curad::Vec<float, 3>> vol_origins;
    std::vector<curad::Vec<float, 3>> delta_xs;
    std::vector<curad::Vec<float, 3>> delta_ys;
    std::vector<curad::Vec<float, 3>> delta_zs;
    for (int j = start_proj; j < start_proj + num_projections; ++j) {
        float angle = angles[j] * M_PI / 180.f;

        curad::Vec<float, 3> init_vol_origin = -vol_size / 2.f + vol_spacing / 2.f + vol_offset;
        auto vol_origin = curad::detail::rotate_yzy(init_vol_origin, angle, 0.f, 0.f);
        vol_origins.push_back(vol_origin);

        curad::Vec<float, 3> init_delta;
        init_delta = init_vol_origin;
        init_delta[0] += vol_spacing[0];
        init_delta = curad::detail::rotate_yzy(init_delta, angle, 0.f, 0.f);
        delta_xs.push_back(init_delta - vol_origin);

        init_delta = init_vol_origin;
        init_delta[1] += vol_spacing[1];
        init_delta = curad::detail::rotate_yzy(init_delta, angle, 0.f, 0.f);
        delta_ys.push_back(init_delta - vol_origin);

        init_delta = init_vol_origin;
        init_delta[2] += vol_spacing[2];
        init_delta = curad::detail::rotate_yzy(init_delta, angle, 0.f, 0.f);
        delta_zs.push_back(init_delta - vol_origin);
    }

    cudaMemcpyToSymbol(curad::dev_vol_origin, vol_origins.data(),
                       sizeof(curad::Vec<float, 3>) * curad::num_projects_per_kernel, 0,
                       cudaMemcpyDefault);
    cudaMemcpyToSymbol(curad::dev_delta_x, delta_xs.data(),
                       sizeof(curad::Vec<float, 3>) * curad::num_projects_per_kernel, 0,
                       cudaMemcpyDefault);
    cudaMemcpyToSymbol(curad::dev_delta_y, delta_ys.data(),
                       sizeof(curad::Vec<float, 3>) * curad::num_projects_per_kernel, 0,
                       cudaMemcpyDefault);
    cudaMemcpyToSymbol(curad::dev_delta_z, delta_zs.data(),
                       sizeof(curad::Vec<float, 3>) * curad::num_projects_per_kernel, 0,
                       cudaMemcpyDefault);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
} // namespace detail

template <class T, class U>
void backproject_3d(T *volume, std::int64_t stride_x, std::int64_t stride_y, std::int64_t stride_z,
                    Vec<std::uint64_t, 3> vol_shape, Vec<float, 3> vol_size,
                    Vec<float, 3> vol_spacing, Vec<float, 3> vol_offset, U *sinogram,
                    std::int64_t sino_width, std::int64_t sino_height,
                    Vec<std::uint64_t, 2> det_shape, std::vector<float> angles,
                    Vec<float, 3> source, float DSD, float DSO) {
    const auto nangles = angles.size();

    // allocate cuda array for sinogram
    auto array_cu = detail::allocate_cuarray(sino_width, sino_height, num_projects_per_kernel);

    cudaTextureObject_t tex;

    auto num_kernel_calls =
        (nangles + curad::num_projects_per_kernel - 1) / curad::num_projects_per_kernel;
    for (int i = 0; i < num_kernel_calls; ++i) {
        auto proj_idx = i * curad::num_projects_per_kernel;

        auto projections_left = nangles - (i * curad::num_projects_per_kernel);

        // On how many projections do we work on this call? either num_projects_per_kernel, or
        // what ever is left
        const auto num_projections =
            std::min<int>(curad::num_projects_per_kernel, projections_left);

        // Copy projection data necessary for the next kernel to cuda array
        detail::copy_projections_to_array(sinogram, proj_idx * sino_width * sino_height, array_cu,
                                          sino_width, sino_height, num_projections);

        // Bind cuda array containing the sinogram to the texture
        detail::copy_to_texture(&tex, array_cu);

        // kernel uses variables stored in __constant__ memory, e.g. volume origin, volume
        // deltas Compute them here and upload them
        detail::setup_constants(proj_idx, num_projections, vol_size, vol_spacing, vol_offset,
                                angles);

        int divx = 16;
        int divy = 32;
        int divz = curad::num_voxels_per_thread;

        dim3 threads_per_block(divx, divy, 1);

        int block_x = (vol_shape[0] + divx - 1) / divx;
        int block_y = (vol_shape[1] + divy - 1) / divy;
        int block_z = (vol_shape[2] + divz - 1) / divz;
        dim3 num_blocks(block_x, block_y, block_z);
        kernel_backprojection<<<num_blocks, threads_per_block>>>(volume, stride_x, stride_y,
                                                                 stride_z, source, vol_shape, DSD,
                                                                 DSO, det_shape, i, nangles, tex);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    cudaDestroyTextureObject(tex);
}

} // namespace curad
