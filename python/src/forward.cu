#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "curadon/forward.hpp"
#include "curadon/image.hpp"
#include "curadon/math/vector.hpp"
#include "curadon/measurement.hpp"
#include "curadon/types.hpp"

#include <vector>

namespace nb = nanobind;

void forward_3d_cuda(
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any, nb::any>, nb::device::cuda, nb::c_contig>
        volume,
    nb::ndarray<curad::u64, nb::shape<3>, nb::device::cpu> vol_shape,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> vol_spacing,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> vol_offset,
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any, nb::any>, nb::device::cuda, nb::c_contig>
        sinogram,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> angles,
    nb::ndarray<curad::u64, nb::shape<2>, nb::device::cpu> det_shape,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> det_spacing,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> det_offset,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> det_rotation, curad::f32 DSO,
    curad::f32 DSD, curad::f32 COR) {

    // Never forgetti, Python calls (z, y, x), we do (x, y, z)
    curad::vec3u curad_vol_shape{vol_shape(2), vol_shape(1), vol_shape(0)};
    curad::vec3f curad_vol_spacing{vol_spacing(2), vol_spacing(1), vol_spacing(0)};
    curad::vec3f curad_vol_offset{vol_offset(2), vol_offset(1), vol_offset(0)};

    curad::device_volume<float> vol_span(volume.data(), curad_vol_shape, curad_vol_spacing,
                                         curad_vol_offset);

    // never forgetti, python calls (v, u), we do (u, v)
    curad::vec3u curad_det_shape{det_shape(1), det_shape(0), angles.size()};
    curad::vec2f curad_det_spacing{det_spacing(1), det_spacing(0)};
    curad::vec2f curad_det_offset{det_offset(1), det_offset(0)};

    std::vector<float> cpu_angles(angles.data(), angles.data() + angles.size());

    curad::device_measurement<float> sino_span(sinogram.data(), curad_det_shape, curad_det_spacing,
                                               curad_det_offset);
    sino_span.set_angles(cpu_angles);
    sino_span.set_distance_source_to_detector(DSD);
    sino_span.set_distance_source_to_object(DSO);
    sino_span.set_center_of_rotation_correction(COR);
    sino_span.set_pitch(det_rotation(0));
    sino_span.set_roll(det_rotation(1));
    sino_span.set_yaw(det_rotation(2));

    // printf("Volume data pointer : %p\n", volume.data());
    // printf("Volume dimension : %zu\n", volume.ndim());
    // printf("vol_shape: %zu, %zu, %zu\n", curad_vol_shape[0], curad_vol_shape[1],
    //        curad_vol_shape[2]);
    // printf("vol_spacing: %f, %f, %f\n", curad_vol_spacing[0], curad_vol_spacing[1],
    //        curad_vol_spacing[2]);
    // printf("vol_offset: %f, %f, %f\n", curad_vol_offset[0], curad_vol_offset[1],
    //        curad_vol_offset[2]);
    //
    // printf("Sinogram data pointer : %p\n", sinogram.data());
    // printf("Sinogram dimension : %zu\n", sinogram.ndim());
    // std::cout << "det_shape: " << det_shape(0) << ", " << det_shape(1) << "\n";
    // std::cout << "det_shape: " << curad_det_shape[0] << ", " << curad_det_shape[1] << ", "
    //           << angles.size() << "\n";
    // std::cout << "det_spacing: " << curad_det_spacing[0] << ", " << curad_det_spacing[1] << "\n";
    // std::cout << "det_offset: " << curad_det_offset[0] << ", " << curad_det_offset[1] << "\n";
    // std::cout << "det_rotation: " << det_rotation(0) << ", " << det_rotation(1) << ", "
    //           << det_rotation(2) << "\n";
    //
    // printf("DSO: %f\n", DSO);
    // printf("DSD: %f\n", DSD);
    // printf("COR: %f\n", COR);

    curad::fp::forward_3d(vol_span, sino_span);
}

void forward_2d_cuda(
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> vol,
    nb::ndarray<curad::u64, nb::shape<2>, nb::device::cpu> vol_shape,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> vol_spacing,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> vol_offset,
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> sino,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> angles, curad::u64 det_shape,
    curad::f32 det_spacing, curad::f32 det_offset, curad::f32 det_rotation, curad::f32 DSO,
    curad::f32 DSD, curad::f32 COR) {

    // Never forgetti, Python calls (z, y, x), we do (x, y, z)
    curad::vec2u curad_vol_shape{vol_shape(1), vol_shape(0)};
    curad::vec2f curad_vol_spacing{vol_spacing(1), vol_spacing(0)};
    curad::vec2f curad_vol_offset{vol_offset(1), vol_offset(0)};
    curad::vec2f curad_vol_extent = curad_vol_shape * curad_vol_spacing;

    std::vector<curad::f32> cpu_angles(angles.data(), angles.data() + angles.size());

    std::cout << "vol_shape: " << curad_vol_shape[0] << ", " << curad_vol_shape[1] << "\n";
    std::cout << "vol_extent: " << curad_vol_extent[0] << ", " << curad_vol_extent[1] << "\n";
    std::cout << "vol_spacing: " << curad_vol_spacing[0] << ", " << curad_vol_spacing[1] << "\n";
    std::cout << "vol_offset: " << curad_vol_offset[0] << ", " << curad_vol_offset[1] << "\n";
    std::cout << "sino_shape: " << angles.size() << "\n";
    std::cout << "det_shape: " << det_shape << "\n";
    std::cout << "det_spacing: " << det_spacing << "\n";
    std::cout << "det_offset: " << det_offset << "\n";
    std::cout << "det_rotation: " << det_rotation << "\n";
    std::cout << "DSO: " << DSO << "\n";
    std::cout << "DSD: " << DSD << "\n";
    std::cout << "vol stride: " << vol.stride(0) << ", " << vol.stride(1) << "\n";
    std::cout << "sino stride: " << sino.stride(0) << ", " << sino.stride(1) << "\n";

    std::cout << "angles: " << cpu_angles[0] << ", " << cpu_angles[1] << ", " << cpu_angles[2]
              << " ... " << cpu_angles[cpu_angles.size() - 2] << ", "
              << cpu_angles[cpu_angles.size() - 1] << "\n";

    curad::fp::forward_2d(vol.data(), curad_vol_shape, curad_vol_extent, curad_vol_spacing,
                          curad_vol_offset, sino.data(), det_shape, det_spacing, cpu_angles, DSD,
                          DSO);
}
