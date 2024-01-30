#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "curadon/detail/image_2d.hpp"
#include "curadon/detail/texture_cache.hpp"
#include "curadon/detail/vec.hpp"
#include "curadon/forward.hpp"
#include "curadon/image.hpp"
#include "curadon/measurement.hpp"
#include "curadon/types.hpp"

#include <vector>

#include <cuda_fp16.h>

#include "common.h"

namespace nb = nanobind;

void forward_3d_cuda(
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any, nb::any>, nb::device::cuda, nb::c_contig>
        volume,
    nb::ndarray<curad::u64, nb::shape<3>, nb::device::cpu> vol_shape,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> vol_spacing,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> vol_offset,
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any, nb::any>, nb::device::cuda, nb::c_contig>
        sinogram,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> phi,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> theta,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> psi,
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
    curad::vec3u curad_det_shape{det_shape(1), det_shape(0), phi.size()};
    curad::vec2f curad_det_spacing{det_spacing(1), det_spacing(0)};
    curad::vec2f curad_det_offset{det_offset(1), det_offset(0)};

    std::vector<float> cpu_psi(phi.data(), phi.data() + phi.size());
    std::vector<float> cpu_theta(theta.data(), theta.data() + theta.size());
    std::vector<float> cpu_phi(psi.data(), psi.data() + psi.size());

    curad::device_measurement<float> sino_span(sinogram.data(), curad_det_shape, curad_det_spacing,
                                               curad_det_offset);
    sino_span.set_angles(cpu_psi, cpu_theta, cpu_phi);
    sino_span.set_distance_source_to_detector(DSD);
    sino_span.set_distance_source_to_object(DSO);
    sino_span.set_center_of_rotation_correction(COR);
    sino_span.set_pitch(det_rotation(0));
    sino_span.set_roll(det_rotation(1));
    sino_span.set_yaw(det_rotation(2));

    curad::fp::forward_3d(vol_span, sino_span);
}

void forward_2d_cuda(nb::ndarray<nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> vol,
                     nb::ndarray<curad::u64, nb::shape<2>, nb::device::cpu> vol_shape,
                     nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> vol_spacing,
                     nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> vol_offset,
                     nb::ndarray<nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> sino,
                     nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cuda> angles,
                     curad::u64 det_shape, curad::f32 det_spacing, curad::f32 det_offset,
                     curad::f32 det_rotation, curad::f32 DSO, curad::f32 DSD, curad::f32 COR,
                     curad::texture_cache &tex_cache) {
    // TODO: make dispatch possible based on types
    if (vol.dtype() != nb::dtype<curad::f32>()) {
        throw nb::type_error("Input image must for of type float32");
    }

    if (sino.dtype() != nb::dtype<curad::f32>()) {
        throw nb::type_error("Input sinogram must for of type float32");
    }

    // Never forgetti, Python calls (z, y, x), we do (x, y, z)
    curad::vec2u curad_vol_shape{vol_shape(1), vol_shape(0)};
    curad::vec2f curad_vol_spacing{vol_spacing(1), vol_spacing(0)};
    curad::vec2f curad_vol_offset{vol_offset(1), vol_offset(0)};
    curad::vec2f curad_vol_extent = curad_vol_shape * curad_vol_spacing;

    curad::image_2d<curad::f32> vol_span((curad::f32 *)vol.data(), curad_vol_shape,
                                         curad_vol_spacing, curad_vol_offset);

    curad::measurement_2d<curad::f32> sino_span((curad::f32 *)sino.data(), det_shape, angles.size(),
                                                det_spacing, det_offset);

    curad::span<curad::f32> angles_span(angles.data(), angles.size());
    sino_span.set_angles(angles_span);
    sino_span.set_distance_source_to_object(DSO);
    sino_span.set_distance_source_to_detector(DSD);
    sino_span.set_center_of_rotation_correction(COR);
    sino_span.set_pitch(det_rotation);

    curad::fp::forward_2d(vol_span, sino_span, tex_cache);
}
