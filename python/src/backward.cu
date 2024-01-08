#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "curadon/backward.hpp"
#include "curadon/detail/vec.hpp"
#include "curadon/image.hpp"
#include "curadon/measurement.hpp"
#include "curadon/types.hpp"

#include <vector>

namespace nb = nanobind;

void backward_3d_cuda(
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

    curad::bp::backproject_3d(vol_span, sino_span);
}

void backward_2d_cuda(
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

    curad::image_2d<curad::f32> vol_span(vol.data(), curad_vol_shape, curad_vol_spacing,
                                         curad_vol_offset);

    // TODO: Why do I need the source really here?,
    const auto source = curad::vec2f{0, -DSO};

    // TODO: make this call the same as forward_2d (both order and types of arguments)
    // TODO: Remove vol_extent, just compute it your-freaking-self :D
    curad::bp::backproject_2d(vol_span, sino.data(), det_shape, DSD, DSO, source, cpu_angles);
}
