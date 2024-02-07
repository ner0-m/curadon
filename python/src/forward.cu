#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "curadon/detail/device_span.hpp"
#include "curadon/detail/plan/plan_2d.hpp"
#include "curadon/detail/vec.hpp"
#include "curadon/forward.hpp"
#include "curadon/image.hpp"
#include "curadon/measurement.hpp"
#include "curadon/types.hpp"

#include <vector>

#include <cuda_fp16.h>

#include "common.h"

namespace nb = nanobind;

using namespace nb::literals;

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

    if (volume.device_id() != sinogram.device_id()) {
        throw nb::attribute_error("Volume and sinogram must be on the same device");
    }

    auto vol_device = volume.device_id();
    auto sino_device = sinogram.device_id();

    // Never forgetti, Python calls (z, y, x), we do (x, y, z)
    curad::vec3u curad_vol_shape{vol_shape(2), vol_shape(1), vol_shape(0)};
    curad::vec3f curad_vol_spacing{vol_spacing(2), vol_spacing(1), vol_spacing(0)};
    curad::vec3f curad_vol_offset{vol_offset(2), vol_offset(1), vol_offset(0)};

    curad::device_volume<float> vol_span(vol_device, volume.data(), curad_vol_shape,
                                         curad_vol_spacing, curad_vol_offset);

    // never forgetti, python calls (v, u), we do (u, v)
    curad::vec3u curad_det_shape{det_shape(1), det_shape(0), phi.size()};
    curad::vec2f curad_det_spacing{det_spacing(1), det_spacing(0)};
    curad::vec2f curad_det_offset{det_offset(1), det_offset(0)};

    std::vector<float> cpu_psi(phi.data(), phi.data() + phi.size());
    std::vector<float> cpu_theta(theta.data(), theta.data() + theta.size());
    std::vector<float> cpu_phi(psi.data(), psi.data() + psi.size());

    curad::device_measurement<float> sino_span(sino_device, sinogram.data(), curad_det_shape,
                                               curad_det_spacing, curad_det_offset);
    sino_span.set_angles(cpu_psi, cpu_theta, cpu_phi);
    sino_span.set_distance_source_to_detector(DSD);
    sino_span.set_distance_source_to_object(DSO);
    sino_span.set_center_of_rotation_correction(COR);
    sino_span.set_pitch(det_rotation(0));
    sino_span.set_roll(det_rotation(1));
    sino_span.set_yaw(det_rotation(2));

    curad::fp::forward_3d(vol_span, sino_span);
}

template <class T, class U>
void forward_2d_cuda_typed(
    nb::ndarray<nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> vol,
    nb::ndarray<nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> sino,
    curad::forward_plan_2d &plan) {

    curad::usize device = vol.device_id();

    curad::device_span_2d<T> vol_span(device, (T *)vol.data(), plan.vol_shape());
    curad::device_span_2d<U> sino_span(device, (U *)sino.data(),
                                       {plan.det_count(), plan.nangles()});

    curad::fp::forward_2d<T, U>(vol_span, sino_span, plan);
}

void forward_2d_cuda(nb::ndarray<nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> vol,
                     nb::ndarray<nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> sino,
                     curad::forward_plan_2d &plan) {
    if (vol.device_id() != sino.device_id()) {
        throw nb::attribute_error("Volume and sinogram must be on the same device");
    }

    if (vol.device_id() != plan.device_id()) {
        throw nb::attribute_error("Volume and plan are not on the same device");
    }

    if (sino.device_id() != plan.device_id()) {
        throw nb::attribute_error("Sinogram and plan are not on the same device");
    }

    if (vol.dtype() == nb::dtype<curad::f32>()) {
        if (sino.dtype() == nb::dtype<curad::f32>()) {
            forward_2d_cuda_typed<curad::f32, curad::f32>(vol, sino, plan);
        }
    } else {
        throw nb::type_error("Only float32 is supported");
    }
}

void add_forward(nb::module_ &m) {
    m.def("forward_2d", &forward_2d_cuda, "volume"_a, "sinogram"_a, "plan"_a);

    m.def("forward_3d", &forward_3d_cuda, "x"_a, "vol_shape"_a, "vol_spacing"_a, "vol_offset"_a,
          "sinogram"_a, "psi"_a, "theta"_a, "phi"_a, "sino_shape"_a, "det_spacing"_a,
          "det_offset"_a, "det_rotation"_a, "DSO"_a, "DSD"_a, "COR"_a);
}
