#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "curadon/detail/plan/plan_2d.hpp"
#include "curadon/detail/texture_cache.hpp"
#include "curadon/types.hpp"

namespace nb = nanobind;

using namespace nb::literals;

void forward_3d_cuda(
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any, nb::any>, nb::device::cuda, nb::c_contig> x,
    nb::ndarray<curad::u64, nb::shape<3>, nb::device::cpu> vol_shape,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> vol_spacing,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> vol_offset,
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any, nb::any>, nb::device::cuda, nb::c_contig> y,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> phi,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> theta,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> psi,
    nb::ndarray<curad::u64, nb::shape<2>, nb::device::cpu> det_shape,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> det_spacing,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> det_offset,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> det_rotation, curad::f32 DSO,
    curad::f32 DSD, curad::f32 COR);

void forward_2d_cuda(nb::ndarray<nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> x,
                     nb::ndarray<nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> y,
                     curad::forward_plan_2d &plan);

void backward_3d_cuda(
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any, nb::any>, nb::device::cuda, nb::c_contig> x,
    nb::ndarray<curad::u64, nb::shape<3>, nb::device::cpu> vol_shape,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> vol_spacing,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> vol_offset,
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any, nb::any>, nb::device::cuda, nb::c_contig> y,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> phi,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> theta,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> psi,
    nb::ndarray<curad::u64, nb::shape<2>, nb::device::cpu> det_shape,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> det_spacing,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> det_offset,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> det_rotation, curad::f32 DSO,
    curad::f32 DSD, curad::f32 COR);

void backward_2d_cuda(nb::ndarray<nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> vol,
                      nb::ndarray<nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> sino,
                      curad::forward_plan_2d &plan);

void add_texture(nb::module_ &m);

void add_stream(nb::module_ &m);

void add_plan(nb::module_ &m);

NB_MODULE(curadon_ext, m) {
    add_texture(m);

    add_stream(m);

    add_plan(m);

    m.def("forward_2d", &forward_2d_cuda, "volume"_a, "sinogram"_a, "plan"_a);

    m.def("backward_2d", &backward_2d_cuda, "volume"_a, "sinogram"_a, "plan"_a);

    m.def("backward_3d", &backward_3d_cuda, "x"_a, "vol_shape"_a, "vol_spacing"_a, "vol_offset"_a,
          "sinogram"_a, "phi"_a, "theta"_a, "psi"_a, "sino_shape"_a, "det_spacing"_a,
          "det_offset"_a, "det_rotation"_a, "DSO"_a, "DSD"_a, "COR"_a);

    m.def("forward_3d", &forward_3d_cuda, "x"_a, "vol_shape"_a, "vol_spacing"_a, "vol_offset"_a,
          "sinogram"_a, "psi"_a, "theta"_a, "phi"_a, "sino_shape"_a, "det_spacing"_a,
          "det_offset"_a, "det_rotation"_a, "DSO"_a, "DSD"_a, "COR"_a);
}
