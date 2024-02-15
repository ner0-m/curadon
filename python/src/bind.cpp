#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "curadon/types.hpp"

namespace curad {
class plan_2d;
}

namespace nb = nanobind;

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
                     curad::plan_2d &plan);

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
                      curad::plan_2d &plan);

void add_stream(nb::module_ &m);

void add_plan(nb::module_ &m);

void add_forward(nb::module_ &m);

void add_backward(nb::module_ &m);

NB_MODULE(curadon_ext, m) {
    add_stream(m);

    add_plan(m);

    add_forward(m);

    add_backward(m);
}
