#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <curadon/types.hpp>

namespace nb = nanobind;

using namespace nb::literals;

void forward_3d_cuda(
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any, nb::any>, nb::device::cuda, nb::c_contig> x,
    nb::ndarray<curad::u64, nb::shape<3>, nb::device::cpu> vol_shape,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> vol_spacing,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> vol_offset,
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any, nb::any>, nb::device::cuda, nb::c_contig> y,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> angles,
    nb::ndarray<curad::u64, nb::shape<2>, nb::device::cpu> det_shape,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> det_spacing,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> det_offset,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> det_rotation, curad::f32 DSO,
    curad::f32 DSD, curad::f32 COR);

void forward_2d_cuda(
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> x,
    nb::ndarray<curad::u64, nb::shape<2>, nb::device::cpu> vol_shape,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> vol_spacing,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> vol_offset,
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> y,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> angles, curad::u64 det_shape,
    curad::f32 det_spacing, curad::f32 det_offset, curad::f32 det_rotation, curad::f32 DSO,
    curad::f32 DSD, curad::f32 COR);

void backward_3d_cuda(
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any, nb::any>, nb::device::cuda, nb::c_contig> x,
    nb::ndarray<curad::u64, nb::shape<3>, nb::device::cpu> vol_shape,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> vol_spacing,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> vol_offset,
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any, nb::any>, nb::device::cuda, nb::c_contig> y,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> angles,
    nb::ndarray<curad::u64, nb::shape<2>, nb::device::cpu> det_shape,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> det_spacing,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> det_offset,
    nb::ndarray<curad::f32, nb::shape<3>, nb::device::cpu> det_rotation, curad::f32 DSO,
    curad::f32 DSD, curad::f32 COR);

void backward_2d_cuda(
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> x,
    nb::ndarray<curad::u64, nb::shape<2>, nb::device::cpu> vol_shape,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> vol_spacing,
    nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> vol_offset,
    nb::ndarray<curad::f32, nb::shape<nb::any, nb::any>, nb::device::cuda, nb::c_contig> y,
    nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cpu> angles, curad::u64 det_shape,
    curad::f32 det_spacing, curad::f32 det_offset, curad::f32 det_rotation, curad::f32 DSO,
    curad::f32 DSD, curad::f32 COR);

NB_MODULE(curadon_ext, m) {
    m.def("forward_3d", &forward_3d_cuda, "x"_a, "vol_shape"_a, "vol_spacing"_a, "vol_offset"_a,
          "sinogram"_a, "angles"_a, "sino_shape"_a, "det_spacing"_a, "det_offset"_a,
          "det_rotation"_a, "DSO"_a, "DSD"_a, "COR"_a);

    m.def("forward_2d", &forward_2d_cuda, "x"_a, "vol_shape"_a, "vol_spacing"_a, "vol_offset"_a,
          "sinogram"_a, "angles"_a, "sino_shape"_a, "det_spacing"_a, "det_offset"_a,
          "det_rotation"_a, "DSO"_a, "DSD"_a, "COR"_a);

    m.def("backward_3d", &backward_3d_cuda, "x"_a, "vol_shape"_a, "vol_spacing"_a, "vol_offset"_a,
          "sinogram"_a, "angles"_a, "sino_shape"_a, "det_spacing"_a, "det_offset"_a,
          "det_rotation"_a, "DSO"_a, "DSD"_a, "COR"_a);

    m.def("backward_2d", &backward_2d_cuda, "x"_a, "vol_shape"_a, "vol_spacing"_a, "vol_offset"_a,
          "sinogram"_a, "angles"_a, "det_shape"_a, "det_spacing"_a, "det_offset"_a,
          "det_rotation"_a, "DSO"_a, "DSD"_a, "COR"_a);
}
