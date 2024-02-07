#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "curadon/detail/plan/plan_2d.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void add_plan(nb::module_ &m) {
    nb::class_<curad::forward_plan_2d>(m, "forward_plan_2d")
        // Ugly, but that's how it works see
        // https://nanobind.readthedocs.io/en/latest/porting.html#custom-constructors
        .def("__init__",
             [](curad::forward_plan_2d *plan, curad::usize device,
                nb::ndarray<curad::u64, nb::shape<2>, nb::device::cpu> py_vol_shape,
                nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> py_vol_spacing,
                nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> py_vol_offset,
                curad::u64 det_count, curad::f32 det_spacing, curad::f32 det_offset, curad::f32 DSO,
                curad::f32 DSD,
                nb::ndarray<curad::f32, nb::shape<nb::any>, nb::device::cuda> angles,
                curad::f32 det_pitch, curad::f32 COR) {
                 thrust::device_vector<curad::f32> owning_angles(angles.data(),
                                                                 angles.data() + angles.size());
                 curad::vec2u vol_shape{py_vol_shape(1), py_vol_shape(0)};
                 curad::vec2f vol_spacing{py_vol_spacing(1), py_vol_spacing(0)};
                 curad::vec2f vol_offset{py_vol_offset(1), py_vol_offset(0)};

                 new (plan) curad::forward_plan_2d(device, vol_shape, vol_spacing, vol_offset,
                                                   det_count, det_spacing, det_offset, DSO, DSD,
                                                   owning_angles, det_pitch, COR);
             })
        .def_rw("forward_block_x", &curad::forward_plan_2d::forward_block_x)
        .def_rw("forward_block_y", &curad::forward_plan_2d::forward_block_y);
}
