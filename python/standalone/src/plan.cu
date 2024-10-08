#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "curadon/detail/plan/plan_2d.hpp"

namespace nanobind {
template <>
struct ndarray_traits<curad::f16> {
    static constexpr bool is_complex = false;
    static constexpr bool is_float = true;
    static constexpr bool is_bool = false;
    static constexpr bool is_int = false;
    static constexpr bool is_signed = true;
};
}; // namespace nanobind

namespace nb = nanobind;
using namespace nb::literals;

curad::precision dtype_to_precision(curad::u32 dtype) {
    if (dtype == 32) {
        return curad::precision::SINGLE;
    } else if (dtype == 16) {
        return curad::precision::HALF;
    } else {
        throw nb::attribute_error("Precision must either be f32 or f16");
    }
}

void add_plan(nb::module_ &m) {
    nb::class_<curad::plan_2d>(m, "plan_2d")
        // Ugly, but that's how it works see
        // https://nanobind.readthedocs.io/en/latest/porting.html#custom-constructors
        .def("__init__",
             [](curad::plan_2d *plan, curad::usize device, curad::u32 vol_prec,
                nb::ndarray<curad::u64, nb::shape<2>, nb::device::cpu> py_vol_shape,
                nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> py_vol_spacing,
                nb::ndarray<curad::f32, nb::shape<2>, nb::device::cpu> py_vol_offset,
                curad::u32 det_prec, curad::u64 det_count, curad::f32 det_spacing,
                curad::f32 det_offset, nb::ndarray<curad::f32, nb::shape<-1>, nb::device::cpu> DSO,
                nb::ndarray<curad::f32, nb::shape<-1>, nb::device::cpu> DSD,
                nb::ndarray<curad::f32, nb::shape<-1>, nb::device::cpu> angles,
                curad::f32 det_pitch, curad::f32 COR) {
                 thrust::device_vector<curad::f32> gpu_angles(angles.data(),
                                                              angles.data() + angles.size());
                 thrust::device_vector<curad::f32> gpu_dso(DSO.data(), DSO.data() + DSO.size());
                 thrust::device_vector<curad::f32> gpu_dsd(DSD.data(), DSD.data() + DSD.size());
                 curad::vec2u vol_shape{py_vol_shape(1), py_vol_shape(0)};
                 curad::vec2f vol_spacing{py_vol_spacing(1), py_vol_spacing(0)};
                 curad::vec2f vol_offset{py_vol_offset(1), py_vol_offset(0)};

                 new (plan) curad::plan_2d(device, dtype_to_precision(vol_prec), vol_shape,
                                           vol_spacing, vol_offset, dtype_to_precision(det_prec),
                                           det_count, det_spacing, det_offset, gpu_dso, gpu_dsd,
                                           gpu_angles, det_pitch, COR);
             })
        .def_rw("forward_block_x", &curad::plan_2d::forward_block_x)
        .def_rw("forward_block_y", &curad::plan_2d::forward_block_y);
}
