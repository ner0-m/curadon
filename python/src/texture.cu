#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_map.h>

#include "curadon/detail/texture_cache.hpp"

namespace nb = nanobind;

void add_texture(nb::module_ &m) {
    nb::class_<curad::texture_cache>(m, "texture_cache").def(nb::init<curad::usize>());
}
