#include <nanobind/nanobind.h>

#include "curadon/pool.hpp"

namespace nb = nanobind;

void add_stream(nb::module_ &m) {
    nb::class_<curad::cuda::event>(m, "event").def("record", &curad::cuda::event::record);

    nb::class_<curad::cuda::event_view>(m, "event_view")
        .def("record", &curad::cuda::event_view::record);

    nb::class_<curad::cuda::event_pool>(m, "event_pool")
        .def(nb::init<curad::usize>())
        .def("get_next_event", &curad::cuda::event_pool::get_next_event)
        .def("get_event", &curad::cuda::event_pool::get_event);

    m.def("get_next_event", curad::cuda::get_next_event);
    m.def("get_event", curad::cuda::get_event);

    nb::class_<curad::cuda::stream>(m, "stream")
        .def("synchronize", &curad::cuda::stream::synchronize)
        .def("wait_for_event", &curad::cuda::stream::wait_for_event);

    nb::class_<curad::cuda::stream_view>(m, "stream_view")
        .def("synchronize", &curad::cuda::stream_view::synchronize)
        .def("wait_for_event", &curad::cuda::stream_view::wait_for_event);

    nb::class_<curad::cuda::stream_pool>(m, "stream_pool")
        .def(nb::init<curad::usize>())
        .def("get_next_stream", &curad::cuda::stream_pool::get_next_stream)
        .def("get_stream", &curad::cuda::stream_pool::get_stream);

    m.def("get_next_stream", curad::cuda::get_next_stream);
    m.def("get_stream", curad::cuda::get_stream);
}
