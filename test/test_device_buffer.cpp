#include "curadon/cuda/device_buffer.hpp"
#include "doctest/doctest.h"

TEST_CASE("Empty device_buffer") {
    curad::device_buffer buff;
    CHECK_UNARY(buff.is_empty());
}

TEST_CASE("Default Memory Resource") {
    const std::size_t size = 10;
    curad::device_buffer buff(size, curad::stream_view{});

    CHECK_NE(nullptr, buff.data());
    CHECK_EQ(size, buff.size());
    CHECK_EQ(size, buff.capacity());

    CHECK_EQ(curad::mr::get_current_device_resource(), buff.memory_resource());
    CHECK_EQ(curad::stream_view{}, buff.stream());
}
