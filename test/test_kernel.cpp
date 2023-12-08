#include "curadon/curadon.h"
#include "doctest/doctest.h"
#include <iostream>

TEST_CASE("kernel: some easy and simple kernel call test") {
    // Calling the kernel
    auto v = curad::test_kernel();

    // for (std::size_t i = 0; i < v.size(); ++i) {
    //     CAPTURE(i);
    //     REQUIRE_EQ(v[i], 20);
    // }

    // for(int i=0; i<5; ++i) {
    //     for(int j=0; j<5; ++j) {
    //         std::cout << v[i * 5 + j] << " ";
    //     }
    //     std::cout << "\n";
    // }
    // CHECK_EQ(0, 1);
}
