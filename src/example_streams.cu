#include "curadon/cuda/stream.hpp"
#include "curadon/curadon.h"

#include <cstdio>
#include <iostream>
#include <vector>

#include "curadon/cuda/device_uvector.hpp"
#include "curadon/cuda/stream_view.hpp"

namespace curad {
template <class T>
class device_span {
  public:
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;

    __host__ __device__ device_span(T *ptr, std::size_t size) : ptr_{ptr}, size_{size} {}

    __host__ device_span(curad::device_uvector<T> &x) : ptr_{x.data()}, size_{x.size()} {}

    __host__ __device__ std::size_t size() const { return size_; }

    __host__ __device__ pointer device_data() { return ptr_; }

    __host__ __device__ const_pointer device_data() const { return ptr_; }

    __host__ __device__ reference operator[](std::size_t i) { return ptr_[i]; }

    __host__ __device__ const_reference operator[](std::size_t i) const { return ptr_[i]; }

    __host__ __device__ device_span<value_type> subspan(std::size_t offset,
                                                        std::size_t length) const {
        return device_span(ptr_ + offset, length);
    }

  private:
    T *ptr_;
    std::size_t size_;
};

using clock_value_t = long long;

__device__ void sleep(clock_value_t sleep_cycles) {
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do {
        cycles_elapsed = clock64() - start;
    } while (cycles_elapsed < sleep_cycles);
}

__global__ void cuda_hello(device_span<float> x) {
    const auto y = threadIdx.x;
    const auto stride = blockDim.x;
    const auto block = blockIdx.x;

    const auto i = block * stride + y;

    if (i < x.size()) {
        x[i] = 0;
        for (int j = 0; j < 20; ++j) {
            x[i] += 1;
        }

        sleep(100000);
    }
}

std::vector<float> test_kernel() {
    const std::size_t size = 1 << 20;

    std::vector<float> av(size, 10);
    std::vector<float> bv(size, 10);
    std::vector<float> cv(size, 2);

    const std::size_t nbytes = sizeof(float) * size;

    curad::stream stream_a;
    curad::stream stream_b;
    curad::stream stream_c;

    curad::device_uvector<float> a(size, stream_a);
    curad::device_uvector<float> b(size, stream_b);
    curad::device_uvector<float> c(size, stream_c);

    const auto block = 256;
    const auto grid = (size + block - 1) / block;

    device_span<float> a_span{a};
    cudaMemcpyAsync(a.data(), av.data(), nbytes, cudaMemcpyDefault, stream_a.value());
    cuda_hello<<<grid, block, 0, stream_a.value()>>>(a_span);

    device_span<float> b_span{b};
    cudaMemcpyAsync(b.data(), bv.data(), nbytes, cudaMemcpyDefault, stream_b.value());
    cuda_hello<<<grid, block, 0, stream_b.value()>>>(b_span);

    device_span<float> c_span{c};
    cudaMemcpyAsync(c.data(), cv.data(), nbytes, cudaMemcpyDefault, stream_c.value());
    cuda_hello<<<grid, block, 0, stream_c.value()>>>(c_span);

    stream_a.synchronize();
    stream_b.synchronize();
    stream_c.synchronize();

    std::vector<float> result(size);
    cudaMemcpy(result.data(), c.data(), nbytes, cudaMemcpyDefault);

    cudaDeviceSynchronize();

    return result;
}

// void forward(view volume, view sinogram) { cuda_hello<<<1, 1>>>(); }
//
// void backward(view sinogram, view volume) {}
} // namespace curad
