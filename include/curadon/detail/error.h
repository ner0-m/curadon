#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>

#define gpuErrchk(answer)                                                                          \
    { ::curad::cuda::detail::gpuAssert((answer), __FILE__, __LINE__); }

namespace curad::cuda::detail {
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: '%s: %s' (%d) in %s:%d\n", cudaGetErrorName(code),
                cudaGetErrorString(code), code, file, line);
        if (abort) {
            exit(code);
        }
    }
}
} // namespace curad::cuda::detail
