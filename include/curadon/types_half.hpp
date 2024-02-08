#pragma once

#include <cuda_fp16.h>

// This is only separately here, as I want to keep types.h clean from CUDA related
// issues to use it in binding code without the need to compile the bind file with NVCC.
// Maybe this is a bad decision, and I'll regret it, but then I'll change it

namespace curad {
using f16 = __half;
}
