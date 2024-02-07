#pragma once

#include <cstdint>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

namespace curad {
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

#ifdef __CUDACC__
using f16 = __half;
#endif
using f32 = float;
using f64 = double;

using isize = std::intptr_t;
using usize = std::uintptr_t;
} // namespace curad
