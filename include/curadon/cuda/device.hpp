#pragma once

#include "curadon/cuda/defines.h"
#include <cuda_runtime_api.h>

namespace curad {
struct cuda_device_id {
    using value_type = int;

    explicit constexpr cuda_device_id(value_type dev_id) noexcept : id_{dev_id} {}

    [[nodiscard]]
    constexpr value_type value() const noexcept {
        return id_;
    }

  private:
    value_type id_;
};

inline cuda_device_id current_device() {
    int dev_id{};
    CURADON_CUDA_TRY(cudaGetDevice(&dev_id));
    return cuda_device_id{dev_id};
}
} // namespace curad
