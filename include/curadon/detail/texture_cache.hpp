#pragma once

#include <cstddef>
#include <cstring>
#include <functional>
#include <unordered_map>

#include <cuda.h>
#include <cuda_runtime.h>

#include "curadon/detail/error.h"
#include "curadon/detail/texture.hpp"
#include "curadon/types.hpp"

namespace curad {

using texture_cache_key = std::tuple<void *, texture_config>;
} // namespace curad

namespace curad::detail {
template <class T>
inline void hash_combine(std::size_t &seed, const T &v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
} // namespace curad::detail

namespace std {
template <>
struct std::hash<curad::texture_config> {
    std::size_t operator()(const curad::texture_config &k) const {
        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:
        auto seed = std::hash<::curad::usize>{}(k.device_id);
        ::curad::detail::hash_combine(seed, k.width);
        ::curad::detail::hash_combine(seed, k.height);
        ::curad::detail::hash_combine(seed, k.depth);
        ::curad::detail::hash_combine(seed, k.is_layered);

        return seed;
    }
};

template <>
struct std::hash<curad::texture_cache_key> {
    std::size_t operator()(const curad::texture_cache_key &k) const {
        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:
        auto seed = std::hash<void *>{}(std::get<0>(k));
        ::curad::detail::hash_combine(seed, std::get<1>(k));

        return seed;
    }
};
} // namespace std

namespace curad {
class texture_cache {
  public:
    texture_cache(u64 cache_size)
        : cache_(cache_size) {}

    bool try_emplace(texture_cache_key key, texture_config config) {
        auto [_, inserted] = cache_.try_emplace(std::move(key), std::move(config));
        return inserted;
    }

    texture &at(const texture_cache_key &key) { return cache_.at(key); }

    const texture &at(const texture_cache_key &key) const { return cache_.at(key); }

  private:
    std::unordered_map<texture_cache_key, texture> cache_;
};

inline texture_cache &get_texture_cache() {
    static texture_cache cache(10);
    return cache;
}
} // namespace curad
