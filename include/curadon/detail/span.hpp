#pragma once

#include <cstdint>

#include "curadon/types.hpp"

namespace curad {
/// Simple std::span like abstraction, but way less features
template <class T>
class span {
  public:
    using value_type = T;
    using size_type = u64;
    using difference_type = i64;
    using pointer = T *;
    using const_pointer = T const *;
    using reference = T &;
    using const_reference = T const &;

    span() = default;

    span(T *data, size_type size)
        : data_(data)
        , size_(size) {}

    size_type size() const { return size_; }

    u64 nbytes() const { return sizeof(T) * size_; }

    pointer data() { return data_; }

    const_pointer data() const { return data_; }

    reference operator[](size_type i) { return data_[i]; }

    const_reference operator[](size_type i) const { return data_[i]; }

    span subspan(size_type offset, size_type size) { return span(data_ + offset, size); }

  private:
    pointer data_ = nullptr;
    size_type size_= 0;
};
} // namespace curad
