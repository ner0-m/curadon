#pragma once

#include <cstdint>

namespace curad {
/// Simple std::span like abstraction, but way less features
template <class T>
class span {
  public:
    using value_type = T;
    using size_type = std::uint64_t;
    using difference_type = std::int64_t;
    using pointer = T *;
    using const_pointer = T const *;
    using reference = T &;
    using const_reference = T const &;

    span(T *data, size_type size)
        : data_(data)
        , size_(size) {}

    size_type size() const { return size_; }

    std::uint64_t nbytes() const { return sizeof(T) * size_; }

    pointer data() { return data_; }

    const_pointer data() const { return data_; }

    reference operator[](size_type i) { return data_[i]; }

    const_reference operator[](size_type i) const { return data_[i]; }

    span subspan(size_type offset, size_type size) { return span(data_ + offset, size); }

  private:
    pointer data_;
    size_type size_;
};
} // namespace curad
