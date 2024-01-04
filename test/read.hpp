#pragma once

#include <istream>
#include <fstream>
#include <vector>

#include "curadon/math/vector.hpp"

namespace curad {
namespace easy {
namespace detail {
inline std::tuple<std::vector<float>, std::size_t, std::size_t, std::size_t, std::vector<float>,
                  float, float>
read_sino(std::istream &stream) {
    std::string aux;

    std::size_t width;
    std::size_t height;
    stream >> aux >> width >> height;

    std::size_t nangles;
    stream >> aux >> nangles;

    std::vector<float> angles;
    angles.reserve(nangles);
    stream >> aux;
    for (int i = 0; i < nangles; ++i) {
        float angle;
        stream >> angle;
        angles.push_back(angle);
    }

    float DSO;
    stream >> aux >> DSO;
    float DSD;
    stream >> aux >> DSD;

    auto size = width * height * nangles;
    std::vector<float> data;
    data.reserve(size);

    float val;
    while (stream >> val) {
        data.push_back(val);
    }

    return {data, width, height, nangles, angles, DSO, DSD};
}

inline std::tuple<std::vector<float>, std::size_t, std::size_t, std::size_t, std::vector<float>,
                  float, float>
read_vol(std::istream &stream) {
    std::string aux;

    std::size_t width;
    std::size_t height;
    std::size_t depth;
    stream >> aux >> width >> height >> depth;

    auto size = width * height * depth;
    std::vector<float> data;
    data.reserve(size);

    float val;
    while (stream >> val) {
        data.push_back(val);
    }

    return {data, width, height, depth, std::vector<float>{}, 0, 0};
}
} // namespace detail
inline std::tuple<std::vector<float>, std::size_t, std::size_t, std::size_t, std::vector<float>,
                  float, float>
read(std::istream &stream) {
    // parse type
    std::string aux;
    std::string type;
    stream >> aux >> type;

    if (type == "sino") {
        return detail::read_sino(stream);
    } else if (type == "vol") {
        return detail::read_vol(stream);
    } else {
        throw std::runtime_error("Unknown type to read from");
    }
}

inline std::tuple<std::vector<float>, std::size_t, std::size_t, std::size_t, std::vector<float>,
                  float, float>
read(std::string filename) {
    std::ifstream ofs(filename, std::ios_base::out);

    return read(ofs);
}
} // namespace easy

} // namespace curad
