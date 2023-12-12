#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <curadon/math/vector.hpp>

namespace curad {
template <typename T>
inline void write(std::ostream &stream, T const *data, std::size_t size, std::size_t width,
                  std::size_t height) {
    // P2: Magic number specifying grey scale, then the two dimensions in the next line
    // Then the maximum value of the image in our case always 255
    stream << "P2\n" << width << " " << height << "\n" << 255 << "\n";

    auto min_elem = *std::min_element(data, data + size);
    auto max_elem = *std::max_element(data, data + size);

    std::int64_t newmin = 0;
    std::int64_t newmax = 255;

    // write all image pixels
    for (int i = 0; i < size; ++i) {
        stream << static_cast<int>(
                      (newmax - newmin) / (max_elem - min_elem) * (data[i] - min_elem) + newmin)
               << " ";
        // stream << static_cast<int>(data[i] * scale) << " ";
    }
}

template <typename T>
inline void write(std::string filename, T const *data, std::size_t size, std::size_t width,
                  std::size_t height) {
    std::ofstream ofs(filename, std::ios_base::out);

    write(ofs, data, size, width, height);
}

inline std::tuple<std::vector<int>, std::size_t, std::size_t> read(std::istream &stream) {
    std::string magic_number;
    std::size_t width;
    std::size_t height;
    stream >> magic_number >> width >> height;

    if (magic_number != "P2") {
        throw std::runtime_error("Invalid magic number! Only P2 (ASCI gray values) supported");
    }

    auto size = width * height;
    std::vector<int> data;
    data.reserve(size);

    std::int64_t val;
    while (stream >> val) {
        data.push_back(val);
    }

    return {data, width, height};
}

inline std::tuple<std::vector<int>, std::size_t, std::size_t> read(std::string filename) {
    std::ifstream ofs(filename, std::ios_base::out);

    return read(ofs);
}

namespace easy {
inline std::tuple<std::vector<float>, std::size_t, std::size_t, std::size_t, std::vector<float>,
                  float, float>
read(std::istream &stream) {
    // parse type
    std::string aux;
    std::string type;
    stream >> aux >> type;

    if (type != "sino") {
        throw std::runtime_error("can only read type 'sino'");
    }

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

    std::cout << "DSO: " << DSO << "\n";
    // std::cout << "aux: " << aux << "\n";
    std::cout << "DSD: " << DSD << "\n";

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
read(std::string filename) {
    std::ifstream ofs(filename, std::ios_base::out);

    return read(ofs);
}
} // namespace easy

} // namespace curad
