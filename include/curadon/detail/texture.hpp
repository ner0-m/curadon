#pragma once

#include "curadon/device_span.hpp"

namespace curad::detail {
inline cudaArray_t allocate_cuarray(std::size_t width, std::size_t height, std::size_t depth) {
    const cudaExtent extent_alloc = make_cudaExtent(width, height, depth);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t array;
    cudaMalloc3DArray(&array, &channelDesc, extent_alloc);
    gpuErrchk(cudaPeekAtLastError());

    return array;
}

// TODO: this is very specific currently, make it more configurable (without exposing
// basically the same API again ;D)
inline void bind_texture_to_array(cudaTextureObject_t *tex, cudaArray_t array_cu) {
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = array_cu;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.addressMode[2] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(tex, &texRes, &texDescr, NULL);
    gpuErrchk(cudaPeekAtLastError());
}

template <class T>
inline void copy_projections_to_array(device_span_3d<T> data, cudaArray_t array_cu) {

    auto ptr = data.device_data();
    const auto width = data.shape()[0];
    const auto height = data.shape()[1];
    const auto width_bytes = sizeof(T) * width;
    const auto num_projections = data.shape()[2];

    cudaMemcpy3DParms copyParams = {0};

    copyParams.srcPtr = make_cudaPitchedPtr((void *)ptr, width_bytes, width, height);

    const cudaExtent extent = make_cudaExtent(width, height, num_projections);
    gpuErrchk(cudaPeekAtLastError());
    copyParams.dstArray = array_cu;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDefault;
    cudaMemcpy3DAsync(&copyParams, 0); // TODO: use stream pool
    cudaStreamSynchronize(0);
    gpuErrchk(cudaPeekAtLastError());
}
} // namespace curad::detail
