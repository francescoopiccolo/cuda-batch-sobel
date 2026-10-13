#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void cudaCheck(cudaError_t status, const char* call) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// Input and output are contiguous batches of count grayscale images.
// Returns kernel-only elapsed time in milliseconds (CUDA events).
float sobelGPU(const unsigned char* input, unsigned char* output,
               int width, int height, int count);
