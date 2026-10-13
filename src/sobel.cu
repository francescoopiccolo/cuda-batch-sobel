#include "sobel.h"

// One thread per pixel; grid.z selects the image in the batch.
__global__ void sobelKernel(const unsigned char* input, unsigned char* output,
                            int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int i = blockIdx.z * width * height + y * width + x;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        output[i] = 0;
        return;
    }
    const int gx = -input[i-width-1] + input[i-width+1]
                   -2*input[i-1] + 2*input[i+1]
                   -input[i+width-1] + input[i+width+1];
    const int gy = -input[i-width-1] - 2*input[i-width] - input[i-width+1]
                   +input[i+width-1] + 2*input[i+width] + input[i+width+1];
    const int magnitude = abs(gx) + abs(gy);
    output[i] = static_cast<unsigned char>(magnitude > 255 ? 255 : magnitude);
}

float sobelGPU(const unsigned char* input, unsigned char* output,
               int width, int height, int count) {
    const size_t bytes = static_cast<size_t>(width) * height * count;
    unsigned char *deviceInput = nullptr, *deviceOutput = nullptr;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaMemcpy(deviceInput, input, bytes, cudaMemcpyHostToDevice));
    const dim3 block(16, 16);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, count);
    CUDA_CHECK(cudaEventRecord(start));
    sobelKernel<<<grid, block>>>(deviceInput, deviceOutput, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaMemcpy(output, deviceOutput, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return milliseconds;
}
