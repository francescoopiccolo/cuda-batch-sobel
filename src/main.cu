#include "sobel.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <cerrno>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

constexpr int WIDTH = 128, HEIGHT = 128, COUNT = 256;
constexpr int PIXELS = WIDTH * HEIGHT;
using Clock = std::chrono::steady_clock;

void makeDirectory(const std::string& path) {
#ifdef _WIN32
    const int result = _mkdir(path.c_str());
#else
    const int result = mkdir(path.c_str(), 0755);
#endif
    if (result != 0 && errno != EEXIST)
        throw std::runtime_error("Cannot create directory: " + path);
}

std::string imagePath(const std::string& directory, int index) {
    std::ostringstream name;
    name << directory << "/image_" << std::setfill('0') << std::setw(3) << index << ".pgm";
    return name.str();
}

void writePGM(const std::string& path, const unsigned char* pixels) {
    std::ofstream file(path, std::ios::binary);
    file << "P5\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    file.write(reinterpret_cast<const char*>(pixels), PIXELS);
    if (!file) throw std::runtime_error("Cannot write: " + path);
}

// These files use a simple P5 header without comments, followed by 8-bit pixels.
void readPGM(const std::string& path, unsigned char* pixels) {
    std::ifstream file(path, std::ios::binary);
    std::string magic;
    int width = 0, height = 0, maximum = 0;
    file >> magic >> width >> height >> maximum;
    if (!file || magic != "P5" || width != WIDTH || height != HEIGHT || maximum != 255)
        throw std::runtime_error("Expected 128x128 P5 PGM without comments: " + path);
    const int separator = file.get();
    if (separator == '\r' && file.peek() == '\n') file.get();
    else if (separator != '\n' && separator != ' ' && separator != '\t')
        throw std::runtime_error("Invalid PGM separator: " + path);
    file.read(reinterpret_cast<char*>(pixels), PIXELS);
    if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Incorrect pixel count: " + path);
}

// Original, reproducible synthetic images: noisy gradients, rectangles and circles.
void generateDataset(const std::string& directory) {
    makeDirectory(directory);
    std::vector<unsigned char> image(PIXELS);
    for (int n = 0; n < COUNT; ++n) {
        uint32_t random = 1234u + n;
        const int cx = 30 + n % 65, cy = 30 + (n * 7) % 65, radius = 12 + n % 15;
        const int left = 10 + n % 35, top = 12 + (n * 3) % 40;
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                random = 1664525u * random + 1013904223u;
                int value = 20 + (2*x + y + n) % 50 + ((random >> 24) % 16);
                if (x >= left && x < left + 45 && y >= top && y < top + 30) value = 150;
                if ((x-cx)*(x-cx) + (y-cy)*(y-cy) <= radius*radius) value = 230;
                image[y * WIDTH + x] = static_cast<unsigned char>(value);
            }
        }
        writePGM(imagePath(directory, n), image.data());
    }
    std::cout << "Generated " << COUNT << " original 128x128 images in " << directory << '\n';
}

// Independent CPU reference using convolution masks, only for correctness checking.
void sobelCPU(const std::vector<unsigned char>& input, std::vector<unsigned char>& output) {
    const int kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    for (int n = 0; n < COUNT; ++n)
        for (int y = 1; y < HEIGHT - 1; ++y)
            for (int x = 1; x < WIDTH - 1; ++x) {
                int gx = 0, gy = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int value = input[n*PIXELS + (y+dy)*WIDTH + x+dx];
                        gx += value * kx[dy+1][dx+1];
                        gy += value * ky[dy+1][dx+1];
                    }
                output[n*PIXELS + y*WIDTH + x] = static_cast<unsigned char>(
                    std::min(255, std::abs(gx) + std::abs(gy)));
            }
}

int main(int argc, char** argv) {
    try {
        if (argc > 3) throw std::runtime_error("Usage: sobel [input_dir output_dir] | --generate [data_dir]");
        if (argc >= 2 && std::string(argv[1]) == "--generate") {
            generateDataset(argc == 3 ? argv[2] : "data");
            return 0;
        }
        const std::string inputDirectory = argc >= 2 ? argv[1] : "data";
        const std::string outputDirectory = argc == 3 ? argv[2] : "output";
        if (inputDirectory == outputDirectory)
            throw std::runtime_error("Use different input and output directories");
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) throw std::runtime_error("An NVIDIA CUDA GPU is required");
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
        const size_t bytes = static_cast<size_t>(PIXELS) * COUNT;
        std::vector<unsigned char> input(bytes), output(bytes), reference(bytes, 0);
        for (int n = 0; n < COUNT; ++n)
            readPGM(imagePath(inputDirectory, n), input.data() + n*PIXELS);
        const auto gpuStart = Clock::now();
        const float kernelMs = sobelGPU(input.data(), output.data(), WIDTH, HEIGHT, COUNT);
        const double gpuMs = std::chrono::duration<double, std::milli>(Clock::now() - gpuStart).count();
        const auto cpuStart = Clock::now();
        sobelCPU(input, reference);
        const double cpuMs = std::chrono::duration<double, std::milli>(Clock::now() - cpuStart).count();
        size_t mismatches = 0;
        for (size_t i = 0; i < bytes; ++i) mismatches += output[i] != reference[i];
        makeDirectory(outputDirectory);
        for (int n = 0; n < COUNT; ++n)
            writePGM(imagePath(outputDirectory, n), output.data() + n*PIXELS);
        std::ostringstream report;
        report << std::fixed << std::setprecision(3)
               << "CUDA batch Sobel edge detection\nGPU: " << device.name
               << "\nCompute capability: " << device.major << '.' << device.minor
               << "\nImages processed: " << COUNT << "\nImage size: " << WIDTH << 'x' << HEIGHT
               << "\nPixels processed: " << bytes << "\nInput: " << inputDirectory
               << "\nOutput: " << outputDirectory << "\nKernel time (CUDA events): " << kernelMs << " ms"
               << "\nGPU allocation + transfers + kernel + cleanup: " << gpuMs << " ms"
               << "\nCPU reference time: " << cpuMs << " ms"
               << "\nPixel mismatches: " << mismatches
               << "\nValidation: " << (mismatches == 0 ? "PASS" : "FAIL") << '\n';
        std::ofstream log(outputDirectory + "/run.log");
        log << report.str();
        log.close();
        if (!log) throw std::runtime_error("Cannot write run.log");
        std::cout << report.str();
        return mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
