# CUDA Batch Sobel

A small image-processing project for **CUDA at Scale for the Enterprise**.
It applies Sobel edge detection to **256 grayscale images, each 128 x 128 pixels**
(4,194,304 pixels in total), using a custom CUDA kernel.

## What it does

1. Reads the 256 included PGM images into a contiguous batch.
2. Copies the batch to the GPU and launches one thread per pixel.
3. Computes horizontal and vertical Sobel gradients, then writes
   `min(255, abs(Gx) + abs(Gy))`. Border pixels are zero.
4. Copies the results back and checks every pixel against an independent CPU convolution.
5. Saves 256 edge images and `output/run.log`, including the GPU name, timings and validation result.

The CPU implementation is a correctness reference. Image processing uses the
CUDA kernel; there is no CPU fallback when a GPU is unavailable.

## Build and run in the lab

Requires an NVIDIA GPU, a compatible driver, the CUDA Toolkit (`nvcc`),
a supported C++ compiler, and GNU Make. No NPP, OpenCV or Python dependency.

```bash
make
./bin/sobel data output
```

Or run `bash run.sh` to build, record `nvidia-smi` output, and execute.
The program returns zero only when all GPU pixels match the CPU reference.
Run from the repository root. Input and output directories must be different.

To target a specific lab GPU, override the flags, for example for compute capability 7.5:

```bash
make NVCCFLAGS="-O2 -std=c++14 -arch=sm_75"
```

On Windows, use an x64 Visual Studio developer terminal with CUDA on `PATH`:

```bat
mkdir bin
nvcc -O2 -std=c++14 src/main.cu src/sobel.cu -o bin/sobel.exe
bin\sobel.exe data output
```

## Data and source files

The original synthetic dataset contains noisy gradients, rectangles and circles
with varying positions and sizes. All 256 input images are included in `data/`;
no external download or dataset license is needed. Regenerate them deterministically:

```bash
./bin/sobel --generate data
```

Files use binary PGM (`P5`), an 8-bit maximum of 255, and a header without comments.
The batch is deliberately fixed at 256 images of 128 x 128 pixels, named
`image_000.pgm` through `image_255.pgm`, to keep the code simple.

| File | Purpose |
| --- | --- |
| `src/main.cu` | PGM input/output, dataset generation, CPU check, saved log |
| `src/sobel.cu` | CUDA kernel, GPU memory transfers and CUDA event timing |
| `src/sobel.h` | Shared declaration and CUDA error checking |
| `data/` | All 256 original input images |
| `output/` | Actual GPU outputs and execution evidence |

## Execution evidence

See [output/run.log](output/run.log) for the actual run and validation,
[output/device.txt](output/device.txt) for GPU/driver information, and
[output/build.log](output/build.log) for the compiler command/version.
All 256 GPU-generated PGM outputs are included. A few input/output pairs are
shown in [output/comparison.png](output/comparison.png).

Kernel time uses CUDA events. The separately reported GPU operation time includes
allocations, transfers, launch and cleanup, but excludes disk I/O and CPU validation.
It is a single run, not a benchmark; timings vary by hardware and run.

After rerunning in the lab, preserve the new evidence in your public repository:

```bash
git add data output src README.md Makefile run.sh SUBMISSION.md
git commit -m "Save CUDA execution evidence"
git push
```

## Short project description

I implemented batch Sobel edge detection with a custom CUDA kernel in C++.
The program processes 256 original 128 x 128 grayscale images, using one GPU
thread per pixel and a three-dimensional grid to select each image. It copies
the results back to the host, saves the edge images and execution log, and
verifies all 4,194,304 output pixels against a CPU reference.
