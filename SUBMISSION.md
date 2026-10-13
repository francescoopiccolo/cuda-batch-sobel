# CUDA Batch Sobel submission

**Repository:** https://github.com/francescoopiccolo/cuda-batch-sobel

**Execution proof:** `output/run.log`, `output/device.txt`, `output/build.log`,
`output/comparison.png`, and all 256 output PGM images.

## Short description

I built a simple CUDA program that applies Sobel edge detection to 256 grayscale
images of 128 x 128 pixels. The dataset contains original synthetic images with
noisy backgrounds, rectangles and circles. A custom CUDA kernel uses one thread
per pixel to compute horizontal and vertical gradients across the entire batch.
The program transfers the edge images back to the CPU and saves them as PGM files.

I ran the project on an NVIDIA GeForce GTX 1650 Ti with CUDA Toolkit 11.8.
All 4,194,304 output pixels matched an independent CPU reference exactly.
The saved run recorded a kernel time of 0.292 ms and a GPU operation time of
85.831 ms including allocation, transfers and cleanup. These are single-run
measurements; the project demonstrates GPU processing and correctness rather
than claiming an overall speedup. The implementation is two `.cu` files and
one `.h` file, with no external image-processing dependencies.

## Reproduce

```bash
make
./bin/sobel data output
```

Expected validation: `Pixel mismatches: 0` and `Validation: PASS`.
The command requires an NVIDIA CUDA GPU and the CUDA Toolkit.
