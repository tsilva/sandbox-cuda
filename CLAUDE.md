# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CUDA benchmark project that compares CPU vs GPU performance for repetitive multiplication operations. The project consists of a single CUDA executable (`multiply_benchmark`) that demonstrates parallel computing capabilities.

## Build System

The project uses CMake with CUDA support:

```bash
# Standard build workflow
cmake ..
make

# Clean build
rm -rf bin/
cmake ..
make
```

Output executables are placed in `bin/` at the project root (configured via `CMAKE_RUNTIME_OUTPUT_DIRECTORY`).

### Build Configuration

- CMake minimum version: 3.16
- CUDA and C++ standard: C++14
- CUDA architectures: 52 and 72 (Maxwell and Volta/Turing)
- Build target: `multiply_benchmark` from `src/multiply_benchmark.cu`

## Running the Benchmark

```bash
# From build directory
../bin/multiply_benchmark

# From project root
./bin/multiply_benchmark
```

No command-line arguments are supported; parameters must be modified in source code.

## Code Architecture

### Single-File Structure

The entire benchmark is contained in `src/multiply_benchmark.cu` with three main components:

1. **GPU Kernel** (`multiplyKernel`): CUDA kernel that performs parallel multiplications across thread blocks
2. **CPU Implementation** (`multiplyCPU`): Sequential reference implementation for comparison
3. **Main Function**: Orchestrates memory allocation, kernel execution, timing, and result verification

### Benchmark Parameters

Three compile-time constants control the workload (lines 29-31 in multiply_benchmark.cu):
- `NUM_ELEMENTS`: Number of parallel computations (affects grid size)
- `ITERATIONS`: Multiplication operations per thread/element (affects compute intensity)
- `VALUE`: The floating-point multiplier

### CUDA Execution Configuration

- Block size: 256 threads per block
- Grid size: Calculated as `(NUM_ELEMENTS + blockSize - 1) / blockSize`
- Thread indexing: `blockIdx.x * blockDim.x + threadIdx.x`

### Memory Management

The code follows standard CUDA patterns:
- Host memory: Allocated with `new[]` for both GPU and CPU results
- Device memory: Allocated with `cudaMalloc`
- Data transfer: `cudaMemcpy` with `cudaMemcpyDeviceToHost` after kernel execution
- Synchronization: `cudaDeviceSynchronize()` ensures kernel completion before timing measurement

## Important Notes

- The benchmark measures total execution time including kernel launch overhead but excludes memory transfer time in the GPU measurement
- Results are verified by comparing first and last elements between CPU and GPU implementations
- No error checking is implemented for CUDA API calls
- README.md must be kept up to date with any significant project changes
