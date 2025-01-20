# CUDA Multiplication Benchmark

This project demonstrates a simple CUDA benchmark comparing CPU vs GPU performance for repetitive multiplication operations.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with version 11.0+)
- CMake (version 3.18 or higher)
- C++ compiler (supporting C++14)

## Building the Project

1. Clone the repository:
```bash
git clone https://github.com/tsilva/sandbox-cuda.git
cd sandbox-cuda
```

2. Build the project:
```bash
cmake ..
make
```

The executable will be created in the `bin` directory at the project root.

## Running the Benchmark

After building, run the benchmark executable:
```bash
../bin/multiply_benchmark
```

The program will output:
- GPU execution time
- CPU execution time
- Speedup factor (CPU time / GPU time)
- First and last elements computed by both CPU and GPU for verification

## Modifying Parameters

You can modify the following constants in `src/multiply_benchmark.cu` to experiment with different workloads:

- `NUM_ELEMENTS`: Number of parallel computations (default: 1,000,000)
- `ITERATIONS`: Number of multiplications per element (default: 1,000)
- `VALUE`: The number being multiplied (default: 1.001f)

## Understanding the Code

The benchmark performs the following operations:
1. Allocates memory on both CPU and GPU
2. Executes the same multiplication task on both devices
3. Measures and compares execution time
4. Verifies results match between CPU and GPU implementations
